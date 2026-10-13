"""MNIST training evidence schema, capture and guarded publication.

CPU fixtures accepted by :func:`validate_structure` are structural test inputs,
not evidence of Vulkan execution.  Capture is the only wrapper-first runtime path.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import warnings

import torch

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = (
    "mnist.synthetic.batch1.sgd-none", "mnist.synthetic.batch1.sgd-zero",
    "mnist.synthetic.batch2.sgd-none", "mnist.synthetic.batch2.sgd-zero",
    "mnist.synthetic.batch16.sgd-none", "mnist.synthetic.batch16.sgd-zero",
    "mnist.real.subset257-129.sgd-none", "mnist.real.subset257-129.sgd-zero",
)
MODES = ("async", "sync")
DOCUMENT_KEYS = {"schema_version", "pytorch_version", "pytorch_git_revision", "records"}
RECORD_KEYS = {"scenario_id", "execution_mode", "recipe", "runtime", "source_identity",
               "reference_receipt", "data_identity", "steps", "evaluation",
               "phase_counters", "parity", "raw_receipt"}
PARAMETERS = ("1.weight", "1.bias", "3.weight", "3.bias")
SHAPES = {"1.weight": [32, 784], "1.bias": [32], "3.weight": [10, 32], "3.bias": [10]}
RTOL_ATOL = {"logits": (2e-4, 2e-4), "loss": (2e-4, 2e-4),
             "gradients": (2e-4, 2e-4), "momentum_buffer": (2e-4, 2e-4),
             "parameters": (1e-4, 1e-5)}
REFERENCE = ROOT / ".superpowers/sdd/2026-10-12-vulkan-mnist-cross-entropy-training/task-1-cpu-reference-fix1.json"
APPROVED_REFERENCE_SHA256 = "61174bd5290b61d256c215ec402f2bfdb62257a69c0864b2ed0aeaf06e224927"
OPTIMIZER = {"lr": .01, "momentum": .9, "weight_decay": 0, "dampening": 0,
             "nesterov": False, "maximize": False, "foreach": None,
             "differentiable": False, "fused": None}
ENVIRONMENT_KEYS = {"PYTORCH_VULKAN_ASYNC_EXECUTION", "PYTHONPATH", "VK_INSTANCE_LAYERS",
                    "PYTORCH_VULKAN_MNIST_ROOT"}
FRAGMENT_FILE = "capture.json"
RECEIPT_FILE = "receipt.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path):
    def fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON literal {value}")

    return json.loads(path.read_text(), object_pairs_hook=fields, parse_constant=invalid_constant)


def source_paths(repo_root=None):
    """Complete current semantic/control inventory for this Task 5 capture."""
    from composed_evidence_common import DEPENDENCIES
    paths = set(DEPENDENCIES)
    paths.update((
        "CMakeLists.txt",
        "src/vulkan/shaders/glsl/linear_relu_backward.comp",
        "src/vulkan/shaders/generated/linear_relu_backward_spv.h",
        "src/vulkan/shaders/generated/linear_relu_backward_spv.sha256",
        "tools/generate_linear_relu_backward_spv.py", "tools/verify_linear_relu_backward_spv.py",
        "tools/verify_operator_spv.py",
        "src/vulkan/operators/classification.cpp", "src/vulkan/operators/classification.h",
        "src/vulkan/shaders/glsl/classification.comp", "src/vulkan/shaders/glsl/normalization.comp",
        "tools/generate_operator_spv.py", "src/vulkan/descriptor_arena.cpp",
        "src/vulkan/descriptor_arena.h", "src/vulkan/pipeline_cache.cpp", "src/vulkan/pipeline_cache.h",
        "tests/python/mnist_training_cases.py", "tests/python/mnist_training_parity.py",
        "tests/python/test_mnist_training_accounting.py", "tests/python/test_mnist_training_reference.py",
        "tests/python/test_vulkan_mnist_training.py", "tests/manual_vulkan_mnist_training.py",
        "tests/python/test_mnist_training_evidence.py", "tests/python/mnist_training_evidence.py",
        "tools/mnist_data.py", "tools/prepare_mnist_data.py", "tools/probe_mnist_training_reference.py",
        "tools/validate_mnist_training_evidence.py", "tools/mnist_training_coverage.py",
        "tools/validate_vulkan_bootstrap_provenance.py",
    ))
    root = ROOT if repo_root is None else Path(repo_root)
    # Names are declared before Task 6 exists; absent files are never hashed.
    # Once either control exists both are required, invalidating prior captures.
    future = ("tools/run_mnist_training_qualification.py", "tests/python/test_mnist_training_qualification.py")
    if any((root / p).exists() for p in future):
        paths.update(future)
        paths.update(("tools/run_vulkan_qualification.py", "tools/shader_toolchain.py",
                      "tools/shader-toolchain-bin/glslc"))
    paths.update(("python/pytorch_vulkan/compiler.py", "python/pytorch_vulkan/compiler_metadata.py",
                  "python/pytorch_vulkan/serialization.py"))
    absent = sorted(p for p in paths if not (root / p).is_file())
    if absent:
        raise ValueError("source inventory has missing paths: " + ", ".join(absent))
    return tuple(sorted(paths))


def _tensor(value, expected_device, path, *, expected_shape=None, requires_grad=False, history=False):
    keys = {"values", "shape", "dtype", "device", "requires_grad", "history"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{path}: invalid tensor snapshot fields")
    if value["device"] != expected_device:
        raise ValueError(f"{path}: wrong tensor device")
    if type(value["shape"]) is not list or any(type(x) is not int or x < 0 for x in value["shape"]):
        raise ValueError(f"{path}: invalid tensor shape")
    if expected_shape is not None and value["shape"] != expected_shape:
        raise ValueError(f"{path}: wrong tensor shape")
    if value["dtype"] != "torch.float32" or type(value["requires_grad"]) is not bool or type(value["history"]) is not bool:
        raise ValueError(f"{path}: invalid tensor metadata")
    if value["requires_grad"] is not requires_grad or value["history"] is not history:
        raise ValueError(f"{path}: invalid gradient history metadata")
    if type(value["values"]) is not list or len(value["values"]) != math.prod(value["shape"]):
        raise ValueError(f"{path}: tensor values/shape mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value["values"]):
        raise ValueError(f"{path}: nonfinite tensor")


def _hash(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label}: invalid SHA256")


def _exact(actual, expected, label):
    """Exact JSON types and values (bool is not an int); recursively allowlist fields."""
    if type(actual) is not type(expected):
        raise ValueError(f"{label}: wrong type")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: wrong fields")
        for key in expected:
            _exact(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label}: wrong length")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _exact(a, b, f"{label}.{index}")
    elif actual != expected:
        raise ValueError(f"{label}: differs from source reference")


def _compare_tensor_pair(cpu, vk, path):
    if cpu["shape"] != vk["shape"] or cpu["dtype"] != vk["dtype"]:
        raise ValueError(f"parity shape/dtype mismatch at {path}")
    rtol, atol = RTOL_ATOL["logits"]
    for a, b in zip(cpu["values"], vk["values"]):
        if abs(a - b) > atol + rtol * abs(a):
            raise ValueError(f"numerical parity tolerance exceeded at {path}")


def _expected_phases(real, updates):
    phases = ["setup.model-staging"]
    for step in range(updates):
        phases.extend(f"step-{step}.{suffix}" for suffix in (
            "staging", "reset", "reset-diagnostics", "forward-backward",
            "backward-diagnostics", "update", "update-diagnostics"))
        if real and step in (16, 33):
            epoch = 0 if step == 16 else 1
            for batch in range(9):
                phases.extend((f"epoch-{epoch}.evaluation-staging-{batch}",
                               f"epoch-{epoch}.evaluation-{batch}",
                               f"epoch-{epoch}.evaluation-diagnostics-{batch}",
                               f"epoch-{epoch}.evaluation-readback-{batch}"))
    return phases


def _phase_category(name):
    if name.startswith("setup."): return "setup"
    if "diagnostics" in name: return "diagnostics"
    if "staging" in name: return "staging"
    if "readback" in name: return "readback"
    if name.endswith(".reset"): return "reset"
    if name.endswith(".forward-backward") or name.endswith(".update") or ".evaluation-" in name:
        return "compute"
    return "other"


def _validate_data(data, real):
    if not real:
        if data != {"kind": "synthetic"}:
            raise ValueError("invalid synthetic data identity")
        return
    if type(data) is not dict or set(data) != {"train", "test"}:
        raise ValueError("invalid data identity")
    canonical_md5 = {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }
    for split, count in (("train", 257), ("test", 129)):
        item = data[split]
        if type(item) is not dict or set(item) != {"dataset", "split", "count", "indices", "label_histogram", "input_digest_sha256", "data_identity"}:
            raise ValueError("invalid data identity")
        for key, expected in (("dataset", "MNIST"), ("split", split), ("count", count), ("indices", list(range(count)))):
            _exact(item[key], expected, f"data identity.{key}")
        hist = item["label_histogram"]
        if type(hist) is not dict or set(hist) != set(map(str, range(10))) or any(type(v) is not int or v < 0 for v in hist.values()) or sum(hist.values()) != count:
            raise ValueError("invalid data identity histogram")
        _hash(item["input_digest_sha256"], "data digest")
        identity = item["data_identity"]
        if type(identity) is not dict or set(identity) != {"canonical_resource_table", "resources"} or identity["canonical_resource_table"] != "torchvision-v0.19.0-MNIST":
            raise ValueError("invalid data identity resources")
        resources = identity["resources"]
        names = {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"}
        if type(resources) is not dict or set(resources) != names:
            raise ValueError("invalid data identity resources")
        for name, resource in resources.items():
            if type(resource) is not dict or set(resource) != {"md5", "sha256", "compressed_bytes"} or type(resource["md5"]) is not str or len(resource["md5"]) != 32 or type(resource["compressed_bytes"]) is not int or resource["compressed_bytes"] <= 0:
                raise ValueError("invalid data identity resource")
            _hash(resource["sha256"], "resource hash")
            if resource["md5"] != canonical_md5[name]:
                raise ValueError("invalid canonical MNIST MD5")
    _exact(data["train"]["data_identity"], data["test"]["data_identity"], "common canonical resource identity")


def _recipe(sid):
    real = sid.startswith("mnist.real.")
    batch = 16 if real else int(sid.split("batch")[1].split(".")[0])
    recipe = {"kind": "real" if real else "synthetic", "batch_size": batch,
              "set_to_none": sid.endswith("sgd-none"), "model_seed": 12012,
              "updates": 34 if real else 3, "optimizer": dict(OPTIMIZER),
              "train_batch_sizes": ([16] * 16 + [1]) * 2 if real else [batch] * 3}
    if real:
        recipe.update({"epochs": 2, "evaluation_batch_sizes": [[16] * 8 + [1]] * 2,
                       "shuffle": False, "drop_last": False, "num_workers": 0, "scale_divisor": 255})
    else:
        recipe.update({"image_seed": 12013 + batch, "labels": [9] if batch == 1 else
                       [0, 9] if batch == 2 else [i % 10 for i in range(16)]})
    return recipe


def validate_structure(document: dict) -> None:
    _validate_document(document, {(s, m) for s in SCENARIOS for m in MODES})


def validate_fragment(fragment: dict) -> None:
    """Exact 6/2-record stage schema; cannot satisfy complete-document validation."""
    keys = DOCUMENT_KEYS | {"stage", "execution_mode"}
    if type(fragment) is not dict or set(fragment) != keys:
        raise ValueError("invalid stage fragment fields")
    stage, mode = fragment["stage"], fragment["execution_mode"]
    if stage not in ("synthetic", "real") or mode not in MODES:
        raise ValueError("invalid fragment stage/mode")
    document = {key: fragment[key] for key in DOCUMENT_KEYS}
    _validate_document(document, {(s, mode) for s in SCENARIOS if s.startswith(f"mnist.{stage}.")})


def validate_record(record: dict) -> None:
    """Same structural contract for one record, useful for bounded diagnostics."""
    if type(record) is not dict or record.get("scenario_id") not in SCENARIOS or record.get("execution_mode") not in MODES:
        raise ValueError("invalid scenario/mode record")
    document = {"schema_version": 1, "pytorch_version": "2.4.0+cpu",
                "pytorch_git_revision": "e4ee3be4063b7c430974252fdf7db42273388d86", "records": [record]}
    _validate_document(document, {(record["scenario_id"], record["execution_mode"])})


def _validate_document(document, expected):
    if type(document) is not dict or set(document) != DOCUMENT_KEYS or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("invalid MNIST evidence document schema")
    _exact(document["pytorch_version"], "2.4.0+cpu", "PyTorch version")
    _exact(document["pytorch_git_revision"], "e4ee3be4063b7c430974252fdf7db42273388d86", "PyTorch revision")
    records = document["records"]
    if type(records) is not list:
        raise ValueError("invalid records")
    for record in records:
        if type(record) is not dict or set(record) != RECORD_KEYS:
            raise ValueError("invalid MNIST evidence record fields")
        if type(record["scenario_id"]) is not str or record["scenario_id"] not in SCENARIOS or type(record["execution_mode"]) is not str or record["execution_mode"] not in MODES:
            raise ValueError("invalid scenario/mode identity")
    actual = [(r.get("scenario_id"), r.get("execution_mode")) for r in records if type(r) is dict]
    if len(actual) != len(records) or len(actual) != len(set(actual)):
        raise ValueError("duplicate or malformed MNIST scenario/mode matrix")
    if set(actual) != expected:
        raise ValueError("incomplete MNIST scenario/mode matrix")
    for record in records:
        sid, mode = record["scenario_id"], record["execution_mode"]
        real = sid.startswith("mnist.real.")
        recipe = record["recipe"]
        _exact(recipe, _recipe(sid), "ordinary source-owned recipe")
        updates = 34 if real else 3
        runtime = record["runtime"]
        runtime_keys = {"pytorch_version", "pytorch_git_revision", "checkout_head", "extension_sha256", "extension_path", "device", "hardware", "driver", "vulkan_api_version", "vulkan_instance_version", "execution_mode"}
        if type(runtime) is not dict or set(runtime) != runtime_keys or any(type(v) is not str or not v for v in runtime.values()) or runtime["device"] != "vk:0" or runtime["execution_mode"] != mode:
            raise ValueError("invalid runtime identity/device/mode")
        if runtime["pytorch_version"] != document["pytorch_version"] or runtime["pytorch_git_revision"] != document["pytorch_git_revision"]:
            raise ValueError("document/runtime PyTorch identity differs")
        _hash(runtime["extension_sha256"], "extension hash")
        for key in ("checkout_head", "pytorch_git_revision"):
            if len(runtime[key]) != 40 or any(c not in "0123456789abcdef" for c in runtime[key]):
                raise ValueError("invalid runtime revision")
        if not Path(runtime["extension_path"]).is_absolute():
            raise ValueError("invalid runtime extension path")
        sources = record["source_identity"]
        if type(sources) is not dict or not sources:
            raise ValueError("invalid source identity")
        for path, value in sources.items():
            if type(path) is not str or path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("invalid source identity path")
            _hash(value, "source identity")
        ref = record["reference_receipt"]
        if type(ref) is not dict or set(ref) != {"path", "sha256", "size", "pytorch_version", "pytorch_git_revision", "source_snapshots"} or type(ref["size"]) is not int or ref["size"] <= 0:
            raise ValueError("invalid reference receipt")
        _hash(ref["sha256"], "reference receipt")
        if type(ref["path"]) is not str or not Path(ref["path"]).is_absolute():
            raise ValueError("invalid reference receipt path")
        _exact(ref["pytorch_version"], document["pytorch_version"], "reference receipt version")
        _exact(ref["pytorch_git_revision"], document["pytorch_git_revision"], "reference receipt revision")
        from tools.probe_mnist_training_reference import EXPECTED_SOURCE_SHA256
        _exact(ref["source_snapshots"], EXPECTED_SOURCE_SHA256, "reference source inventory")
        _validate_data(record["data_identity"], real)
        steps = record["steps"]
        if type(steps) is not list or len(steps) != updates:
            raise ValueError("invalid step count")
        for index, step in enumerate(steps):
            if type(step) is not dict or set(step) != {"cpu", "vulkan"}:
                raise ValueError("invalid step diagnostics")
            for side, device in (("cpu", "cpu"), ("vulkan", "vk:0")):
                evidence = step[side]
                if type(evidence) is not dict or set(evidence) != {"reset", "backward", "update"}:
                    raise ValueError("invalid step diagnostics")
                reset = evidence["reset"]
                if type(reset) is not dict or set(reset) != {"gradients", "retained_objects"} or type(reset["gradients"]) is not dict or set(reset["gradients"]) != set(PARAMETERS):
                    raise ValueError("invalid reset diagnostics")
                expected_retained = {} if index == 0 else {p: not recipe["set_to_none"] for p in PARAMETERS}
                _exact(reset["retained_objects"], expected_retained, "reset retained-object observations")
                if (index == 0 or recipe["set_to_none"]) and any(v is not None for v in reset["gradients"].values()):
                    raise ValueError("changed reset contract")
                if index and not recipe["set_to_none"]:
                    if any(type(v) is not dict for v in reset["gradients"].values()) or set(reset["retained_objects"]) != set(PARAMETERS) or any(v is not True for v in reset["retained_objects"].values()):
                        raise ValueError("changed reset contract")
                    for p, value in reset["gradients"].items():
                        _tensor(value, device, "reset", expected_shape=SHAPES[p])
                        if any(value["values"]): raise ValueError("reset gradient is nonzero")
                back, update = evidence["backward"], evidence["update"]
                if type(back) is not dict or set(back) != {"logits", "loss", "gradients"} or type(update) is not dict or set(update) != {"parameters", "state", "optimizer_options"}:
                    raise ValueError("invalid step diagnostics")
                _exact(update["optimizer_options"], {**OPTIMIZER, "parameter_names": list(PARAMETERS)}, "observed optimizer options")
                if type(back["gradients"]) is not dict or set(back["gradients"]) != set(PARAMETERS): raise ValueError("invalid gradient keys")
                for p in PARAMETERS: _tensor(back["gradients"][p], device, "gradient", expected_shape=SHAPES[p])
                batch_count = recipe["train_batch_sizes"][index]
                _tensor(back["logits"], device, "logits", expected_shape=[batch_count, 10], requires_grad=True, history=True)
                _tensor(back["loss"], device, "loss", expected_shape=[], requires_grad=True, history=True)
                if type(update["parameters"]) is not dict or type(update["state"]) is not dict or set(update["parameters"]) != set(PARAMETERS) or set(update["state"]) != set(PARAMETERS):
                    raise ValueError("invalid parameter/state keys")
                for p in PARAMETERS:
                    _tensor(update["parameters"][p], device, "parameter", expected_shape=SHAPES[p], requires_grad=True)
                    state = update["state"][p]
                    if type(state) is not dict or set(state) != {"momentum_buffer"}:
                        raise ValueError("missing momentum state")
                    _tensor(state["momentum_buffer"], device, "momentum", expected_shape=SHAPES[p])
        diagnostics = {side: {f"step-{i}.{event}": step[side][event]
                             for i, step in enumerate(steps) for event in ("reset", "backward", "update")}
                       for side in ("cpu", "vulkan")}
        counters = record["phase_counters"]
        if type(counters) is not dict or set(counters) != {"cpu", "vulkan"}:
            raise ValueError("invalid phase counters")
        expected_order = _expected_phases(real, updates)
        for side in ("cpu", "vulkan"):
            summary = counters[side]
            if type(summary) is not dict or set(summary) != {"phase_order", "phase_routes", "counters", "counter_categories", "tensor_metadata"}:
                raise ValueError("invalid phase counter summary")
            if summary["phase_order"] != expected_order:
                raise ValueError("invalid phase order/counter coverage")
            routes = {name: _phase_category(name) for name in expected_order}
            if summary["phase_routes"] != routes:
                raise ValueError("invalid phase routes")
            categories = {key: sorted(name for name in expected_order if routes[name] == key)
                          for key in ("compute", "reset", "staging", "diagnostics", "readback", "setup")}
            _exact(summary["counter_categories"], categories, "phase counter categories")
            if side == "cpu":
                _exact(summary["counters"], {}, "CPU counters")
                continue
            if type(summary["counters"]) is not dict or set(summary["counters"]) != set(expected_order):
                raise ValueError("missing Vulkan phase counters")
            routes = counters["vulkan"]["phase_routes"]
            for name, row in counters["vulkan"]["counters"].items():
                if type(row) is not list or len(row) != 4 or any(type(x) is not int or x < 0 for x in row):
                    raise ValueError("invalid phase counters")
                if routes[name] == "compute":
                    if row[0] <= 0 or row[2:] != [0, 0]: raise ValueError("compute counters/fallback invalid")
                elif routes[name] == "reset":
                    if row[2:] != [0, 0]: raise ValueError("reset transfer/fallback counter is nonzero")
                elif row[3] != 0:
                    raise ValueError("fallback counter is nonzero")
        if type(record["evaluation"]) is not list or (real and len(record["evaluation"]) != 2) or (not real and record["evaluation"]):
            raise ValueError("invalid evaluation evidence")
        for epoch, evaluation in enumerate(record["evaluation"]):
            if type(evaluation) is not dict or set(evaluation) != {"epoch", "metric_device", "comparison", "cpu", "vulkan", "diagnostic_logits"} or type(evaluation["epoch"]) is not int or evaluation["epoch"] != epoch:
                raise ValueError("invalid evaluation evidence")
            _exact(evaluation["metric_device"], "cpu", "disclosed evaluation metric device")
            for side in ("cpu", "vulkan"):
                metrics = evaluation[side]
                if type(metrics) is not dict or set(metrics) != {"predictions", "correct", "count", "accuracy"}:
                    raise ValueError("invalid evaluation metrics")
                if metrics["count"] != 129 or type(metrics["predictions"]) is not list or len(metrics["predictions"]) != 129 or any(type(x) is not int or x < 0 or x > 9 for x in metrics["predictions"]):
                    raise ValueError("invalid evaluation predictions/count")
                if type(metrics["correct"]) is not int or type(metrics["count"]) is not int or type(metrics["accuracy"]) is not float or metrics["correct"] < 0 or metrics["correct"] > 129 or metrics["accuracy"] != metrics["correct"] / 129:
                    raise ValueError("invalid evaluation metrics")
            if evaluation["cpu"]["predictions"] != evaluation["vulkan"]["predictions"] or evaluation["cpu"]["correct"] != evaluation["vulkan"]["correct"] or evaluation["cpu"]["accuracy"] != evaluation["vulkan"]["accuracy"]:
                raise ValueError("evaluation parity mismatch")
            logits = evaluation["diagnostic_logits"]
            if type(logits) is not list or len(logits) != 9:
                raise ValueError("invalid evaluation logits")
            for index, pair in enumerate(logits):
                if type(pair) is not dict or set(pair) != {"cpu", "vulkan"}:
                    raise ValueError("invalid evaluation logits")
                size = 16 if index < 8 else 1
                _tensor(pair["cpu"], "cpu", "evaluation logits", expected_shape=[size, 10])
                _tensor(pair["vulkan"], "vk:0", "evaluation logits", expected_shape=[size, 10])
                _compare_tensor_pair(pair["cpu"], pair["vulkan"], "evaluation logits")
                for side in ("cpu", "vulkan"):
                    diagnostics[side][f"epoch-{epoch}.evaluation-{index}"] = {"logits": pair[side]}
            for side in ("cpu", "vulkan"):
                predictions = [prediction for item in logits for prediction in _predictions(item[side])]
                _exact(evaluation[side]["predictions"], predictions, "logit-derived predictions")
            _exact(evaluation["comparison"], {"passed": True, **{key: evaluation["cpu"][key]
                   for key in ("correct", "count", "accuracy")}}, "evaluation comparison")
        for side in ("cpu", "vulkan"):
            _exact(counters[side]["tensor_metadata"], _snapshot_metadata(diagnostics[side]),
                   f"{side} observed tensor metadata")
        _exact(record["parity"], _numeric_parity(diagnostics["cpu"], diagnostics["vulkan"]), "parity summaries")
        raw = record["raw_receipt"]
        if type(raw) is not dict or set(raw) != {"attempt", "file", "record_id"} or type(raw["attempt"]) is not str or type(raw["file"]) is not str:
            raise ValueError("invalid raw receipt linkage")
        if raw["record_id"] != sid or Path(raw["attempt"]).is_absolute() is False or raw["file"] != "capture.json":
            raise ValueError("invalid raw receipt locator")


def _tree(value, metadata, path="value"):
    if isinstance(value, torch.Tensor):
        vals = value.detach().cpu().reshape(-1).tolist()
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError(f"nonfinite snapshot at {path}")
        meta = metadata
        if type(meta) is not dict or set(meta) != {"device", "shape", "dtype", "requires_grad", "history"}:
            raise ValueError(f"{path}: missing live metadata")
        _exact(meta["shape"], list(value.shape), f"{path} snapshot/live shape")
        _exact(meta["dtype"], str(value.dtype), f"{path} snapshot/live dtype")
        return {"values": vals, "shape": list(value.shape), "dtype": str(value.dtype),
                "device": meta["device"], "requires_grad": meta["requires_grad"], "history": meta["history"]}
    if isinstance(value, dict):
        if type(metadata) is not dict or metadata.keys() != value.keys():
            raise ValueError(f"{path}: metadata tree keys mismatch")
        return {k: _tree(v, metadata[k], f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not isinstance(metadata, (list, tuple)) or len(value) != len(metadata):
            raise ValueError(f"{path}: metadata tree length mismatch")
        return [_tree(v, metadata[i], f"{path}.{i}") for i, v in enumerate(value)]
    if metadata is not None:
        raise ValueError(f"{path}: scalar metadata must be null")
    return value


def _snapshot_metadata(tree):
    if isinstance(tree, dict):
        if "values" in tree:
            return {key: tree[key] for key in ("device", "dtype", "shape", "requires_grad", "history")}
        return {key: _snapshot_metadata(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_snapshot_metadata(value) for value in tree]
    return None


def _predictions(logits):
    values = logits["values"]
    return [max(range(10), key=lambda i: values[start + i]) for start in range(0, len(values), 10)]


def _resolve_reference(path):
    path = Path(path).resolve(strict=True)
    receipt = validate_reference_receipt(path)
    return {"path": str(path), "sha256": _sha(path.read_bytes()), "size": path.stat().st_size,
            "pytorch_version": receipt["torch_version"], "pytorch_git_revision": receipt["upstream_revision"],
            "source_snapshots": {k: v["sha256"] for k, v in receipt["source_snapshots"].items()}}


def validate_reference_receipt(path: Path) -> dict:
    """Strict offline Task 1 receipt preflight; never fetches sources or bootstraps Vulkan."""
    path = Path(path).resolve(strict=True)
    receipt = _json(path)
    keys = {"evidence_kind", "torch_version", "upstream_revision", "source_snapshots", "installed",
            "profiler_phase_operators_and_shapes", "optimizer_defaults", "cpu_comparisons",
            "numerical_rationale", "process_settings"}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError("invalid Task 1 CPU reference receipt fields")
    if (receipt["evidence_kind"] != "CPU reference only; not Vulkan execution evidence"
            or receipt["torch_version"] != "2.4.0+cpu"
            or receipt["upstream_revision"] != "e4ee3be4063b7c430974252fdf7db42273388d86"):
        raise ValueError("unapproved Task 1 CPU reference receipt")
    from tools.probe_mnist_training_reference import EXPECTED_SOURCE_SHA256, SOURCE_LINES, OPERATORS
    snapshots = receipt["source_snapshots"]
    if type(snapshots) is not dict or set(snapshots) != set(EXPECTED_SOURCE_SHA256):
        raise ValueError("Task 1 receipt has incomplete upstream source closure")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        item = snapshots[name]
        if type(item) is not dict or set(item) != {"sha256", "url", "line_references"}:
            raise ValueError("invalid Task 1 upstream source snapshot")
        if item["sha256"] != expected or item["url"] != f"https://raw.githubusercontent.com/pytorch/pytorch/e4ee3be4063b7c430974252fdf7db42273388d86/{name}":
            raise ValueError("Task 1 upstream source identity differs")
        if not isinstance(item["line_references"], dict) or any(type(k) is not str or type(v) is not list or not v or any(type(n) is not int or n <= 0 for n in v) for k, v in item["line_references"].items()):
            raise ValueError("incomplete Task 1 source reference strings")
        if set(item["line_references"]) != set(SOURCE_LINES.get(name, [])):
            raise ValueError("incomplete Task 1 semantic source needles")
        # Fail closed on the explicit unavailable:* schema used by the CPU probe.
        if "unavailable" in json.dumps(item, sort_keys=True).lower():
            raise ValueError("Task 1 source reference strings are unavailable")
    for field in ("installed", "profiler_phase_operators_and_shapes", "optimizer_defaults",
                  "cpu_comparisons", "numerical_rationale", "process_settings"):
        if type(receipt[field]) is not dict or not receipt[field]:
            raise ValueError(f"Task 1 receipt has empty {field}")
    installed = receipt["installed"]
    if set(installed) != {"schemas", "dispatch_tables"}:
        raise ValueError("invalid installed closure fields")
    for key in ("schemas", "dispatch_tables"):
        entries = installed[key]
        if type(entries) is not dict or set(entries) != set(OPERATORS):
            raise ValueError("missing installed schema/dispatch closure")
        if any(type(v) is not str or not v or "unavailable:" in v.lower() for v in entries.values()):
            raise ValueError("unavailable installed schema/dispatch closure")
    _exact(str(torch.__version__), receipt["torch_version"], "actual installed PyTorch version")
    _exact(torch.version.git_version, receipt["upstream_revision"], "actual installed PyTorch revision")
    for operator in OPERATORS:
        name, _, requested_overload = operator.partition(".")
        overload = "" if requested_overload in ("", "default") else requested_overload
        try:
            current_schema = str(torch._C._get_schema(name, overload))
        except (AttributeError, RuntimeError) as error:
            raise ValueError("actual installed schema closure is unavailable") from error
        _exact(current_schema, installed["schemas"][operator], "actual installed schema")
        dispatch_name = name if requested_overload in ("", "default") else operator
        actual_dispatch = torch._C._dispatch_dump_table(dispatch_name)
        # An integrated wrapper may add PrivateUse1 kernels. The ordinary CPU
        # and generated CPU autograd registrations must still match the pinned
        # CPU-only receipt; do not initialize Vulkan to compare its dispatch.
        def cpu_registrations(table):
            return [line for line in table.splitlines() if line.startswith(("CPU:", "AutogradCPU:"))]
        _exact(cpu_registrations(actual_dispatch), cpu_registrations(installed["dispatch_tables"][operator]),
               "actual installed CPU dispatch/autograd closure")
    # This source-owned approved CPU receipt is an immutable trust anchor, not
    # whatever bytes happen to occupy the default filename today. Its exact
    # semantic tables also constrain nested profiler shapes/needles/rationale.
    approved_bytes = REFERENCE.read_bytes()
    if _sha(approved_bytes) != APPROVED_REFERENCE_SHA256:
        raise ValueError("approved Task 1 CPU receipt trust anchor changed")
    approved = json.loads(approved_bytes)
    for key in keys - {"process_settings"}:
        _exact(receipt[key], approved[key], f"Task 1 semantic closure.{key}")
    settings = receipt["process_settings"]
    if set(settings) != {"num_threads", "num_interop_threads", "deterministic_algorithms",
                         "cudnn_deterministic", "cudnn_benchmark"}:
        raise ValueError("invalid CPU process settings")
    for key in ("num_threads", "num_interop_threads"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError("invalid CPU thread settings")
    for key in ("deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
        if type(settings[key]) is not bool:
            raise ValueError("invalid CPU deterministic settings")
    return receipt


def _safe_attempt(attempt):
    raw = Path(attempt)
    if not raw.is_absolute(): raise ValueError("attempt must be an absolute path")
    if raw.parent.resolve(strict=True) != raw.parent:
        raise ValueError("attempt parent must not contain symlink aliases")
    if raw.exists() or raw.is_symlink(): raise ValueError("attempt already exists")
    resolved = raw.parent.resolve(strict=True) / raw.name
    if not resolved.is_relative_to(ROOT / ".superpowers"):
        raise ValueError("attempt must be excluded under .superpowers")
    rel = resolved.relative_to(ROOT)
    if subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=ROOT).returncode:
        raise ValueError("attempt directory is not git-excluded")
    return resolved


def select_execution_mode(mode: str) -> None:
    """Fresh-process policy: set an unset mode; reject conflicts/loaded wrappers."""
    if mode not in MODES:
        raise ValueError("invalid capture mode")
    if any(name == "pytorch_vulkan" or name.startswith("pytorch_vulkan.") for name in sys.modules):
        raise ValueError("capture requires a fresh process before wrapper initialization")
    selected = "1" if mode == "async" else "0"
    inherited = os.environ.get("PYTORCH_VULKAN_ASYNC_EXECUTION")
    if inherited is not None and inherited != selected:
        raise ValueError("inherited execution mode conflicts with requested mode")
    os.environ["PYTORCH_VULKAN_ASYNC_EXECUTION"] = selected


def _pythonpath(root):
    return os.pathsep.join(str(root / p) for p in (".", "build", "tests/python"))


def _configure_environment(stage, mode, root):
    select_execution_mode(mode)
    expected = {"PYTHONPATH": _pythonpath(ROOT), "VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation",
                "PYTORCH_VULKAN_MNIST_ROOT": None if stage == "synthetic" else str(root)}
    for key, selected in expected.items():
        inherited = os.environ.get(key)
        if inherited is not None and inherited != selected:
            raise ValueError(f"inherited {key} conflicts with canonical capture environment")
        if selected is not None:
            os.environ[key] = selected


def _command_root(receipt, attempt, root):
    """Authenticate exact argv and environment, retaining actual spellings."""
    command = receipt["command"]
    if type(command) is not list or len(command) not in (11, 13) or any(type(v) is not str or not v for v in command):
        raise ValueError("invalid capture command argv")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK) or not executable.samefile(sys.executable):
        raise ValueError("capture command Python differs from validation interpreter")
    script = command[1]
    if script not in ("tools/mnist_training_coverage.py", str(root / "tools/mnist_training_coverage.py")):
        raise ValueError("capture command script differs from semantic control")
    _exact(receipt["working_directory"], str(root), "capture working directory")
    stage, mode = receipt["stage"], receipt["mode"]
    required = ["capture", "--stage", stage, "--mode", mode, "--output", str(attempt),
                "--reference-receipt", receipt["reference_receipt"]["path"]]
    real_root = None
    if stage == "real":
        if len(command) != 13 or command[-2] != "--mnist-root":
            raise ValueError("real capture command is missing data root")
        from tools.mnist_data import external_data_root
        raw = Path(command[-1])
        if not raw.is_absolute():
            raise ValueError("real data root argv must be absolute")
        real_root = external_data_root(raw, root)
        if str(real_root) != str(raw):
            raise ValueError("real data root argv must be canonical")
        required.extend(["--mnist-root", str(real_root)])
    _exact(command[2:], required, "capture command options")
    expected_environment = {"PYTORCH_VULKAN_ASYNC_EXECUTION": "1" if mode == "async" else "0",
                            "PYTHONPATH": _pythonpath(root), "VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation",
                            "PYTORCH_VULKAN_MNIST_ROOT": None if real_root is None else str(real_root)}
    _exact(receipt["environment"], expected_environment, "capture environment")
    return real_root


def capture(stage: str, mode: str, root: Path | None, attempt: Path, *, reference_receipt: Path) -> dict:
    """Wrapper-first real capture. Must run in a fresh process for each mode."""
    if stage not in ("synthetic", "real") or mode not in MODES:
        raise ValueError("invalid capture stage/mode")
    if stage == "real" and root is None:
        raise ValueError("real capture requires external MNIST root")
    if stage == "synthetic" and root is not None:
        raise ValueError("synthetic capture does not accept a data root")
    _configure_environment(stage, mode, root)
    # Setting PYTHONPATH after Python starts does not update sys.path. Select
    # the same canonical package paths now, before the first wrapper import.
    selected_paths = _pythonpath(ROOT).split(os.pathsep)
    sys.path[:] = selected_paths + [path for path in sys.path if path not in selected_paths]
    attempt = _safe_attempt(attempt)
    reference = _resolve_reference(reference_receipt)
    invocation = {"stage": stage, "mode": mode, "command": [sys.executable, *sys.argv],
                  "working_directory": str(Path.cwd()), "reference_receipt": reference,
                  "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}}
    _command_root(invocation, attempt, ROOT)
    before = {p: _sha((ROOT / p).read_bytes()) for p in source_paths()}
    if stage == "real":
        from tools.mnist_data import external_data_root, load_mnist_subset
        root = external_data_root(root, ROOT)
        # Verify all four canonical resources before wrapper/backend initialization.
        load_mnist_subset(root, split="train", repo_root=ROOT)
        load_mnist_subset(root, split="test", repo_root=ROOT)
    # Retain an empty partial attempt on any later execution/serialization failure.
    attempt.mkdir(parents=False, exist_ok=False)
    # Bootstrap precedes imports that can execute CPU backward or parity helpers.
    wrapper = importlib.import_module("pytorch_vulkan")
    if wrapper._C.execution_mode() != mode:
        raise ValueError("initialized execution mode does not match request")
    from vector_matmul_capability_evidence import runtime_identity
    from mnist_training_parity import run_real_pair, run_synthetic_pair
    runtime = runtime_identity(ROOT)
    # TorchVersion is a str subclass; serialize an ordinary JSON string while
    # leaving the shared runtime producer and strict schema contract unchanged.
    runtime = {**runtime, "pytorch_version": str(runtime["pytorch_version"])}
    if runtime["execution_mode"] != mode:
        raise ValueError("runtime mode differs from request")
    selected = _selected_extension(ROOT, os.environ["PYTHONPATH"])
    _exact(str(Path(wrapper._C.__file__).resolve()), str(selected), "loaded vs selected extension")
    _exact(runtime["extension_sha256"], _sha(selected.read_bytes()), "loaded extension SHA256")
    pairs = []
    if stage == "synthetic":
        for batch in (1, 2, 16):
            for set_none in (True, False):
                pairs.append((f"mnist.synthetic.batch{batch}.sgd-{'none' if set_none else 'zero'}",
                              run_synthetic_pair(batch, set_none, backend=wrapper)))
    else:
        for set_none in (True, False):
            pairs.append((f"mnist.real.subset257-129.sgd-{'none' if set_none else 'zero'}",
                          run_real_pair(root, set_none, backend=wrapper)))
    after = {p: _sha((ROOT / p).read_bytes()) for p in source_paths()}
    if before != after: raise ValueError("semantic source changed during capture")
    _exact(runtime["extension_sha256"], _sha(selected.read_bytes()), "extension changed during capture")
    records = []
    for sid, pair in pairs:
        cpu, vk = pair["cpu"], pair["vulkan"]
        snapshots = pair["diagnostic_snapshots"]
        diagnostics = {}
        for name, vals in snapshots.items():
            obs = cpu if name == "cpu" else vk
            diagnostics[name] = {event: _tree(value, obs["tensor_metadata"][event]) for event, value in vals.items()}
        record = _record_from_pair(sid, mode, pair, runtime, before, reference, diagnostics)
        record["raw_receipt"] = {"attempt": str(attempt), "file": "capture.json", "record_id": sid}
        records.append(record)
    payload = {"schema_version": 1, "stage": stage, "execution_mode": mode,
               "pytorch_version": str(torch.__version__), "pytorch_git_revision": torch.version.git_version,
               "records": records}
    # Genuine producer schema must succeed before any completion artifact.
    validate_fragment(payload)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (attempt / FRAGMENT_FILE).open("xb") as stream:
        stream.write(encoded)
    receipt = {"schema_version": 1, "complete": True, "stage": stage, "mode": mode,
                "command": [sys.executable, *sys.argv],
                "working_directory": str(Path.cwd()),
               "environment": {name: os.environ.get(name) for name in (
                   "PYTORCH_VULKAN_ASYNC_EXECUTION", "PYTHONPATH", "VK_INSTANCE_LAYERS",
                   "PYTORCH_VULKAN_MNIST_ROOT")},
               "source_identity": before, "runtime": runtime, "reference_receipt": reference,
                "outputs": {"capture.json": {"sha256": _sha(encoded), "size": len(encoded)}}}
    _command_root(receipt, attempt, ROOT)
    from composed_evidence_common import source_identity
    _exact(source_identity(source_paths(), ROOT), before, "source changed before capture completion")
    _exact(reference, _resolve_reference(reference_receipt), "reference changed before capture completion")
    _exact(runtime["extension_sha256"], _sha(selected.read_bytes()), "extension changed before capture completion")
    if wrapper._C.execution_mode() != mode:
        raise ValueError("initialized mode changed before capture completion")
    with (attempt / RECEIPT_FILE).open("x") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    for filename in (FRAGMENT_FILE, RECEIPT_FILE):
        (attempt / filename).chmod(0o444)
    attempt.chmod(0o555)
    return payload


def _record_from_pair(sid, mode, pair, runtime, sources, reference, diagnostics):
    real = sid.startswith("mnist.real.")
    batch = 16 if real else int(sid.split("batch")[1].split(".")[0])
    set_none = sid.endswith("sgd-none")
    _exact(pair["set_to_none"], sid.endswith("sgd-none"), "observed reset recipe")
    _exact(pair["parameter_names"], list(PARAMETERS), "observed parameter inventory")
    steps = []
    count = pair["updates"]
    for index in range(count):
        steps.append({side: {event: diagnostics[side][f"step-{index}.{event}"]
                      for event in ("reset", "backward", "update")}
                      for side in ("cpu", "vulkan")})
    recipe = {"kind": "real" if real else "synthetic", "batch_size": batch,
              "set_to_none": set_none, "model_seed": 12012,
              "updates": count, "optimizer": {"lr": .01, "momentum": .9, "weight_decay": 0,
              "dampening": 0, "nesterov": False, "maximize": False, "foreach": None,
              "differentiable": False, "fused": None},
              "train_batch_sizes": pair.get("train_batch_sizes", [batch] * 3)}
    if not real:
        recipe["image_seed"] = 12013 + batch
        recipe["labels"] = [9] if batch == 1 else [0, 9] if batch == 2 else [i % 10 for i in range(16)]
    else:
        recipe.update({"epochs": 2, "evaluation_batch_sizes": pair["evaluation_batch_sizes"],
                       "shuffle": False, "drop_last": False, "num_workers": 0,
                       "scale_divisor": 255})
    evaluations = []
    if real:
        _exact(pair["metric_device"], "cpu", "observed evaluation metric device")
        for item in pair["evaluations"]:
            epoch = item["epoch"]
            logits = [{side: diagnostics[side][f"epoch-{epoch}.evaluation-{batch}"]["logits"]
                       for side in ("cpu", "vulkan")} for batch in range(9)]
            evaluations.append({**item, "metric_device": pair["metric_device"], "diagnostic_logits": logits})
    record = {"scenario_id": sid, "execution_mode": mode, "recipe": recipe, "runtime": runtime,
              "source_identity": sources, "reference_receipt": reference,
              "data_identity": pair.get("data_identity", {"kind": "synthetic"}),
              "steps": steps,
              "evaluation": evaluations,
              "phase_counters": {"cpu": pair["cpu"], "vulkan": pair["vulkan"]},
              "parity": _numeric_parity(diagnostics["cpu"], diagnostics["vulkan"]),
              "raw_receipt": {"attempt": "", "file": "capture.json", "record_id": sid}}
    return record


def _numeric_parity(cpu, vk):
    errors = {}

    def visit(a, b, path, field):
        if type(a) is not type(b):
            raise ValueError(f"parity type mismatch at {path}")
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f"parity keys mismatch at {path}")
            if "values" in a:
                if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
                    raise ValueError(f"parity shape/dtype mismatch at {path}")
                rtol, atol = RTOL_ATOL[field]
                av, bv = a["values"], b["values"]
                if len(av) != len(bv):
                    raise ValueError(f"parity length mismatch at {path}")
                delta = [abs(x - y) for x, y in zip(av, bv)]
                if any(d > atol + rtol * abs(x) for d, x in zip(delta, av)):
                    raise ValueError(f"numerical parity tolerance exceeded at {path}")
                if ".reset." in path and any(delta):
                    raise ValueError(f"reset parity mismatch at {path}")
                errors[path] = {"maximum_absolute_error": float(max(delta, default=0.0)),
                                "maximum_relative_error": float(max((d / max(abs(x), atol)
                                    for d, x in zip(delta, av)), default=0.0))}
                return
            for key in a:
                visit(a[key], b[key], f"{path}.{key}", key if key in RTOL_ATOL else field)
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f"parity length mismatch at {path}")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}.{index}", field)
        elif a != b:
            raise ValueError(f"parity mismatch at {path}")

    if cpu.keys() != vk.keys():
        raise ValueError("parity event inventory mismatch")
    for event in sorted(cpu):
        visit(cpu[event], vk[event], event, None)
    return {"tolerances": {key: {"rtol": rtol, "atol": atol}
                           for key, (rtol, atol) in RTOL_ATOL.items()}, "errors": errors}


def validate_current(document: dict, *, repo_root: Path, receipt_roots: list[Path]) -> None:
    """One CPU-only authentication path for full raw attempts and current files."""
    assembled = _authenticated_assembly(receipt_roots, repo_root=repo_root)
    if type(document) is not dict or set(document) != DOCUMENT_KEYS:
        raise ValueError("invalid MNIST evidence document fields")
    for key in DOCUMENT_KEYS - {"records"}:
        _exact(document[key], assembled[key], f"current document.{key}")
    records = document["records"]
    if type(records) is not list or len(records) != 16:
        raise ValueError("incomplete current document matrix")
    for record in records:
        if (type(record) is not dict or set(record) != RECORD_KEYS
                or type(record["scenario_id"]) is not str or type(record["execution_mode"]) is not str):
            raise ValueError("invalid current record fields/identity types")
    expected = {(r["scenario_id"], r["execution_mode"]): r for r in assembled["records"]}
    actual = [(r.get("scenario_id"), r.get("execution_mode")) for r in records if isinstance(r, dict)]
    if len(actual) != 16 or len(set(actual)) != 16 or set(actual) != set(expected):
        raise ValueError("incomplete or duplicate current document matrix")
    for record in records:
        _exact(record, expected[(record["scenario_id"], record["execution_mode"])],
               "aggregate record differs from authenticated raw capture")


def _attempt_path(path, root):
    supplied = Path(path)
    if not supplied.is_absolute() or supplied.is_symlink() or supplied.resolve(strict=True) != supplied:
        raise ValueError("attempt path must be absolute and alias-free")
    path = supplied
    if not path.is_relative_to(root / ".superpowers"):
        raise ValueError("attempt must be under excluded .superpowers")
    if subprocess.run(["git", "check-ignore", "-q", str(path.relative_to(root))], cwd=root).returncode:
        raise ValueError("attempt is not git-excluded")
    if not path.is_dir():
        raise ValueError("attempt must be an existing directory")
    return path


def _validate_attempt(path: Path, *, repo_root=None):
    root = ROOT if repo_root is None else Path(repo_root).resolve(strict=True)
    path = _attempt_path(path, root)
    receipt_path, capture_path = path / "receipt.json", path / "capture.json"
    if not receipt_path.is_file() or not capture_path.is_file():
        raise ValueError("missing raw capture receipt/archive")
    if receipt_path.is_symlink() or capture_path.is_symlink():
        raise ValueError("raw capture files must not be symlinks")
    receipt = _json(receipt_path)
    raw = capture_path.read_bytes()
    receipt_keys = {"schema_version", "complete", "stage", "mode", "command", "working_directory", "environment",
                    "source_identity", "runtime", "reference_receipt", "outputs"}
    if type(receipt) is not dict or set(receipt) != receipt_keys or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("invalid completed capture receipt schema")
    if receipt.get("complete") is not True:
        raise ValueError("capture receipt is not complete")
    _exact(receipt["outputs"], {"capture.json": {"sha256": _sha(raw), "size": len(raw)}}, "raw capture receipt hash/size")
    fragment = _json(capture_path)
    if type(fragment) is not dict or set(fragment) != DOCUMENT_KEYS | {"stage", "execution_mode"}:
        raise ValueError("invalid stage fragment fields")
    validate_fragment(fragment)
    mode, stage = fragment["execution_mode"], fragment["stage"]
    wanted = {s for s in SCENARIOS if s.startswith(f"mnist.{stage}.")}
    ids = [r["scenario_id"] for r in fragment["records"]]
    if mode not in MODES or stage not in ("synthetic", "real") or len(ids) != len(set(ids)) or set(ids) != wanted:
        raise ValueError("incomplete or duplicate stage fragment")
    if receipt.get("mode") != mode or receipt.get("stage") != stage:
        raise ValueError("fragment/receipt mode or stage mismatch")
    _command_root(receipt, path, root)
    for record in fragment["records"]:
        if record.get("execution_mode") != mode:
            raise ValueError("fragment record mode differs")
        for key in ("source_identity", "runtime", "reference_receipt"):
            _exact(record.get(key), receipt[key], f"capture receipt provenance.{key}")
        _exact(record["raw_receipt"], {"attempt": str(path), "file": "capture.json", "record_id": record["scenario_id"]}, "immutable raw receipt locator")
    return fragment, receipt, raw


def aggregate_attempts(attempts: list[Path], *, repo_root: Path = ROOT) -> dict:
    """Combine four authentic stage fragments in memory and validate current linkage.

    This is deliberately nonpublishing: callers receive a candidate document but
    no output path is created or replaced. CPU-only captures cannot qualify it.
    """
    return _authenticated_assembly(attempts, repo_root=repo_root)


def _authenticated_assembly(attempts, *, repo_root):
    root = Path(repo_root).resolve(strict=True)
    if len(attempts) != 4 or len(set(map(str, attempts))) != 4:
        raise ValueError("authentication requires four unique stage/mode attempts")
    # Cheap current control checks precede large tensor JSON parsing. Complete
    # immutable readers still authenticate every fragment below.
    from composed_evidence_common import source_identity
    current = source_identity(source_paths(root), root)
    receipt_fingerprints = {}
    for attempt in attempts:
        attempt = _attempt_path(attempt, root)
        receipt_path = Path(attempt) / RECEIPT_FILE
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError("missing completed raw capture receipt")
        receipt_fingerprints[Path(attempt)] = _sha(receipt_path.read_bytes())
        preliminary = _json(receipt_path)
        if type(preliminary) is not dict:
            raise ValueError("invalid completed raw receipt fields")
        _exact(preliminary.get("source_identity"), current, "current source identity")
    fragments = [_validate_attempt(Path(p), repo_root=root) for p in attempts]
    identities = {(f[0]["stage"], f[0]["execution_mode"]) for f in fragments}
    expected = {(stage, mode) for stage in ("synthetic", "real") for mode in MODES}
    if identities != expected:
        raise ValueError("aggregation requires all four stage/mode attempts")
    records = [r for fragment, _, _ in fragments for r in fragment["records"]]
    first = fragments[0][0]
    document = {"schema_version": 1, "pytorch_version": first["pytorch_version"],
                "pytorch_git_revision": first["pytorch_git_revision"], "records": records}
    # validate_fragment already checked every record; this assembly separately
    # proves the complete scenario/mode set and shared identities.
    reference = records[0]["reference_receipt"]
    for record in records:
        if record["source_identity"] != records[0]["source_identity"]:
            raise ValueError("stage fragments have different semantic sources")
        if record["reference_receipt"] != reference:
            raise ValueError("stage fragments have different CPU reference receipts")
        a, b = dict(record["runtime"]), dict(records[0]["runtime"])
        a.pop("execution_mode"); b.pop("execution_mode")
        _exact(a, b, "common runtime identity (excluding only mode)")
    for scenario in SCENARIOS:
        pair = [r for r in records if r["scenario_id"] == scenario]
        if len(pair) != 2:
            raise ValueError("incomplete current scenario/mode matrix")
        _exact(pair[0]["recipe"], pair[1]["recipe"], "same-scenario cross-mode recipe")
        _exact(pair[0]["data_identity"], pair[1]["data_identity"], "same-scenario data identity")
    _exact(reference, _resolve_reference(Path(reference["path"])), "current validated reference receipt metadata")
    for _, receipt, _ in fragments:
        extension = _selected_extension(root, receipt["environment"]["PYTHONPATH"])
        _exact(receipt["runtime"]["extension_path"], str(extension), "actual selected extension path")
        _exact(receipt["runtime"]["extension_sha256"], _sha(extension.read_bytes()), "actual selected extension SHA256")
    real_roots = {_command_root(receipt, Path(attempt), root)
                  for attempt, (_, receipt, _) in zip(attempts, fragments) if receipt["stage"] == "real"}
    if len(real_roots) != 1:
        raise ValueError("real capture roots differ across modes")
    replay_cpu_reference(document, real_root=real_roots.pop(), repo_root=root)
    # No hash re-labeling: check all source/receipt/capture/binary bytes again
    # after replay, so publication sees the same authenticated inputs.
    _exact(source_identity(source_paths(root), root), current, "source changed during authentication")
    for attempt, (_, receipt, raw) in zip(attempts, fragments):
        _exact(_sha((Path(attempt) / RECEIPT_FILE).read_bytes()), receipt_fingerprints[Path(attempt)], "receipt changed during authentication")
        if (Path(attempt) / FRAGMENT_FILE).read_bytes() != raw:
            raise ValueError("raw capture changed during authentication")
        if _sha(Path(receipt["runtime"]["extension_path"]).read_bytes()) != receipt["runtime"]["extension_sha256"]:
            raise ValueError("extension changed during authentication")
    _exact(reference, _resolve_reference(Path(reference["path"])), "reference changed during authentication")
    return document


def _selected_extension(root, pythonpath):
    """Resolve the package/extension the canonical capture path selects; no import."""
    paths = pythonpath.split(os.pathsep)
    spec = importlib.machinery.PathFinder.find_spec("pytorch_vulkan", paths)
    if spec is None or not spec.origin or spec.submodule_search_locations is None:
        raise ValueError("canonical wrapper package is unavailable")
    if _sha(Path(spec.origin).read_bytes()) != _sha((root / "python/pytorch_vulkan/__init__.py").read_bytes()):
        raise ValueError("selected wrapper differs from authenticated source")
    for name in ("compiler.py", "compiler_metadata.py", "serialization.py"):
        target = Path(spec.origin).parent / name
        if not target.is_file() or _sha(target.read_bytes()) != _sha((root / "python/pytorch_vulkan" / name).read_bytes()):
            raise ValueError("selected wrapper helper differs from authenticated source")
    extension_spec = importlib.machinery.PathFinder.find_spec("pytorch_vulkan._C", spec.submodule_search_locations)
    if extension_spec is None or not isinstance(extension_spec.loader, importlib.machinery.ExtensionFileLoader):
        raise ValueError("selected Vulkan extension is unavailable")
    candidates = [Path(folder) / ("_C" + suffix) for folder in spec.submodule_search_locations
                  for suffix in importlib.machinery.EXTENSION_SUFFIXES
                  if (Path(folder) / ("_C" + suffix)).is_file()]
    if len(candidates) != 1:
        raise ValueError("selected wrapper has ambiguous extension artifacts")
    return Path(extension_spec.origin).resolve(strict=True)


def replay_cpu_reference(document: dict, *, real_root: Path | None, repo_root: Path) -> dict:
    """Reconstruct the ordinary CPU recipe; never imports or initializes Vulkan.

    Returns per-scenario `verified` or explicitly warned `data-unavailable`.
    Missing canonical resources are not acquired; corruption is a hard failure.
    This does not claim GPU execution or real replay when data is unavailable.
    """
    from mnist_training_cases import make_model, synthetic_batch, train
    from mnist_training_parity import TrainingObserver, _real_observers
    from tools.mnist_data import load_mnist_subset, mnist_loader
    statuses, cache = {}, {}
    real_data = None
    if any(r["scenario_id"].startswith("mnist.real.") for r in document["records"]):
        if real_root is not None:
            try:
                real_data = {split: load_mnist_subset(real_root, split=split, repo_root=repo_root)
                             for split in ("train", "test")}
            except FileNotFoundError:
                real_data = None
        if real_data is None:
            warnings.warn("real MNIST CPU replay: data-unavailable; no download or replay claim", RuntimeWarning, stacklevel=2)
    for record in document["records"]:
        sid = record["scenario_id"]
        real = sid.startswith("mnist.real.")
        if real and real_data is None:
            statuses[sid] = "data-unavailable"
            continue
        if sid not in cache:
            observer = TrainingObserver("cpu")
            with observer.phase("setup.model-staging"):
                model = make_model("cpu")
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            if real:
                observer, _, _, evaluations = _real_observers(
                    model, optimizer, "cpu", None, mnist_loader(real_data["train"][0]),
                    mnist_loader(real_data["test"][0]), sid.endswith("sgd-none"), observer=observer)
            else:
                batch = int(sid.split("batch")[1].split(".")[0])
                train(model, optimizer, lambda: [synthetic_batch(batch)] * 3, device="cpu",
                      epochs=1, set_to_none=sid.endswith("sgd-none"), observer=observer)
                evaluations = []
            expected = {name: _tree(value, observer.tensor_metadata[name])
                        for name, value in observer.snapshots.items()}
            cache[sid] = (expected, evaluations)
        expected, evaluations = cache[sid]
        actual = {f"step-{i}.{event}": step["cpu"][event] for i, step in enumerate(record["steps"])
                  for event in ("reset", "backward", "update")}
        if real:
            _exact(record["data_identity"], {split: item[1] for split, item in real_data.items()}, "CPU replay verified dataset identity")
            for epoch, evaluation in enumerate(record["evaluation"]):
                _exact(evaluation["cpu"], evaluations[epoch], "CPU replay verified evaluation metrics")
                for batch, logits in enumerate(evaluation["diagnostic_logits"]):
                    actual[f"epoch-{epoch}.evaluation-{batch}"] = {"logits": logits["cpu"]}
        _exact(actual, expected, f"CPU replay {sid}")
        statuses[sid] = "verified"
    return statuses


def _excluded_new_directory(path: Path):
    raw = Path(path)
    if not raw.is_absolute() or raw.exists() or raw.is_symlink():
        raise ValueError("archive must be an absolute new directory")
    resolved_parent = raw.parent.resolve(strict=True)
    if resolved_parent != raw.parent:
        raise ValueError("archive parent must not contain symlink aliases")
    target = resolved_parent / raw.name
    if not target.is_relative_to(ROOT / ".superpowers"):
        raise ValueError("archive must be under excluded .superpowers")
    if subprocess.run(["git", "check-ignore", "-q", str(target.relative_to(ROOT))], cwd=ROOT).returncode:
        raise ValueError("archive directory is not git-excluded")
    return target


def _file_identity(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError("missing/unreadable raw capture file") from error
    return {"sha256": _sha(raw), "size": len(raw)}


def _publication_recheck(document, attempts, fingerprints):
    """Cheap final byte/control checks after archive creation and serialization."""
    from composed_evidence_common import source_identity
    for attempt in attempts:
        for filename in (RECEIPT_FILE, FRAGMENT_FILE):
            _exact(_file_identity(Path(attempt) / filename), fingerprints[(Path(attempt), filename)], "publication raw input changed")
    first = document["records"][0]
    _exact(source_identity(source_paths(), ROOT), first["source_identity"], "publication current source identity")
    _exact(_resolve_reference(Path(first["reference_receipt"]["path"])), first["reference_receipt"], "publication reference changed")
    selected = _selected_extension(ROOT, _pythonpath(ROOT))
    _exact(str(selected), first["runtime"]["extension_path"], "publication selected extension changed")
    _exact(_sha(selected.read_bytes()), first["runtime"]["extension_sha256"], "publication extension changed")


def publish(attempts: list[Path], output: Path, *, archive: Path) -> None:
    """CPU-only complete-matrix publisher with lock, immutable byte archive and rollback."""
    if len(attempts) != 4:
        raise ValueError("publication requires four stage/mode attempts")
    archive_path = _excluded_new_directory(Path(archive))
    output_path = Path(output)
    if not output_path.is_absolute() or output_path.is_symlink():
        raise ValueError("output must be an absolute regular path")
    if output_path.exists() and not stat.S_ISREG(output_path.lstat().st_mode):
        raise ValueError("output must be a regular file")
    resolved_parent = output_path.parent.resolve(strict=True)
    if resolved_parent != output_path.parent:
        raise ValueError("output parent must not contain symlink aliases")
    output_path = resolved_parent / output_path.name
    resolved_attempts = [Path(a).resolve(strict=True) for a in attempts]
    def overlap(left, right):
        return left == right or left.is_relative_to(right) or right.is_relative_to(left)
    if (len(set(resolved_attempts)) != len(resolved_attempts)
            or any(overlap(archive_path, item) or overlap(output_path, item) for item in resolved_attempts)
            or overlap(archive_path, output_path)):
        raise ValueError("publication paths alias an input")
    lock_path = output_path.with_name(output_path.name + ".lock")
    if lock_path.is_symlink():
        raise ValueError("publication lock must not be a symlink")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another publisher owns the output") from error
        # Authentication runs under sole writer ownership, before archival.
        fingerprints = {(Path(p), filename): _file_identity(Path(p) / filename)
                        for p in attempts for filename in (RECEIPT_FILE, FRAGMENT_FILE)}
        doc = aggregate_attempts([Path(p) for p in attempts], repo_root=ROOT)
        original = output_path.read_bytes() if output_path.exists() else None
        archive_path.mkdir(parents=False, exist_ok=False)
        if original is not None:
            baseline = archive_path / "baseline.bin"
            baseline.write_bytes(original)
            baseline.chmod(0o444)
        manifest = {"schema_version": 1, "output": str(output_path),
                    "baseline": {"present": original is not None,
                                 "sha256": None if original is None else _sha(original),
                                 "size": None if original is None else len(original)}}
        manifest_path = archive_path / "archive.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        manifest_path.chmod(0o444)
        archive_path.chmod(0o555)
        payload = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        staging = None
        replacement_started = False
        try:
            fd, name = tempfile.mkstemp(prefix=".mnist-evidence-", dir=output_path.parent)
            staging = Path(name)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            _publication_recheck(doc, attempts, fingerprints)
            if output_path.is_symlink() or (output_path.exists() and not stat.S_ISREG(output_path.lstat().st_mode)):
                raise ValueError("publication output became unsafe")
            _exact(output_path.read_bytes() if output_path.exists() else None, original, "publication baseline changed")
            _exact(_json(manifest_path), manifest, "publication archive manifest changed")
            if original is not None and (archive_path / "baseline.bin").read_bytes() != original:
                raise ValueError("publication archive baseline bytes changed")
            replacement_started = True
            os.replace(staging, output_path)
        except BaseException:
            if staging is not None and staging.exists(): staging.unlink()
            if not replacement_started:
                raise
            if original is None:
                output_path.unlink(missing_ok=True)
            else:
                fd, name = tempfile.mkstemp(prefix=".mnist-rollback-", dir=output_path.parent)
                with os.fdopen(fd, "wb") as stream: stream.write(original)
                os.replace(name, output_path)
            raise
