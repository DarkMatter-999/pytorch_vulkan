"""Strict tensor-tree and phase evidence machinery, independent of Vulkan import.

Pinned 2.4 LinearAlgebra.cpp matmul folds/expanded-batch bmm, derivatives.yaml
mm/bmm/mv, MSE generated first/second rules and FunctionsManual.cpp's mean
double-backward Scalar division establish the semantic closure below.
"""
import hashlib
import math
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = (
    "tools/vulkan_wrapper_pytest.py", "tools/capture_vulkan_composed_evidence.py",
    "tests/python/composed_evidence_common.py",
    "tests/python/vector_matmul_capability_evidence.py",
    "python/pytorch_vulkan/__init__.py",
    "src/vulkan_compute.cpp", "src/vulkan_compute.h",
    "src/vulkan_layout.cpp", "src/vulkan_layout.h", "src/vulkan_tensor_layout.h",
    "src/vulkan_platform.cpp", "src/vulkan_platform.h",
    "src/vulkan_execution.cpp", "src/vulkan_execution.h",
    "src/vulkan_transfer.cpp", "src/vulkan_transfer.h",
    "src/vulkan_allocator.cpp", "src/vulkan_allocator.h",
    "src/vulkan_buffer.cpp", "src/vulkan_buffer.h", "src/python_module.cpp",
    "src/vulkan/shader_registry.cpp", "src/vulkan/shader_registry.h",
    *(f"src/vulkan/operators/{name}" for name in (
        "add.cpp", "linear.cpp", "linear.h", "out.cpp", "out.h", "mul.cpp",
        "sub.cpp", "binary.h", "unary.cpp", "unary.h", "reduction.cpp", "reduction.h",
        "view.cpp", "view.h", "loss.cpp", "loss.h", "scalar_division.cpp",
        "optimizer.cpp", "optimizer.h", "autograd.cpp", "autograd.h")),
    *(f"src/vulkan/shaders/{folder}/{name}" for folder, names in (
        ("glsl", ("loss.comp", "pointwise.comp", "gemm.comp", "model.comp",
                  "broadcast.comp", "reduction.comp", "reduction_backward.comp")),
        ("generated", ("loss_spv.h", "loss_spv.sha256", "pointwise_spv.h",
                       "pointwise_spv.sha256", "gemm_spv.h", "gemm_spv.sha256",
                       "model_spv.h", "model_spv.sha256", "operator_spv.h",
                       "operator_spv.sha256", "reduction_indexing_spv.h",
                       "reduction_indexing_spv.sha256"))) for name in names),
)
MATRIX_LEAVES = {"aten::mm", "aten::bmm", "aten::mv", "aten::dot"}
SEMANTIC_OPS = MATRIX_LEAVES | {
    "aten::matmul", "aten::mse_loss", "aten::mse_loss_backward", "aten::div_",
    "aten::mul", "aten::mul_", "aten::neg", "aten::sub", "aten::add", "aten::add_",
    "aten::sum", "aten::expand", "aten::reshape", "aten::view", "aten::transpose",
    "aten::t", "aten::slice", "aten::slice_backward", "aten::select_backward",
}


def source_identity(paths, root=ROOT):
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def history_nodes(tensor):
    todo = [tensor.grad_fn] if tensor.grad_fn is not None else []
    seen, names = set(), []
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        todo.extend(n for n, _ in node.next_functions)
    return sorted(names)


def tree(value, *, empty_mean=False, path=""):
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
        # NaN is never serialized as an unvalidated numeric JSON value.
        exceptional = empty_mean and path == "loss" and len(values) == 1 and math.isnan(values[0])
        return {"values": [] if exceptional else values, "shape": list(value.shape),
                "stride": list(value.stride()), "offset": value.storage_offset(),
                "dtype": str(value.dtype), "device": str(value.device),
                "requires_grad": value.requires_grad,
                "history": None if value.grad_fn is None else type(value.grad_fn).__name__,
                "history_nodes": history_nodes(value),
                "exception": "empty-mean-loss-NaN" if exceptional else None}
    if isinstance(value, dict):
        return {k: tree(v, empty_mean=empty_mean, path=f"{path}.{k}" if path else k)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree(v, empty_mean=empty_mean, path=f"{path}.{i}") for i, v in enumerate(value)]
    return value


def equal(actual, expected, label="record", *, parity=False, numeric=False):
    """Exact types/keys/CPU replay; only Vulkan tensor entries admit tolerance."""
    if type(actual) is not type(expected):
        raise ValueError(f"{label}: wrong type")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: wrong fields")
        tensor = "values" in expected and "dtype" in expected
        for key in expected:
            if parity and tensor and key == "device":
                equal(actual[key], "vk:0", label + ".device")
            else:
                equal(actual[key], expected[key], label + "." + key,
                      parity=parity and (not tensor or key == "values"),
                      numeric=parity and tensor and key == "values")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(label + ": wrong length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f"{label}.{i}", parity=parity, numeric=numeric)
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(label + ": nonfinite")
        if not (math.isclose(actual, expected, rel_tol=.003, abs_tol=.003) if numeric else actual == expected):
            raise ValueError(label + ": not CPU reference")
    elif actual != expected:
        raise ValueError(label + ": not source reference")


def phases(destination):
    @contextmanager
    def phase(name):
        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as trace:
                yield
        finally:
            # Only semantic operators/generated AD; SGD's Python wrapper label is
            # process-global instrumentation, not a different optimizer policy.
            destination[name] = {
                "ops": sorted({e.key for e in trace.key_averages()
                               if e.key.startswith("aten::") or "Backward" in e.key}),
                "leaves": [{"op": e.name, "inputs": e.input_shapes}
                           for e in trace.events() if e.name in MATRIX_LEAVES]}
    return phase


def combine_phases(cpu, vk):
    if cpu.keys() != vk.keys():
        raise ValueError("phase inventory mismatch")
    return {name: {"cpu_ops": cpu[name]["ops"], "cpu_leaves": cpu[name]["leaves"],
                   "vulkan_ops": vk[name]["ops"], "vulkan_leaves": vk[name]["leaves"]}
            for name in cpu}


def validate_phases(actual, cpu, *, model=False, mse_case=None):
    if not isinstance(actual, dict) or actual.keys() != cpu.keys():
        raise ValueError("phase inventory mismatch")
    for name, reference in cpu.items():
        phase = actual[name]
        if not isinstance(phase, dict) or set(phase) != {"cpu_ops", "cpu_leaves", "vulkan_ops", "vulkan_leaves"}:
            raise ValueError(name + ": invalid phase")
        equal(phase["cpu_ops"], reference["ops"], name + ".cpu_ops")
        equal(phase["cpu_leaves"], reference["leaves"], name + ".cpu_leaves")
        ops = phase["vulkan_ops"]
        if not isinstance(ops, list) or any(type(x) is not str for x in ops) or ops != sorted(set(ops)):
            raise ValueError(name + ": invalid Vulkan ops")
        required = set(reference["ops"]) & SEMANTIC_OPS
        if name.endswith("forward"):
            # CPU native MSE reduction is sum/div_; Vulkan numerical loss leaf
            # owns that reduction in its source-bound loss shader.
            required -= {"aten::sum", "aten::div_"}
        if not name.endswith("update"):
            # Generated gradient accumulation may allocate add instead of add_.
            required.discard("aten::add_")
        if model and not name.endswith("update"):
            required |= set(reference["ops"]) & {"aten::clone", "aten::copy_"}
        if mse_case is not None:
            if (name == "forward" and mse_case["kind"] == "broadcast") or (
                    name == "first" and (mse_case["kind"] == "broadcast" or
                    (mse_case["kind"] == "expanded-upstream" and mse_case["reduction"] == "none"))):
                # Scalar seeds are read directly. Public broadcast operands and
                # genuinely multi-element expanded upstreams still materialize.
                required |= {"aten::clone", "aten::copy_"}
        # Generated node witnesses bind manual MSE double backward and inverse
        # layout/reduction composition beyond superficial aten-prefix checks.
        required |= {x for x in reference["ops"] if "Backward" in x}
        if not required <= set(ops):
            raise ValueError(name + ": missing derivative/semantic closure " + str(required - set(ops)))
        if set(ops) & MATRIX_LEAVES != set(reference["ops"]) & MATRIX_LEAVES:
            raise ValueError(name + ": wrong matrix leaf route")
        equal(phase["vulkan_leaves"], reference["leaves"], name + ".matrix leaf inventory")
        if model and name.endswith("forward") and not {"aten::clone", "aten::copy_"} <= set(ops):
            # Pinned should_fold and cross-batch expand+reshape both materialize
            # noncontiguous original bases in every model forward.
            raise ValueError(name + ": missing fold/cross-broadcast materialization")


def validate_counters(counters, *, minimum_copies=0):
    if not isinstance(counters, list) or len(counters) != 4 or any(type(x) is not int or x < 0 for x in counters):
        raise ValueError("invalid counters")
    if counters[0] <= 0 or counters[2:] != [0, 0]:
        raise ValueError("missing device work/copy or fallback/transfer")
    if counters[1] < minimum_copies:
        raise ValueError("copy counter below source-established materialization minimum")


def validate_runtime(runtime):
    keys = {"pytorch_version", "pytorch_git_revision", "checkout_head", "extension_sha256",
            "extension_path", "device", "hardware", "driver", "vulkan_api_version",
            "vulkan_instance_version", "execution_mode"}
    if not isinstance(runtime, dict) or runtime.keys() != keys or any(type(v) is not str or not v for v in runtime.values()):
        raise ValueError("invalid runtime identity")
    if runtime["pytorch_version"] != torch.__version__ or runtime["pytorch_git_revision"] != torch.version.git_version:
        raise ValueError("wrong PyTorch reference")
    if runtime["device"] != "vk:0" or runtime["execution_mode"] not in ("async", "sync"):
        raise ValueError("wrong captured mode/device")
    for key, size in (("extension_sha256", 64), ("checkout_head", 40), ("pytorch_git_revision", 40)):
        if not re.fullmatch("[0-9a-f]{" + str(size) + "}", runtime[key]):
            raise ValueError("invalid runtime hash")


def validate_fd(rows):
    for row in rows:
        if not math.isclose(row["analytic"], row["fd"], rel_tol=.008, abs_tol=.002):
            raise ValueError("FD disagrees with analytic second reverse")


def qualify_runtime(recorded, root=ROOT):
    from vector_matmul_capability_evidence import runtime_identity
    current = {k: str(v) for k, v in runtime_identity(root).items()}
    for key in ("pytorch_version", "pytorch_git_revision", "extension_sha256", "device",
                "hardware", "driver", "vulkan_api_version", "vulkan_instance_version", "execution_mode"):
        equal(recorded[key], current[key], "live." + key)
