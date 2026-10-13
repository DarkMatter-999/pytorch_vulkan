#!/usr/bin/env python3
"""Acquire no sources; inspect pinned snapshots and record a CPU-only oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


REVISION = "e4ee3be4063b7c430974252fdf7db42273388d86"
VERSION = "2.4.0+cpu"
EXPECTED_SOURCE_SHA256 = {
    "aten/src/ATen/native/native_functions.yaml": "4b3babc71d75a966028092fdbb8a52398c103f6e2c6bb8b63e6ca7d15b82f768",
    "tools/autograd/derivatives.yaml": "9b9c2ddabf4ca49d3483c0ebff179886dc21ae1db46abd493c72073c6bacc972",
    "aten/src/ATen/native/Linear.cpp": "3264a545db3f6da93dad03e4f65364b980572e936485117e2376b2a7bafa0895",
    "aten/src/ATen/native/LossNLL.cpp": "fe5d5996e6fee506cad9aea21c198233e32d5b57dde8eac56830c21f23801154",
    "aten/src/ATen/native/SoftMax.cpp": "a01775813dfe2bb66d0c76a390c950dd64c87ffe27cb3333a2c85201ce885a68",
    "torch/csrc/autograd/FunctionsManual.cpp": "f7c074589357d685a4910110de79072cd1daa044f5d5d598110099ee04b6010e",
    "torch/optim/sgd.py": "ec6cda6d3cee92721b4dae4c54825313c5b405af0c97a109e7f6b53badc64005",
    "torch/optim/optimizer.py": "46f911d59955d779b3323a61cbc35c33725da64edfff6b9b2fa7e59b732b1a5b",
}
SOURCE_LINES = {
    "aten/src/ATen/native/Linear.cpp": ["addmm", "Tensor linear"],
    "aten/src/ATen/native/LossNLL.cpp": ["cross_entropy_loss", "nll_loss_forward", "nll_loss_backward"],
    "aten/src/ATen/native/SoftMax.cpp": ["log_softmax", "_log_softmax_backward_data"],
    "tools/autograd/derivatives.yaml": ["nll_loss_backward", "_log_softmax_backward_data"],
    "torch/optim/sgd.py": ["class SGD", "momentum_buffer", "def step"],
    "torch/optim/optimizer.py": ["def zero_grad", "set_to_none"],
    "torch/csrc/autograd/FunctionsManual.cpp": ["log_softmax"],
    "aten/src/ATen/native/native_functions.yaml": ["addmm", "log_softmax", "nll_loss_forward", "nll_loss_backward"],
}
OPERATORS = [
    "aten::addmm", "aten::linear", "aten::relu", "aten::log_softmax.int",
    "aten::nll_loss_forward.default", "aten::nll_loss_backward.default",
    "aten::cross_entropy_loss.default", "aten::_log_softmax_backward_data.default",
    "aten::sum.dim_IntList", "aten::view.default", "aten::t.default",
]


def source_inventory(root: Path, identities: dict | None = None,
                     source_lines: dict | None = None) -> dict:
    identities = identities or {
        name: {
            "sha256": digest,
            "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{REVISION}/{name}",
        }
        for name, digest in EXPECTED_SOURCE_SHA256.items()
    }
    source_lines = SOURCE_LINES if source_lines is None else source_lines
    inventory = {}
    for name, expected in identities.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required pinned source snapshot missing: {path}")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for pinned source {name}: expected {expected['sha256']}, got {digest}")
        text = content.decode("utf-8")
        line_refs = {}
        for needle in source_lines.get(name, []):
            hits = [i for i, line in enumerate(text.splitlines(), 1) if needle in line]
            if not hits:
                raise ValueError(f"missing expected source reference {needle!r} in {name}")
            line_refs[needle] = hits[:12]
        inventory[name] = {"sha256": digest,
                           "url": expected["url"],
                           "line_references": line_refs}
    return inventory


def validate_versions(torch_version: str, git_version: str | None) -> None:
    if torch_version != VERSION:
        raise RuntimeError(f"expected torch version {VERSION}, found {torch_version}")
    if git_version != REVISION:
        raise RuntimeError(f"expected torch git revision {REVISION}, found {git_version}")


def profiler_phases() -> dict:
    torch.manual_seed(12012)
    model = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(784, 32),
                                 torch.nn.ReLU(), torch.nn.Linear(32, 10))
    x = torch.rand((2, 1, 28, 28), generator=torch.Generator().manual_seed(12015))
    y = torch.tensor([0, 9], dtype=torch.int64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    events = {}
    actions = {
        "forward": lambda: model(x),
        "cross_entropy": lambda: F.cross_entropy(model(x), y),
    }
    for name, action in actions.items():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function(name):
                result = action()
        events[name] = _event_table(prof)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("backward"):
            loss.backward()
    events["backward"] = _event_table(prof)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("optimizer_step"):
            optimizer.step()
    events["optimizer_step"] = _event_table(prof)
    return events


def _event_table(prof) -> list:
    table = []
    for event in prof.events():
        if event.name.startswith("ProfilerStep") or event.name.startswith("record_function"):
            continue
        shapes = getattr(event, "input_shapes", [])
        table.append({"operator": event.name, "input_shapes": shapes})
    # Keep the observed route while avoiding repeated event records.
    seen = set()
    result = []
    for item in table:
        key = (item["operator"], json.dumps(item["input_shapes"], sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def cpu_comparisons() -> dict:
    output = {}
    for batch_size in (1, 2, 16):
        generator = torch.Generator(device="cpu").manual_seed(12013 + batch_size)
        images = torch.rand((batch_size, 1, 28, 28), generator=generator, dtype=torch.float32)
        labels = ([9] if batch_size == 1 else [0, 9] if batch_size == 2
                  else [i % 10 for i in range(16)])
        labels = torch.tensor(labels, dtype=torch.int64)
        logits_a = torch.randn((batch_size, 10), generator=generator, dtype=torch.float32).requires_grad_()
        logits_b = logits_a.detach().clone().requires_grad_()
        default_loss = torch.nn.CrossEntropyLoss()(logits_a, labels)
        composite_loss = F.nll_loss(F.log_softmax(logits_b, dim=1), labels)
        default_grad, = torch.autograd.grad(default_loss, logits_a)
        composite_grad, = torch.autograd.grad(composite_loss, logits_b)
        loss_delta = abs(float(default_loss) - float(composite_loss))
        grad_delta = float((default_grad - composite_grad).abs().max())
        output[str(batch_size)] = {
            "images_shape": list(images.shape), "labels": labels.tolist(),
            "loss_default": float(default_loss), "loss_logsoftmax_nll": float(composite_loss),
            "loss_abs_error": loss_delta, "gradient_max_abs_error": grad_delta,
            "finite": bool(torch.isfinite(default_loss) and torch.isfinite(default_grad).all()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_versions(torch.__version__, torch.version.git_version)
    if args.source_root.name != REVISION:
        raise RuntimeError(f"expected source revision directory {REVISION}, found {args.source_root.name}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    source = source_inventory(args.source_root)
    schemas, dispatch = {}, {}
    for operator in OPERATORS:
        schema_name, _, requested_overload = operator.partition(".")
        try:
            overload_name = "" if requested_overload in ("", "default") else requested_overload
            schemas[operator] = str(torch._C._get_schema(schema_name, overload_name))
        except (AttributeError, RuntimeError) as error:
            schemas[operator] = f"unavailable: {error}"
        dispatch_name = schema_name if requested_overload in ("", "default") else operator
        dispatch[operator] = torch._C._dispatch_dump_table(dispatch_name)
    defaults_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(defaults_model.parameters(), lr=0.01, momentum=0.9)
    options = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    comparisons = cpu_comparisons()
    max_loss_error = max(row["loss_abs_error"] for row in comparisons.values())
    max_gradient_error = max(row["gradient_max_abs_error"] for row in comparisons.values())
    receipt = {
        "evidence_kind": "CPU reference only; not Vulkan execution evidence",
        "torch_version": torch.__version__, "upstream_revision": REVISION,
        "source_snapshots": source,
        "installed": {"schemas": schemas, "dispatch_tables": dispatch},
        "profiler_phase_operators_and_shapes": profiler_phases(),
        "optimizer_defaults": options,
        "cpu_comparisons": comparisons,
        "numerical_rationale": {
            "default_cross_entropy_composition": "F.cross_entropy(logits, labels) is compared directly with F.nll_loss(F.log_softmax(logits, dim=1), labels).",
            "measured_ce_max_abs_error": max_loss_error,
            "measured_gradient_max_abs_error": max_gradient_error,
            "training_parity_ceilings": {"logits_loss_gradients_momentum": {"rtol": 2e-4, "atol": 2e-4},
                                          "parameters": {"rtol": 1e-4, "atol": 1e-5}},
            "basis": "CPU composition must be identical within measured float32 error; fixed Vulkan ceilings are the approved existing first-order CrossEntropy and momentum-SGD budgets, not measured device error.",
        },
        "process_settings": {"num_threads": torch.get_num_threads(),
                             "num_interop_threads": torch.get_num_interop_threads(),
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "cudnn_benchmark": torch.backends.cudnn.benchmark},
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        target.write(serialized)
    print(f"CPU reference receipt: {args.output}")
    print(f"batches=1,2,16; source={torch.__version__}@{torch.version.git_version}; Vulkan initialized=no")


if __name__ == "__main__":
    main()
