"""Source-owned capture factories for finite G1 vector/matmul evidence.

This module is deliberately separate from the broad conformance case table:
these cases capture CPU/Vulkan execution, generated reverse history, and named
stock composite routes as durable capability evidence.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_IDENTITY_PATHS = (
    "src/vulkan/operators/add.cpp",
    "src/vulkan/operators/linear.cpp",
    "src/vulkan/operators/linear.h",
    "tests/python/test_vulkan_mul_broadcast.py",
    "tests/python/test_vulkan_vector_matrix.py",
    "tests/python/vector_matmul_capability_evidence.py",
)


def runtime_identity(root: Path = ROOT) -> dict[str, str]:
    """Measure the installed build and selected Vulkan device identity."""
    import pytorch_vulkan

    summary = subprocess.run(
        ["vulkaninfo", "--summary"], check=True, capture_output=True, text=True
    ).stdout
    lines = summary.splitlines()
    device_headers = [i for i, line in enumerate(lines) if line.strip() == "Devices:"]
    if len(device_headers) != 1:
        raise RuntimeError("vulkaninfo must report exactly one Devices section")
    gpu_headers = [i for i in range(device_headers[0] + 1, len(lines))
                   if re.fullmatch(r"\s*GPU\d+:\s*", lines[i])]
    if len(gpu_headers) != 1:
        raise RuntimeError("vulkaninfo must report exactly one GPU device")
    fields: dict[str, str] = {}
    for line in lines[gpu_headers[0] + 1:]:
        match = re.fullmatch(
            r"\s*(deviceName|driverName|driverInfo|apiVersion)\s*=\s*(.*?)\s*", line
        )
        if match:
            if match.group(1) in fields:
                raise RuntimeError(f"duplicate Vulkan identity field {match.group(1)}")
            fields[match.group(1)] = match.group(2)
    if set(fields) != {"deviceName", "driverName", "driverInfo", "apiVersion"}:
        raise RuntimeError("vulkaninfo device identity is incomplete")
    instance = re.search(r"^Vulkan Instance Version:\s*(\S+)\s*$", summary, re.M)
    if instance is None:
        raise RuntimeError("vulkaninfo instance API version is missing")
    extension = Path(pytorch_vulkan._C.__file__).resolve()
    checkout = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "pytorch_version": torch.__version__,
        "pytorch_git_revision": str(torch.version.git_version),
        "checkout_head": checkout,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "extension_path": str(extension),
        "device": "vk:0",
        "hardware": fields["deviceName"],
        "driver": f"{fields['driverName']} {fields['driverInfo']}",
        "vulkan_api_version": fields["apiVersion"],
        "vulkan_instance_version": instance.group(1),
        "execution_mode": pytorch_vulkan._C.execution_mode(),
    }


def _case(name: str, schema: str, api: str, shapes: tuple[tuple[int, ...], ...],
          seed: int, *, route: str | None = None,
          mixed: tuple[int, int] | None = None,
          layouts: tuple[str, ...] | None = None) -> dict[str, Any]:
    if schema == "aten::mul.Tensor":
        output_shape = tuple(torch.broadcast_shapes(*shapes))
    elif schema == "aten::dot.default":
        output_shape = ()
    elif schema == "aten::mv.default":
        output_shape = (shapes[0][0],)
    elif len(shapes[0]) == 1 and len(shapes[1]) == 1:
        output_shape = ()
    elif len(shapes[0]) == 2 and len(shapes[1]) == 1:
        output_shape = (shapes[0][0],)
    else:
        output_shape = (shapes[1][1],)
    return {"name": name, "schema": schema, "api": api,
            "shapes": [list(shape) for shape in shapes], "seed": seed,
            "route": route, "mixed": list(mixed) if mixed is not None else None,
            "layouts": list(layouts or ("contiguous",) * len(shapes)),
            "output_shape": list(output_shape)}


VECTOR_MATMUL_CASES = (
    _case("g1.mul.broadcast.rank0-rank1", "aten::mul.Tensor", "mul",
          ((), (3,)), 611, mixed=(0, 1)),
    _case("g1.mul.broadcast.rank2", "aten::mul.Tensor", "mul",
          ((2, 1), (1, 3)), 612, mixed=(0, 1), layouts=("offset", "offset")),
    _case("g1.mul.broadcast.rank3-rank2", "aten::mul.Tensor", "mul",
          ((2, 1, 3), (4, 3)), 613),
    _case("g1.dot.rank1.forward", "aten::dot.default", "dot",
          ((3,), (3,)), 621, layouts=("nonunit", "contiguous")),
    _case("g1.dot.rank1.grad-a-wrt-b", "aten::dot.default", "dot",
          ((3,), (3,)), 622, mixed=(0, 1)),
    _case("g1.dot.rank1.grad-b-wrt-a", "aten::dot.default", "dot",
          ((3,), (3,)), 623, mixed=(1, 0)),
    _case("g1.mv.rank2-rank1.forward", "aten::mv.default", "mv",
          ((2, 3), (3,)), 631, layouts=("transpose", "nonunit")),
    _case("g1.mv.rank2-rank1.grad-m-wrt-x", "aten::mv.default", "mv",
          ((2, 3), (3,)), 632, mixed=(0, 1)),
    _case("g1.mv.rank2-rank1.grad-x-wrt-m", "aten::mv.default", "mv",
          ((2, 3), (3,)), 633, mixed=(1, 0)),
    _case("g1.matmul.vv.torch-matmul", "aten::matmul.default", "matmul",
          ((3,), (3,)), 641, route="dot", layouts=("offset", "nonunit")),
    _case("g1.matmul.mv.torch-matmul", "aten::matmul.default", "matmul",
          ((2, 3), (3,)), 642, route="mv", layouts=("transpose", "nonunit")),
    _case("g1.matmul.vm.torch-matmul", "aten::matmul.default", "matmul",
          ((3,), (3, 2)), 643, route="mm", layouts=("nonunit", "transpose")),
    _case("g1.matmul.vv.at", "aten::matmul.default", "at",
          ((3,), (3,)), 651, route="dot"),
    _case("g1.matmul.mv.at", "aten::matmul.default", "at",
          ((2, 3), (3,)), 652, route="mv"),
    _case("g1.matmul.vm.at", "aten::matmul.default", "at",
          ((3,), (3, 2)), 653, route="mm"),
)
REQUIRED_CASES = {case["name"]: case for case in VECTOR_MATMUL_CASES}


def source_identity(root: Path = ROOT) -> dict[str, str]:
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in SOURCE_IDENTITY_PATHS
    }


def _operation(api: str):
    if api == "mul":
        return torch.mul
    if api == "dot":
        return torch.dot
    if api == "mv":
        return torch.mv
    if api == "matmul":
        return torch.matmul
    if api == "at":
        return lambda left, right: left @ right
    raise ValueError(f"unknown source-owned vector/matmul API: {api}")


def _inputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])
    values = []
    for shape, layout in zip(case["shapes"], case["layouts"]):
        if layout == "contiguous":
            value = torch.randn(shape, generator=generator) + 0.35
        elif layout == "offset":
            strides = []
            stride = 1
            for extent in reversed(shape):
                strides.append(stride)
                stride *= extent
            value = (torch.randn(stride + 1, generator=generator) + 0.35).as_strided(
                shape, tuple(reversed(strides)), 1
            )
        elif layout == "nonunit":
            count = torch.empty(shape).numel()
            value = (torch.randn(max(1, count * 2 - 1), generator=generator) + 0.35).as_strided(
                shape, (2,), 0
            )
        elif layout == "transpose":
            if len(shape) != 2:
                raise ValueError(f"{case['name']}: transpose fixture must have rank two")
            base = torch.randn(tuple(reversed(shape)), generator=generator) + 0.35
            value = base.t()
        else:
            raise ValueError(f"{case['name']}: unsupported source-owned layout {layout}")
        values.append(value.requires_grad_())
    return tuple(values)


def source_fixture_input_values(case: dict[str, Any]) -> list[list[float]]:
    """Rebuild expected CPU values from the deterministic source-owned fixture."""
    return [value.detach().reshape(-1).tolist() for value in _inputs(case)]


def source_mixed_direction_values(case: dict[str, Any]) -> list[float]:
    if case["mixed"] is None:
        raise ValueError(f"{case['name']}: fixture has no selected mixed direction")
    target = case["mixed"][1]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"] + 3000)
    direction = torch.randn(case["shapes"][target], generator=generator) + 0.3
    return direction.reshape(-1).tolist()


def _seeded_tensor(shape, seed: int, shift: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator) + shift


def _cpu_tensor_fact(value: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(value.shape), "values": value.detach().reshape(-1).tolist()}


def _cpu_centered_fd(operation, inputs, first_target, second_target,
                     seed, probe, direction, eps=0.001) -> float:
    """Independent first-gradient differences; never uses the analytic mixed tensor."""
    base_inputs = tuple(value.detach() for value in inputs)

    def projected_first(step):
        shifted = [value.clone().requires_grad_() for value in base_inputs]
        shifted[second_target] = (
            base_inputs[second_target] + step * direction
        ).requires_grad_()
        output = operation(*shifted)
        first = torch.autograd.grad(output, shifted[first_target], seed)[0]
        return (first * probe).sum()

    return float((projected_first(eps) - projected_first(-eps)) / (2 * eps))


def cpu_fixture_reference(case: dict[str, Any]) -> dict[str, Any]:
    """Execute deterministic ordinary ATen/autograd on CPU, with independent FD.

    This oracle has no Vulkan import, capture call, persisted values, or formula
    table. All local semantic dependencies live in this source-hashed module;
    the external ATen/autograd implementation is identified by PyTorch revision.
    """
    inputs = _inputs(case)
    operation = _operation(case["api"])
    output = operation(*inputs)
    seed = _seeded_tensor(output.shape, case["seed"] + 1000, 0.4)
    first = torch.autograd.grad(output, inputs, seed, create_graph=True)
    reference = {
        "inputs": [value.detach().reshape(-1).tolist() for value in inputs],
        "output": output.detach().reshape(-1).tolist(),
        "first": [value.detach().reshape(-1).tolist() for value in first],
        "recipe": {"upstream_seed": _cpu_tensor_fact(seed),
                   "mixed_probe": None, "fd_epsilon": None},
    }
    if case["mixed"] is not None:
        first_target, second_target = case["mixed"]
        probe = _seeded_tensor(first[first_target].shape, case["seed"] + 2000, 0.25)
        direction = _seeded_tensor(inputs[second_target].shape, case["seed"] + 3000, 0.3)
        mixed = torch.autograd.grad(first[first_target], inputs[second_target], probe)[0]
        reference.update(
            mixed=mixed.detach().reshape(-1).tolist(),
            direction=direction.reshape(-1).tolist(),
            fd=_cpu_centered_fd(operation, inputs, first_target, second_target,
                                seed, probe, direction),
        )
        reference["recipe"].update(mixed_probe=_cpu_tensor_fact(probe), fd_epsilon=0.001)
    return reference


def _to_device_preserving_layout(value: torch.Tensor, device: str) -> torch.Tensor:
    storage_elements = value.untyped_storage().nbytes() // value.element_size()
    cpu_storage = torch.as_strided(value.detach(), (storage_elements,), (1,), 0)
    vk_storage = cpu_storage.to(device)
    return vk_storage.as_strided(value.shape, value.stride(), value.storage_offset()).requires_grad_()


def _meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "rank": tensor.dim(),
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "device": str(tensor.device),
    }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    actual_values = actual.detach().cpu().reshape(-1).tolist()
    expected_values = expected.detach().cpu().reshape(-1).tolist()
    return max(abs(left - right) for left, right in zip(actual_values, expected_values))


def _value_fact(cpu: torch.Tensor, vulkan: torch.Tensor) -> dict[str, Any]:
    cpu_values = cpu.detach().cpu().reshape(-1).tolist()
    vulkan_values = vulkan.detach().cpu().reshape(-1).tolist()
    return {
        "value_class": "finite",
        "shape": list(cpu.shape),
        "cpu_values": cpu_values,
        "vulkan_values": vulkan_values,
    }


def _sync() -> None:
    import pytorch_vulkan

    pytorch_vulkan._C.synchronize()


def _runtime_delta(before: tuple[int, ...], after: tuple[int, ...]) -> dict[str, int]:
    names = ("compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks")
    return {name: int(end - start) for name, start, end in zip(names, before, after)}


def _snapshot() -> tuple[int, ...]:
    import pytorch_vulkan

    return tuple(pytorch_vulkan._C.execution_counter_snapshot())


def _profiled_call(operation, inputs):
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        output = operation(*inputs)
    ops = sorted({event.key for event in trace.key_averages()
                  if event.key.startswith("aten::")})
    return output, ops


def _allclose_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = _error(actual, expected)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=0.003, atol=0.003)
    return error


def capture_vector_matmul_case(case: dict[str, Any], device: str = "vk:0",
                               root: Path = ROOT,
                               measured_runtime: dict[str, str] | None = None) -> dict[str, Any]:
    """Execute one source-owned case through its public PyTorch API on CPU and vk."""
    import pytorch_vulkan

    if case != REQUIRED_CASES.get(case.get("name")):
        raise ValueError(f"{case.get('name')}: case does not match the source-owned factory")
    operation = _operation(case["api"])
    cpu_inputs = _inputs(case)
    vk_inputs = tuple(_to_device_preserving_layout(value, device) for value in cpu_inputs)
    _sync()
    cpu_output, cpu_ops = _profiled_call(operation, cpu_inputs)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output, vk_ops = _profiled_call(operation, vk_inputs)
    _sync()
    forward_runtime = _snapshot()
    output_error = _allclose_error(vk_output, cpu_output)
    if (vk_output.device != torch.device(device)
            or list(cpu_output.shape) != case["output_shape"]):
        raise AssertionError(f"{case['name']}: output is not resident on {device}")

    cpu_seed = _seeded_tensor(cpu_output.shape, case["seed"] + 1000, 0.4)
    cpu_recipe = {"upstream_seed": _cpu_tensor_fact(cpu_seed),
                  "mixed_probe": None, "fd_epsilon": None}
    vk_seed = cpu_seed.to(device)
    cpu_first = torch.autograd.grad(cpu_output, cpu_inputs, cpu_seed, create_graph=True)
    pytorch_vulkan._C.reset_execution_counters()
    vk_first = torch.autograd.grad(vk_output, vk_inputs, vk_seed, create_graph=True)
    _sync()
    first_runtime = _snapshot()
    first_errors = [_allclose_error(actual, expected)
                    for actual, expected in zip(vk_first, cpu_first)]
    first_history = [
        {"target": f"input{index}",
         "cpu_requires_grad": bool(expected.requires_grad),
         "vulkan_requires_grad": bool(actual.requires_grad),
         "cpu_grad_fn": type(expected.grad_fn).__name__ if expected.grad_fn else None,
         "vulkan_grad_fn": type(actual.grad_fn).__name__ if actual.grad_fn else None,
         "parity": bool(expected.requires_grad == actual.requires_grad)}
        for index, (expected, actual) in enumerate(zip(cpu_first, vk_first))
    ]
    first_values = [
        _value_fact(expected, actual)
        for expected, actual in zip(cpu_first, vk_first)
    ]
    graph: dict[str, Any] = {
        "directions": ["first_reverse"],
        "first_reverse": {
            "targets": [f"input{i}" for i in range(len(cpu_inputs))],
            "max_abs_errors": first_errors,
            "history": first_history,
            "values": first_values,
        },
    }
    mixed_error = None
    if case["mixed"] is not None:
        first_target, second_target = case["mixed"]
        cpu_probe = _seeded_tensor(cpu_first[first_target].shape, case["seed"] + 2000, 0.25)
        cpu_recipe.update(mixed_probe=_cpu_tensor_fact(cpu_probe), fd_epsilon=0.001)
        vk_probe = cpu_probe.to(device)
        cpu_mixed = torch.autograd.grad(cpu_first[first_target], cpu_inputs[second_target], cpu_probe)[0]
        pytorch_vulkan._C.reset_execution_counters()
        mixed_before = _snapshot()
        vk_mixed = torch.autograd.grad(vk_first[first_target], vk_inputs[second_target], vk_probe)[0]
        _sync()
        mixed_after = _snapshot()
        mixed_error = _allclose_error(vk_mixed, cpu_mixed)

        # Independent nonzero CPU finite-difference projection in the second-input direction.
        direction = _seeded_tensor(cpu_inputs[second_target].shape, case["seed"] + 3000, 0.3)
        fd_value = _cpu_centered_fd(operation, cpu_inputs, first_target, second_target,
                                    cpu_seed, cpu_probe, direction)
        analytic_tensor = (cpu_mixed * direction).sum()
        cpu_mixed_values = cpu_mixed.detach().cpu().reshape(-1).tolist()
        direction_values = direction.detach().cpu().reshape(-1).tolist()
        analytic = sum(float(value) * float(delta)
                       for value, delta in zip(cpu_mixed_values, direction_values))
        torch.testing.assert_close(analytic_tensor, torch.tensor(fd_value), rtol=0.008, atol=0.002)
        # Count after synchronized readback because count_nonzero has no Vulkan leaf.
        if (not bool(torch.count_nonzero(cpu_mixed))
                or not bool(torch.count_nonzero(vk_mixed.detach().cpu()))):
            raise AssertionError(f"{case['name']}: mixed derivative lacks a nonzero signal")
        graph["directions"].append("selected_mixed_reverse")
        graph["selected_mixed_reverse"] = {
            "direction": f"grad_input{first_target}_wrt_input{second_target}",
            "target": f"input{second_target}",
            "cpu_nonzero": True,
            "vulkan_nonzero": True,
            "cpu_analytic_projection": analytic,
            "cpu_centered_fd_projection": fd_value,
            "cpu_fd_abs_error": abs(analytic - fd_value),
            "vulkan_projection": sum(
                float(value) * float(delta)
                for value, delta in zip(
                    vk_mixed.detach().cpu().reshape(-1).tolist(), direction_values
                )
            ),
            "max_abs_error": mixed_error,
            "values": _value_fact(cpu_mixed, vk_mixed),
            "direction_values": direction_values,
            "runtime": _runtime_delta(mixed_before, mixed_after),
        }

    output_route = case["route"]
    if output_route:
        leaf = f"aten::{output_route}"
        if leaf not in cpu_ops or leaf not in vk_ops:
            raise AssertionError(f"{case['name']}: stock route lacks {leaf} in profiler observations")
    records = [cpu_inputs[0], *cpu_inputs[1:]]
    record = {
        "schema": case["schema"],
        "operands": [
            {"role": "primary_input" if index == 0 else "operand",
             "dtype": str(t.dtype).removeprefix("torch."), "rank": t.dim()}
            for index, t in enumerate(records)
        ],
        "primary_input": {"dtype": "float32", "rank": cpu_inputs[0].dim()},
        "input_dtypes": ["float32"],
        "input_ranks": sorted({t.dim() for t in cpu_inputs}),
        "input_shapes": sorted({"x".join(map(str, t.shape)) for t in cpu_inputs if t.dim()}),
        "gradients": True,
        "parity": True,
        "output_dtype": str(cpu_output.dtype).removeprefix("torch."),
        "output_rank": cpu_output.dim(),
        "vector_matmul_evidence": {
            "case_id": case["name"],
            "api": case["api"],
            "route_leaf": output_route,
            "seed": case["seed"],
            "layouts": case["layouts"],
            "inputs": [{"cpu": _meta(cpu), "vulkan": _meta(vk),
                        "values": _value_fact(cpu, vk)}
                       for cpu, vk in zip(cpu_inputs, vk_inputs)],
            "output": {"cpu": _meta(cpu_output), "vulkan": _meta(vk_output)},
            "output_values": _value_fact(cpu_output, vk_output),
            "output_max_abs_error": output_error,
            "cpu_forward_ops": cpu_ops,
            "vulkan_forward_ops": vk_ops,
            "runtime": _runtime_delta((0, 0, 0, 0), forward_runtime),
            "first_backward_runtime": _runtime_delta((0, 0, 0, 0), first_runtime),
            "graph": graph,
            "cpu_reference_recipe": cpu_recipe,
            "source_identity": source_identity(root),
            "runtime_identity": measured_runtime or runtime_identity(root),
            "capture_command": (
                "vector_matmul_capability_evidence.capture_all_vector_matmul_cases(device='vk:0')"
            ),
        },
    }
    if case["mixed"] is not None:
        record["reverse_autograd"] = {"order": 2, "graph_preserved": True}
    return record


def capture_all_vector_matmul_cases(device: str = "vk:0", root: Path = ROOT
                                    ) -> dict[str, dict[str, Any]]:
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        raise RuntimeError("Vulkan is unavailable for vector/matmul evidence capture")
    measured_runtime = runtime_identity(root)
    return {case["name"]: capture_vector_matmul_case(case, device, root, measured_runtime)
            for case in VECTOR_MATMUL_CASES}
