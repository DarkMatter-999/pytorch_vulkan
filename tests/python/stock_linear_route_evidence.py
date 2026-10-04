"""Measured stock Linear route capture shared by tests and the refresh tool."""

from __future__ import annotations

from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from tools.vulkan_capability_declarations import STOCK_LINEAR_ROUTE_CASES


def _sync() -> None:
    import pytorch_vulkan

    pytorch_vulkan._C.synchronize()


def _counters() -> dict[str, int]:
    import pytorch_vulkan

    return {
        "compute_dispatches": pytorch_vulkan._C.compute_dispatch_count(),
        "vulkan_copies": pytorch_vulkan._C.vulkan_copy_count(),
        "explicit_transfers": pytorch_vulkan._C.explicit_transfer_count(),
        "fallbacks": pytorch_vulkan._C.fallback_count(),
    }


def _profile_call(module: torch.nn.Module, input: torch.Tensor, no_grad: bool):
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        with profile(activities=[ProfilerActivity.CPU]) as trace:
            output = module(input)
    return output, sorted({event.key for event in trace.key_averages() if event.key.startswith("aten::")})


def _grad_fact(name: str, cpu: torch.Tensor, vk: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "cpu_shape": list(cpu.shape),
        "vulkan_shape": list(vk.shape),
        "cpu_requires_grad": cpu.requires_grad,
        "vulkan_requires_grad": vk.requires_grad,
        "cpu_grad_fn": type(cpu.grad_fn).__name__ if cpu.grad_fn is not None else None,
        "vulkan_grad_fn": type(vk.grad_fn).__name__ if vk.grad_fn is not None else None,
        "parity": True,
    }


def capture_stock_linear_route(case: dict[str, Any], device: str = "vk:0") -> dict[str, Any]:
    """Execute one ordinary CPU/Vulkan Linear route and return durable evidence."""
    import pytorch_vulkan

    rank = case["rank"]
    state = case["state"]
    bias_present = case["bias_present"]
    input_shape = tuple(case["input_shape"])
    generator = torch.Generator(device="cpu").manual_seed(
        91237 + rank * 101 + int(bias_present) * 7 + (0 if state == "trainable" else 1 if state == "frozen-weight" else 2)
        + (37 if case["layout"] == "offset-transposed" else 0)
    )
    if case["layout"] == "offset-transposed":
        base_shape = (2,) * (rank - 1) + (5,)
        cpu_base = torch.randn(base_shape, generator=generator)
        cpu_input = cpu_base.transpose(0, 1)[..., 1:4]
    else:
        cpu_base = None
        cpu_input = torch.randn(input_shape, generator=generator)
    cpu_weight = torch.randn((4, 3), generator=generator)
    cpu_bias = torch.randn((4,), generator=generator) if bias_present else None
    needs_grad = state != "no-grad"
    cpu_input.requires_grad_(needs_grad)

    cpu_module = torch.nn.Linear(3, 4, bias=bias_present)
    with torch.no_grad():
        cpu_module.weight.copy_(cpu_weight)
        if bias_present:
            cpu_module.bias.copy_(cpu_bias)
    cpu_module.weight.requires_grad_(state == "trainable")
    if bias_present:
        cpu_module.bias.requires_grad_(state == "trainable")

    if cpu_base is None:
        vk_input = cpu_input.detach().to(device)
    else:
        vk_base = cpu_base.to(device)
        vk_input = vk_base.transpose(0, 1)[..., 1:4]
    vk_input.requires_grad_(needs_grad)
    vk_module = torch.nn.Linear(3, 4, bias=bias_present)
    with torch.no_grad():
        vk_module.weight.copy_(cpu_weight)
        if bias_present:
            vk_module.bias.copy_(cpu_bias)
    vk_module.weight.requires_grad_(state == "trainable")
    if bias_present:
        vk_module.bias.requires_grad_(state == "trainable")
    vk_module.to(device)
    _sync()

    cpu_output, cpu_ops = _profile_call(cpu_module, cpu_input, state == "no-grad")
    pytorch_vulkan._C.reset_execution_counters()
    vk_output, vk_ops = _profile_call(vk_module, vk_input, state == "no-grad")
    _sync()
    forward_counters = _counters()
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=0.003, atol=0.003)
    if vk_output.device != torch.device(device):
        raise AssertionError(f"{case['name']}: Linear output left {device}")
    if (case["matrix_leaf"].removesuffix(".default") not in vk_ops
            or case["matrix_leaf"].removesuffix(".default") not in cpu_ops):
        raise AssertionError(f"{case['name']}: matrix leaf missing from CPU/Vulkan profiler route")
    required_ops = {name.removesuffix(".default") for name in case["required_forward_ops"]}
    if not required_ops <= set(vk_ops) or not required_ops <= set(cpu_ops):
        raise AssertionError(f"{case['name']}: required profiler route operators are missing")

    backward_counters = None
    second_counters = None
    gradients: list[dict[str, Any]] = []
    second_direction: dict[str, Any] | None = None
    selected_second = case["second_direction"]
    if state != "no-grad":
        targets_cpu = [cpu_input]
        targets_vk = [vk_input]
        target_names = ["input"]
        if state == "trainable":
            targets_cpu.append(cpu_module.weight)
            targets_vk.append(vk_module.weight)
            target_names.append("weight")
            if bias_present:
                targets_cpu.append(cpu_module.bias)
                targets_vk.append(vk_module.bias)
                target_names.append("bias")
        cpu_first = torch.autograd.grad(
            (cpu_output * cpu_output).sum(), targets_cpu, create_graph=state == "trainable"
        )
        pytorch_vulkan._C.reset_execution_counters()
        vk_first = torch.autograd.grad(
            (vk_output * vk_output).sum(), targets_vk, create_graph=state == "trainable"
        )
        _sync()
        backward_counters = _counters()
        for name, expected, actual in zip(target_names, cpu_first, vk_first):
            if actual.device != torch.device(device):
                raise AssertionError(f"{case['name']}: {name} gradient left {device}")
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
            if state == "trainable" and not (expected.requires_grad and actual.requires_grad):
                raise AssertionError(f"{case['name']}: requested first-gradient graph history was lost")
            gradients.append(_grad_fact(name, expected, actual))
        if selected_second:
            if not all(expected.requires_grad == actual.requires_grad for expected, actual in zip(cpu_first, vk_first)):
                raise AssertionError(f"{case['name']}: first-gradient graph history differs")
            cpu_directions = tuple(torch.full_like(g, 0.25 * (i + 1)) for i, g in enumerate(cpu_first))
            vk_directions = tuple(d.to(device) for d in cpu_directions)
            cpu_scalar = sum((g * direction).sum() for g, direction in zip(cpu_first, cpu_directions))
            vk_scalar = sum((g * direction).sum() for g, direction in zip(vk_first, vk_directions))
            cpu_second = torch.autograd.grad(cpu_scalar, targets_cpu)
            pytorch_vulkan._C.reset_execution_counters()
            vk_second = torch.autograd.grad(vk_scalar, targets_vk)
            _sync()
            second_counters = _counters()
            for expected, actual in zip(cpu_second, vk_second):
                if actual.device != torch.device(device):
                    raise AssertionError(f"{case['name']}: selected second derivative left {device}")
                torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
            second_direction = {
                "direction": ["0.25", "0.50", "0.75"],
                "target_names": target_names,
                "cpu_vulkan_parity": True,
                "cpu_requires_grad": [g.requires_grad for g in cpu_first],
                "vulkan_requires_grad": [g.requires_grad for g in vk_first],
            }

    primary = {"dtype": "float32", "rank": rank}
    tensors = [cpu_input, cpu_weight] + ([cpu_bias] if cpu_bias is not None else [])
    route_record = {
        "case_id": case["name"],
        "schema": "aten::linear.default",
        "rank": rank,
        "input_shape": list(cpu_input.shape),
        "layout": case["layout"],
        "state": state,
        "bias_present": bias_present,
        "input_metadata": {
            "cpu": {"shape": list(cpu_input.shape), "strides": list(cpu_input.stride()),
                    "storage_offset": cpu_input.storage_offset()},
            "vulkan": {"shape": list(vk_input.shape), "strides": list(vk_input.stride()),
                       "storage_offset": vk_input.storage_offset()},
        },
        "cpu_forward_ops": cpu_ops,
        "vulkan_forward_ops": vk_ops,
        "matrix_leaf": case["matrix_leaf"],
        "route": case["route"],
        "required_forward_ops": case["required_forward_ops"],
        "forward_counters": forward_counters,
        "first_gradient_facts": gradients,
        "first_backward_counters": backward_counters,
        "first_gradient_graph_preserved": state == "trainable",
        "selected_second_direction": second_direction,
        "selected_second_counters": second_counters,
        "output_device": str(vk_output.device),
        "output_requires_grad": vk_output.requires_grad,
        "grad_mode": "no_grad" if state == "no-grad" else "enabled",
    }
    input_ranks = sorted({tensor.dim() for tensor in tensors})
    input_shapes = sorted({"x".join(map(str, tensor.shape)) for tensor in tensors if tensor.dim()})
    return {
        "schema": "aten::linear.default",
        "operands": [
            {"role": "primary_input" if index == 0 else "operand",
             "dtype": str(tensor.dtype).removeprefix("torch."), "rank": tensor.dim()}
            for index, tensor in enumerate(tensors)
        ],
        "primary_input": primary,
        "input_dtypes": ["float32"],
        "input_ranks": input_ranks,
        "input_shapes": input_shapes,
        "gradients": state != "no-grad",
        "parity": True,
        "output_dtype": "float32",
        "output_rank": rank,
        "stock_linear_route": route_record,
    }


def capture_all_stock_linear_routes(device: str = "vk:0") -> dict[str, dict[str, Any]]:
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        raise RuntimeError("Vulkan is unavailable for stock Linear route capture")
    records = {
        case["name"]: capture_stock_linear_route(case, device)
        for case in STOCK_LINEAR_ROUTE_CASES
    }
    return records
