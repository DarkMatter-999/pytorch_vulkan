"""Minimal torch.compile backend for the Vulkan operator subset."""

from collections.abc import Mapping
import time

import torch

from .compiler_metadata import validate_compiler_tensor_metadata
from .compiler_metadata import _is_symbolic_dimension
from torch._subclasses.fake_tensor import FakeTensor


class VulkanCompilerError(RuntimeError):
    """Raised when a graph is outside the Vulkan compiler contract."""


_SUPPORTED_MODULES = (torch.nn.Flatten, torch.nn.Linear, torch.nn.ReLU)
_SUPPORTED_FUNCTIONS = {
    torch.ops.aten.linear.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
}
_LAST_STATS = {
    "node_count": 0,
    "setup_time": 0.0,
    "calls": 0,
    "replay_time": 0.0,
    "call_metrics": [],
    "fusion_applied": False,
}


def _counter_snapshot():
    from . import _C

    return {
        "dispatches": _C.compute_dispatch_count(),
        "submissions": _C.compute_submitted_count(),
        "completions": _C.compute_completed_count(),
        "waits": _C.compute_wait_count(),
        "transfers": _C.explicit_transfer_count(),
    }


def _counter_delta(before, after):
    return {name: after[name] - before[name] for name in before}


def _validate_value(value):
    if isinstance(value, torch.Tensor):
        if isinstance(value, FakeTensor):
            if value.device != torch.device("vk:0"):
                raise VulkanCompilerError("Vulkan compiler requires device vk:0")
            if value.dtype != torch.float32 or value.layout != torch.strided:
                raise VulkanCompilerError(
                    "Vulkan compiler requires strided contiguous F32 tensors"
                )
            if any(_is_symbolic_dimension(dim) for dim in value.shape):
                raise VulkanCompilerError(
                    "Vulkan compiler requires static contiguous F32 metadata"
                )
            if not value.is_contiguous():
                raise VulkanCompilerError(
                    "Vulkan compiler requires zero-offset contiguous tensors"
                )
            return
        validate_compiler_tensor_metadata(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_value(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_value(item)


def _validate_graph(gm):
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node.op == "call_module":
            try:
                module = gm.get_submodule(node.target)
            except AttributeError as error:
                raise VulkanCompilerError(
                    f"unsupported FX node call_module {node.target!r}"
                ) from error
            if not isinstance(module, _SUPPORTED_MODULES):
                raise VulkanCompilerError(
                    f"unsupported FX node call_module {type(module).__name__}"
                )
            if isinstance(module, torch.nn.Flatten) and (module.start_dim != 1 or module.end_dim != -1):
                raise VulkanCompilerError("unsupported FX node Flatten configuration")
            continue
        if node.op == "call_function" and node.target in _SUPPORTED_FUNCTIONS:
            continue
        raise VulkanCompilerError(f"unsupported FX node {node.op} {node.target!r}")


def compiler_stats():
    """Return a snapshot of the most recently lowered graph and replay calls."""
    return {
        **_LAST_STATS,
        "call_metrics": [dict(metrics) for metrics in _LAST_STATS["call_metrics"]],
    }


def _linear_relu_replay(gm, inputs):
    output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
    nodes = [node for node in gm.graph.nodes if node.op != "output"]
    if len(output_nodes) != 1 or len(output_nodes[0].args) != 1:
        return None
    returned = output_nodes[0].args[0]
    if isinstance(returned, (tuple, list)):
        if len(returned) != 1:
            return None
        returned = returned[0]
    if returned is not nodes[-1]:
        return None
    if len(nodes) == 4:
        flatten = None
        placeholder, first_node, relu_node, second_node = nodes
    elif len(nodes) == 5:
        placeholder, flatten, first_node, relu_node, second_node = nodes
        if flatten.op != "call_module":
            return None
        flatten_module = gm.get_submodule(flatten.target)
        if not isinstance(flatten_module, torch.nn.Flatten):
            return None
        if flatten_module.start_dim != 1 or flatten_module.end_dim != -1:
            return None
        if flatten.args != (placeholder,) or flatten.kwargs:
            return None
    else:
        return None
    if [node.op for node in nodes if node is not flatten] != [
        "placeholder", "call_module", "call_module", "call_module"
    ]:
        return None
    if (
        first_node.args != ((flatten if flatten is not None else placeholder),)
        or relu_node.args != (first_node,)
        or second_node.args != (relu_node,)
        or any(node.kwargs for node in (first_node, relu_node, second_node))
    ):
        return None
    if len({node.target for node in (first_node, relu_node, second_node)}) != 3:
        return None
    first = gm.get_submodule(first_node.target)
    relu = gm.get_submodule(relu_node.target)
    second = gm.get_submodule(second_node.target)
    if not isinstance(first, torch.nn.Linear) or not isinstance(relu, torch.nn.ReLU) or not isinstance(second, torch.nn.Linear):
        return None
    if len(inputs) != 1 or inputs[0].ndim < 2:
        return None
    if any(_is_symbolic_dimension(dim) for dim in inputs[0].shape):
        return None
    if inputs[0].ndim == 2:
        input_features = inputs[0].shape[1]
    else:
        input_features = 1
        for dimension in inputs[0].shape[1:]:
            input_features *= dimension
    if (
        first.in_features != input_features
        or first.out_features != second.in_features
        or first.weight.ndim != 2
        or tuple(first.weight.shape) != (first.out_features, first.in_features)
        or second.weight.ndim != 2
        or tuple(second.weight.shape) != (second.out_features, second.in_features)
        or first.bias is None
        or second.bias is None
        or first.bias.ndim != 1
        or tuple(first.bias.shape) != (first.out_features,)
        or second.bias.ndim != 1
        or tuple(second.bias.shape) != (second.out_features,)
        or first.weight.dtype != second.weight.dtype
        or first.weight.device != second.weight.device
        or first.weight.layout != second.weight.layout
        or first.bias.dtype != second.bias.dtype
        or first.bias.device != second.bias.device
        or first.bias.layout != second.bias.layout
    ):
        return None
    fused_input = inputs[0]
    if flatten is not None:
        fused_input = inputs[0].reshape(inputs[0].shape[0], first.in_features)
    return (torch.nn.functional.linear(
        torch.ops.pytorch_vulkan.linear_relu(fused_input, first.weight, first.bias),
        second.weight,
        second.bias,
    ),)


def vulkan_backend(gm, example_inputs):
    """Lower a static FX graph to direct execution of Vulkan-dispatched ATen ops."""
    started = time.perf_counter()
    _validate_graph(gm)
    for parameter in gm.parameters():
        _validate_value(parameter)
    for value in example_inputs:
        if isinstance(value, torch.SymInt):
            raise VulkanCompilerError("dynamic symbolic shapes are unsupported")
        _validate_value(value)

    _LAST_STATS.update(
        node_count=sum(1 for node in gm.graph.nodes if node.op != "output"),
        setup_time=time.perf_counter() - started,
        calls=0,
        replay_time=0.0,
        call_metrics=[],
        fusion_applied=False,
    )

    def replay(*inputs):
        call_started = time.perf_counter()
        for value in inputs:
            _validate_value(value)
        before = _counter_snapshot()
        result = _linear_relu_replay(gm, inputs)
        if result is not None:
            _LAST_STATS["fusion_applied"] = True
        if result is None:
            result = gm(*inputs)
        _validate_value(result)
        after = _counter_snapshot()
        _LAST_STATS["calls"] += 1
        _LAST_STATS["replay_time"] += time.perf_counter() - call_started
        _LAST_STATS["call_metrics"].append(_counter_delta(before, after))
        return result

    return replay
