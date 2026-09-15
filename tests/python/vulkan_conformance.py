"""Small, executable operator cases shared by Vulkan conformance tests."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import pytorch_vulkan


TensorFactory = Callable[[], tuple[Any, ...]]


# These identifiers are the deliberately narrow operation surface in the
# capability matrix. Unsupported/deferred schemas are listed separately in
# the capability tests rather than inferred from registry case names.
DECLARED_OPERATION_MANIFEST = frozenset({
    "aten::sum.dim_IntList", "aten::mean.dim", "aten::argmax.default",
    "aten::linear.default", "aten::convolution.default",
    "aten::_adaptive_avg_pool2d.default", "aten::neg.default",
    "aten::abs.default", "aten::relu.default", "aten::add.Tensor",
    "aten::sub.Tensor", "aten::mul.Tensor", "aten::as_strided.default",
    "aten::view.default", "aten::_reshape_alias.default", "aten::reshape.default",
    "aten::masked_select.default",
    "aten::div.Tensor", "aten::lerp.Scalar_out", "aten::lerp_.Scalar",
    "aten::sqrt.out", "aten::add_.Tensor", "aten::mul_.Scalar",
    "aten::addcmul_.default", "aten::addcdiv_.default", "aten::zero_.default",
})


@dataclass(frozen=True)
class ConformanceCase:
    name: str
    family: str
    declaration_id: str
    operation: Callable[..., Any]
    input_factory: TensorFactory
    cpu_reference: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    supported: bool = True
    error_pattern: str = r"Vulkan"
    expected_dtype: torch.dtype = torch.float32
    expected_shape: tuple[int, ...] | None = None
    check_gradients: bool = False
    autograd_supported: bool = True
    autograd_error_type: type[Exception] | None = None
    autograd_error_pattern: str | None = None
    requires_grad_inputs: bool = False
    rtol: float = 1e-5
    atol: float = 1e-8
    execution_mode: str = "compute"
    convert_inputs: bool = True
    setup_inputs: Callable[[Any, str], Any] | None = None

    def inputs(self) -> tuple[Any, ...]:
        return self.input_factory(
            requires_grad=self.check_gradients or self.requires_grad_inputs
        )


def vulkan_backend() -> str:
    """Return the canonical device string used by the test harness."""
    return "vk:0"


def to_vulkan_inputs(value: Any, device: str = "vk:0") -> Any:
    """Move tensor operands recursively, leaving scalar metadata untouched."""
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        converted = value.to(device)
        if (tuple(value.stride()) != tuple(converted.stride()) or
                value.storage_offset() != converted.storage_offset()):
            converted = torch.empty_strided(
                value.size(), value.stride(), dtype=value.dtype, device=device
            )
            converted.copy_(value)
        return converted.detach().requires_grad_(value.requires_grad)
    if isinstance(value, tuple):
        return tuple(to_vulkan_inputs(item, device) for item in value)
    if isinstance(value, list):
        return [to_vulkan_inputs(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_vulkan_inputs(item, device) for key, item in value.items()}
    return value


def run_case(case: ConformanceCase, device: str = "vk:0") -> Any:
    inputs = case.inputs()
    if case.convert_inputs:
        inputs = to_vulkan_inputs(inputs, device)
    kwargs = case.kwargs or {}
    return case.operation(*inputs, *case.args, **kwargs)


def run_and_compare(case: ConformanceCase, device: str = "vk:0") -> tuple[Any, Any]:
    """Compute the CPU reference and Vulkan result with counters scoped to execution."""
    cpu_inputs = case.inputs()
    reference_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value for value in cpu_inputs
    )
    cpu_result = case.cpu_reference(*reference_inputs, *case.args, **(case.kwargs or {}))
    inputs = to_vulkan_inputs(cpu_inputs, device)
    pytorch_vulkan._C.reset_execution_counters()
    result = case.operation(*inputs, *case.args, **(case.kwargs or {}))
    return result, cpu_result


def assert_cpu_parity(case: ConformanceCase, result: torch.Tensor) -> None:
    inputs = case.inputs()
    expected = case.cpu_reference(*inputs, *case.args, **(case.kwargs or {}))
    torch.testing.assert_close(result.cpu(), expected)


def assert_vulkan_result(result: torch.Tensor, case: ConformanceCase) -> None:
    assert isinstance(result, torch.Tensor)
    assert result.device == torch.device("vk:0")
    assert result.dtype == case.expected_dtype
    if case.expected_shape is not None:
        assert tuple(result.shape) == case.expected_shape


def assert_gradients(case: ConformanceCase, device: str = "vk:0") -> None:
    cpu_inputs = case.inputs()
    vk_inputs = to_vulkan_inputs(cpu_inputs, device)
    kwargs = case.kwargs or {}
    cpu_result = case.cpu_reference(*cpu_inputs, *case.args, **kwargs)
    gradient = torch.ones_like(cpu_result)
    vk_gradient = gradient.to(device)
    pytorch_vulkan._C.reset_execution_counters()
    vk_result = case.operation(*vk_inputs, *case.args, **kwargs)
    cpu_result.backward(gradient)
    pytorch_vulkan._C.reset_execution_counters()
    vk_result.backward(vk_gradient)
    dispatches, vulkan_copies, explicit_transfers = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert case.execution_mode in {"compute", "copy"}
    assert dispatches > 0 or vulkan_copies > 0, (
        f"{case.name} backward performed neither Vulkan dispatch nor copy work"
    )
    assert explicit_transfers == 0, f"{case.name} backward transferred explicitly"
    for cpu_input, vk_input in zip(cpu_inputs, vk_inputs):
        if not isinstance(cpu_input, torch.Tensor) or not cpu_input.requires_grad:
            continue
        assert cpu_input.grad is not None
        assert vk_input.grad is not None
        assert vk_input.grad.device == torch.device(device)
        torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)


def _unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=requires_grad),)


def _unary_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    return (value.detach().requires_grad_(requires_grad),)


def _binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([3.0, 4.0], dtype=torch.float32, requires_grad=requires_grad),
    )


def _binary_strided(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    lhs = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    rhs = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, 1::2]
    return (lhs.detach().requires_grad_(requires_grad),
            rhs.detach().requires_grad_(requires_grad))


def _reduction(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=requires_grad),)


def _reduction_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    return (value.detach().requires_grad_(requires_grad),)


def _indexing(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([[1.0, 5.0], [9.0, 3.0]], dtype=torch.float32),)


def _indexing_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.tensor([[1.0, 8.0, 2.0, 7.0], [9.0, 3.0, 6.0, 4.0],
                          [5.0, 0.0, 11.0, 10.0]], dtype=torch.float32).t()
    return (value.detach().requires_grad_(requires_grad),)


def _view(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return (value.t().detach().requires_grad_(requires_grad),)


def _metadata_view(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return (value.detach().requires_grad_(requires_grad),)


def _linear(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    return (
        value.detach().requires_grad_(requires_grad),
        torch.ones((3, 4), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(3, dtype=torch.float32, requires_grad=requires_grad),
    )


def _linear_strided(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    bias = torch.arange(6, dtype=torch.float32)[::2]
    return tuple(item.detach().requires_grad_(requires_grad)
                 for item in (value, weight, bias))


def _convolution(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.ones((2, 1, 8, 8), dtype=torch.float32, requires_grad=requires_grad),
        torch.ones((4, 1, 3, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _convolution_strided(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, weight, bias = _convolution()
    value = value.transpose(2, 3)
    weight = weight.transpose(2, 3)
    return tuple(item.detach().requires_grad_(requires_grad)
                 for item in (value, weight, bias))


def _pooling(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),)


def _pooling_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).transpose(2, 3)
    return (value.detach().requires_grad_(requires_grad),)


def _masked_select(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([True, False, True]),
    )


def _masked_select_view(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    mask = torch.tensor([[True, False], [False, True], [True, False]])
    return (value.detach().requires_grad_(requires_grad), mask)


def _empty_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),)


def _empty_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),
            torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad))


def _empty_reduction(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),)


def _bool_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([True, False]),)


def _float16_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.0, -2.0], dtype=torch.float16),)


def _double_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))


def _mixed_device_add(value):
    return torch.add(value, torch.tensor([1.0, 1.0], dtype=value.dtype))


def _broadcast_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.ones((2, 1), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.float32))


def _optimizer_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor(2.0, dtype=torch.float32))


def _optimizer_single(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),)


def _optimizer_triple(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([0.5, 1.5], dtype=torch.float32),
            torch.tensor([2.0, 3.0], dtype=torch.float32))


def _optimizer_value_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32))


def _wrong_offset(*, requires_grad=False) -> tuple[torch.Tensor]:
    base = torch.ones(3, dtype=torch.float32)
    return (base[1:], torch.tensor(2.0, dtype=torch.float32))


def _setup_double_offset(inputs, device):
    value, _ = to_vulkan_inputs(inputs, device)
    value = value.to(torch.float64)
    return (value[1:], value[0])


def _overlapping(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty_strided((2, 2), (1, 0), dtype=torch.float32),)


def _invalid_out(value):
    return torch.neg(value, out=torch.empty_like(value, dtype=torch.int64))


def _cpu_out(value):
    return torch.neg(value, out=torch.empty(value.shape, dtype=value.dtype))


def _bad_pool_params(value):
    return torch.ops.aten._adaptive_avg_pool2d.default(value, (2, 2))


def _wrong_convolution(*, requires_grad=False):
    value, weight, bias = _convolution()
    return (value[:, :, :-1, :], weight, bias)


def _argmax_bad_out(value):
    out = torch.empty((1,), dtype=torch.int64, device=value.device)
    return torch.argmax(value, dim=1, out=out)


def _argmax_offset_out(value):
    out = torch.empty((3,), dtype=torch.int64, device=value.device)[1:]
    return torch.argmax(value, dim=1, out=out)


def _unsupported_overload(value):
    return value.neg_()


def _adaptive_pool(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.ones((2, 4, 3, 5), dtype=torch.float32, requires_grad=requires_grad),)


def _cpu_neg(value):
    return torch.neg(value)


def _cpu_add(lhs, rhs):
    return torch.add(lhs, rhs)


def _cpu_lerp(lhs, rhs):
    return torch.lerp(lhs, rhs, 0.25)


def _lerp_out(lhs, rhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.lerp.Scalar_out(lhs, rhs, 0.25, out=out)


def _lerp_inplace(lhs, rhs):
    return torch.ops.aten.lerp_.Scalar(lhs, rhs, 0.25)


def _addcmul_inplace(lhs, tensor1, tensor2):
    return torch.ops.aten.addcmul_.default(lhs, tensor1, tensor2, value=0.25)


def _addcdiv_inplace(lhs, tensor1, tensor2):
    return torch.ops.aten.addcdiv_.default(lhs, tensor1, tensor2, value=0.25)


def _add_inplace(lhs, rhs):
    return torch.ops.aten.add_.Tensor(lhs, rhs, alpha=1.0)


def _mul_scalar_inplace(lhs):
    return torch.ops.aten.mul_.Scalar(lhs, 2.0)


def _zero_inplace(lhs):
    return torch.ops.aten.zero_.default(lhs)


def _sqrt_out(lhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.sqrt.out(lhs, out=out)


def _sqrt_out(lhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.sqrt.out(lhs, out=out)


def _cpu_sum(value, dim, keepdim=False):
    return torch.sum(value, dim=dim, keepdim=keepdim)


def _cpu_mean(value, dim=None, keepdim=False):
    return torch.mean(value, dim=dim, keepdim=keepdim)


def _cpu_argmax(value, dim=None, keepdim=False):
    return torch.argmax(value, dim=dim, keepdim=keepdim)


def _cpu_reshape(value, shape):
    return torch.reshape(value, shape)


def _cpu_as_strided(value, size, stride):
    return torch.as_strided(value, size, stride)


def _cpu_linear(value, weight, bias):
    return torch.nn.functional.linear(value, weight, bias)


def _cpu_convolution(value, weight, bias, **kwargs):
    return torch.nn.functional.conv2d(value, weight, bias, **kwargs)


def _cpu_max_pool(value, kernel_size):
    return torch.nn.functional.max_pool2d(value, kernel_size)


def _cpu_masked_select(value, mask):
    return torch.masked_select(value, mask)


def _cpu_adaptive_pool(value, output_size):
    return torch.nn.functional.adaptive_avg_pool2d(value, output_size)


DECLARATION_ID_BY_CASE = {
    "unary.neg.float32": "aten::neg.default",
    "unary.neg.float32.strided": "aten::neg.default",
    "unary.abs.float32": "aten::abs.default",
    "unary.relu.float32.empty": "aten::relu.default",
    "binary.add.float32": "aten::add.Tensor",
    "binary.sub.float32.strided": "aten::sub.Tensor",
    "binary.mul.float32.empty": "aten::mul.Tensor",
    "reduction.sum.dim": "aten::sum.dim_IntList",
    "reduction.mean.dim": "aten::mean.dim",
    "reduction.sum.keepdim.strided": "aten::sum.dim_IntList",
    "reduction.mean.optional-dim": "aten::mean.dim",
    "reduction.sum.empty-dim": "aten::sum.dim_IntList",
    "reduction.mean.empty-dim": "aten::mean.dim",
    "indexing.argmax.dim": "aten::argmax.default",
    "indexing.argmax.optional-dim.keepdim": "aten::argmax.default",
    "indexing.argmax.strided": "aten::argmax.default",
    "view.reshape.float32": "aten::reshape.default",
    "view.as-strided.metadata": "aten::as_strided.default",
    "view.view.metadata": "aten::view.default",
    "view.reshape-alias.metadata": "aten::_reshape_alias.default",
    "linear.forward": "aten::linear.default",
    "linear.forward.strided": "aten::linear.default",
    "convolution.forward": "aten::convolution.default",
    "convolution.forward.strided": "aten::convolution.default",
    "pooling.max.rejected": "aten::max_pool2d_with_indices.default",
    "masked-select.bool-mask": "aten::masked_select.default",
    "masked-select.strided-value-view": "aten::masked_select.default",
    "optimizer.div.scalar": "aten::div.Tensor",
    "optimizer.lerp.out": "aten::lerp.Scalar_out",
    "optimizer.lerp.inplace": "aten::lerp_.Scalar",
    "optimizer.sqrt.out": "aten::sqrt.out",
    "optimizer.add.inplace": "aten::add_.Tensor",
    "optimizer.mul.scalar.inplace": "aten::mul_.Scalar",
    "optimizer.addcmul.inplace": "aten::addcmul_.default",
    "optimizer.addcdiv.inplace": "aten::addcdiv_.default",
    "optimizer.zero.inplace": "aten::zero_.default",
    "aten._adaptive_avg_pool2d.global": "aten::_adaptive_avg_pool2d.default",
    "aten._adaptive_avg_pool2d.global.strided": "aten::_adaptive_avg_pool2d.default",
    "unary.neg.bool.rejected": "aten::neg.default",
    "unary.neg.float16.rejected": "aten::neg.default",
    "binary.add.double.rejected": "aten::add.Tensor",
    "binary.add.mixed-device.rejected": "aten::add.Tensor",
    "binary.add.broadcast.rejected": "aten::add.Tensor",
    "reduction.sum.non-contiguous-overlap.rejected": "aten::sum.dim_IntList",
    "comparison.ne.nonzero-offset-input.rejected": "aten::ne.Tensor",
    "unary.neg.invalid-out.rejected": "aten::neg.out",
    "unary.neg.cpu-out.rejected": "aten::neg.out",
    "pooling.parameters.rejected": "aten::_adaptive_avg_pool2d.default",
    "convolution.shape.rejected": "aten::convolution.default",
    "unary.neg_.unsupported-overload.rejected": "aten::neg_.default",
}


def _case(name, family, operation, factory, *, args=(), kwargs=None,
          supported=True, error_pattern=r"Vulkan", expected_dtype=torch.float32,
          expected_shape=None, check_gradients=False, cpu_reference=None,
           autograd_supported=True, autograd_error_type=None,
           autograd_error_pattern=None, requires_grad_inputs=False, rtol=1e-5, atol=1e-8,
           execution_mode="compute", convert_inputs=True, setup_inputs=None):
    case = ConformanceCase(name, family, DECLARATION_ID_BY_CASE[name], operation, factory, cpu_reference, args,
                           kwargs, supported, error_pattern, expected_dtype,
                           expected_shape, check_gradients, autograd_supported,
                           autograd_error_type, autograd_error_pattern,
                           requires_grad_inputs, rtol, atol, execution_mode, convert_inputs,
                           setup_inputs)
    return case


ALL_CASES = (
    _case("unary.neg.float32", "unary", torch.neg, _unary, cpu_reference=_cpu_neg,
          expected_shape=(3,), check_gradients=True),
    _case("unary.neg.float32.strided", "unary", torch.neg, _unary_strided,
          cpu_reference=_cpu_neg, expected_shape=(2, 2), check_gradients=True),
    _case("unary.abs.float32", "unary", torch.abs, _unary,
          cpu_reference=torch.abs, expected_shape=(3,), check_gradients=True),
    _case("unary.relu.float32.empty", "unary", torch.relu, _empty_unary,
          cpu_reference=torch.relu, expected_shape=(0, 3), execution_mode="empty"),
    _case("binary.add.float32", "binary", torch.add, _binary, cpu_reference=_cpu_add,
          expected_shape=(2,), check_gradients=True),
    _case("binary.sub.float32.strided", "binary", torch.sub, _binary_strided,
          cpu_reference=torch.sub, expected_shape=(2, 2), check_gradients=True),
    _case("binary.mul.float32.empty", "binary", torch.mul, _empty_binary,
          cpu_reference=torch.mul, expected_shape=(0, 3), execution_mode="empty"),
    _case("reduction.sum.dim", "reduction", torch.sum, _reduction, args=(1,),
          cpu_reference=_cpu_sum, expected_shape=(2,), check_gradients=True),
    _case("reduction.mean.dim", "reduction", torch.mean, _reduction, args=(1,),
          cpu_reference=_cpu_mean, expected_shape=(2,), check_gradients=True),
    _case("reduction.sum.keepdim.strided", "reduction", torch.sum, _reduction_strided,
          args=(1,), kwargs={"keepdim": True}, cpu_reference=_cpu_sum,
          expected_shape=(3, 1, 4), check_gradients=True),
    _case("reduction.mean.optional-dim", "reduction", torch.mean, _reduction,
           cpu_reference=_cpu_mean, expected_shape=(), check_gradients=True),
    _case("reduction.sum.empty-dim", "reduction", torch.sum, _empty_reduction,
           args=(0,), cpu_reference=_cpu_sum, expected_shape=(3,), execution_mode="empty"),
    _case("reduction.mean.empty-dim", "reduction", torch.mean, _empty_reduction,
           args=(0,), cpu_reference=_cpu_mean, expected_shape=(3,), execution_mode="empty",
           rtol=0, atol=0),
    _case("indexing.argmax.dim", "indexing", torch.argmax, _indexing, args=(1,),
          cpu_reference=_cpu_argmax, expected_dtype=torch.int64, expected_shape=(2,)),
    _case("indexing.argmax.optional-dim.keepdim", "indexing", torch.argmax, _indexing,
          kwargs={"keepdim": True}, cpu_reference=_cpu_argmax,
          expected_dtype=torch.int64, expected_shape=(1, 1)),
    _case("indexing.argmax.strided", "indexing", torch.argmax, _indexing_strided,
          args=(-1,), cpu_reference=_cpu_argmax, expected_dtype=torch.int64,
          expected_shape=(4,)),
    _case("view.reshape.float32", "view", torch.reshape, _view, args=((6,),),
          cpu_reference=_cpu_reshape, expected_shape=(6,), check_gradients=True,
          execution_mode="copy"),
    _case("view.as-strided.metadata", "view", torch.as_strided, _metadata_view,
           args=((2, 3), (3, 1)), cpu_reference=_cpu_as_strided, expected_shape=(2, 3),
           execution_mode="metadata"),
    _case("view.view.metadata", "view", torch.ops.aten.view.default, _metadata_view,
           args=((6,),), cpu_reference=lambda value, shape: torch.ops.aten.view.default(value, shape),
           expected_shape=(6,), execution_mode="metadata"),
    _case("view.reshape-alias.metadata", "view", torch.ops.aten._reshape_alias.default,
           _metadata_view, args=((2, 3), (3, 1)),
           cpu_reference=lambda value, size, stride: torch.ops.aten._reshape_alias.default(
               value, size, stride), expected_shape=(2, 3), execution_mode="metadata"),
    _case("linear.forward", "linear", torch.nn.functional.linear, _linear,
          cpu_reference=_cpu_linear, expected_shape=(2, 3), check_gradients=True),
    _case("linear.forward.strided", "linear", torch.nn.functional.linear, _linear_strided,
          cpu_reference=_cpu_linear, expected_shape=(2, 3), check_gradients=True),
    _case("convolution.forward", "convolution", torch.nn.functional.conv2d, _convolution,
          kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
          cpu_reference=_cpu_convolution, expected_shape=(2, 4, 8, 8), check_gradients=True),
    _case("convolution.forward.strided", "convolution", torch.nn.functional.conv2d,
          _convolution_strided, kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
          cpu_reference=_cpu_convolution, expected_shape=(2, 4, 8, 8), check_gradients=True),
    _case("pooling.max.rejected", "pooling", torch.nn.functional.max_pool2d, _pooling,
          args=(2,), cpu_reference=_cpu_max_pool, supported=False,
          error_pattern=r"Vulkan pooling is not declared"),
    _case("masked-select.bool-mask", "masked-select", torch.masked_select, _masked_select,
          cpu_reference=_cpu_masked_select, expected_shape=(2,),
          autograd_supported=False, autograd_error_type=NotImplementedError,
           autograd_error_pattern=r"aten::(zero_|masked_scatter_)", requires_grad_inputs=True),
    _case("masked-select.strided-value-view", "masked-select", torch.masked_select,
           _masked_select_view, cpu_reference=_cpu_masked_select, expected_shape=(3,),
           execution_mode="copy"),
    _case("optimizer.div.scalar", "optimizer", torch.ops.aten.div.Tensor,
           _optimizer_pair, cpu_reference=torch.div, expected_shape=(2,)),
    _case("optimizer.lerp.out", "optimizer", _lerp_out, _optimizer_value_pair,
           cpu_reference=_cpu_lerp, expected_shape=(2,)),
    _case("optimizer.lerp.inplace", "optimizer", _lerp_inplace, _optimizer_value_pair,
           cpu_reference=_cpu_lerp, expected_shape=(2,)),
    _case("optimizer.sqrt.out", "optimizer", _sqrt_out, _optimizer_single,
           cpu_reference=torch.sqrt, expected_shape=(2,)),
    _case("optimizer.add.inplace", "optimizer", _add_inplace, _optimizer_value_pair,
           cpu_reference=torch.add, expected_shape=(2,)),
    _case("optimizer.mul.scalar.inplace", "optimizer", _mul_scalar_inplace,
           _optimizer_single, cpu_reference=lambda value: value * 2,
           expected_shape=(2,)),
    _case("optimizer.addcmul.inplace", "optimizer", _addcmul_inplace,
           _optimizer_triple, cpu_reference=lambda lhs, a, b: torch.addcmul(
               lhs, a, b, value=0.25), expected_shape=(2,)),
    _case("optimizer.addcdiv.inplace", "optimizer", _addcdiv_inplace,
           _optimizer_triple, cpu_reference=lambda lhs, a, b: torch.addcdiv(
               lhs, a, b, value=0.25), expected_shape=(2,)),
    _case("optimizer.zero.inplace", "optimizer", _zero_inplace,
           _optimizer_single, cpu_reference=lambda value: value.zero_(),
           expected_shape=(2,)),
    _case("aten._adaptive_avg_pool2d.global", "pooling",
          torch.ops.aten._adaptive_avg_pool2d.default, _adaptive_pool,
          args=((1, 1),), cpu_reference=_cpu_adaptive_pool, expected_shape=(2, 4, 1, 1),
           check_gradients=True),
    _case("aten._adaptive_avg_pool2d.global.strided", "pooling",
          torch.ops.aten._adaptive_avg_pool2d.default, _pooling_strided,
          args=((1, 1),), cpu_reference=_cpu_adaptive_pool, expected_shape=(1, 1, 1, 1),
          check_gradients=True),
    _case("unary.neg.bool.rejected", "unary", torch.neg, _bool_unary, supported=False,
          cpu_reference=_cpu_neg,
           error_pattern=r"supports only float32 tensors"),
    _case("unary.neg.float16.rejected", "unary", torch.neg, _float16_unary, supported=False,
           cpu_reference=_cpu_neg,
           error_pattern=r"float16 support is deferred"),
    _case("binary.add.double.rejected", "binary", torch.add, _double_binary,
           supported=False, convert_inputs=True, cpu_reference=_cpu_add,
           error_pattern=r"formatter conversion supports only Vulkan float32 to Vulkan Double"),
    _case("binary.add.mixed-device.rejected", "binary", _mixed_device_add, _unary,
           supported=False, cpu_reference=_cpu_add,
           error_pattern=r"requires operands on the same device; got"),
    _case("binary.add.broadcast.rejected", "binary", torch.add, _broadcast_binary,
           supported=False, cpu_reference=_cpu_add,
           error_pattern=r"requires equal tensor sizes; broadcasting is unsupported"),
    _case("reduction.sum.non-contiguous-overlap.rejected", "reduction", torch.sum, _overlapping,
           args=(1,), supported=False, cpu_reference=_cpu_sum,
           error_pattern=r"rejects overlapping input views"),
    _case("comparison.ne.nonzero-offset-input.rejected", "comparison",
            torch.ne, _wrong_offset, supported=False,
            cpu_reference=torch.ne,
            error_pattern=r"formatter Double ne\.Tensor requires a contiguous zero-offset strided input",
            setup_inputs=_setup_double_offset),
    _case("unary.neg.invalid-out.rejected", "unary", _invalid_out, _unary,
           supported=False, cpu_reference=_cpu_neg,
           error_pattern=r"supports only float32 and bool tensors, got Long"),
    _case("unary.neg.cpu-out.rejected", "unary", _cpu_out, _unary,
           supported=False, cpu_reference=_cpu_neg,
           error_pattern=r"requires a Vulkan output tensor"),
    _case("pooling.parameters.rejected", "pooling", _bad_pool_params, _adaptive_pool,
           args=(), supported=False, cpu_reference=_cpu_adaptive_pool,
           error_pattern=r"output size|\(1, 1\)"),
    _case("convolution.shape.rejected", "convolution", torch.nn.functional.conv2d,
           _wrong_convolution, kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
           supported=False, cpu_reference=_cpu_convolution,
           error_pattern=r"unsupported fixed shape|shape"),
    _case("unary.neg_.unsupported-overload.rejected", "unary", _unsupported_overload,
           _unary, supported=False, cpu_reference=_cpu_neg,
           error_pattern=r"Vulkan neg in-place variants are unsupported"),
)


SUPPORTED_CASES = tuple(case for case in ALL_CASES if case.supported)
