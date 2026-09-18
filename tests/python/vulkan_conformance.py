"""Small, executable operator cases shared by Vulkan conformance tests."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import pytorch_vulkan


TensorFactory = Callable[[], tuple[Any, ...]]


# These identifiers are the deliberately narrow operation surface in the
# capability matrix. Unsupported/deferred schemas are listed separately in
# the capability tests rather than inferred from registry case names.
DECLARED_OPERATION_MANIFEST = frozenset(
    {
        "aten::sum.dim_IntList",
        "aten::mean.dim",
        "aten::argmax.default",
        "aten::amax.default",
        "aten::amax.out",
        "aten::amin.default",
        "aten::amin.out",
        "aten::prod.dim_int",
        "aten::prod.int_out",
        "aten::_softmax.default",
        "aten::_softmax.out",
        "aten::_log_softmax.default",
        "aten::_log_softmax.out",
        "aten::_softmax_backward_data.out",
        "aten::_log_softmax_backward_data.out",
        "aten::linear.default",
        "aten::mm.default",
        "aten::addmm.default",
        "aten::addmm.out",
        "aten::convolution.default",
        "aten::convolution_backward.default",
        "aten::_adaptive_avg_pool2d.default",
        "aten::_adaptive_avg_pool2d_backward.default",
        "aten::neg.default",
        "aten::native_batch_norm.default",
        "aten::native_batch_norm_backward.default",
        "aten::nll_loss_forward.default",
        "aten::nll_loss_backward.default",
        "aten::abs.default",
        "aten::relu.default",
        "aten::add.Tensor",
        "aten::sub.Tensor",
        "aten::mul.Tensor",
        "aten::as_strided.default",
        "aten::view.default",
        "aten::_reshape_alias.default",
        "aten::reshape.default",
        "aten::masked_select.default",
        "aten::mse_loss.default",
        "aten::mse_loss_backward.default",
        "aten::sigmoid.default",
        "aten::tanh.default",
        "aten::gelu.default",
        "aten::sigmoid_backward.grad_input",
        "aten::tanh_backward.grad_input",
        "aten::gelu_backward.grad_input",
        "aten::div.Tensor",
        "aten::lerp.Scalar_out",
        "aten::lerp_.Scalar",
        "aten::add.Scalar",
        "aten::add.Scalar_out",
        "aten::add.out",
        "aten::sub.Scalar",
        "aten::sub.Scalar_out",
        "aten::sub.out",
        "aten::mul.Scalar",
        "aten::mul.Scalar_out",
        "aten::mul.out",
        "aten::rsub.Scalar",
        "aten::rsub.Scalar_out",
        "aten::sqrt.out",
        "aten::add_.Tensor",
        "aten::mul_.Scalar",
        "aten::addcmul_.default",
        "aten::addcdiv_.default",
        "aten::zero_.default",
        "aten::_copy_from.default",
        "aten::_to_copy.default",
        "aten::copy_.default",
        "aten::empty.memory_format",
        "aten::empty_strided.default",
    }
)

# Roadmap inventory only: these schemas are registered by the backend but remain
# deferred. Keeping this separate from DECLARED_OPERATION_MANIFEST prevents an
# inventory entry from being interpreted as a support declaration.
ROADMAP_OPERATION_FAMILIES = {
    "transfer/creation": frozenset(
        {
            "aten::_copy_from_and_resize.default",
            "aten::_local_scalar_dense.default",
            "aten::resize_.default",
            "aten::set_.source_Storage",
            "aten::set_.source_Storage_storage_offset",
        }
    ),
    "scalar and out= pointwise": frozenset(
        {
            "aten::abs.out",
            "aten::add_.Scalar",
            "aten::div.out",
            "aten::exp.out",
            "aten::fill_.Scalar",
            "aten::log.out",
            "aten::neg.out",
            "aten::relu.out",
            "aten::sigmoid.out",
            "aten::sigmoid_.default",
            "aten::tanh.out",
            "aten::tanh_.default",
            "aten::sub_.Scalar",
            "aten::sub_.Tensor",
            "aten::mul_.Tensor",
        }
    ),
    "reductions/indexing": frozenset(
        {
            "aten::argmax.out",
            "aten::max.default",
            "aten::mean.default",
            "aten::mean.out",
            "aten::min.default",
            "aten::sum.IntList_out",
            "aten::sum.default",
            "aten::maximum.out",
            "aten::minimum.out",
        }
    ),
    "MSE loss": frozenset(
        {
            "aten::binary_cross_entropy.default",
            "aten::binary_cross_entropy_backward.default",
            "aten::binary_cross_entropy_backward.grad_input",
        }
    ),
    "sigmoid/tanh/GELU": frozenset(
        {
            "aten::gelu.out",
            "aten::silu.out",
            "aten::silu_backward.grad_input",
        }
    ),
    "convolution/pooling backward": frozenset(
        {
            "aten::avg_pool2d.out",
            "aten::avg_pool2d_backward.grad_input",
            "aten::convolution_backward_overrideable.default",
            "aten::convolution_overrideable.default",
            "aten::max_pool2d_with_indices.default",
            "aten::upsample_bilinear2d.out",
            "aten::upsample_bilinear2d_backward.grad_input",
            "aten::upsample_nearest2d.out",
            "aten::upsample_nearest2d_backward.grad_input",
            "aten::_upsample_nearest_exact2d_backward.grad_input",
            "aten::_upsample_nearest_exact2d.out",
        }
    ),
    "normalization": frozenset(
        {
            "aten::native_layer_norm.default",
            "aten::native_layer_norm_backward.default",
        }
    ),
    # Cross-entropy is represented by the deferred log-softmax and NLL pieces.
    "cross-entropy/NLL": frozenset(
        {
            "aten::nll_loss_forward.output",
            "aten::nll_loss_backward.grad_input",
        }
    ),
    "attention": frozenset(
        {
            "aten::_native_multi_head_attention.default",
            "aten::_native_multi_head_attention.out",
            "aten::_transform_bias_rescale_qkv.default",
        }
    ),
    "tensor algebra": frozenset(
        {
            "aten::addcdiv.out",
            "aten::addcmul.out",
            "aten::bmm.out",
            "aten::dot.default",
            "aten::mm.out",
        }
    ),
    "tensor construction and indexing": frozenset(
        {
            "aten::_cat.default",
            "aten::arange.start_out",
            "aten::cat.out",
        }
    ),
    "comparison and math": frozenset(
        {
            "aten::atan.out",
            "aten::ceil.default",
            "aten::ceil.out",
            "aten::clamp.out",
            "aten::clamp_min.out",
            "aten::eq.Scalar_out",
            "aten::eq.Tensor_out",
            "aten::ge.Scalar_out",
            "aten::ge.Tensor_out",
            "aten::gt.Scalar",
            "aten::gt.Scalar_out",
            "aten::gt.Tensor_out",
            "aten::isfinite.out",
            "aten::le.Scalar_out",
            "aten::le.Tensor_out",
            "aten::le.Tensor_out",
            "aten::logit.default",
            "aten::logit.out",
            "aten::lt.Scalar",
            "aten::lt.Scalar_out",
            "aten::lt.Tensor_out",
            "aten::ne.Scalar_out",
            "aten::ne.Tensor",
            "aten::ne.Tensor_out",
            "aten::round.out",
            "aten::sgn.out",
        }
    ),
    "bitwise and random": frozenset(
        {
            "aten::bernoulli_.float",
            "aten::bitwise_and.Tensor_out",
            "aten::bitwise_not.out",
            "aten::bitwise_or.Tensor_out",
            "aten::bitwise_xor.Tensor_out",
            "aten::normal_.default",
            "aten::uniform_.default",
        }
    ),
    "activation and scalar math": frozenset(
        {
            "aten::hardsigmoid.out",
            "aten::hardsigmoid_backward.grad_input",
            "aten::hardswish_.default",
            "aten::hardswish_backward.default",
            "aten::hardtanh.default",
            "aten::hardtanh_.default",
            "aten::hardtanh_backward.default",
            "aten::leaky_relu.out",
            "aten::leaky_relu_backward.grad_input",
            "aten::log_sigmoid_backward.default",
            "aten::log_sigmoid_backward.grad_input",
            "aten::log_sigmoid_forward.default",
            "aten::log_sigmoid_forward.output",
            "aten::pow.Tensor_Scalar_out",
            "aten::reciprocal.out",
            "aten::threshold_backward.grad_input",
        }
    ),
    "dropout": frozenset(
        {
            "aten::native_dropout.default",
            "aten::native_dropout_backward.default",
        }
    ),
}

ROADMAP_DEFERRED_SCHEMAS = frozenset().union(*ROADMAP_OPERATION_FAMILIES.values())

ROADMAP_DEFERRED_REASON_BY_FAMILY = {
    "transfer/creation": "storage resizing, alias rebinding, and scalar readback contracts are deferred",
    "scalar and out= pointwise": "the overload-specific scalar, out, and in-place validation contracts are deferred",
    "reductions/indexing": "the deferred reduction and indexing shapes, outputs, or empty-input contracts are not implemented",
    "MSE loss": "binary cross-entropy support and its backward/reduction contracts are deferred",
    "sigmoid/tanh/GELU": "additional activation overloads and in-place paths are deferred",
    "convolution/pooling backward": "additional pooling, upsampling, and overrideable convolution forms are deferred",
    "normalization": "layer-normalization forward and backward contracts are deferred",
    "cross-entropy/NLL": "the output-form NLL contracts are deferred",
    "attention": "attention and QKV-rescaling contracts are deferred",
    "tensor algebra": "matrix, batched-matrix, and compound tensor algebra contracts are deferred",
    "tensor construction and indexing": "tensor concatenation and range-construction contracts are deferred",
    "comparison and math": "the additional comparison, clamp, and scalar-math overload contracts are deferred",
    "bitwise and random": "bitwise and random-generation contracts are deferred",
    "activation and scalar math": "additional activation backward and scalar-math contracts are deferred",
    "dropout": "dropout forward and backward contracts are deferred",
}


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
        if (
            tuple(value.stride()) != tuple(converted.stride())
            or value.storage_offset() != converted.storage_offset()
        ):
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
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in cpu_inputs
    )
    cpu_result = case.cpu_reference(
        *reference_inputs, *case.args, **(case.kwargs or {})
    )
    inputs = to_vulkan_inputs(cpu_inputs, device)
    pytorch_vulkan._C.reset_execution_counters()
    result = case.operation(*inputs, *case.args, **(case.kwargs or {}))
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    if case.execution_mode == "compute":
        assert dispatches > 0
    elif case.execution_mode == "copy":
        assert vulkan_copies > 0
    else:
        assert dispatches == 0 and vulkan_copies == 0
    assert explicit_transfers == 0
    assert fallbacks == 0
    return result, cpu_result


def assert_cpu_parity(case: ConformanceCase, result: torch.Tensor) -> None:
    inputs = case.inputs()
    expected = case.cpu_reference(*inputs, *case.args, **(case.kwargs or {}))
    torch.testing.assert_close(result.cpu(), expected)


def assert_vulkan_result(result: torch.Tensor, case: ConformanceCase) -> None:
    assert isinstance(result, torch.Tensor)
    assert result.device == torch.device(
        f"{torch._C._get_privateuse1_backend_name()}:0"
    )
    assert result.dtype == case.expected_dtype
    if case.expected_shape is not None:
        assert tuple(result.shape) == case.expected_shape


def assert_no_vulkan_work() -> None:
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert dispatches == 0
    assert vulkan_copies == 0
    assert explicit_transfers == 0
    assert fallbacks == 0


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
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert case.execution_mode in {"compute", "copy"}
    assert dispatches > 0 or vulkan_copies > 0, (
        f"{case.name} backward performed neither Vulkan dispatch nor copy work"
    )
    assert explicit_transfers == 0, f"{case.name} backward transferred explicitly"
    assert fallbacks == 0, f"{case.name} backward used implicit CPU fallback"
    for cpu_input, vk_input in zip(cpu_inputs, vk_inputs):
        if not isinstance(cpu_input, torch.Tensor) or not cpu_input.requires_grad:
            continue
        assert cpu_input.grad is not None
        assert vk_input.grad is not None
        assert vk_input.grad.device == torch.device(device)
        torch.testing.assert_close(
            vk_input.grad.cpu(), cpu_input.grad, rtol=case.rtol, atol=case.atol
        )


def _unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (
        torch.tensor(
            [-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=requires_grad
        ),
    )


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
    return (
        lhs.detach().requires_grad_(requires_grad),
        rhs.detach().requires_grad_(requires_grad),
    )


def _loss(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=requires_grad
        ),
        torch.tensor([[0.5, 1.0], [2.0, 5.0]], dtype=torch.float32),
    )


def _mse_none(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="none")


def _mse_sum(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="sum")


def _mse_mean(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="mean")


def _mse_backward(*, requires_grad=False):
    input, target = _loss(requires_grad=False)
    return (torch.ones_like(input), input, target)


def _mse_backward_op(grad, input, target):
    return torch.ops.aten.mse_loss_backward.default(grad, input, target, 0)


def _reduction(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=requires_grad
        ),
    )


def _reduction_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    return (value.detach().requires_grad_(requires_grad),)


def _indexing(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([[1.0, 5.0], [9.0, 3.0]], dtype=torch.float32),)


def _indexing_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.tensor(
        [[1.0, 8.0, 2.0, 7.0], [9.0, 3.0, 6.0, 4.0], [5.0, 0.0, 11.0, 10.0]],
        dtype=torch.float32,
    ).t()
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


def _linear_strided(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    bias = torch.arange(6, dtype=torch.float32)[::2]
    return tuple(
        item.detach().requires_grad_(requires_grad) for item in (value, weight, bias)
    )


def _mm(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(10, dtype=torch.float32).reshape(2, 5),
        torch.ones((5, 3), dtype=torch.float32),
    )


def _addmm(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((2, 3), dtype=torch.float32),
        torch.arange(10, dtype=torch.float32).reshape(2, 5),
        torch.ones((5, 3), dtype=torch.float32),
    )


def _addmm_out(value, mat1, mat2):
    return torch.addmm(value, mat1, mat2, out=torch.empty_like(value))


def _cpu_addmm(value, mat1, mat2):
    return torch.addmm(value, mat1, mat2)


def _cpu_mm(value, other):
    return torch.mm(value, other)


def _convolution(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.ones((2, 1, 8, 8), dtype=torch.float32, requires_grad=requires_grad),
        torch.ones((4, 1, 3, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _convolution_strided(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, weight, bias = _convolution()
    value = value.transpose(2, 3)
    weight = weight.transpose(2, 3)
    return tuple(
        item.detach().requires_grad_(requires_grad) for item in (value, weight, bias)
    )


def _convolution_backward_inputs(*, requires_grad=False) -> tuple[torch.Tensor, ...]:
    value, weight, _ = _convolution()
    return torch.ones((2, 4, 8, 8), dtype=torch.float32), value, weight


def _convolution_backward(grad, value, weight):
    return torch.ops.aten.convolution_backward.default(
        grad,
        value,
        weight,
        [4],
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        1,
        [True, True, True],
    )[0]


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


def _gelu_tanh(value):
    return torch.nn.functional.gelu(value, approximate="tanh")


def _activation_backward(*, requires_grad=False):
    value = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
    return torch.ones_like(value), value


def _sigmoid_backward_op(grad, output):
    return torch.ops.aten.sigmoid_backward.grad_input(
        grad, output, grad_input=torch.empty_like(output)
    )


def _tanh_backward_op(grad, output):
    return torch.ops.aten.tanh_backward.grad_input(
        grad, output, grad_input=torch.empty_like(output)
    )


def _gelu_backward_op(grad, input):
    return torch.ops.aten.gelu_backward.grad_input(
        grad, input, approximate="tanh", grad_input=torch.empty_like(input)
    )


def _sigmoid_backward_cpu(grad, output):
    return grad * output * (1.0 - output)


def _tanh_backward_cpu(grad, output):
    return grad * (1.0 - output * output)


def _gelu_backward_cpu(grad, input):
    input = input.detach().requires_grad_()
    result = torch.nn.functional.gelu(input, approximate="tanh")
    return torch.autograd.grad(result, input, grad)[0]


def _empty_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),
    )


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
    return (
        torch.ones((2, 1), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.float32),
    )


def _optimizer_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor(2.0, dtype=torch.float32),
    )


def _optimizer_single(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),)


def _optimizer_triple(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([0.5, 1.5], dtype=torch.float32),
        torch.tensor([2.0, 3.0], dtype=torch.float32),
    )


def _optimizer_value_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )


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


def _adaptive_pool_backward_inputs(*, requires_grad=False) -> tuple[torch.Tensor, ...]:
    return torch.ones((2, 4, 1, 1), dtype=torch.float32), _adaptive_pool()[0]


def _adaptive_pool_backward(grad, value):
    return torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, value)


def _normalization(*, requires_grad=False):
    return (
        torch.randn(2, 4, dtype=torch.float32, requires_grad=requires_grad),
        torch.ones(4, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )


def _native_batch_norm_output(input, weight, bias, running_mean, running_var):
    return torch.ops.aten.native_batch_norm.default(
        input, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )[0]


def _normalization_backward(*, requires_grad=False):
    input, weight, bias, running_mean, running_var = _normalization()
    _, save_mean, save_inv = torch.ops.aten.native_batch_norm.default(
        input, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )
    return (
        torch.ones_like(input),
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_inv,
    )


def _native_batch_norm_backward_output(
    grad, input, weight, running_mean, running_var, save_mean, save_inv
):
    return torch.ops.aten.native_batch_norm_backward.default(
        grad,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_inv,
        True,
        1e-5,
        [True, True, True],
    )[0]


def _nll_forward(*, requires_grad=False):
    return (
        torch.randn(2, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([1, 2], dtype=torch.int64),
    )


def _nll_forward_output(logits, labels):
    return torch.ops.aten.nll_loss_forward.default(
        torch.log_softmax(logits, dim=1), labels, None, 1, -100
    )[0]


def _nll_backward(*, requires_grad=False):
    logits = torch.randn(2, 3, dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.int64)
    log_probs = torch.log_softmax(logits, dim=1)
    _, total = torch.ops.aten.nll_loss_forward.default(log_probs, labels, None, 1, -100)
    return torch.ones(1, dtype=torch.float32), log_probs, labels, total.reshape(1)


def _nll_backward_output(grad, log_probs, labels, total):
    return torch.ops.aten.nll_loss_backward.default(
        grad.reshape(()), log_probs, labels, None, 1, -100, total.reshape(())
    )


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
    owns_training_step = not pytorch_vulkan._C.training_step_active()
    if owns_training_step:
        pytorch_vulkan._C.begin_training_step()
    try:
        result = torch.ops.aten.add_.Tensor(lhs, rhs, alpha=1.0)
        if owns_training_step:
            pytorch_vulkan._C.end_training_step()
        return result
    except Exception:
        if owns_training_step:
            pytorch_vulkan._C.cancel_training_step()
        raise


def _mul_scalar_inplace(lhs):
    owns_training_step = not pytorch_vulkan._C.training_step_active()
    if owns_training_step:
        pytorch_vulkan._C.begin_training_step()
    try:
        result = torch.ops.aten.mul_.Scalar(lhs, 2.0)
        if owns_training_step:
            pytorch_vulkan._C.end_training_step()
        return result
    except Exception:
        if owns_training_step:
            pytorch_vulkan._C.cancel_training_step()
        raise


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


def _cpu_prod(value, dim, keepdim=False):
    return torch.prod(value, dim=dim, keepdim=keepdim)


def _softmax(value, dim):
    return torch.softmax(value, dim=dim)


def _log_softmax(value, dim):
    return torch.log_softmax(value, dim=dim)


def _amax_out(value, dim):
    return torch.ops.aten.amax.out(
        value,
        [dim],
        False,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _amin_out(value, dim):
    return torch.ops.aten.amin.out(
        value,
        [dim],
        False,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _prod_int_out(value, dim):
    return torch.ops.aten.prod.int_out(
        value,
        dim,
        False,
        dtype=value.dtype,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _softmax_out(value, dim):
    return torch.ops.aten._softmax.out(value, dim, False, out=torch.empty_like(value))


def _log_softmax_out(value, dim):
    return torch.ops.aten._log_softmax.out(
        value, dim, False, out=torch.empty_like(value)
    )


def _softmax_backward_inputs(*, requires_grad=False):
    output = torch.softmax(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), dim=1
    )
    grad = torch.ones_like(output)
    return grad, output


def _softmax_backward_out(grad, output):
    return torch.ops.aten._softmax_backward_data.out(
        grad, output, 1, torch.float32, grad_input=torch.empty_like(output)
    )


def _log_softmax_backward_out(grad, output):
    return torch.ops.aten._log_softmax_backward_data.out(
        grad, output, 1, torch.float32, out=torch.empty_like(output)
    )


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
    "unary.sigmoid.float32": "aten::sigmoid.default",
    "unary.tanh.float32": "aten::tanh.default",
    "unary.gelu.tanh.float32": "aten::gelu.default",
    "unary.sigmoid.float32.empty": "aten::sigmoid.default",
    "unary.sigmoid.backward": "aten::sigmoid_backward.grad_input",
    "unary.tanh.backward": "aten::tanh_backward.grad_input",
    "unary.gelu.tanh.backward": "aten::gelu_backward.grad_input",
    "binary.add.float32": "aten::add.Tensor",
    "binary.sub.float32.strided": "aten::sub.Tensor",
    "binary.mul.float32.empty": "aten::mul.Tensor",
    "scalar.add.float32": "aten::add.Scalar",
    "scalar.sub.float32": "aten::sub.Scalar",
    "scalar.mul.float32": "aten::mul.Scalar",
    "out.add.tensor.float32": "aten::add.out",
    "out.sub.tensor.float32": "aten::sub.out",
    "out.mul.tensor.float32": "aten::mul.out",
    "out.add.scalar.float32": "aten::add.Scalar_out",
    "out.sub.scalar.float32": "aten::sub.Scalar_out",
    "out.mul.scalar.float32": "aten::mul.Scalar_out",
    "scalar.rsub.float32": "aten::rsub.Scalar",
    "out.rsub.scalar.float32": "aten::rsub.Scalar_out",
    "reduction.sum.dim": "aten::sum.dim_IntList",
    "reduction.mean.dim": "aten::mean.dim",
    "reduction.sum.keepdim.strided": "aten::sum.dim_IntList",
    "reduction.mean.optional-dim": "aten::mean.dim",
    "reduction.sum.empty-dim": "aten::sum.dim_IntList",
    "reduction.mean.empty-dim": "aten::mean.dim",
    "reduction.amax.dim": "aten::amax.out",
    "reduction.amax.default": "aten::amax.default",
    "reduction.amin.dim": "aten::amin.out",
    "reduction.amin.default": "aten::amin.default",
    "reduction.prod.dim": "aten::prod.int_out",
    "reduction.prod.default": "aten::prod.dim_int",
    "reduction.softmax.dim": "aten::_softmax.out",
    "reduction.softmax.default": "aten::_softmax.default",
    "reduction.log-softmax.dim": "aten::_log_softmax.out",
    "reduction.log-softmax.default": "aten::_log_softmax.default",
    "reduction.softmax.backward": "aten::_softmax_backward_data.out",
    "reduction.log-softmax.backward": "aten::_log_softmax_backward_data.out",
    "indexing.argmax.dim": "aten::argmax.default",
    "indexing.argmax.optional-dim.keepdim": "aten::argmax.default",
    "indexing.argmax.strided": "aten::argmax.default",
    "view.reshape.float32": "aten::reshape.default",
    "view.as-strided.metadata": "aten::as_strided.default",
    "view.view.metadata": "aten::view.default",
    "view.reshape-alias.metadata": "aten::_reshape_alias.default",
    "linear.forward": "aten::linear.default",
    "linear.forward.strided": "aten::linear.default",
    "mm.forward": "aten::mm.default",
    "addmm.forward": "aten::addmm.default",
    "addmm.out": "aten::addmm.out",
    "convolution.forward": "aten::convolution.default",
    "convolution.forward.strided": "aten::convolution.default",
    "convolution.backward": "aten::convolution_backward.default",
    "pooling.max.rejected": "aten::max_pool2d_with_indices.default",
    "masked-select.bool-mask": "aten::masked_select.default",
    "loss.mse.none": "aten::mse_loss.default",
    "loss.mse.sum": "aten::mse_loss.default",
    "loss.mse.mean": "aten::mse_loss.default",
    "loss.mse.backward": "aten::mse_loss_backward.default",
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
    "aten._adaptive_avg_pool2d.backward": "aten::_adaptive_avg_pool2d_backward.default",
    "normalization.native-batch-norm": "aten::native_batch_norm.default",
    "normalization.native-batch-norm-backward": "aten::native_batch_norm_backward.default",
    "classification.nll-forward": "aten::nll_loss_forward.default",
    "classification.nll-backward": "aten::nll_loss_backward.default",
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
    "transfer.copy_from.float32": "aten::_copy_from.default",
    "transfer.to_copy.float32": "aten::_to_copy.default",
    "transfer.copy.float32": "aten::copy_.default",
    "transfer.empty.float32": "aten::empty.memory_format",
    "transfer.empty_strided.float32": "aten::empty_strided.default",
}


def _case(
    name,
    family,
    operation,
    factory,
    *,
    args=(),
    kwargs=None,
    supported=True,
    error_pattern=r"Vulkan",
    expected_dtype=torch.float32,
    expected_shape=None,
    check_gradients=False,
    cpu_reference=None,
    autograd_supported=True,
    autograd_error_type=None,
    autograd_error_pattern=None,
    requires_grad_inputs=False,
    rtol=1e-5,
    atol=1e-8,
    execution_mode="compute",
    convert_inputs=True,
    setup_inputs=None,
):
    case = ConformanceCase(
        name,
        family,
        DECLARATION_ID_BY_CASE[name],
        operation,
        factory,
        cpu_reference,
        args,
        kwargs,
        supported,
        error_pattern,
        expected_dtype,
        expected_shape,
        check_gradients,
        autograd_supported,
        autograd_error_type,
        autograd_error_pattern,
        requires_grad_inputs,
        rtol,
        atol,
        execution_mode,
        convert_inputs,
        setup_inputs,
    )
    return case


def _transfer_pair(*, requires_grad=False):
    source = torch.arange(4, dtype=torch.float32, requires_grad=requires_grad)
    destination = torch.empty_like(source)
    return destination, source


def _copy_from(destination, source):
    return torch.ops.aten._copy_from.default(source, destination)


def _copy_from_cpu_reference(destination, source):
    return destination.copy_(source)


def _copy_in_place(destination, source):
    return torch.ops.aten.copy_.default(destination, source)


def _empty_like_template(template):
    return torch.empty_like(template, device=template.device)


def _empty_strided_like_template(template):
    return torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )


def _empty_strided_zero_template(*, requires_grad=False):
    return (torch.empty_strided((0, 2), (1, 2), dtype=torch.float32),)


def _scalar_add(value):
    return torch.add(value, 2.0, alpha=1.0)


def _scalar_sub(value):
    return torch.sub(value, 2.0, alpha=1.0)


def _scalar_mul(value):
    return torch.mul(value, 2.0)


def _scalar_add_out(value):
    return torch.add(value, 2.0, alpha=1.0, out=torch.empty_like(value))


def _scalar_sub_out(value):
    return torch.sub(value, 2.0, alpha=1.0, out=torch.empty_like(value))


def _scalar_mul_out(value):
    return torch.mul(value, 2.0, out=torch.empty_like(value))


def _tensor_add_out(lhs, rhs):
    return torch.add(lhs, rhs, alpha=1.0, out=torch.empty_like(lhs))


def _tensor_sub_out(lhs, rhs):
    return torch.sub(lhs, rhs, alpha=1.0, out=torch.empty_like(lhs))


def _tensor_mul_out(lhs, rhs):
    return torch.mul(lhs, rhs, out=torch.empty_like(lhs))


def _rscalar(value):
    return torch.ops.aten.rsub.Scalar(value, 2.0, 1.0)


def _rscalar_out(value):
    return torch.ops.aten.rsub.Scalar_out(value, 2.0, 1.0, out=torch.empty_like(value))


ALL_CASES = (
    _case(
        "unary.neg.float32",
        "unary",
        torch.neg,
        _unary,
        cpu_reference=_cpu_neg,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.neg.float32.strided",
        "unary",
        torch.neg,
        _unary_strided,
        cpu_reference=_cpu_neg,
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "unary.abs.float32",
        "unary",
        torch.abs,
        _unary,
        cpu_reference=torch.abs,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.relu.float32.empty",
        "unary",
        torch.relu,
        _empty_unary,
        cpu_reference=torch.relu,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "unary.sigmoid.float32",
        "unary",
        torch.sigmoid,
        _unary,
        cpu_reference=torch.sigmoid,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.tanh.float32",
        "unary",
        torch.tanh,
        _unary,
        cpu_reference=torch.tanh,
        expected_shape=(3,),
        check_gradients=True,
        rtol=1e-4,
        atol=2e-5,
    ),
    _case(
        "unary.gelu.tanh.float32",
        "unary",
        _gelu_tanh,
        _unary,
        cpu_reference=_gelu_tanh,
        expected_shape=(3,),
        check_gradients=True,
        rtol=2e-5,
        atol=2e-6,
    ),
    _case(
        "unary.sigmoid.float32.empty",
        "unary",
        torch.sigmoid,
        _empty_unary,
        cpu_reference=torch.sigmoid,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "unary.sigmoid.backward",
        "unary",
        _sigmoid_backward_op,
        _activation_backward,
        cpu_reference=_sigmoid_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "unary.tanh.backward",
        "unary",
        _tanh_backward_op,
        _activation_backward,
        cpu_reference=_tanh_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
        rtol=1e-4,
        atol=2e-5,
    ),
    _case(
        "unary.gelu.tanh.backward",
        "unary",
        _gelu_backward_op,
        _activation_backward,
        cpu_reference=_gelu_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
        rtol=3e-4,
        atol=2e-5,
    ),
    _case(
        "binary.add.float32",
        "binary",
        torch.add,
        _binary,
        cpu_reference=_cpu_add,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "binary.sub.float32.strided",
        "binary",
        torch.sub,
        _binary_strided,
        cpu_reference=torch.sub,
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "binary.mul.float32.empty",
        "binary",
        torch.mul,
        _empty_binary,
        cpu_reference=torch.mul,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "loss.mse.none",
        "loss",
        _mse_none,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="none"),
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "loss.mse.sum",
        "loss",
        _mse_sum,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="sum"),
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "loss.mse.mean",
        "loss",
        _mse_mean,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="mean"),
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "loss.mse.backward",
        "loss",
        _mse_backward_op,
        _mse_backward,
        cpu_reference=lambda g, x, y: torch.ops.aten.mse_loss_backward.default(
            g, x, y, 0
        ),
        expected_shape=(2, 2),
    ),
    _case(
        "scalar.add.float32",
        "scalar and out",
        _scalar_add,
        _unary,
        cpu_reference=lambda value: torch.add(value, 2.0, alpha=1.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "scalar.sub.float32",
        "scalar and out",
        _scalar_sub,
        _unary,
        cpu_reference=lambda value: torch.sub(value, 2.0, alpha=1.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "scalar.mul.float32",
        "scalar and out",
        _scalar_mul,
        _unary,
        cpu_reference=lambda value: torch.mul(value, 2.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "out.add.scalar.float32",
        "scalar and out",
        _scalar_add_out,
        _unary,
        cpu_reference=lambda value: torch.add(
            value, 2.0, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "out.sub.scalar.float32",
        "scalar and out",
        _scalar_sub_out,
        _unary,
        cpu_reference=lambda value: torch.sub(
            value, 2.0, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "out.mul.scalar.float32",
        "scalar and out",
        _scalar_mul_out,
        _unary,
        cpu_reference=lambda value: torch.mul(value, 2.0, out=torch.empty_like(value)),
        expected_shape=(3,),
    ),
    _case(
        "out.add.tensor.float32",
        "scalar and out",
        _tensor_add_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.add(
            lhs, rhs, alpha=1.0, out=torch.empty_like(lhs)
        ),
        expected_shape=(2,),
    ),
    _case(
        "out.sub.tensor.float32",
        "scalar and out",
        _tensor_sub_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.sub(
            lhs, rhs, alpha=1.0, out=torch.empty_like(lhs)
        ),
        expected_shape=(2,),
    ),
    _case(
        "out.mul.tensor.float32",
        "scalar and out",
        _tensor_mul_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.mul(lhs, rhs, out=torch.empty_like(lhs)),
        expected_shape=(2,),
    ),
    _case(
        "scalar.rsub.float32",
        "scalar and out",
        _rscalar,
        _unary,
        cpu_reference=lambda value: torch.sub(2.0, value, alpha=1.0),
        expected_shape=(3,),
    ),
    _case(
        "out.rsub.scalar.float32",
        "scalar and out",
        _rscalar_out,
        _unary,
        cpu_reference=lambda value: torch.sub(
            2.0, value, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "reduction.sum.dim",
        "reduction",
        torch.sum,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_sum,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "reduction.mean.dim",
        "reduction",
        torch.mean,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_mean,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "reduction.sum.keepdim.strided",
        "reduction",
        torch.sum,
        _reduction_strided,
        args=(1,),
        kwargs={"keepdim": True},
        cpu_reference=_cpu_sum,
        expected_shape=(3, 1, 4),
        check_gradients=True,
    ),
    _case(
        "reduction.mean.optional-dim",
        "reduction",
        torch.mean,
        _reduction,
        cpu_reference=_cpu_mean,
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "reduction.sum.empty-dim",
        "reduction",
        torch.sum,
        _empty_reduction,
        args=(0,),
        cpu_reference=_cpu_sum,
        expected_shape=(3,),
        execution_mode="empty",
    ),
    _case(
        "reduction.mean.empty-dim",
        "reduction",
        torch.mean,
        _empty_reduction,
        args=(0,),
        cpu_reference=_cpu_mean,
        expected_shape=(3,),
        execution_mode="empty",
        rtol=0,
        atol=0,
    ),
    _case(
        "reduction.amax.dim",
        "reduction",
        _amax_out,
        _reduction,
        args=(1,),
        cpu_reference=torch.amax,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amax.default",
        "reduction",
        torch.amax,
        _reduction,
        args=(1,),
        cpu_reference=torch.amax,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amin.dim",
        "reduction",
        _amin_out,
        _reduction,
        args=(1,),
        cpu_reference=torch.amin,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amin.default",
        "reduction",
        torch.amin,
        _reduction,
        args=(1,),
        cpu_reference=torch.amin,
        expected_shape=(2,),
    ),
    _case(
        "reduction.prod.dim",
        "reduction",
        _prod_int_out,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_prod,
        expected_shape=(2,),
    ),
    _case(
        "reduction.prod.default",
        "reduction",
        torch.prod,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_prod,
        expected_shape=(2,),
    ),
    _case(
        "reduction.softmax.dim",
        "reduction",
        _softmax_out,
        _reduction,
        args=(1,),
        cpu_reference=_softmax,
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.softmax.default",
        "reduction",
        _softmax,
        _reduction,
        args=(1,),
        cpu_reference=_softmax,
        expected_shape=(2, 2),
    ),
    _case(
        "reduction.log-softmax.dim",
        "reduction",
        _log_softmax_out,
        _reduction,
        args=(1,),
        cpu_reference=_log_softmax,
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.log-softmax.default",
        "reduction",
        _log_softmax,
        _reduction,
        args=(1,),
        cpu_reference=_log_softmax,
        expected_shape=(2, 2),
    ),
    _case(
        "reduction.softmax.backward",
        "reduction",
        _softmax_backward_out,
        _softmax_backward_inputs,
        cpu_reference=lambda grad, output: torch.ops.aten._softmax_backward_data(
            grad, output, 1, torch.float32
        ),
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.log-softmax.backward",
        "reduction",
        _log_softmax_backward_out,
        _softmax_backward_inputs,
        cpu_reference=lambda grad, output: torch.ops.aten._log_softmax_backward_data(
            grad,
            output,
            1,
            torch.float32,
        ),
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "indexing.argmax.dim",
        "indexing",
        torch.argmax,
        _indexing,
        args=(1,),
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(2,),
    ),
    _case(
        "indexing.argmax.optional-dim.keepdim",
        "indexing",
        torch.argmax,
        _indexing,
        kwargs={"keepdim": True},
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(1, 1),
    ),
    _case(
        "indexing.argmax.strided",
        "indexing",
        torch.argmax,
        _indexing_strided,
        args=(-1,),
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(4,),
    ),
    _case(
        "view.reshape.float32",
        "view",
        torch.reshape,
        _view,
        args=((6,),),
        cpu_reference=_cpu_reshape,
        expected_shape=(6,),
        check_gradients=True,
        execution_mode="copy",
    ),
    _case(
        "view.as-strided.metadata",
        "view",
        torch.as_strided,
        _metadata_view,
        args=((2, 3), (3, 1)),
        cpu_reference=_cpu_as_strided,
        expected_shape=(2, 3),
        execution_mode="metadata",
    ),
    _case(
        "view.view.metadata",
        "view",
        torch.ops.aten.view.default,
        _metadata_view,
        args=((6,),),
        cpu_reference=lambda value, shape: torch.ops.aten.view.default(value, shape),
        expected_shape=(6,),
        execution_mode="metadata",
    ),
    _case(
        "view.reshape-alias.metadata",
        "view",
        torch.ops.aten._reshape_alias.default,
        _metadata_view,
        args=((2, 3), (3, 1)),
        cpu_reference=lambda value, size, stride: torch.ops.aten._reshape_alias.default(
            value, size, stride
        ),
        expected_shape=(2, 3),
        execution_mode="metadata",
    ),
    _case(
        "linear.forward",
        "linear",
        torch.nn.functional.linear,
        _linear,
        cpu_reference=_cpu_linear,
        expected_shape=(2, 3),
        check_gradients=True,
    ),
    _case(
        "linear.forward.strided",
        "linear",
        torch.nn.functional.linear,
        _linear_strided,
        cpu_reference=_cpu_linear,
        expected_shape=(2, 3),
        check_gradients=True,
    ),
    _case(
        "mm.forward",
        "linear",
        torch.mm,
        _mm,
        cpu_reference=_cpu_mm,
        expected_shape=(2, 3),
    ),
    _case(
        "addmm.forward",
        "linear",
        torch.addmm,
        _addmm,
        cpu_reference=_cpu_addmm,
        expected_shape=(2, 3),
    ),
    _case(
        "addmm.out",
        "linear",
        _addmm_out,
        _addmm,
        cpu_reference=_addmm_out,
        expected_shape=(2, 3),
    ),
    _case(
        "convolution.forward",
        "convolution",
        torch.nn.functional.conv2d,
        _convolution,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 8, 8),
        check_gradients=True,
    ),
    _case(
        "convolution.forward.strided",
        "convolution",
        torch.nn.functional.conv2d,
        _convolution_strided,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 8, 8),
        check_gradients=True,
    ),
    _case(
        "convolution.backward",
        "convolution",
        _convolution_backward,
        _convolution_backward_inputs,
        cpu_reference=lambda grad,
        value,
        weight: torch.ops.aten.convolution_backward.default(
            grad,
            value,
            weight,
            [4],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )[0],
        expected_shape=(2, 1, 8, 8),
    ),
    _case(
        "pooling.max.rejected",
        "pooling",
        torch.nn.functional.max_pool2d,
        _pooling,
        args=(2,),
        cpu_reference=_cpu_max_pool,
        supported=False,
        error_pattern=r"Vulkan pooling is not declared",
    ),
    _case(
        "masked-select.bool-mask",
        "masked-select",
        torch.masked_select,
        _masked_select,
        cpu_reference=_cpu_masked_select,
        expected_shape=(2,),
        autograd_supported=False,
        autograd_error_type=NotImplementedError,
        autograd_error_pattern=r"aten::(zero_|masked_scatter_)",
        requires_grad_inputs=True,
    ),
    _case(
        "masked-select.strided-value-view",
        "masked-select",
        torch.masked_select,
        _masked_select_view,
        cpu_reference=_cpu_masked_select,
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "optimizer.div.scalar",
        "optimizer",
        torch.ops.aten.div.Tensor,
        _optimizer_pair,
        cpu_reference=torch.div,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.lerp.out",
        "optimizer",
        _lerp_out,
        _optimizer_value_pair,
        cpu_reference=_cpu_lerp,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.lerp.inplace",
        "optimizer",
        _lerp_inplace,
        _optimizer_value_pair,
        cpu_reference=_cpu_lerp,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.sqrt.out",
        "optimizer",
        _sqrt_out,
        _optimizer_single,
        cpu_reference=torch.sqrt,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.add.inplace",
        "optimizer",
        _add_inplace,
        _optimizer_value_pair,
        cpu_reference=torch.add,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.mul.scalar.inplace",
        "optimizer",
        _mul_scalar_inplace,
        _optimizer_single,
        cpu_reference=lambda value: value * 2,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.addcmul.inplace",
        "optimizer",
        _addcmul_inplace,
        _optimizer_triple,
        cpu_reference=lambda lhs, a, b: torch.addcmul(lhs, a, b, value=0.25),
        expected_shape=(2,),
    ),
    _case(
        "optimizer.addcdiv.inplace",
        "optimizer",
        _addcdiv_inplace,
        _optimizer_triple,
        cpu_reference=lambda lhs, a, b: torch.addcdiv(lhs, a, b, value=0.25),
        expected_shape=(2,),
    ),
    _case(
        "optimizer.zero.inplace",
        "optimizer",
        _zero_inplace,
        _optimizer_single,
        cpu_reference=lambda value: value.zero_(),
        expected_shape=(2,),
    ),
    _case(
        "aten._adaptive_avg_pool2d.global",
        "pooling",
        torch.ops.aten._adaptive_avg_pool2d.default,
        _adaptive_pool,
        args=((1, 1),),
        cpu_reference=_cpu_adaptive_pool,
        expected_shape=(2, 4, 1, 1),
        check_gradients=True,
    ),
    _case(
        "aten._adaptive_avg_pool2d.global.strided",
        "pooling",
        torch.ops.aten._adaptive_avg_pool2d.default,
        _pooling_strided,
        args=((1, 1),),
        cpu_reference=_cpu_adaptive_pool,
        expected_shape=(1, 1, 1, 1),
        check_gradients=True,
    ),
    _case(
        "aten._adaptive_avg_pool2d.backward",
        "pooling",
        _adaptive_pool_backward,
        _adaptive_pool_backward_inputs,
        cpu_reference=lambda grad,
        value: torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, value),
        expected_shape=(2, 4, 3, 5),
    ),
    _case(
        "normalization.native-batch-norm",
        "normalization",
        _native_batch_norm_output,
        _normalization,
        cpu_reference=_native_batch_norm_output,
        expected_shape=(2, 4),
    ),
    _case(
        "normalization.native-batch-norm-backward",
        "normalization",
        _native_batch_norm_backward_output,
        _normalization_backward,
        cpu_reference=_native_batch_norm_backward_output,
        expected_shape=(2, 4),
    ),
    _case(
        "classification.nll-forward",
        "cross-entropy/NLL",
        _nll_forward_output,
        _nll_forward,
        cpu_reference=_nll_forward_output,
        expected_shape=(),
    ),
    _case(
        "classification.nll-backward",
        "cross-entropy/NLL",
        _nll_backward_output,
        _nll_backward,
        cpu_reference=_nll_backward_output,
        expected_shape=(2, 3),
    ),
    _case(
        "unary.neg.bool.rejected",
        "unary",
        torch.neg,
        _bool_unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"supports only float32 tensors",
    ),
    _case(
        "unary.neg.float16.rejected",
        "unary",
        torch.neg,
        _float16_unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"float16 support is deferred",
    ),
    _case(
        "binary.add.double.rejected",
        "binary",
        torch.add,
        _double_binary,
        supported=False,
        convert_inputs=True,
        cpu_reference=_cpu_add,
        error_pattern=r"formatter conversion supports only Vulkan float32 to Vulkan Double",
    ),
    _case(
        "binary.add.mixed-device.rejected",
        "binary",
        _mixed_device_add,
        _unary,
        supported=False,
        cpu_reference=_cpu_add,
        error_pattern=r"requires operands on the same device; got",
    ),
    _case(
        "binary.add.broadcast.rejected",
        "binary",
        torch.add,
        _broadcast_binary,
        supported=False,
        cpu_reference=_cpu_add,
        error_pattern=r"requires equal tensor sizes; broadcasting is unsupported",
    ),
    _case(
        "reduction.sum.non-contiguous-overlap.rejected",
        "reduction",
        torch.sum,
        _overlapping,
        args=(1,),
        supported=False,
        cpu_reference=_cpu_sum,
        error_pattern=r"rejects overlapping input views",
    ),
    _case(
        "comparison.ne.nonzero-offset-input.rejected",
        "comparison",
        torch.ne,
        _wrong_offset,
        supported=False,
        cpu_reference=torch.ne,
        error_pattern=r"formatter Double ne\.Tensor requires a contiguous zero-offset strided input",
        setup_inputs=_setup_double_offset,
    ),
    _case(
        "unary.neg.invalid-out.rejected",
        "unary",
        _invalid_out,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"supports only float32 and bool tensors, got Long",
    ),
    _case(
        "unary.neg.cpu-out.rejected",
        "unary",
        _cpu_out,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"requires a Vulkan output tensor",
    ),
    _case(
        "pooling.parameters.rejected",
        "pooling",
        _bad_pool_params,
        _adaptive_pool,
        args=(),
        supported=False,
        cpu_reference=_cpu_adaptive_pool,
        error_pattern=r"output size|\(1, 1\)",
    ),
    _case(
        "convolution.shape.rejected",
        "convolution",
        torch.nn.functional.conv2d,
        _wrong_convolution,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        supported=False,
        cpu_reference=_cpu_convolution,
        error_pattern=r"unsupported fixed shape|shape",
    ),
    _case(
        "unary.neg_.unsupported-overload.rejected",
        "unary",
        _unsupported_overload,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"Vulkan neg in-place variants are unsupported",
    ),
    _case(
        "transfer.copy_from.float32",
        "transfer",
        _copy_from,
        _transfer_pair,
        cpu_reference=_copy_from_cpu_reference,
        expected_shape=(4,),
        execution_mode="copy",
    ),
    _case(
        "transfer.to_copy.float32",
        "transfer",
        lambda value: value.to(value.device, copy=True),
        _unary,
        cpu_reference=lambda value: value.to(value.device, copy=True),
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "transfer.copy.float32",
        "transfer",
        _copy_in_place,
        _transfer_pair,
        cpu_reference=_copy_in_place,
        expected_shape=(4,),
        execution_mode="copy",
    ),
    _case(
        "transfer.empty.float32",
        "transfer",
        _empty_like_template,
        _empty_unary,
        cpu_reference=_empty_like_template,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "transfer.empty_strided.float32",
        "transfer",
        _empty_strided_like_template,
        _empty_strided_zero_template,
        cpu_reference=_empty_strided_like_template,
        expected_shape=(0, 2),
        execution_mode="empty",
    ),
)


SUPPORTED_CASES = tuple(case for case in ALL_CASES if case.supported)
