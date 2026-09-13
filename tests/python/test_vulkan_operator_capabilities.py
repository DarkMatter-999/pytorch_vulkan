import re

import pytest
import torch

import pytorch_vulkan


def test_reduction_indexing_matrix_is_declared():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    for operation in ("`aten::sum`", "`aten::mean`", "`aten::argmax`"):
        assert operation in matrix


def test_model_matrix_declares_fixed_phase_6_slices():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    for operation in (
        "`aten::linear`",
        "`aten::convolution`",
        "`aten::_adaptive_avg_pool2d`",
        "fixed MLP",
        "fixed CNN",
    ):
        assert operation in matrix


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def assert_vulkan_capability(operation, inputs, *, supported, error_pattern=None):
    """Run an operation and assert explicit support or rejection."""
    if supported:
        result = operation(*inputs)
        assert result.device.type == "vk"
        assert result.device.index == 0
        return result

    pattern = error_pattern or r"Vulkan"
    with pytest.raises(RuntimeError, match=pattern) as error:
        operation(*inputs)
    assert type(error.value) is RuntimeError


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_baseline_remains_supported(vulkan_backend, operation):
    tensor = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32).to(vulkan_backend)

    result = assert_vulkan_capability(operation, (tensor,), supported=True)

    assert result.dtype is torch.float32
    torch.testing.assert_close(result.cpu(), operation(tensor.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_baseline_remains_supported(vulkan_backend, operation):
    lhs = torch.tensor([1.0, -2.0], dtype=torch.float32).to(vulkan_backend)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32).to(vulkan_backend)

    result = assert_vulkan_capability(operation, (lhs, rhs), supported=True)

    assert result.dtype is torch.float32
    torch.testing.assert_close(
        result.cpu(), operation(lhs.cpu(), rhs.cpu()), rtol=0, atol=0
    )


def test_float16_operator_request_is_rejected_without_cpu_fallback(vulkan_backend):
    expected = (
        "Vulkan float16 support is deferred; allocation cannot use float16 until its "
        "storage, transfer, shader, promotion, and autograd contracts are implemented"
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        torch.neg(torch.empty((2,), dtype=torch.float16, device=vulkan_backend))


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu, torch.sub])
def test_unsupported_bool_operator_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    inputs = (tensor,) if operation in [torch.neg, torch.abs, torch.relu] else (tensor, tensor)

    assert_vulkan_capability(
        operation,
        inputs,
        supported=False,
        error_pattern=r"Vulkan.*float32|Vulkan.*dtype|support.*bool",
    )


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_gradient_contract(vulkan_backend, operation):
    cpu_input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
    vk_input = cpu_input.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu_input)
    vk_result = operation(vk_input)
    grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cpu_result.backward(grad)
    vk_result.backward(grad.to(vulkan_backend))

    assert vk_result.device == vk_input.device
    assert vk_result.dtype is torch.float32
    assert vk_result.shape == vk_input.shape
    assert vk_input.grad is not None
    assert vk_input.grad.device == vk_input.device
    assert vk_input.grad.dtype is torch.float32
    assert vk_input.grad.shape == vk_input.shape
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_gradient_contract(vulkan_backend, operation):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32, requires_grad=True)
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float32, requires_grad=True)
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_()
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_result = operation(vk_lhs, vk_rhs)
    grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cpu_result.backward(grad)
    vk_result.backward(grad.to(vulkan_backend))

    assert vk_result.device == vk_lhs.device
    assert vk_result.dtype is torch.float32
    assert vk_result.shape == vk_lhs.shape
    for vk_input, cpu_input in ((vk_lhs, cpu_lhs), (vk_rhs, cpu_rhs)):
        assert vk_input.grad is not None
        assert vk_input.grad.device == vk_input.device
        assert vk_input.grad.dtype is torch.float32
        assert vk_input.grad.shape == vk_input.shape
        torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_layout_shape_and_explicit_transfer_contract(
    vulkan_backend, operation
):
    non_contiguous = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    assert not non_contiguous.is_contiguous()
    assert_vulkan_capability(
        operation, (non_contiguous,), supported=False, error_pattern=r"contiguous"
    )

    zero_dimensional = torch.empty((), dtype=torch.float32, device=vulkan_backend)
    assert_vulkan_capability(
        operation, (zero_dimensional,), supported=False, error_pattern=r"zero-dimensional"
    )

    source = torch.tensor([1.0, -2.0], dtype=torch.float32, device=vulkan_backend)
    result = assert_vulkan_capability(operation, (source,), supported=True)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    transferred = result.to("cpu")
    assert transferred.device.type == "cpu"
    torch.testing.assert_close(transferred, operation(source.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_shape_and_explicit_transfer_contract(vulkan_backend, operation):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    mismatched_rhs = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    assert_vulkan_capability(
        operation, (lhs, mismatched_rhs), supported=False, error_pattern=r"size|shape"
    )

    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    result = assert_vulkan_capability(operation, (lhs, rhs), supported=True)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.to("cpu").device.type == "cpu"
