import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _vk(values, backend, requires_grad=False):
    return torch.tensor(values, dtype=torch.float32).to(backend).requires_grad_(
        requires_grad
    )


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_tensor_gradients_match_cpu(vulkan_backend, operation):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], requires_grad=True)
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_()
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_result = operation(vk_lhs, vk_rhs)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_lhs.grad.cpu(), cpu_lhs.grad)
    torch.testing.assert_close(vk_rhs.grad.cpu(), cpu_rhs.grad)


@pytest.mark.parametrize("operation", [torch.add, torch.sub])
@pytest.mark.parametrize("requires_grad_operand", ["lhs", "rhs"])
def test_binary_tensor_single_operand_gradient_matches_cpu(
    vulkan_backend, operation, requires_grad_operand
):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], requires_grad=requires_grad_operand == "lhs")
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], requires_grad=requires_grad_operand == "rhs")
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_(
        requires_grad_operand == "lhs"
    )
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_(
        requires_grad_operand == "rhs"
    )

    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_result = operation(vk_lhs, vk_rhs)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    if requires_grad_operand == "lhs":
        torch.testing.assert_close(vk_lhs.grad.cpu(), cpu_lhs.grad)
        assert vk_rhs.grad is None
    else:
        assert vk_lhs.grad is None
        torch.testing.assert_close(vk_rhs.grad.cpu(), cpu_rhs.grad)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_binary_scalar_gradient_matches_cpu(vulkan_backend, operation, scalar_left):
    cpu_tensor = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    vk_tensor = cpu_tensor.detach().clone().to(vulkan_backend).requires_grad_()
    scalar = 2.5
    cpu_operands = (scalar, cpu_tensor) if scalar_left else (cpu_tensor, scalar)
    vk_operands = (scalar, vk_tensor) if scalar_left else (vk_tensor, scalar)

    cpu_result = operation(*cpu_operands)
    vk_result = operation(*vk_operands)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_tensor.grad.cpu(), cpu_tensor.grad)


def test_rsub_scalar_left_gradient_matches_cpu(vulkan_backend):
    cpu_tensor = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    vk_tensor = cpu_tensor.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = torch.rsub(cpu_tensor, 2.5)
    vk_result = torch.rsub(vk_tensor, 2.5)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_tensor.grad.cpu(), cpu_tensor.grad)


def test_binary_rejections_remain_explicit(vulkan_backend):
    lhs = torch.empty((2, 1), device=vulkan_backend)
    rhs = torch.empty((2, 3), device=vulkan_backend)
    with pytest.raises(RuntimeError, match="broadcast"):
        torch.mul(lhs, rhs)

    equal_lhs = torch.empty((2,), device=vulkan_backend)
    equal_rhs = torch.empty((2,), device=vulkan_backend)
    with pytest.raises(RuntimeError, match="alpha"):
        torch.add(equal_lhs, equal_rhs, alpha=2)

    grad_lhs = equal_lhs.detach().requires_grad_()
    grad_rhs = equal_rhs.detach().requires_grad_()
    with pytest.raises((RuntimeError, NotImplementedError), match="out|requires grad"):
        torch.mul(grad_lhs, grad_rhs, out=torch.empty_like(equal_lhs))


def test_binary_inplace_rejection_remains_explicit(vulkan_backend):
    tensor = _vk([1.0, 2.0], vulkan_backend)
    other = _vk([3.0, 4.0], vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError), match="in-place|Could not run"):
        tensor.mul_(other)
