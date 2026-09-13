import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_vulkan_operation_builds_autograd_graph(vulkan_backend):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    y = torch.neg(x)

    assert y.grad_fn is not None
    y.backward(torch.ones(y.shape, dtype=y.dtype).to(vulkan_backend))
    assert x.grad is not None
    assert x.grad.device == x.device
    torch.testing.assert_close(x.grad.cpu(), -torch.ones_like(x.cpu()))


@pytest.mark.parametrize("operation", [torch.abs, torch.relu])
def test_vulkan_unary_backward_matches_cpu(vulkan_backend, operation):
    cpu = torch.tensor([-2.0, -0.25, 0.0, 0.5, 3.0], requires_grad=True)
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu)
    vk_result = operation(vk)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_result.cpu(), cpu_result.detach())
    assert vk.grad is not None
    assert vk.grad.device == vk.device
    assert vk.grad.dtype is torch.float32
    assert vk.grad.shape == vk.shape
    assert vk.grad.is_contiguous()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad)


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_vulkan_unary_out_requires_grad_is_rejected(vulkan_backend, operation):
    input = torch.tensor([-2.0, 0.0, 3.0], device=vulkan_backend, requires_grad=True)
    output = torch.empty_like(input)

    with pytest.raises((RuntimeError, TypeError)):
        operation(input, out=output)


def test_cpu_operation_keeps_normal_autograd_behavior():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = torch.neg(x)

    assert y.grad_fn is not None
    y.backward(torch.ones_like(y))
    torch.testing.assert_close(x.grad, torch.tensor([-1.0, -1.0]))


def test_detached_vulkan_input_does_not_build_autograd_graph(vulkan_backend):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend).detach()
    y = torch.neg(x)

    assert y.grad_fn is None
    assert not y.requires_grad


def test_vulkan_input_without_requires_grad_does_not_build_autograd_graph(
    vulkan_backend,
):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=False)
    y = torch.relu(x)

    assert y.grad_fn is None
    assert not y.requires_grad


def test_vulkan_backward_can_reuse_retained_graph(vulkan_backend):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    y = torch.neg(x)
    grad_output = torch.ones_like(x.cpu()).to(vulkan_backend)

    y.backward(grad_output, retain_graph=True)
    x.grad = None
    # Existing Vulkan in-place rejection prevents accumulated leaf gradients.
    y.backward(grad_output, retain_graph=True)

    torch.testing.assert_close(x.grad.cpu(), torch.tensor([-1.0, -1.0]))


def test_vulkan_non_scalar_backward_accepts_explicit_vulkan_gradient(
    vulkan_backend,
):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    y = torch.relu(x)
    grad_output = torch.tensor([3.0, 4.0], device=vulkan_backend)

    y.backward(grad_output)

    assert x.grad is not None
    assert x.grad.device == x.device
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([3.0, 0.0]))


def test_unsupported_vulkan_operator_fails_without_cpu_fallback(vulkan_backend):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)

    with pytest.raises(RuntimeError, match="Could not run|not implemented|Vulkan"):
        torch.sin(x)
