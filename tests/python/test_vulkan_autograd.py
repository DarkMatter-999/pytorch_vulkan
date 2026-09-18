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


def test_saved_vulkan_view_survives_forward_scope_and_backward(vulkan_backend):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = cpu.to(vulkan_backend).detach().requires_grad_()

    def build_result(value):
        saved_view = value.transpose(0, 1)
        return torch.neg(saved_view)

    result = build_result(source)
    result.backward(torch.ones(result.shape, dtype=result.dtype).to(vulkan_backend))

    torch.testing.assert_close(source.grad.cpu(), -torch.ones_like(cpu))


def test_migrated_linear_backward_accepts_a_view(vulkan_backend):
    cpu_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cpu_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    input_view = cpu_input[:, ::2]
    weight_view = cpu_weight[:, ::2]
    vk_input = input_view.detach().to(vulkan_backend).requires_grad_()
    vk_weight = weight_view.detach().to(vulkan_backend).requires_grad_()

    output = torch.nn.functional.linear(vk_input, vk_weight)
    output.backward(torch.ones(output.shape, dtype=output.dtype).to(vulkan_backend))

    assert vk_input.grad is not None
    assert vk_weight.grad is not None
    torch.testing.assert_close(vk_input.grad.cpu(), torch.ones_like(input_view) @ weight_view)
    torch.testing.assert_close(vk_weight.grad.cpu(), torch.ones_like(output.cpu()).t() @ input_view)


@pytest.mark.parametrize(
    ("operation", "cpu_operation"),
    [
        (torch.sigmoid, torch.sigmoid),
        (torch.tanh, torch.tanh),
        (
            lambda value: torch.nn.functional.gelu(value, approximate="tanh"),
            lambda value: torch.nn.functional.gelu(value, approximate="tanh"),
        ),
    ],
)
def test_vulkan_activation_backward_matches_cpu(vulkan_backend, operation, cpu_operation):
    cpu = torch.tensor([-2.0, -0.25, 0.0, 0.5, 3.0], requires_grad=True)
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    cpu_result = cpu_operation(cpu)
    vk_result = operation(vk)
    cpu_result.backward(grad)
    vk_result.backward(grad.to(vulkan_backend))

    assert vk.grad is not None
    assert vk.grad.device == vk.device
    torch.testing.assert_close(vk_result.cpu(), cpu_result.detach(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=3e-4, atol=2e-5)


@pytest.mark.parametrize(
    "operation",
    [
        torch.sigmoid,
        torch.tanh,
        lambda value: torch.nn.functional.gelu(value, approximate="tanh"),
    ],
)
def test_vulkan_activation_higher_order_backward_is_rejected(vulkan_backend, operation):
    value = torch.tensor([-1.0, 0.5], dtype=torch.float32, device=vulkan_backend, requires_grad=True)
    result = operation(value)
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(result.sum(), value, create_graph=True)
