import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _assert_vulkan(tensor):
    assert tensor.device.type == "vk"
    assert tensor.device.index == 0
    assert tensor.is_contiguous()


def test_fixed_mlp_forward_and_first_order_gradients(vulkan_backend):
    torch.manual_seed(7)
    cpu_input = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    cpu_weight = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    cpu_bias = torch.randn(16, dtype=torch.float32, requires_grad=True)
    cpu_weight2 = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    cpu_bias2 = torch.randn(4, dtype=torch.float32, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()
    vk_weight2 = cpu_weight2.detach().to(vulkan_backend).requires_grad_()
    vk_bias2 = cpu_bias2.detach().to(vulkan_backend).requires_grad_()

    cpu_hidden = torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))
    cpu_output = torch.nn.functional.linear(cpu_hidden, cpu_weight2, cpu_bias2)
    cpu_loss = cpu_output.sum()
    cpu_loss.backward()
    vk_hidden = torch.relu(torch.nn.functional.linear(vk_input, vk_weight, vk_bias))
    vk_output = torch.nn.functional.linear(vk_hidden, vk_weight2, vk_bias2)
    _assert_vulkan(vk_hidden)
    _assert_vulkan(vk_output)
    vk_output.sum().backward()

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)
    torch.testing.assert_close(vk_weight2.grad.cpu(), cpu_weight2.grad)
    torch.testing.assert_close(vk_bias2.grad.cpu(), cpu_bias2.grad)


def test_fixed_cnn_forward_and_first_order_gradients(vulkan_backend):
    torch.manual_seed(11)
    cpu_input = torch.randn(2, 1, 8, 8, dtype=torch.float32, requires_grad=True)
    cpu_weight = torch.randn(4, 1, 3, 3, dtype=torch.float32, requires_grad=True)
    cpu_bias = torch.randn(4, dtype=torch.float32, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()

    cpu_conv = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    cpu_pool = torch.nn.functional.adaptive_avg_pool2d(torch.relu(cpu_conv), 1)
    cpu_pool.sum().backward()
    vk_conv = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
    _assert_vulkan(vk_conv)
    vk_pool = torch.nn.functional.adaptive_avg_pool2d(torch.relu(vk_conv), 1)
    _assert_vulkan(vk_pool)
    vk_pool.sum().backward()

    torch.testing.assert_close(vk_pool.cpu(), cpu_pool.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)


def test_model_workload_float16_is_explicitly_unsupported(vulkan_backend):
    with pytest.raises((RuntimeError, TypeError), match="float16|Float16|dtype|Vulkan"):
        x = torch.ones(2, 4, dtype=torch.float16, device=vulkan_backend)
        weight = torch.ones(3, 4, dtype=torch.float16, device=vulkan_backend)
        torch.nn.functional.linear(x, weight)
