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
    assert tensor.dtype is torch.float32


def test_fixed_model_forward_and_first_order_gradients(vulkan_backend):
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
    cpu_output.backward(torch.ones_like(cpu_output))
    vk_hidden = torch.relu(torch.nn.functional.linear(vk_input, vk_weight, vk_bias))
    vk_output = torch.nn.functional.linear(vk_hidden, vk_weight2, vk_bias2)
    _assert_vulkan(vk_hidden)
    _assert_vulkan(vk_output)
    grad_output = torch.add(torch.mul(vk_output.detach(), 0.0), 1.0)
    _assert_vulkan(grad_output)
    vk_output.backward(grad_output)

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)
    torch.testing.assert_close(vk_weight2.grad.cpu(), cpu_weight2.grad)
    torch.testing.assert_close(vk_bias2.grad.cpu(), cpu_bias2.grad)
    for gradient in (
        vk_input.grad,
        vk_weight.grad,
        vk_bias.grad,
        vk_weight2.grad,
        vk_bias2.grad,
    ):
        _assert_vulkan(gradient)


def test_fixed_normalization_classifier_workload(vulkan_backend):
    torch.manual_seed(107)
    cpu_input = torch.randn(2, 4, requires_grad=True)
    cpu_labels = torch.tensor([1, 2], dtype=torch.int64)
    cpu_weight = torch.randn(3, 4, requires_grad=True)
    cpu_bias = torch.randn(3, requires_grad=True)
    running_mean = torch.zeros(4)
    running_var = torch.ones(4)
    cpu_norm = torch.ops.aten.native_batch_norm.default(
        cpu_input,
        torch.ones(4, requires_grad=True),
        torch.zeros(4, requires_grad=True),
        running_mean,
        running_var,
        True,
        0.1,
        1e-5,
    )[0]
    cpu_logits = torch.nn.functional.linear(cpu_norm, cpu_weight, cpu_bias)
    cpu_loss = torch.nn.functional.cross_entropy(cpu_logits, cpu_labels)
    cpu_loss.backward()
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()
    vk_norm = torch.ops.aten.native_batch_norm.default(
        vk_input,
        torch.ones(4).to(vulkan_backend).requires_grad_(),
        torch.zeros(4).to(vulkan_backend).requires_grad_(),
        running_mean.to(vulkan_backend),
        running_var.to(vulkan_backend),
        True,
        0.1,
        1e-5,
    )[0]
    vk_logits = torch.nn.functional.linear(vk_norm, vk_weight, vk_bias)
    vk_loss = torch.nn.functional.cross_entropy(
        vk_logits, cpu_labels.to(vulkan_backend)
    )
    vk_loss.backward()
    _assert_vulkan(vk_loss)
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss.detach(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        vk_weight.grad.cpu(), cpu_weight.grad, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad, rtol=2e-4, atol=2e-4)


def test_fixed_mlp_forward(vulkan_backend):
    torch.manual_seed(7)
    cpu_input = torch.randn(2, 8, dtype=torch.float32)
    cpu_weight = torch.randn(16, 8, dtype=torch.float32)
    cpu_bias = torch.randn(16, dtype=torch.float32)
    cpu_weight2 = torch.randn(4, 16, dtype=torch.float32)
    cpu_bias2 = torch.randn(4, dtype=torch.float32)

    cpu_output = torch.nn.functional.linear(
        torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)),
        cpu_weight2,
        cpu_bias2,
    )
    vk_hidden = torch.relu(
        torch.nn.functional.linear(
            cpu_input.to(vulkan_backend),
            cpu_weight.to(vulkan_backend),
            cpu_bias.to(vulkan_backend),
        )
    )
    vk_output = torch.nn.functional.linear(
        vk_hidden, cpu_weight2.to(vulkan_backend), cpu_bias2.to(vulkan_backend)
    )

    _assert_vulkan(vk_hidden)
    _assert_vulkan(vk_output)
    torch.testing.assert_close(vk_output.cpu(), cpu_output)


def test_fixed_mlp_bounded_repeated_forward_parity(vulkan_backend):
    for iteration in range(3):
        torch.manual_seed(700 + iteration)
        cpu_input = torch.randn(2, 8, dtype=torch.float32)
        cpu_weight = torch.randn(16, 8, dtype=torch.float32)
        cpu_bias = torch.randn(16, dtype=torch.float32)
        cpu_weight2 = torch.randn(4, 16, dtype=torch.float32)
        cpu_bias2 = torch.randn(4, dtype=torch.float32)
        cpu_output = torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)),
            cpu_weight2,
            cpu_bias2,
        )
        vk_input = cpu_input.to(vulkan_backend)
        vk_output = torch.nn.functional.linear(
            torch.relu(
                torch.nn.functional.linear(
                    vk_input, cpu_weight.to(vulkan_backend), cpu_bias.to(vulkan_backend)
                )
            ),
            cpu_weight2.to(vulkan_backend),
            cpu_bias2.to(vulkan_backend),
        )
        torch.testing.assert_close(vk_output.cpu(), cpu_output)


def test_fixed_cnn_forward_and_first_order_gradients(vulkan_backend):
    torch.manual_seed(11)
    cpu_input = torch.randn(2, 1, 8, 8, dtype=torch.float32, requires_grad=True)
    cpu_weight = torch.randn(4, 1, 3, 3, dtype=torch.float32, requires_grad=True)
    cpu_bias = torch.randn(4, dtype=torch.float32, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()

    cpu_conv = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    cpu_output = torch.nn.functional.adaptive_avg_pool2d(torch.relu(cpu_conv), (1, 1))
    grad_output = torch.randn_like(cpu_output)
    vk_conv = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
    vk_relu = torch.relu(vk_conv)
    vk_output = torch.nn.functional.adaptive_avg_pool2d(vk_relu, (1, 1))
    _assert_vulkan(vk_conv)
    _assert_vulkan(vk_relu)
    _assert_vulkan(vk_output)
    cpu_output.backward(grad_output)
    vk_output.backward(grad_output.to(vulkan_backend))

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)
    _assert_vulkan(vk_input.grad)
    _assert_vulkan(vk_weight.grad)
    _assert_vulkan(vk_bias.grad)


def test_model_workload_float16_is_explicitly_unsupported(vulkan_backend):
    with pytest.raises((RuntimeError, TypeError), match="float16|Float16|dtype|Vulkan"):
        x = torch.ones(2, 4, dtype=torch.float16, device=vulkan_backend)
        weight = torch.ones(3, 4, dtype=torch.float16, device=vulkan_backend)
        torch.nn.functional.linear(x, weight)


def test_model_workload_unsupported_variants_are_rejected(vulkan_backend):
    input = torch.randn(2, 1, 8, 8, dtype=torch.float32).to(vulkan_backend)
    weight = torch.randn(4, 1, 3, 3, dtype=torch.float32).to(vulkan_backend)
    bias = torch.randn(4, dtype=torch.float32).to(vulkan_backend)

    with pytest.raises(RuntimeError, match="convolution|stride|fixed"):
        torch.nn.functional.conv2d(input, weight, bias, stride=2, padding=1)

    with pytest.raises(RuntimeError, match="pooling|output|size|fixed"):
        torch.nn.functional.adaptive_avg_pool2d(
            torch.relu(torch.nn.functional.conv2d(input, weight, bias, padding=1)),
            (2, 2),
        )
