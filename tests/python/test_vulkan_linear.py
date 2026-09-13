import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _linear_inputs(device):
    torch.manual_seed(23)
    return (
        torch.randn(2, 8, dtype=torch.float32, device=device),
        torch.randn(16, 8, dtype=torch.float32, device=device),
        torch.randn(16, dtype=torch.float32, device=device),
    )


def test_linear_forward_matches_cpu_and_returns_contiguous_f32(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    result = torch.nn.functional.linear(
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )

    assert result.device == torch.device("vk:0")
    assert result.dtype is torch.float32
    assert result.shape == (2, 16)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))


def test_linear_forward_without_bias_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, _ = _linear_inputs("cpu")
    result = torch.nn.functional.linear(
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    )

    assert result.device == torch.device("vk:0")
    assert result.dtype is torch.float32
    assert result.shape == (2, 16)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight))


def test_linear_backward_matches_cpu_with_explicit_vulkan_gradient(vulkan_backend):
    torch.manual_seed(31)
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    cpu_input.requires_grad_()
    cpu_weight.requires_grad_()
    cpu_bias.requires_grad_()
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()
    cpu_output = torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight, vk_bias)
    grad_output = torch.randn_like(cpu_output)

    cpu_output.backward(grad_output)
    vk_output.backward(grad_output.to(vulkan_backend))

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    for actual, expected in ((vk_input.grad, cpu_input.grad),
                             (vk_weight.grad, cpu_weight.grad),
                             (vk_bias.grad, cpu_bias.grad)):
        assert actual.device == torch.device("vk:0")
        assert actual.dtype is torch.float32
        assert actual.is_contiguous()
        torch.testing.assert_close(actual.cpu(), expected)


def test_linear_backward_without_bias_returns_undefined_bias_gradient(vulkan_backend):
    cpu_input, cpu_weight, _ = _linear_inputs("cpu")
    cpu_input.requires_grad_()
    cpu_weight.requires_grad_()
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()

    output = torch.nn.functional.linear(vk_input, vk_weight)
    output.backward(torch.ones_like(output.cpu()).to(vulkan_backend))

    assert vk_input.grad is not None
    assert vk_weight.grad is not None
    torch.testing.assert_close(vk_input.grad.cpu(), torch.autograd.grad(
        torch.nn.functional.linear(cpu_input, cpu_weight).sum(), cpu_input,
        retain_graph=True)[0])


def test_linear_backward_is_first_order_only(vulkan_backend):
    input = torch.randn(2, 8, dtype=torch.float32).to(vulkan_backend).requires_grad_()
    weight = torch.randn(16, 8, dtype=torch.float32).to(vulkan_backend).requires_grad_()
    output = torch.nn.functional.linear(input, weight)

    gradient = torch.autograd.grad(
        output, input, torch.ones_like(output.cpu()).to(vulkan_backend), create_graph=True
    )[0]

    assert gradient.device == torch.device("vk:0")
    assert not gradient.requires_grad


def test_addmm_forward_and_out_are_reachable(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input_vk, weight_vk, bias_vk = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )
    expected = torch.addmm(cpu_bias, cpu_input, cpu_weight.t())
    result = torch.addmm(bias_vk, input_vk, weight_vk.t())
    output = torch.empty_like(result)
    assert torch.addmm(bias_vk, input_vk, weight_vk.t(), out=output) is output
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(output.cpu(), expected)


@pytest.mark.parametrize("keyword", ["alpha", "beta"])
def test_addmm_out_rejects_non_default_scalars(vulkan_backend, keyword):
    _, _, bias = _linear_inputs("cpu")
    mat1 = torch.randn((2, 8), dtype=torch.float32).to(vulkan_backend)
    mat2 = torch.randn((8, 16), dtype=torch.float32).to(vulkan_backend)
    output = torch.empty((2, 16), dtype=torch.float32, device=vulkan_backend)

    with pytest.raises(RuntimeError, match=rf"{keyword} == 1"):
        torch.addmm(bias.to(vulkan_backend), mat1, mat2, out=output, **{keyword: 2})


def test_addmm_rejects_arbitrary_non_contiguous_mat2(vulkan_backend):
    _, _, bias = _linear_inputs("cpu")
    mat1 = torch.randn((2, 8), dtype=torch.float32).to(vulkan_backend)
    mat2 = torch.randn((8, 32), dtype=torch.float32).to(vulkan_backend)[:, ::2]
    assert not mat2.is_contiguous()
    assert mat2.stride() != (1, 8)

    with pytest.raises(RuntimeError, match="contiguous|transpose|layout"):
        torch.addmm(bias.to(vulkan_backend), mat1, mat2)


def test_linear_rejects_invalid_rank_shape_dtype_layout_device_and_offset(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input_vk, weight_vk, bias_vk = (
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend), cpu_bias.to(vulkan_backend)
    )

    with pytest.raises(RuntimeError, match="2-D|rank"):
        torch.nn.functional.linear(input_vk.unsqueeze(0), weight_vk, bias_vk)
    with pytest.raises(RuntimeError, match="matching features|size|shape"):
        torch.nn.functional.linear(input_vk, torch.empty((16, 7), device=vulkan_backend), bias_vk)
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.nn.functional.linear(input_vk, weight_vk.to(torch.bool), bias_vk)
    with pytest.raises(RuntimeError, match="contiguous|layout"):
        torch.nn.functional.linear(input_vk.transpose(0, 1), weight_vk, bias_vk)
    with pytest.raises(RuntimeError, match="same device|Vulkan"):
        torch.nn.functional.linear(input_vk, weight_vk.cpu(), bias_vk)

    offset_input = torch.empty((3, 8), dtype=torch.float32, device=vulkan_backend)[1:]
    assert offset_input.storage_offset() != 0
    with pytest.raises(RuntimeError, match="zero storage offset|offset"):
        torch.nn.functional.linear(offset_input, weight_vk, bias_vk)
