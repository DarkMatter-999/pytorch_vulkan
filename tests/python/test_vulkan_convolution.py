import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _conv_inputs(device, requires_grad=False):
    torch.manual_seed(67)
    input = torch.randn(2, 1, 8, 8, dtype=torch.float32).to(device)
    weight = torch.randn(4, 1, 3, 3, dtype=torch.float32).to(device)
    bias = torch.randn(4, dtype=torch.float32).to(device)
    if requires_grad:
        input.requires_grad_()
        weight.requires_grad_()
        bias.requires_grad_()
    return input, weight, bias


def test_conv2d_fixed_contract_forward_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu")
    result = torch.nn.functional.conv2d(
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend), stride=1, padding=1, dilation=1, groups=1,
    )
    expected = torch.nn.functional.conv2d(
        cpu_input, cpu_weight, cpu_bias, stride=1, padding=1, dilation=1, groups=1,
    )

    assert result.device == torch.device("vk:0")
    assert result.dtype is torch.float32
    assert result.shape == (2, 4, 8, 8)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), expected)


def test_conv2d_fixed_contract_backward_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu", requires_grad=True)
    vk_input, vk_weight, vk_bias = _conv_inputs(vulkan_backend, requires_grad=True)
    cpu_output = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    vk_output = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
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


def test_conv2d_backward_is_first_order_only(vulkan_backend):
    input, weight, bias = _conv_inputs(vulkan_backend, requires_grad=True)
    output = torch.nn.functional.conv2d(input, weight, bias, padding=1)

    gradient = torch.autograd.grad(
        output, input, torch.ones_like(output.cpu()).to(vulkan_backend), create_graph=True
    )[0]

    assert gradient.device == torch.device("vk:0")
    assert not gradient.requires_grad


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stride": 2},
        {"padding": 0},
        {"dilation": 2},
        {"groups": 2},
    ],
)
def test_conv2d_rejects_non_fixed_parameters(vulkan_backend, kwargs):
    input, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="Vulkan convolution|fixed|support|channels"):
        torch.nn.functional.conv2d(input, weight, bias, **kwargs)


@pytest.mark.parametrize("bad_input", [
    torch.empty((1, 1, 8, 8)),
    torch.empty((2, 2, 8, 8)),
    torch.empty((2, 1, 7, 8)),
])
def test_conv2d_rejects_non_fixed_input_shape(vulkan_backend, bad_input):
    _, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="shape|size|fixed|support"):
        torch.nn.functional.conv2d(bad_input.to(vulkan_backend), weight, bias, padding=1)


@pytest.mark.parametrize("shape", [(), (8,), (2, 1), (2, 1, 8), (2, 1, 8, 8, 1)])
def test_conv2d_rejects_every_other_input_rank(vulkan_backend, shape):
    _, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|size|Expected|fixed|support"):
        torch.nn.functional.conv2d(torch.empty(shape, device=vulkan_backend), weight, bias, padding=1)


@pytest.mark.parametrize("bad_weight", [
    torch.empty((4, 3, 3, 3)),
    torch.empty((4, 1, 3)),
    torch.empty((4, 1, 3, 3, 1)),
])
def test_conv2d_rejects_every_other_weight_rank_or_shape(vulkan_backend, bad_weight):
    input, _, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|size|Expected|fixed|support|channels"):
        torch.nn.functional.conv2d(input, bad_weight.to(vulkan_backend), bias, padding=1)


@pytest.mark.parametrize("bad_bias", [
    torch.empty(()),
    torch.empty((4, 1)),
    torch.empty((3,)),
    torch.empty((5,)),
])
def test_conv2d_rejects_every_other_bias_rank_or_shape(vulkan_backend, bad_bias):
    input, weight, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="bias|rank|shape|size|fixed|support"):
        torch.nn.functional.conv2d(input, weight, bad_bias.to(vulkan_backend), padding=1)


def test_conv2d_rejects_non_fixed_weight_and_bias_shapes(vulkan_backend):
    input, _, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="shape|size|fixed|support"):
        torch.nn.functional.conv2d(input, torch.empty((3, 1, 3, 3), device=vulkan_backend),
                                   torch.empty((3,), device=vulkan_backend), padding=1)
    with pytest.raises(RuntimeError, match="shape|size|fixed|support"):
        torch.nn.functional.conv2d(input, torch.empty((4, 1, 5, 3), device=vulkan_backend),
                                   torch.empty((4,), device=vulkan_backend), padding=1)
    _, weight, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="bias|shape|size|fixed|support"):
        torch.nn.functional.conv2d(input, weight, torch.empty((1,), device=vulkan_backend), padding=1)
    with pytest.raises(RuntimeError, match="bias|unsupported|Vulkan|device"):
        torch.nn.functional.conv2d(input, weight, None, padding=1)


def test_conv2d_rejects_rank_dtype_layout_device_and_offset(vulkan_backend):
    input, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|4-D|Expected"):
        torch.nn.functional.conv2d(input.unsqueeze(0), weight, bias, padding=1)
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.nn.functional.conv2d(input, weight.to(torch.float64), bias, padding=1)
    with pytest.raises(RuntimeError, match="contiguous|layout"):
        torch.nn.functional.conv2d(input.transpose(2, 3), weight, bias, padding=1)
    with pytest.raises(RuntimeError, match="device|Vulkan"):
        torch.nn.functional.conv2d(input, weight.cpu(), bias, padding=1)
    offset_input = torch.empty((3, 1, 8, 8), device=vulkan_backend)[1:]
    assert offset_input.storage_offset() != 0
    with pytest.raises(RuntimeError, match="offset|contiguous"):
        torch.nn.functional.conv2d(offset_input, weight, bias, padding=1)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_rejects_each_non_f32_operand(vulkan_backend, operand):
    input, weight, bias = _conv_inputs("cpu")
    tensors = {"input": input, "weight": weight, "bias": bias}
    with pytest.raises(RuntimeError, match="float32|dtype|allocation"):
        tensors[operand].to(torch.float64).to(vulkan_backend)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_rejects_each_non_contiguous_operand(vulkan_backend, operand):
    input, weight, bias = _conv_inputs(vulkan_backend)
    if operand == "input":
        input = input.transpose(2, 3)
    elif operand == "weight":
        weight = weight.transpose(2, 3)
    else:
        bias = torch.empty(8, device=vulkan_backend)[::2]
    assert not {"input": input, "weight": weight, "bias": bias}[operand].is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous|layout"):
        torch.nn.functional.conv2d(input, weight, bias, padding=1)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_rejects_each_device_mismatch(vulkan_backend, operand):
    input, weight, bias = _conv_inputs(vulkan_backend)
    if operand == "input":
        input = input.cpu()
    elif operand == "weight":
        weight = weight.cpu()
    else:
        bias = bias.cpu()
    with pytest.raises(RuntimeError, match="device|Vulkan|same"):
        torch.nn.functional.conv2d(input, weight, bias, padding=1)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_rejects_each_nonzero_storage_offset(vulkan_backend, operand):
    input, weight, bias = _conv_inputs(vulkan_backend)
    if operand == "input":
        input = torch.empty((3, 1, 8, 8), device=vulkan_backend)[1:]
    elif operand == "weight":
        weight = torch.empty((5, 1, 3, 3), device=vulkan_backend)[1:]
    else:
        bias = torch.empty(5, device=vulkan_backend)[1:]
    assert {"input": input, "weight": weight, "bias": bias}[operand].storage_offset() != 0
    with pytest.raises(RuntimeError, match="offset|contiguous"):
        torch.nn.functional.conv2d(input, weight, bias, padding=1)


def _conv_transpose_inputs(device):
    cpu_input = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    cpu_weight = torch.randn(4, 1, 3, 3, dtype=torch.float32)
    cpu_bias = torch.randn(1, dtype=torch.float32)
    return cpu_input.to(device), cpu_weight.to(device), cpu_bias.to(device)


@pytest.mark.parametrize("output_padding", [(0, 0), (1, 0)])
def test_conv2d_rejects_transposed_at_vulkan_parameter_boundary(vulkan_backend, output_padding):
    input, weight, bias = _conv_transpose_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="transposed|fixed|support|Vulkan"):
        torch.nn.functional.conv_transpose2d(input, weight, bias, padding=1,
                                             output_padding=output_padding)
