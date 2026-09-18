import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


def _linear_inputs(device):
    torch.manual_seed(23)
    return (
        torch.randn(2, 8, dtype=torch.float32, device=device),
        torch.randn(16, 8, dtype=torch.float32, device=device),
        torch.randn(16, dtype=torch.float32, device=device),
    )


def test_linear_forward_matches_cpu_and_returns_contiguous_f32(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.nn.functional.linear(
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )

    assert result.device == torch.device(vulkan_backend)
    assert result.dtype is torch.float32
    assert result.shape == (2, 16)
    assert result.is_contiguous()
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(
        result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)
    )


def test_linear_forward_without_bias_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, _ = _linear_inputs("cpu")
    result = torch.nn.functional.linear(
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    )

    assert result.device == torch.device(vulkan_backend)
    assert result.dtype is torch.float32
    assert result.shape == (2, 16)
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight)
    )


def test_standalone_linear_is_complete_before_return(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    result = torch.nn.functional.linear(
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )

    # No explicit Vulkan synchronization is needed between dispatch return and
    # consuming the result on the host.
    torch.testing.assert_close(
        result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)
    )


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
    for actual, expected in (
        (vk_input.grad, cpu_input.grad),
        (vk_weight.grad, cpu_weight.grad),
        (vk_bias.grad, cpu_bias.grad),
    ):
        assert actual.device == torch.device(vulkan_backend)
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
    torch.testing.assert_close(
        vk_input.grad.cpu(),
        torch.autograd.grad(
            torch.nn.functional.linear(cpu_input, cpu_weight).sum(),
            cpu_input,
            retain_graph=True,
        )[0],
    )


def test_linear_backward_is_first_order_only(vulkan_backend):
    input = torch.randn(2, 8, dtype=torch.float32).to(vulkan_backend).requires_grad_()
    weight = torch.randn(16, 8, dtype=torch.float32).to(vulkan_backend).requires_grad_()
    output = torch.nn.functional.linear(input, weight)

    gradient = torch.autograd.grad(
        output,
        input,
        torch.ones_like(output.cpu()).to(vulkan_backend),
        create_graph=True,
    )[0]

    assert gradient.device == torch.device(vulkan_backend)
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


def test_addmm_out_accepts_non_default_scalars(vulkan_backend):
    self_cpu = torch.randn((2, 16), dtype=torch.float32)
    mat1_cpu = torch.randn((2, 8), dtype=torch.float32)
    mat2_cpu = torch.randn((8, 16), dtype=torch.float32)
    self, mat1, mat2 = (x.to(vulkan_backend) for x in (self_cpu, mat1_cpu, mat2_cpu))
    output = torch.empty((2, 16), dtype=torch.float32, device=vulkan_backend)
    torch.addmm(self, mat1, mat2, beta=0.25, alpha=2.0, out=output)
    torch.testing.assert_close(
        output.cpu(), torch.addmm(self_cpu, mat1_cpu, mat2_cpu, beta=0.25, alpha=2.0)
    )


def test_addmm_rejects_arbitrary_non_contiguous_mat2(vulkan_backend):
    cpu_input, _, bias = _linear_inputs("cpu")
    cpu_mat2_base = torch.randn((8, 32), dtype=torch.float32)
    cpu_mat2 = cpu_mat2_base[:, ::2]
    mat1 = cpu_input.to(vulkan_backend)
    mat2 = cpu_mat2_base.to(vulkan_backend)[:, ::2]
    assert not mat2.is_contiguous()
    assert mat2.stride() != (1, 8)

    result = torch.addmm(bias.to(vulkan_backend), mat1, mat2)
    torch.testing.assert_close(result.cpu(), torch.addmm(bias, cpu_input, cpu_mat2))


def test_linear_rejects_invalid_rank_shape_dtype_layout_device_and_offset(
    vulkan_backend,
):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input_vk, weight_vk, bias_vk = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )

    with pytest.raises(RuntimeError, match="2-D|rank"):
        torch.nn.functional.linear(input_vk.unsqueeze(0), weight_vk, bias_vk)
    with pytest.raises(RuntimeError, match="matching features|size|shape"):
        torch.nn.functional.linear(
            input_vk, torch.empty((16, 7), device=vulkan_backend), bias_vk
        )
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.nn.functional.linear(input_vk, weight_vk.to(torch.bool), bias_vk)
    with pytest.raises(RuntimeError, match="same device|Vulkan"):
        torch.nn.functional.linear(input_vk, weight_vk.cpu(), bias_vk)

    transposed_input = (
        torch.randn((8, 2), dtype=torch.float32).to(vulkan_backend).transpose(0, 1)
    )
    assert not transposed_input.is_contiguous()
    torch.nn.functional.linear(transposed_input, weight_vk, bias_vk)
    offset_input = torch.empty((3, 8), dtype=torch.float32, device=vulkan_backend)[1:]
    assert offset_input.storage_offset() != 0
    torch.nn.functional.linear(offset_input, weight_vk, bias_vk)


def test_linear_accepts_positive_stride_views_for_all_operands(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    cpu_input_view = cpu_input[:, ::2]
    cpu_weight_view = cpu_weight[:, ::2]
    cpu_bias_view = torch.cat((cpu_bias, torch.zeros_like(cpu_bias)))[::2]
    input_view = cpu_input.to(vulkan_backend)[:, ::2]
    weight_view = cpu_weight.to(vulkan_backend)[:, ::2]
    bias_view = torch.cat((cpu_bias, torch.zeros_like(cpu_bias))).to(vulkan_backend)[
        ::2
    ]
    result = torch.nn.functional.linear(input_view, weight_view, bias_view)
    expected = torch.nn.functional.linear(
        cpu_input_view, cpu_weight_view, cpu_bias_view
    )
    torch.testing.assert_close(result.cpu(), expected)


def test_linear_rejects_overlapping_output_view(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input, weight, bias = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )
    output = torch.empty((1, 16), device=vulkan_backend).expand(2, 16)
    with pytest.raises(RuntimeError, match="overlap|layout|output"):
        torch.addmm(bias.to(vulkan_backend), input, weight.t(), out=output)


def test_linear_accepts_noncontiguous_nonzero_offset_output_view(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input = cpu_input.to(vulkan_backend)
    weight = cpu_weight.to(vulkan_backend)
    bias = cpu_bias.to(vulkan_backend)
    output = torch.empty((17, 2), device=vulkan_backend)[1:].transpose(0, 1)
    assert output.shape == (2, 16)
    assert output.storage_offset() != 0
    assert not output.is_contiguous()
    result = torch.addmm(bias, input, weight.t(), out=output)
    expected = torch.addmm(cpu_bias, cpu_input, cpu_weight.t())
    assert result is output
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_addmm_rejects_output_aliasing_each_operand(vulkan_backend, operand):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input, weight, bias = (
        cpu_input.to(vulkan_backend),
        cpu_weight.t().to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )
    if operand == "input":
        storage = torch.empty((2, 16), device=vulkan_backend)
        input = storage[:, :8]
        output = storage
    elif operand == "weight":
        storage = torch.empty((16, 16), device=vulkan_backend)
        weight = storage[:, :8].t()
        output = storage[:2]
    else:
        storage = torch.empty((32,), device=vulkan_backend)
        bias = storage[:16]
        output = storage.view(2, 16)
    with pytest.raises(RuntimeError, match="alias|overlap|output"):
        torch.addmm(bias, input, weight, out=output)


def test_linear_backward_accepts_view_operands(vulkan_backend):
    cpu_input = torch.randn((2, 16), dtype=torch.float32, requires_grad=True)
    cpu_weight = torch.randn((16, 16), dtype=torch.float32, requires_grad=True)
    cpu_input_view = cpu_input[:, ::2]
    cpu_weight_view = cpu_weight[:, ::2]
    cpu_input_view.retain_grad()
    cpu_weight_view.retain_grad()
    vk_input = cpu_input_view.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight_view.detach().to(vulkan_backend).requires_grad_()
    cpu_output = torch.nn.functional.linear(cpu_input_view, cpu_weight_view)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight)
    cpu_output.backward(torch.ones_like(cpu_output))
    vk_output.backward(torch.ones_like(vk_output.cpu()).to(vulkan_backend))
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input_view.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight_view.grad)


def test_linear_backward_rejects_unsupported_view_rank(vulkan_backend):
    _, cpu_weight, _ = _linear_inputs("cpu")
    weight = cpu_weight.to(vulkan_backend)
    value = torch.randn((2, 8)).unsqueeze(0).to(vulkan_backend).requires_grad_()
    with pytest.raises(RuntimeError, match="2-D|rank|dimension"):
        torch.nn.functional.linear(value, weight)
