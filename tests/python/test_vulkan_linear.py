from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import pytorch_vulkan
from tools.vulkan_capability_declarations import STOCK_LINEAR_ROUTE_CASES


_STOCK_LINEAR_CASES = {case["name"]: case for case in STOCK_LINEAR_ROUTE_CASES}


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


def test_linear_uses_pytorch_composite_dispatch_boundary():
    source_path = Path(__file__).resolve().parents[2] / "src/vulkan/operators/linear.cpp"
    source = source_path.read_text()
    assert 'm.impl("linear",' not in source
    assert 'm.impl("linear", &pytorch_vulkan::autograd_linear)' not in source
    assert 'm.impl("addmm", &pytorch_vulkan::addmm)' in source


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
    assert pytorch_vulkan._C.compute_dispatch_count() == 3
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


def test_linear_forward_accepts_contiguous_singleton_stride_input(vulkan_backend):
    cpu_base = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cpu_input = cpu_base.as_strided((1, 3), (1, 1))
    cpu_weight = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    vk_base = cpu_base.to(vulkan_backend)
    vk_input = vk_base.as_strided((1, 3), (1, 1))
    vk_weight = cpu_weight.to(vulkan_backend)
    assert vk_input.stride() == (1, 1) and vk_input.is_contiguous()

    result = torch.nn.functional.linear(vk_input, vk_weight)
    pytorch_vulkan._C.synchronize()

    torch.testing.assert_close(
        result.cpu(), torch.nn.functional.linear(cpu_input, cpu_weight),
        rtol=0.003, atol=0.003,
    )
    torch.testing.assert_close(vk_base.cpu(), cpu_base)


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


def test_linear_backward_preserves_create_graph_history(vulkan_backend):
    cpu_input = torch.tensor([[0.2, -0.7, 1.1]], requires_grad=True)
    cpu_weight = torch.tensor([[0.3, 0.4, -0.2], [-0.6, 0.8, 0.5]], requires_grad=True)
    cpu_bias = torch.tensor([0.1, -0.3], requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()

    cpu_output = torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight, vk_bias)
    cpu_first = torch.autograd.grad((cpu_output * cpu_output).sum(), (cpu_input, cpu_weight, cpu_bias), create_graph=True)
    vk_first = torch.autograd.grad((vk_output * vk_output).sum(), (vk_input, vk_weight, vk_bias), create_graph=True)
    for actual, expected in zip(vk_first, cpu_first):
        assert actual.requires_grad == expected.requires_grad
        assert actual.device == torch.device(vulkan_backend)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)

    cpu_direction = tuple(torch.full_like(value, 0.25 + index) for index, value in enumerate(cpu_first))
    vk_direction = tuple(value.to(vulkan_backend) for value in cpu_direction)
    cpu_second = torch.autograd.grad(sum((grad * direction).sum() for grad, direction in zip(cpu_first, cpu_direction)), (cpu_input, cpu_weight, cpu_bias))
    vk_second = torch.autograd.grad(sum((grad * direction).sum() for grad, direction in zip(vk_first, vk_direction)), (vk_input, vk_weight, vk_bias))
    for actual, expected in zip(vk_second, cpu_second):
        assert actual.device == torch.device(vulkan_backend)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


def _profile_linear(module, input, no_grad=False):
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        with profile(activities=[ProfilerActivity.CPU]) as trace:
            output = module(input)
    return output, {event.key for event in trace.key_averages()}


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("has_bias", [False, True], ids=["bias-absent", "bias-present"])
@pytest.mark.parametrize("state", ["trainable", "frozen-weight", "no-grad"], ids=lambda value: value)
def test_stock_linear_each_declared_rank(vulkan_backend, rank, has_bias, state):
    shape = (2,) * (rank - 1) + (3,)
    generator = torch.Generator(device="cpu").manual_seed(4100 + rank * 10 + has_bias)
    cpu_input = torch.randn(shape, generator=generator)
    cpu_weight = torch.randn((4, 3), generator=generator)
    cpu_bias = torch.randn((4,), generator=generator) if has_bias else None
    cpu_input.requires_grad_(state != "no-grad")
    cpu_weight.requires_grad_(state == "trainable")
    if cpu_bias is not None:
        cpu_bias.requires_grad_(state == "trainable")
    cpu_module = torch.nn.Linear(3, 4, bias=has_bias)
    with torch.no_grad():
        cpu_module.weight.copy_(cpu_weight)
        if has_bias:
            cpu_module.bias.copy_(cpu_bias)
    cpu_module.weight.requires_grad_(state == "trainable")
    if has_bias:
        cpu_module.bias.requires_grad_(state == "trainable")
    module = torch.nn.Linear(3, 4, bias=has_bias)
    with torch.no_grad():
        module.weight.copy_(cpu_weight)
        if has_bias:
            module.bias.copy_(cpu_bias)
    module.weight.requires_grad_(state == "trainable")
    if has_bias:
        module.bias.requires_grad_(state == "trainable")
    module.to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(state != "no-grad")

    cpu_output, cpu_trace = _profile_linear(cpu_module, cpu_input, no_grad=state == "no-grad")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_output, vk_trace = _profile_linear(module, vk_input, no_grad=state == "no-grad")
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    assert vk_output.device == torch.device(vulkan_backend)
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=0.003, atol=0.003)

    matrix_ops = {schema for schema in vk_trace if schema in {
        "aten::mm", "aten::addmm", "aten::bmm"
    }}
    assert matrix_ops, f"rank {rank} {state} bias={has_bias} did not reach a matrix leaf: {vk_trace}"
    assert matrix_ops == {schema for schema in cpu_trace if schema in {
        "aten::mm", "aten::addmm", "aten::bmm"
    }}
    case_id = (
        f"linear.rank{rank}.contiguous.bias-{'present' if has_bias else 'absent'}.{state}"
    )
    route = _STOCK_LINEAR_CASES[case_id]
    assert route["matrix_leaf"].removesuffix(".default") in matrix_ops
    assert {op.removesuffix(".default") for op in route["required_forward_ops"]} <= vk_trace
    if state == "no-grad":
        assert not vk_output.requires_grad
        return

    cpu_loss, vk_loss = (cpu_output * cpu_output).sum(), (vk_output * vk_output).sum()
    cpu_targets = (cpu_input,)
    vk_targets = (vk_input,)
    if state == "trainable":
        cpu_targets += (cpu_module.weight,)
        vk_targets += (module.weight,)
        if has_bias:
            cpu_targets += (cpu_module.bias,)
            vk_targets += (module.bias,)
    cpu_first = torch.autograd.grad(cpu_loss, cpu_targets, create_graph=state == "trainable")
    vk_first = torch.autograd.grad(vk_loss, vk_targets, create_graph=state == "trainable")
    for actual, expected in zip(vk_first, cpu_first):
        assert actual.device == torch.device(vulkan_backend)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)

    if state == "trainable" and rank == 2 and has_bias:
        directions = tuple(torch.full_like(value, 0.2 + index).to(vulkan_backend) for index, value in enumerate(vk_first))
        cpu_directions = tuple(value.cpu() for value in directions)
        cpu_second = torch.autograd.grad(sum((g * d).sum() for g, d in zip(cpu_first, cpu_directions)), cpu_targets)
        vk_second = torch.autograd.grad(sum((g * d).sum() for g, d in zip(vk_first, directions)), vk_targets)
        for actual, expected in zip(vk_second, cpu_second):
            assert actual.device == torch.device(vulkan_backend)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("rank", range(3, 9))
@pytest.mark.parametrize("has_bias", [False, True], ids=["bias-absent", "bias-present"])
@pytest.mark.parametrize("state", ["trainable", "frozen-weight", "no-grad"])
def test_stock_linear_offset_transpose_uses_observed_bmm_routes(
    vulkan_backend, rank, has_bias, state
):
    base_shape = (2,) * (rank - 1) + (5,)
    generator = torch.Generator(device="cpu").manual_seed(8700 + rank * 10 + has_bias)
    cpu_base = torch.randn(base_shape, generator=generator)
    cpu_input = cpu_base.transpose(0, 1)[..., 1:4]
    assert cpu_input.storage_offset() != 0 and not cpu_input.is_contiguous()
    cpu_input.requires_grad_(state != "no-grad")
    cpu_weight = torch.randn((4, 3), generator=generator)
    cpu_bias = torch.randn((4,), generator=generator) if has_bias else None

    cpu_module = torch.nn.Linear(3, 4, bias=has_bias)
    with torch.no_grad():
        cpu_module.weight.copy_(cpu_weight)
        if has_bias:
            cpu_module.bias.copy_(cpu_bias)
    cpu_module.weight.requires_grad_(state == "trainable")
    if has_bias:
        cpu_module.bias.requires_grad_(state == "trainable")

    vk_base = cpu_base.to(vulkan_backend)
    vk_input = vk_base.transpose(0, 1)[..., 1:4].detach().requires_grad_(state != "no-grad")
    module = torch.nn.Linear(3, 4, bias=has_bias)
    with torch.no_grad():
        module.weight.copy_(cpu_weight)
        if has_bias:
            module.bias.copy_(cpu_bias)
    module.weight.requires_grad_(state == "trainable")
    if has_bias:
        module.bias.requires_grad_(state == "trainable")
    module.to(vulkan_backend)
    assert vk_input.storage_offset() != 0 and not vk_input.is_contiguous()

    cpu_output, cpu_trace = _profile_linear(cpu_module, cpu_input, no_grad=state == "no-grad")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_output, vk_trace = _profile_linear(module, vk_input, no_grad=state == "no-grad")
    pytorch_vulkan._C.synchronize()
    matrix_leaf = "aten::mm" if state == "trainable" else "aten::bmm"
    case_id = (
        f"linear.rank{rank}.offset-transposed.bias-{'present' if has_bias else 'absent'}.{state}"
    )
    route = _STOCK_LINEAR_CASES[case_id]
    assert route["matrix_leaf"].removesuffix(".default") == matrix_leaf
    assert {op.removesuffix(".default") for op in route["required_forward_ops"]} <= vk_trace
    assert matrix_leaf in vk_trace
    assert matrix_leaf in cpu_trace
    if state == "trainable":
        assert "aten::clone" in vk_trace
        assert "aten::clone" in cpu_trace
    else:
        assert "aten::expand" in vk_trace
        assert "aten::expand" in cpu_trace
    assert ("aten::add_" in vk_trace) == has_bias
    assert ("aten::add_" in cpu_trace) == has_bias
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=0.003, atol=0.003)

    if state == "no-grad":
        assert not vk_output.requires_grad
        return
    cpu_targets = (cpu_input,)
    vk_targets = (vk_input,)
    if state == "trainable":
        cpu_targets += (cpu_module.weight,)
        vk_targets += (module.weight,)
        if has_bias:
            cpu_targets += (cpu_module.bias,)
            vk_targets += (module.bias,)
    cpu_first = torch.autograd.grad((cpu_output * cpu_output).sum(), cpu_targets, create_graph=state == "trainable")
    vk_first = torch.autograd.grad((vk_output * vk_output).sum(), vk_targets, create_graph=state == "trainable")
    for actual, expected in zip(vk_first, cpu_first):
        assert actual.device == torch.device(vulkan_backend)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


def test_stock_linear_saved_weight_mutation_matches_cpu_version_check(vulkan_backend):
    cpu_input = torch.randn((2, 3), generator=torch.Generator().manual_seed(9281), requires_grad=True)
    cpu_module = torch.nn.Linear(3, 4)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    module = torch.nn.Linear(3, 4).to(vulkan_backend)
    with torch.no_grad():
        module.weight.copy_(cpu_module.weight)
        module.bias.copy_(cpu_module.bias)
    cpu_output = cpu_module(cpu_input)
    vk_output = module(vk_input)
    with torch.no_grad():
        cpu_module.weight.add_(1)
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        cpu_output.sum().backward()
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        vk_output.sum().backward()


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


def test_linear_rejects_invalid_shape_dtype_device_and_accepts_supported_layouts(
    vulkan_backend,
):
    cpu_input, cpu_weight, cpu_bias = _linear_inputs("cpu")
    input_vk, weight_vk, bias_vk = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )

    pytorch_vulkan._C.reset_execution_counters()
    high_rank = input_vk.unsqueeze(0).unsqueeze(0)
    high_rank_result = torch.nn.functional.linear(high_rank, weight_vk, bias_vk)
    assert high_rank_result.shape == (1, 1, 2, 16)
    torch.testing.assert_close(
        high_rank_result.cpu(),
        torch.nn.functional.linear(high_rank.cpu(), cpu_weight, cpu_bias),
        rtol=0.003, atol=0.003,
    )
    with pytest.raises(RuntimeError, match="matching features|size|shape|matrices"):
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


def test_linear_backward_accepts_high_rank_input(vulkan_backend):
    cpu_input = torch.randn((1, 1, 2, 8), generator=torch.Generator().manual_seed(9291), requires_grad=True)
    cpu_weight = torch.randn((16, 8), generator=torch.Generator().manual_seed(9292), requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    cpu_output = torch.nn.functional.linear(cpu_input, cpu_weight)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight)
    cpu_output.sum().backward()
    vk_output.sum().backward()
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=0.003, atol=0.003)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0.003, atol=0.003)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad, rtol=0.003, atol=0.003)
