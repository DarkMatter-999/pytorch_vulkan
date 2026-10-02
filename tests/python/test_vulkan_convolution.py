import copy
from pathlib import Path

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


def test_conv_bias_gradient_uses_parallel_workgroup_reduction():
    root = Path(__file__).resolve().parents[2]
    shader = (root / "src/vulkan/shaders/glsl/convolution.comp").read_text()
    assert "} else if (params.operation == 3u) {" in shader
    bias_branch = shader.split("} else if (params.operation == 3u) {", 1)[1].split(
        "\n  }", 1
    )[0]
    assert "gl_WorkGroupID.x" in bias_branch
    assert "gl_LocalInvocationID.x" in bias_branch
    assert "reduction_values[lane]" in bias_branch
    assert "barrier();" in bias_branch

    compute = (root / "src/vulkan_compute.cpp").read_text()
    convolution_dispatch = compute.split("void VulkanCompute::convolution(", 1)[1].split(
        "void VulkanCompute::pooling(", 1
    )[0]
    assert "operation == 2 || operation == 3" in convolution_dispatch


def test_conv2d_fixed_contract_forward_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu")
    result = torch.nn.functional.conv2d(
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    expected = torch.nn.functional.conv2d(
        cpu_input,
        cpu_weight,
        cpu_bias,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
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
    for actual, expected in (
        (vk_input.grad, cpu_input.grad),
        (vk_weight.grad, cpu_weight.grad),
        (vk_bias.grad, cpu_bias.grad),
    ):
        assert actual.device == torch.device("vk:0")
        assert actual.dtype is torch.float32
        assert actual.is_contiguous()
        torch.testing.assert_close(actual.cpu(), expected)


def test_conv2d_cnn_batch_weight_gradient_matches_cpu(vulkan_backend):
    torch.manual_seed(1729)
    cpu_input = torch.randn((1024, 3, 32, 32), dtype=torch.float32, requires_grad=True)
    cpu_weight = torch.randn((8, 3, 3, 3), dtype=torch.float32, requires_grad=True)
    cpu_bias = torch.randn((8,), dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn((1024, 8, 32, 32), dtype=torch.float32)

    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()

    cpu_output = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    vk_output = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
    cpu_output.backward(grad_output)
    vk_output.backward(grad_output.to(vulkan_backend))

    for actual, expected in (
        (vk_input.grad, cpu_input.grad),
        (vk_weight.grad, cpu_weight.grad),
        (vk_bias.grad, cpu_bias.grad),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)


def test_conv2d_backward_is_first_order_only(vulkan_backend):
    input, weight, bias = _conv_inputs(vulkan_backend, requires_grad=True)
    output = torch.nn.functional.conv2d(input, weight, bias, padding=1)

    gradient = torch.autograd.grad(
        output,
        input,
        torch.ones_like(output.cpu()).to(vulkan_backend),
        create_graph=True,
    )[0]

    assert gradient.device == torch.device("vk:0")
    assert not gradient.requires_grad


@pytest.mark.parametrize(
    "groups,input_channels,output_channels,kernel,stride,dilation,bias",
    [
        (1, 2, 4, (2, 3), (2, 1), (1, 2), True),
        (1, 2, 4, (2, 3), (2, 1), (1, 2), False),
        (2, 4, 6, (3, 2), (1, 2), (2, 1), False),
        (2, 4, 6, (3, 2), (1, 2), (2, 1), True),
        (4, 4, 8, (2, 3), (2, 1), (1, 2), True),
        (4, 4, 8, (2, 3), (2, 1), (1, 2), False),
    ],
    ids=["dense-bias", "dense-no-bias", "grouped-no-bias", "grouped-bias",
         "depthwise-multiplier-bias", "depthwise-multiplier-no-bias"],
)
@pytest.mark.parametrize(
    "trainable",
    [(True, True, True), (True, False, True), (False, True, False),
     (False, False, True), (True, True, False), (False, False, False)],
    ids=["all", "input-bias", "weight", "bias-only", "input-weight", "frozen"],
)
def test_conv2d_module_first_backward_matches_cpu_with_mixed_trainability(
    vulkan_backend, groups, input_channels, output_channels, kernel, stride, dilation,
    bias, trainable, request,
):
    generator = torch.Generator(device="cpu").manual_seed(
        301 + groups * 17 + output_channels + sum(trainable)
    )
    cpu_module = torch.nn.Conv2d(
        input_channels, output_channels, kernel, stride=stride, dilation=dilation,
        groups=groups, bias=bias,
    )
    if "depthwise-multiplier" in request.node.callspec.id:
        assert groups == input_channels
        assert cpu_module.weight.shape[1] == 1
        assert output_channels % input_channels == 0
    needs = (trainable[0], trainable[1], bias and trainable[2])
    cpu_module.weight.requires_grad_(needs[1])
    if bias:
        cpu_module.bias.requires_grad_(needs[2])
    cpu_input = torch.randn((2, input_channels, 9, 10), generator=generator)
    cpu_input.requires_grad_(needs[0])
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(needs[0])

    cpu_output = cpu_module(cpu_input)
    vk_output = vk_module(vk_input)
    cpu_seed = torch.randn(cpu_output.shape, generator=generator)
    vk_seed = cpu_seed.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    assert vk_output.shape == cpu_output.shape
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=2e-4, atol=2e-4)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    if any(needs):
        cpu_output.backward(cpu_seed)
        vk_output.backward(vk_seed)
        pytorch_vulkan._C.synchronize()
        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert counters == (sum(needs), 0, 0, 0)
    else:
        assert not cpu_output.requires_grad and not vk_output.requires_grad
        assert cpu_input.grad is None and vk_input.grad is None
        assert cpu_module.weight.grad is None and vk_module.weight.grad is None
        if bias:
            assert cpu_module.bias.grad is None and vk_module.bias.grad is None
        assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
        return

    for actual, expected, selected in (
        (vk_input.grad, cpu_input.grad, needs[0]),
        (vk_module.weight.grad, cpu_module.weight.grad, needs[1]),
        (vk_module.bias.grad if bias else None,
         cpu_module.bias.grad if bias else None, needs[2]),
    ):
        assert (actual is not None) is selected
        assert (expected is not None) is selected
        if selected:
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("device", ["cpu", "vk:0"], ids=["cpu", "vulkan"])
def test_conv2d_module_saved_input_mutation_checks_version(device, vulkan_backend):
    actual_device = "vk:0" if device == "vk:0" else "cpu"
    module = torch.nn.Conv2d(2, 3, 3, bias=True)
    if actual_device == "vk:0":
        module = copy.deepcopy(module).to(vulkan_backend)
    cpu_input = torch.randn((1, 2, 5, 6), generator=torch.Generator().manual_seed(451))
    input = cpu_input.to(actual_device).requires_grad_(True)
    output = module(input)
    with torch.no_grad():
        input.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        output.sum().backward()


def test_conv2d_module_expanded_upstream_seed_matches_cpu(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(9817)
    cpu_module = torch.nn.Conv2d(2, 4, (2, 3), stride=(1, 2), padding=(1, 0), groups=2)
    cpu_input = torch.randn((2, 2, 5, 7), generator=generator, requires_grad=True)
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(True)
    cpu_output, vk_output = cpu_module(cpu_input), vk_module(vk_input)
    cpu_seed_base = torch.randn((2, 4, 1, 1), generator=generator)
    cpu_seed = cpu_seed_base.expand_as(cpu_output)
    vk_seed = cpu_seed_base.to(vulkan_backend).expand(vk_output.shape)
    cpu_output.backward(cpu_seed)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_seed)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (3, 0, 0, 0)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(vk_module.weight.grad.cpu(), cpu_module.weight.grad, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(vk_module.bias.grad.cpu(), cpu_module.bias.grad, rtol=3e-4, atol=3e-4)


def test_convolution_accepts_native_transformed_bias_free_group_views(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(117)
    cpu_input_base = torch.randn((4, 1, 6, 7), generator=generator)
    cpu_weight_base = torch.randn((6, 1, 3, 5), generator=generator)
    vk_input_base = cpu_input_base.to(vulkan_backend)
    vk_weight_base = cpu_weight_base.to(vulkan_backend)
    vk_groups = [
        (
            vk_input_base.narrow(0, group * 2, 2),
            vk_weight_base.narrow(0, group * 3, 3),
        )
        for group in range(2)
    ]
    cpu_groups = [
        (cpu_input_base.narrow(0, group * 2, 2), cpu_weight_base.narrow(0, group * 3, 3))
        for group in range(2)
    ]
    for group, ((vk_input, vk_weight), (cpu_input, cpu_weight)) in enumerate(
        zip(vk_groups, cpu_groups)
    ):
        assert tuple(vk_input.shape) == (2, 1, 6, 7)
        assert tuple(vk_weight.shape) == (3, 1, 3, 5)
        assert (vk_input.storage_offset(), vk_weight.storage_offset()) == (
            (0, 0) if group == 0 else (84, 45)
        )
    cpu_results = [
        torch.nn.functional.conv2d(x, w, None, stride=(1, 2), padding=(1, 0),
                                   dilation=(2, 1), groups=1)
        for x, w in cpu_groups
    ]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_results = [
        torch.nn.functional.conv2d(x, w, None, stride=(1, 2), padding=(1, 0),
                                   dilation=(2, 1), groups=1)
        for x, w in vk_groups
    ]
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    for actual, expected in zip(vk_results, cpu_results):
        assert tuple(actual.shape) == (2, 3, 4, 2)
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)
    cpu_cat = torch.cat(cpu_results, dim=1).transpose(0, 1).narrow(2, 0, 3)
    vk_cat = torch.cat(vk_results, dim=1).transpose(0, 1).narrow(2, 0, 3)
    assert tuple(vk_cat.shape) == (6, 2, 3, 2)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_cat.cpu(), cpu_cat, rtol=3e-4, atol=3e-4)


def test_convolution_backward_fixed_schema_matches_cpu(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu", requires_grad=True)
    vk_input, vk_weight, vk_bias = _conv_inputs(vulkan_backend, requires_grad=True)
    cpu_output = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    vk_output = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
    grad = torch.randn_like(cpu_output)
    expected = torch.ops.aten.convolution_backward.default(
        grad,
        cpu_input,
        cpu_weight,
        [4],
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        1,
        [True, True, True],
    )
    actual = torch.ops.aten.convolution_backward.default(
        grad.to(vulkan_backend),
        vk_input,
        vk_weight,
        [4],
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        1,
        [True, True, True],
    )
    assert len(actual) == 3
    for value, reference in zip(actual, expected):
        assert value.device == torch.device("vk:0")
        assert value.dtype is torch.float32
        torch.testing.assert_close(value.cpu(), reference)


@pytest.mark.parametrize("has_forward_bias", [False, True], ids=["bias-absent", "bias-present"])
@pytest.mark.parametrize("mask_bits", range(8), ids=[f"mask-{value:03b}" for value in range(8)])
def test_convolution_backward_all_output_masks_match_cpu(
    vulkan_backend, has_forward_bias, mask_bits
):
    generator = torch.Generator().manual_seed(20261002 + mask_bits)
    cpu_input = torch.randn((2, 4, 7, 8), generator=generator)
    cpu_weight = torch.randn((6, 2, 3, 2), generator=generator)
    cpu_bias = torch.randn((6,), generator=generator) if has_forward_bias else None
    cpu_grad = torch.randn((2, 6, 7, 9), generator=generator)
    input = cpu_input.to(vulkan_backend)
    weight = cpu_weight.to(vulkan_backend)
    bias = cpu_bias.to(vulkan_backend) if cpu_bias is not None else None
    grad = cpu_grad.to(vulkan_backend)
    mask = [(mask_bits & (1 << (2 - index))) != 0 for index in range(3)]

    # The forward bias state is prepared through ordinary convolution; it is not
    # an argument to the direct backward schema.
    torch.nn.functional.conv2d(input, weight, bias, padding=(1, 1), groups=2)
    with torch.backends.mkldnn.flags(enabled=False):
        torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=(1, 1), groups=2)
        expected = torch.ops.aten.convolution_backward.default(
            cpu_grad, cpu_input, cpu_weight, None, [1, 1], [1, 1], [1, 1],
            False, [0, 0], 2, mask
        )

    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    actual = torch.ops.aten.convolution_backward.default(
        grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2, mask
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    after_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    after_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]

    assert len(actual) == 3
    assert counters == (sum(mask), 0, 0, 0)
    if not any(mask):
        assert after_live == before_live
        assert after_creations == before_creations
    for index, (value, reference) in enumerate(zip(actual, expected)):
        assert (value is not None) is mask[index]
        if not mask[index]:
            continue
        assert value.shape == reference.shape
        assert value.dtype is torch.float32
        assert value.device == torch.device("vk:0")
        torch.testing.assert_close(value.cpu(), reference)


@pytest.mark.parametrize("bias_sizes", [None, [], [6], [19]], ids=["none", "empty", "matching", "mismatching"])
def test_convolution_backward_bias_sizes_are_advisory(vulkan_backend, bias_sizes):
    generator = torch.Generator().manual_seed(90210)
    cpu_input = torch.randn((2, 4, 7, 8), generator=generator)
    cpu_weight = torch.randn((6, 2, 3, 2), generator=generator)
    cpu_grad = torch.randn((2, 6, 7, 9), generator=generator)
    input, weight, grad = (value.to(vulkan_backend) for value in (cpu_input, cpu_weight, cpu_grad))
    with torch.backends.mkldnn.flags(enabled=False):
        expected = torch.ops.aten.convolution_backward.default(
            cpu_grad, cpu_input, cpu_weight, bias_sizes, [1, 1], [1, 1], [1, 1],
            False, [0, 0], 2, [False, False, True]
        )[2]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.convolution_backward.default(
        grad, input, weight, bias_sizes, [1, 1], [1, 1], [1, 1], False, [0, 0],
        2, [False, False, True]
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    assert actual[0] is None and actual[1] is None
    assert actual[2].shape == (6,)
    torch.testing.assert_close(actual[2].cpu(), expected)


def test_convolution_backward_bias_only_reduces_actual_spatial_domain(vulkan_backend):
    generator = torch.Generator().manual_seed(314159)
    cpu_input = torch.randn((2, 4, 7, 8), generator=generator)
    cpu_weight = torch.randn((6, 2, 3, 2), generator=generator)
    cpu_grad = torch.randn((2, 6, 4, 5), generator=generator)
    input, weight, grad = (value.to(vulkan_backend) for value in (cpu_input, cpu_weight, cpu_grad))
    with torch.backends.mkldnn.flags(enabled=False):
        expected = torch.ops.aten.convolution_backward.default(
            cpu_grad, cpu_input, cpu_weight, None, [1, 1], [1, 1], [1, 1],
            False, [0, 0], 2, [False, False, True]
        )[2]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.convolution_backward.default(
        grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,
        [False, False, True]
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    assert actual[0] is None and actual[1] is None
    torch.testing.assert_close(actual[2].cpu(), expected)


@pytest.mark.parametrize(
    "mask",
    [[True, False, False], [False, True, False]],
    ids=["dinput", "dweight"],
)
def test_convolution_backward_selected_input_or_weight_rejects_spatial_mismatch(
    vulkan_backend, mask
):
    input = torch.randn((2, 4, 7, 8)).to(vulkan_backend)
    weight = torch.randn((6, 2, 3, 2)).to(vulkan_backend)
    grad = torch.randn((2, 6, 6, 9)).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="grad_output shape expected"):
        torch.ops.aten.convolution_backward.default(
            grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,
            mask
        )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


@pytest.mark.parametrize("bad_extent", ["batch", "channels"], ids=["batch", "channels"])
def test_convolution_backward_bias_only_rejects_batch_or_channel_mismatch(
    vulkan_backend, bad_extent
):
    input = torch.randn((2, 4, 7, 8)).to(vulkan_backend)
    weight = torch.randn((6, 2, 3, 2)).to(vulkan_backend)
    grad_shape = (1, 6, 4, 5) if bad_extent == "batch" else (2, 5, 4, 5)
    grad = torch.randn(grad_shape).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="dBias grad_output requires matching batch and output channels"):
        torch.ops.aten.convolution_backward.default(
            grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,
            [False, False, True]
        )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_convolution_backward_empty_mask_skips_output_dependent_shape_checks(vulkan_backend):
    input = torch.randn((2, 4, 7, 8)).to(vulkan_backend)
    weight = torch.randn((6, 2, 3, 2)).to(vulkan_backend)
    grad = torch.randn((1, 5, 4, 5)).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    actual = torch.ops.aten.convolution_backward.default(
        grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,
        [False, False, False]
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    after_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    after_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    assert actual == (None, None, None)
    assert counters == (0, 0, 0, 0)
    assert after_live == before_live
    assert after_creations == before_creations


def test_convolution_backward_empty_mask_keeps_common_input_weight_checks(vulkan_backend):
    input = torch.randn((2, 4, 7, 8)).to(vulkan_backend)
    weight = torch.randn((6, 1, 3, 2)).to(vulkan_backend)
    grad = torch.randn((2, 6, 7, 9)).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="grouped input channels do not match"):
        torch.ops.aten.convolution_backward.default(
            grad, input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,
            [False, False, False]
        )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_convolution_backward_bias_only_bounds_expanded_grad_reduction_loop(vulkan_backend):
    input = torch.randn((1, 1, 1, 1)).to(vulkan_backend)
    weight = torch.randn((1, 1, 1, 1)).to(vulkan_backend)
    grad = torch.ones((1, 1, 1, 1)).to(vulkan_backend).expand((1, 1, 65535, 65537))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="logical numel/reduction exceeds shader index range"):
        torch.ops.aten.convolution_backward.default(
            grad, input, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1,
            [False, False, True]
        )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    after_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    after_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    assert counters == (0, 0, 0, 0)
    assert after_live == before_live
    assert after_creations == before_creations


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": 2},
    ],
)
def test_conv2d_rejects_unsupported_parameters(vulkan_backend, kwargs):
    input, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="Vulkan convolution|fixed|support|channels"):
        torch.nn.functional.conv2d(input, weight, bias, **kwargs)


def test_conv2d_rejects_geometry_outside_shader_int_range(vulkan_backend):
    input = torch.randn((1, 1, 12, 12)).to(vulkan_backend)
    weight = torch.randn((1, 1, 1, 1)).to(vulkan_backend)
    bias = torch.randn((1,)).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="Vulkan convolution.*(range|represent|32-bit)"):
        torch.nn.functional.conv2d(
            input, weight, bias, stride=2**40, padding=2**40
        )


def test_conv2d_rejects_unrepresentable_shader_intermediates(vulkan_backend):
    input = torch.randn((1, 1, 1, 1)).to(vulkan_backend)
    weight = torch.randn((1, 1, 1, 1)).to(vulkan_backend)
    bias = torch.randn((1,)).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="^Vulkan convolution geometry exceeds shader range$"):
        torch.nn.functional.conv2d(
            input, weight, bias, stride=1_500_000_000, padding=1_500_000_000
        )


def test_convolution_backward_rejects_geometry_outside_shader_int_range(vulkan_backend):
    grad = torch.empty((1, 1, 1, 1)).to(vulkan_backend)
    input = torch.empty((1, 1, 1, 1)).to(vulkan_backend)
    weight = torch.empty((1, 1, 1, 1)).to(vulkan_backend)
    with pytest.raises((RuntimeError, ValueError), match="geometry exceeds shader range|geometry values must fit signed 32-bit shader arithmetic"):
        torch.ops.aten.convolution_backward.default(
            grad,
            input,
            weight,
            [1],
            [2**40, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )


@pytest.mark.parametrize("operand", ["input", "weight", "grad_output"])
def test_convolution_rejects_channels_last_before_output_allocation(
    vulkan_backend, operand
):
    cpu_input = torch.randn((2, 2, 5, 6), dtype=torch.float32)
    cpu_weight = torch.randn((3, 2, 3, 2), dtype=torch.float32)
    cpu_bias = torch.randn((3,), dtype=torch.float32)
    cpu_output = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias)
    input = cpu_input.to(vulkan_backend)
    weight = cpu_weight.to(vulkan_backend)
    bias = cpu_bias.to(vulkan_backend)
    grad = torch.randn_like(cpu_output).to(vulkan_backend)
    if operand == "input":
        input = cpu_input.contiguous(memory_format=torch.channels_last).to(vulkan_backend)
    elif operand == "weight":
        weight = cpu_weight.contiguous(memory_format=torch.channels_last).to(vulkan_backend)
    else:
        grad = cpu_output.contiguous(memory_format=torch.channels_last).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="channels.last|canonical|layout"):
        if operand == "grad_output":
            torch.ops.aten.convolution_backward.default(
                grad, input, weight, [3], [1, 1], [0, 0], [1, 1], False,
                [0, 0], 1, [True, True, True]
            )
        else:
            torch.ops.aten.convolution.default(
                input, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
            )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


@pytest.mark.parametrize(
    "input_shape,weight_shape",
    [((1, 1, 1, 1), (1, 1, 1, 1)), ((2, 1, 7, 8), (6, 1, 3, 2))],
)
def test_convolution_accepts_minimum_and_ambiguous_singleton_layouts(
    vulkan_backend, input_shape, weight_shape
):
    torch.manual_seed(813)
    cpu_input = torch.randn(input_shape)
    cpu_weight = torch.randn(weight_shape)
    cpu_bias = torch.randn(weight_shape[0])
    actual = torch.nn.functional.conv2d(
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend)
    )
    expected = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.is_contiguous()


def test_convolution_rejects_empty_operands_and_extreme_shader_stride(vulkan_backend):
    weight = torch.ones((1, 1, 1, 1)).to(vulkan_backend)
    bias = torch.ones((1,)).to(vulkan_backend)
    empty_input = torch.empty((0, 1, 1, 1)).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="empty|positive|extent"):
        torch.ops.aten.convolution.default(
            empty_input, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
    input = torch.ones((1, 1, 1, 1)).to(vulkan_backend)
    with pytest.raises((RuntimeError, ValueError), match="range|32-bit|shader"):
        torch.ops.aten.convolution.default(
            input, weight, bias, [2**31, 1], [0, 0], [1, 1], False, [0, 0], 1
        )


def test_convolution_backward_rejects_mismatched_grad_output_spatial_shape(vulkan_backend):
    grad = torch.empty((1, 1, 4, 5)).to(vulkan_backend)
    input = torch.empty((1, 1, 5, 5)).to(vulkan_backend)
    weight = torch.empty((1, 1, 3, 3)).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="grad_output shape expected.*actual"):
        torch.ops.aten.convolution_backward.default(
            grad, input, weight, [1], [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
            [True, True, True],
        )


@pytest.mark.parametrize(
    "bad_input",
    [
        torch.empty((2, 1, 8)),
        torch.empty((2, 3, 8, 8)),
        torch.empty((2, 2, 8, 8)),
    ],
)
def test_conv2d_rejects_non_fixed_input_shape(vulkan_backend, bad_input):
    _, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="shape|size|fixed|support"):
        torch.nn.functional.conv2d(
            bad_input.to(vulkan_backend), weight, bias, padding=1
        )


@pytest.mark.parametrize("shape", [(), (8,), (2, 1), (2, 1, 8), (2, 1, 8, 8, 1)])
def test_conv2d_rejects_every_other_input_rank(vulkan_backend, shape):
    _, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|size|Expected|fixed|support"):
        torch.nn.functional.conv2d(
            torch.empty(shape, device=vulkan_backend), weight, bias, padding=1
        )


@pytest.mark.parametrize(
    "bad_weight",
    [
        torch.empty((4, 3, 3, 3)),
        torch.empty((4, 1, 3)),
        torch.empty((4, 1, 3, 3, 1)),
    ],
)
def test_conv2d_rejects_every_other_weight_rank_or_shape(vulkan_backend, bad_weight):
    input, _, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(
        RuntimeError, match="rank|shape|size|Expected|fixed|support|channels"
    ):
        torch.nn.functional.conv2d(
            input, bad_weight.to(vulkan_backend), bias, padding=1
        )


@pytest.mark.parametrize(
    "bad_bias",
    [
        torch.empty(()),
        torch.empty((4, 1)),
        torch.empty((3,)),
        torch.empty((5,)),
    ],
)
def test_conv2d_rejects_every_other_bias_rank_or_shape(vulkan_backend, bad_bias):
    input, weight, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="bias|rank|shape|size|fixed|support"):
        torch.nn.functional.conv2d(
            input, weight, bad_bias.to(vulkan_backend), padding=1
        )


def test_conv2d_rejects_non_fixed_weight_and_bias_shapes(vulkan_backend):
    input, _, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="shape|size|fixed|support|negative dimension|kernel span|geometry exceeds shader range"):
        torch.nn.functional.conv2d(
            input,
            torch.empty((3, 1, 3, 3), device=vulkan_backend),
            torch.empty((2,), device=vulkan_backend),
            padding=1,
        )
    with pytest.raises(RuntimeError, match="shape|size|fixed|support|negative dimension|kernel span|geometry exceeds shader range"):
        torch.nn.functional.conv2d(
            input,
            torch.empty((4, 1, 12, 3), device=vulkan_backend),
            torch.empty((4,), device=vulkan_backend),
            padding=1,
        )
    _, weight, _ = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="bias|shape|size|fixed|support"):
        torch.nn.functional.conv2d(
            input, weight, torch.empty((1,), device=vulkan_backend), padding=1
        )


@pytest.mark.parametrize(
    "groups,input_channels,output_channels,kernel,stride,dilation",
    [
        (1, 2, 4, (2, 3), (2, 1), (1, 2)),
        (2, 4, 6, (3, 2), (1, 2), (2, 1)),
        (2, 4, 8, (2, 3), (2, 1), (1, 2)),
    ],
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_conv2d_without_bias_matches_cpu(
    vulkan_backend, groups, input_channels, output_channels, kernel, stride, dilation,
    requires_grad,
):
    generator = torch.Generator(device="cpu").manual_seed(9241 + groups + output_channels)
    cpu_input = torch.randn((2, input_channels, 9, 10), dtype=torch.float32, generator=generator)
    cpu_weight = torch.randn(
        (output_channels, input_channels // groups, *kernel), dtype=torch.float32,
        generator=generator,
    )
    expected = torch.nn.functional.conv2d(
        cpu_input,
        cpu_weight,
        None,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    vk_input, vk_weight = cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    if requires_grad:
        vk_input.requires_grad_()
        vk_weight.requires_grad_()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()

    actual = torch.nn.functional.conv2d(
        vk_input,
        vk_weight,
        None,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert actual.requires_grad is requires_grad
    assert counters == (1, 0, 0, 0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)


def test_conv2d_rejects_defined_empty_cpu_bias_before_dispatch(vulkan_backend):
    cpu_input = torch.randn((1, 2, 5, 6), dtype=torch.float32)
    cpu_weight = torch.randn((4, 2, 2, 3), dtype=torch.float32)
    vk_input, vk_weight = cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]

    with pytest.raises(RuntimeError, match="bias|device|Vulkan|same"):
        torch.nn.functional.conv2d(vk_input, vk_weight, torch.Tensor())

    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_conv2d_rejects_malformed_defined_bias_before_output_allocation(vulkan_backend):
    cpu_input = torch.randn((1, 2, 5, 6), dtype=torch.float32)
    cpu_weight = torch.randn((4, 2, 2, 3), dtype=torch.float32)
    vk_input, vk_weight = cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    bad_bias = torch.empty((3,), device=vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]

    with pytest.raises(RuntimeError, match="bias"):
        torch.nn.functional.conv2d(vk_input, vk_weight, bad_bias)

    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_conv2d_rejects_rank_dtype_layout_device_and_offset(vulkan_backend):
    input, weight, bias = _conv_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|4-D|Expected"):
        torch.nn.functional.conv2d(input.unsqueeze(0), weight, bias, padding=1)
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.nn.functional.conv2d(input, weight.to(torch.float64), bias, padding=1)
    with pytest.raises(RuntimeError, match="device|Vulkan"):
        torch.nn.functional.conv2d(input, weight.cpu(), bias, padding=1)
    transposed_input = input.transpose(2, 3)
    assert not transposed_input.is_contiguous()
    torch.nn.functional.conv2d(transposed_input, weight, bias, padding=1)
    offset_input = torch.empty((3, 1, 8, 8), device=vulkan_backend)[1:]
    assert offset_input.storage_offset() != 0
    torch.nn.functional.conv2d(offset_input, weight, bias, padding=1)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_rejects_each_non_f32_operand(vulkan_backend, operand):
    input, weight, bias = _conv_inputs("cpu")
    tensors = {"input": input, "weight": weight, "bias": bias}
    with pytest.raises(RuntimeError, match="float32|dtype|allocation"):
        tensors[operand].to(torch.float64).to(vulkan_backend)


@pytest.mark.parametrize("operand", ["input", "weight", "bias"])
def test_conv2d_accepts_each_positive_stride_operand(vulkan_backend, operand):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu")
    input, weight, bias = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )
    if operand == "input":
        cpu_input = cpu_input.transpose(2, 3)
        input = input.transpose(2, 3)
    elif operand == "weight":
        cpu_weight = cpu_weight.transpose(2, 3)
        weight = weight.transpose(2, 3)
    else:
        bias_base = torch.cat((cpu_bias, torch.zeros_like(cpu_bias)))
        cpu_bias = bias_base[::2]
        bias = bias_base.to(vulkan_backend)[::2]
    view = {"input": input, "weight": weight, "bias": bias}[operand]
    assert not view.is_contiguous()
    actual = torch.nn.functional.conv2d(input, weight, bias, padding=1)
    expected = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    torch.testing.assert_close(actual.cpu(), expected)


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
def test_conv2d_accepts_each_nonzero_storage_offset(vulkan_backend, operand):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu")
    base_input, base_weight, base_bias = cpu_input, cpu_weight, cpu_bias
    input, weight, bias = (
        cpu_input.to(vulkan_backend),
        cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend),
    )
    if operand == "input":
        cpu_input = torch.cat((base_input, base_input[:1]), dim=0)[1:]
        input = torch.cat((base_input, base_input[:1]), dim=0).to(vulkan_backend)[1:]
    elif operand == "weight":
        cpu_weight = torch.cat((base_weight, base_weight[:1]), dim=0)[1:]
        weight = torch.cat((base_weight, base_weight[:1]), dim=0).to(vulkan_backend)[1:]
    else:
        cpu_bias = torch.cat((base_bias, base_bias[:1]))[1:]
        bias = torch.cat((base_bias, base_bias[:1])).to(vulkan_backend)[1:]
    view = {"input": input, "weight": weight, "bias": bias}[operand]
    assert view.storage_offset() != 0
    actual = torch.nn.functional.conv2d(input, weight, bias, padding=1)
    expected = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    torch.testing.assert_close(actual.cpu(), expected)


def test_conv2d_accepts_positive_stride_views_for_all_operands(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu")
    input_view = torch.randn((2, 1, 8, 16), dtype=torch.float32)[:, :, :, ::2]
    weight_view = torch.randn((4, 1, 3, 6), dtype=torch.float32)[:, :, :, ::2]
    bias_view = torch.cat((cpu_bias, torch.zeros_like(cpu_bias)))[::2]
    result = torch.nn.functional.conv2d(
        input_view.to(vulkan_backend),
        weight_view.to(vulkan_backend),
        bias_view.to(vulkan_backend),
        padding=1,
    )
    expected = torch.nn.functional.conv2d(input_view, weight_view, bias_view, padding=1)
    torch.testing.assert_close(result.cpu(), expected)


def test_conv2d_backward_accepts_view_operands(vulkan_backend):
    cpu_input, cpu_weight, cpu_bias = _conv_inputs("cpu", requires_grad=True)
    cpu_input = cpu_input.transpose(2, 3)
    cpu_weight = torch.randn((4, 1, 3, 6), dtype=torch.float32, requires_grad=True)[
        :, :, :, ::2
    ]
    cpu_input.retain_grad()
    cpu_weight.retain_grad()
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()
    cpu_output = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
    vk_output = torch.nn.functional.conv2d(vk_input, vk_weight, vk_bias, padding=1)
    grad = torch.ones_like(cpu_output)
    cpu_output.backward(grad)
    vk_output.backward(grad.to(vulkan_backend))
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)


def test_conv2d_backward_rejects_unsupported_view_rank(vulkan_backend):
    _, cpu_weight, cpu_bias = _conv_inputs("cpu")
    weight, bias = cpu_weight.to(vulkan_backend), cpu_bias.to(vulkan_backend)
    value = torch.randn((2, 1, 8, 8)).unsqueeze(0).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|shape|4-D|Expected"):
        torch.nn.functional.conv2d(value, weight, bias, padding=1)


def _conv_transpose_inputs(device):
    cpu_input = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    cpu_weight = torch.randn(4, 1, 3, 3, dtype=torch.float32)
    cpu_bias = torch.randn(1, dtype=torch.float32)
    return cpu_input.to(device), cpu_weight.to(device), cpu_bias.to(device)


@pytest.mark.parametrize("groups,outputs", [(1, 6), (2, 6), (4, 8), (4, 4)])
def test_grouped_convolution_all_gradients_and_launch_extent(vulkan_backend, groups, outputs):
    torch.manual_seed(3187)
    value = torch.randn((2, 4, 8, 8))
    weight = torch.randn((outputs, 4 // groups, 3, 3))
    bias = torch.randn(outputs)
    grad = torch.randn((2, outputs, 8, 8))
    expected_forward = torch.nn.functional.conv2d(value, weight, bias, padding=1, groups=groups)
    actual_forward = torch.nn.functional.conv2d(
        value.to(vulkan_backend), weight.to(vulkan_backend), bias.to(vulkan_backend),
        padding=1, groups=groups,
    )
    args = ([outputs], [1, 1], [1, 1], [1, 1], False, [0, 0], groups, [True, True, True])
    expected = torch.ops.aten.convolution_backward.default(grad, value, weight, *args)
    actual = torch.ops.aten.convolution_backward.default(
        grad.to(vulkan_backend), value.to(vulkan_backend), weight.to(vulkan_backend), *args
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual_forward.cpu(), expected_forward, rtol=2e-4, atol=2e-4)
    assert actual[1].shape == weight.shape
    assert actual[1].numel() == outputs * (4 // groups) * 3 * 3 == weight.numel()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("groups,outputs", [(3, 6), (2, 5)])
def test_grouped_convolution_divisibility_rejections_match_cpu(vulkan_backend, groups, outputs):
    tensors = (torch.ones((2, 4, 8, 8)), torch.ones((outputs, 2, 3, 3)), torch.ones(outputs))
    with pytest.raises(RuntimeError):
        torch.nn.functional.conv2d(*tensors, padding=1, groups=groups)
    with pytest.raises(RuntimeError, match="groups.*divide"):
        torch.nn.functional.conv2d(*(t.to(vulkan_backend) for t in tensors), padding=1, groups=groups)


@pytest.mark.parametrize("transposed,output_padding", [(True, [0, 0]), (False, [1, 0])])
def test_convolution_retains_parameter_guards(vulkan_backend, transposed, output_padding):
    tensors = tuple(torch.ones(shape) for shape in ((2, 4, 8, 8), (4, 4, 3, 3), (4,)))
    args = ([2, 2], [1, 1], [1, 1], transposed, output_padding, 1)
    torch.ops.aten.convolution.default(*tensors, *args)
    with pytest.raises(RuntimeError, match="non-transposed.*zero.*output_padding"):
        torch.ops.aten.convolution.default(*(t.to(vulkan_backend) for t in tensors), *args)


@pytest.mark.parametrize("output_padding", [(0, 0), (1, 0)])
def test_conv_transpose2d_rejects_at_vulkan_parameter_boundary(
    vulkan_backend, output_padding
):
    input, weight, bias = _conv_transpose_inputs(vulkan_backend)
    with pytest.raises(RuntimeError, match="transposed|fixed|support|Vulkan"):
        torch.nn.functional.conv_transpose2d(
            input, weight, bias, stride=2, padding=1, output_padding=output_padding
        )
