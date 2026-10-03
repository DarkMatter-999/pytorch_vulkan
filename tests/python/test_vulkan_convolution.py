import copy
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import pytorch_vulkan


def _trace_tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "strides": list(value.stride()),
                "storage_offset": value.storage_offset(), "device": str(value.device)}
    if isinstance(value, (tuple, list)):
        return [_trace_tensor_metadata(child) for child in value]
    return value


def _flatten_trace_metadata(value):
    if isinstance(value, dict) and "shape" in value:
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _flatten_trace_metadata(child)


def _synchronized_cpu(value):
    pytorch_vulkan._C.synchronize()
    return value.cpu()


class ConvolutionRouteTrace(TorchDispatchMode):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        self.events.append({"phase": self.phase, "operator": str(func),
                            "arguments": _trace_tensor_metadata(args),
                            "output": _trace_tensor_metadata(output)})
        return output


def _assert_grouped_ggi_route(trace, device):
    convolutions = [event for event in trace.events
                    if event["operator"] == "aten.convolution.default"]
    assert convolutions
    inputs = [metadata for event in convolutions
              for metadata in _flatten_trace_metadata(event["arguments"])]
    observed = {(tuple(item["shape"]), tuple(item["strides"]), item["storage_offset"])
                for item in inputs if item["device"] == str(device)}
    assert ((2, 1, 6, 7), (42, 168, 7, 1), 0) in observed
    assert ((2, 1, 6, 7), (42, 168, 7, 1), 84) in observed
    assert ((3, 1, 3, 5), (15, 90, 5, 1), 0) in observed
    assert ((3, 1, 3, 5), (15, 90, 5, 1), 45) in observed
    cats = [event for event in trace.events if event["operator"] == "aten.cat.default"]
    assert any(item["shape"] == [2, 6, 4, 2]
               for event in cats
               for item in _flatten_trace_metadata(event["output"]))


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


def test_conv2d_stock_module_create_graph_first_backward(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4611)
    module = torch.nn.Conv2d(4, 6, (3, 2), stride=(2, 1), padding=(1, 0),
                             dilation=(1, 2), groups=2, bias=True)
    x_cpu = torch.randn((1, 4, 6, 7), generator=generator, requires_grad=True)
    seed_cpu = torch.randn((1, 6, 3, 5), generator=generator, requires_grad=True)
    x_vk = x_cpu.detach().to(vulkan_backend).requires_grad_()
    module_vk = copy.deepcopy(module).to(vulkan_backend)
    module_vk.weight.requires_grad_(True)
    module_vk.bias.requires_grad_(True)
    seed_vk = seed_cpu.detach().to(vulkan_backend).requires_grad_()
    cpu_y = module(x_cpu)
    vk_y = module_vk(x_vk)
    cpu_grads = torch.autograd.grad(cpu_y, (x_cpu, module.weight, module.bias),
                                    seed_cpu, create_graph=True)
    assert [type(value.grad_fn).__name__ for value in cpu_grads] == [
        "ConvolutionBackwardBackward0", "ConvolutionBackwardBackward0",
        "ConvolutionBackwardBackward0",
    ]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_grads = torch.autograd.grad(vk_y, (x_vk, module_vk.weight, module_vk.bias),
                                   seed_vk, create_graph=True)
    assert all(g.requires_grad and g.grad_fn is not None for g in vk_grads)
    assert [type(value.grad_fn).__name__ for value in vk_grads] == [
        type(value.grad_fn).__name__ for value in cpu_grads
    ]
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches == 3
    assert copies == transfers == fallbacks == 0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_y.cpu(), cpu_y.detach(), rtol=3e-4, atol=3e-4)
    for actual, expected in zip(vk_grads, cpu_grads):
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


def test_convolution_overrideable_forward_generated_first_backward(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4615)
    x_cpu = torch.randn((1, 2, 4, 5), generator=generator, requires_grad=True)
    w_cpu = torch.randn((3, 2, 2, 3), generator=generator, requires_grad=True)
    b_cpu = torch.randn((3,), generator=generator, requires_grad=True)
    seed_cpu = torch.randn((1, 3, 3, 3), generator=generator, requires_grad=True)
    xv, wv, bv, sv = (value.detach().to(vulkan_backend).requires_grad_()
                      for value in (x_cpu, w_cpu, b_cpu, seed_cpu))
    vk_y = torch.ops.aten.convolution_overrideable.default(
        xv, wv, bv, [1, 1], [0, 0], [1, 1], False, [0], 1,
    )
    cpu_y = torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu)
    cpu_grads = torch.autograd.grad(cpu_y, (x_cpu, w_cpu, b_cpu), seed_cpu,
                                    create_graph=True)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_grads = torch.autograd.grad(vk_y, (xv, wv, bv), sv, create_graph=True)
    assert all(value.requires_grad and value.grad_fn is not None for value in vk_grads)
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches == 3
    assert copies == transfers == fallbacks == 0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_y.cpu(), cpu_y.detach(), rtol=3e-4, atol=3e-4)
    for actual, expected in zip(vk_grads, cpu_grads):
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("mask_bits", range(8), ids=[f"mask-{value:03b}" for value in range(8)])
def test_convolution_backward_overrideable_exact_schema_adapter(vulkan_backend, mask_bits):
    x = torch.randn((1, 2, 4, 5), generator=torch.Generator().manual_seed(4612))
    w = torch.randn((3, 2, 2, 3), generator=torch.Generator().manual_seed(4613))
    go = torch.randn((1, 3, 3, 3), generator=torch.Generator().manual_seed(4614))
    mask = [(mask_bits & (1 << (2 - index))) != 0 for index in range(3)]
    args = ([1, 1], [0, 0], [1, 1], False, [0], 1, mask)
    expected = torch.ops.aten.convolution_backward.default(
        go, x, w, None, [1, 1], [0, 0], [1, 1], False, [0], 1, mask,
    )
    vk_inputs = (go.to(vulkan_backend), x.to(vulkan_backend), w.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_timing = pytorch_vulkan._C.timing_breakdown()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    actual = torch.ops.aten.convolution_backward_overrideable.default(
        *vk_inputs, *args
    )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (sum(mask), 0, 0, 0)
    if not any(mask):
        assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_timing["buffer_creations"]
        assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    for value, reference in zip(actual, expected):
        assert (value is None) == (reference is None)
        if value is not None:
            assert tuple(value.shape) == tuple(reference.shape)
            pytorch_vulkan._C.synchronize()
            torch.testing.assert_close(value.cpu(), reference, rtol=3e-4, atol=3e-4)


def test_transposed_conv_first_backward_preserves_graph_and_matches_cpu(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4616)
    cpu_module = torch.nn.ConvTranspose2d(
        2, 3, 2, stride=(2, 2), output_padding=(1, 0), bias=True,
    )
    cpu_input = torch.randn((1, 2, 3, 3), generator=generator, requires_grad=True)
    cpu_seed = torch.randn((1, 3, 7, 6), generator=generator, requires_grad=True)
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.detach().to(vulkan_backend).requires_grad_()
    cpu_output = cpu_module(cpu_input)
    vk_output = vk_module(vk_input)
    cpu_grads = torch.autograd.grad(
        cpu_output, (cpu_input, cpu_module.weight, cpu_module.bias), cpu_seed,
        create_graph=True,
    )
    assert [type(value.grad_fn).__name__ for value in cpu_grads] == [
        "ConvolutionBackwardBackward0", "ConvolutionBackwardBackward0",
        "ConvolutionBackwardBackward0",
    ]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_grads = torch.autograd.grad(
        vk_output, (vk_input, vk_module.weight, vk_module.bias), vk_seed,
        create_graph=True,
    )
    assert all(value.requires_grad and value.grad_fn is not None for value in vk_grads)
    assert [type(value.grad_fn).__name__ for value in vk_grads] == [
        type(value.grad_fn).__name__ for value in cpu_grads
    ]
    assert pytorch_vulkan._C.execution_counter_snapshot() == (3, 0, 0, 0)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=3e-4, atol=3e-4)
    for actual, expected in zip(vk_grads, cpu_grads):
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("mutated_role", ["input", "weight"])
def test_generated_convolution_saved_variable_version_checks(vulkan_backend, mutated_role):
    generator = torch.Generator(device="cpu").manual_seed(4617)
    x_cpu = torch.randn((1, 2, 4, 5), generator=generator, requires_grad=True)
    w_cpu = torch.randn((3, 2, 2, 3), generator=generator, requires_grad=True)
    x_vk = x_cpu.detach().to(vulkan_backend).requires_grad_()
    w_vk = w_cpu.detach().to(vulkan_backend).requires_grad_()
    cpu_y = torch.nn.functional.conv2d(x_cpu, w_cpu)
    vk_y = torch.nn.functional.conv2d(x_vk, w_vk)
    cpu_target = x_cpu if mutated_role == "input" else w_cpu
    vk_target = x_vk if mutated_role == "input" else w_vk
    with torch.no_grad():
        cpu_target.add_(1)
        vk_target.add_(1)
    cpu_seed = torch.ones(cpu_y.shape)
    vk_seed = cpu_seed.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(cpu_y, (x_cpu, w_cpu), cpu_seed)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(vk_y, (x_vk, w_vk), vk_seed)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_grouped_generated_convolution_second_roles_and_selected_third(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4701)
    x = torch.randn((1, 4, 6, 7), generator=generator, requires_grad=True)
    w = torch.randn((6, 2, 3, 2), generator=generator, requires_grad=True)
    b = torch.randn((6,), generator=generator, requires_grad=True)
    g = torch.randn((1, 6, 3, 5), generator=generator, requires_grad=True)
    xv, wv, bv, gv = (t.detach().to(vulkan_backend).requires_grad_()
                      for t in (x, w, b, g))
    params = dict(stride=(2, 1), padding=(1, 0), dilation=(1, 2), groups=2)
    y = torch.nn.functional.conv2d(x, w, b, **params)
    yv = torch.nn.functional.conv2d(xv, wv, bv, **params)
    first = torch.autograd.grad(y, (x, w, b), g, create_graph=True, retain_graph=True)
    first_vk = torch.autograd.grad(yv, (xv, wv, bv), gv,
                                   create_graph=True, retain_graph=True)
    assert all(value.requires_grad and value.grad_fn is not None
               for value in first + first_vk)
    assert [type(value.grad_fn).__name__ for value in first_vk] == [
        type(value.grad_fn).__name__ for value in first
    ]
    print(f"grouped y/first shapes={tuple(y.shape)}/{[tuple(v.shape) for v in first]}; "
          f"history={[type(v.grad_fn).__name__ for v in first_vk]}")

    seeds = tuple(torch.randn(t.shape, generator=generator) for t in first)
    seeds_vk = tuple(seed.to(vulkan_backend) for seed in seeds)
    sx = torch.randn(x.shape, generator=generator)
    sx_vk = sx.to(vulkan_backend)
    contractions = tuple((q * seed).sum() for q, seed in zip(first, seeds))
    contractions_vk = tuple((q * seed).sum()
                            for q, seed in zip(first_vk, seeds_vk))

    cpu_trace = ConvolutionRouteTrace("second_reverse_ggI")
    vk_trace = ConvolutionRouteTrace("second_reverse_ggI")
    with cpu_trace:
        cpu_ggi = torch.autograd.grad(contractions[0], (w, g), allow_unused=True,
                                      retain_graph=True)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with vk_trace:
        vk_ggi = torch.autograd.grad(contractions_vk[0], (wv, gv),
                                     allow_unused=True, retain_graph=True)
    _assert_grouped_ggi_route(cpu_trace, "cpu")
    _assert_grouped_ggi_route(vk_trace, vulkan_backend)
    assert tuple(vk_ggi[0].shape) == (6, 2, 3, 2)
    for label, trace in (("cpu", cpu_trace), ("vulkan", vk_trace)):
        route_inputs = sorted({
            (tuple(item["shape"]), tuple(item["strides"]), item["storage_offset"])
            for event in trace.events
            if event["operator"] == "aten.convolution.default"
            for item in _flatten_trace_metadata(event["arguments"])
            if item["device"] == ("cpu" if label == "cpu" else str(vulkan_backend))
        })
        cat_outputs = [item["shape"] for event in trace.events
                       if event["operator"] == "aten.cat.default"
                       for item in _flatten_trace_metadata(event["output"])]
        print(f"grouped {label} ggI convolution inputs={route_inputs} cat outputs={cat_outputs}")

    cpu_ggw = torch.autograd.grad(contractions[1], (x, g), allow_unused=True,
                                  retain_graph=True)
    vk_ggw = torch.autograd.grad(contractions_vk[1], (xv, gv),
                                 allow_unused=True, retain_graph=True)
    cpu_ggb_trace = ConvolutionRouteTrace("second_reverse_ggb")
    vk_ggb_trace = ConvolutionRouteTrace("second_reverse_ggb")
    with cpu_ggb_trace:
        cpu_ggb = torch.autograd.grad(contractions[2], (x, w, g), allow_unused=True,
                                      retain_graph=True)
    with vk_ggb_trace:
        vk_ggb = torch.autograd.grad(contractions_vk[2], (xv, wv, gv),
                                     allow_unused=True, retain_graph=True)
    cpu_combined_trace = ConvolutionRouteTrace("second_combined")
    vk_combined_trace = ConvolutionRouteTrace("second_combined")
    with cpu_combined_trace:
        cpu_combined = torch.autograd.grad(first, (x, w, b, g), grad_outputs=seeds,
                                           allow_unused=True, retain_graph=True)
    with vk_combined_trace:
        vk_combined = torch.autograd.grad(first_vk, (xv, wv, bv, gv),
                                          grad_outputs=seeds_vk, allow_unused=True,
                                          retain_graph=True)
    for label, trace in (("cpu-ggb", cpu_ggb_trace), ("vulkan-ggb", vk_ggb_trace),
                         ("cpu-combined", cpu_combined_trace),
                         ("vulkan-combined", vk_combined_trace)):
        print(f"grouped observed {label} operators="
              f"{sorted({event['operator'] for event in trace.events})}")
    assert cpu_ggb[:2] == (None, None) and vk_ggb[:2] == (None, None)
    assert cpu_combined[2] is None and vk_combined[2] is None

    sw = seeds[1]
    cpu_mixed = torch.autograd.grad(contractions[1], x, create_graph=True,
                                    retain_graph=True)[0]
    vk_mixed = torch.autograd.grad(contractions_vk[1], xv, create_graph=True,
                                   retain_graph=True)[0]
    cpu_third = torch.autograd.grad((cpu_mixed * sx).sum(), g)[0]
    vk_third = torch.autograd.grad((vk_mixed * sx_vk).sum(), gv)[0]
    oracle = torch.nn.functional.conv2d(sx, sw, None, **params)
    assert torch.count_nonzero(cpu_third).item() > 0
    torch.testing.assert_close(cpu_third, oracle, rtol=0, atol=0)

    pytorch_vulkan._C.synchronize()
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches > 0
    assert transfers == fallbacks == 0
    print(f"grouped second/selected-third counters={(dispatches, copies, transfers, fallbacks)}")
    for actual, expected in zip(first_vk, first):
        torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                   rtol=7e-4, atol=7e-4)
    for actual, expected in zip(vk_ggi, cpu_ggi):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.count_nonzero(expected).item() > 0
            torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                       rtol=8e-4, atol=8e-4)
    for actual_group, expected_group in ((vk_ggw, cpu_ggw), (vk_ggb, cpu_ggb),
                                         (vk_combined, cpu_combined)):
        for actual, expected in zip(actual_group, expected_group):
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert torch.count_nonzero(expected).item() > 0
                torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                           rtol=8e-4, atol=8e-4)
    vk_third_cpu = _synchronized_cpu(vk_third)
    torch.testing.assert_close(vk_third_cpu, oracle, rtol=8e-4, atol=8e-4)
    assert torch.count_nonzero(vk_third_cpu).item() > 0


def test_depthwise_generated_convolution_selected_second_and_third(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4702)
    x = torch.randn((1, 2, 5, 6), generator=generator, requires_grad=True)
    w = torch.randn((4, 1, 2, 3), generator=generator, requires_grad=True)
    g = torch.randn((1, 4, 4, 3), generator=generator, requires_grad=True)
    xv, wv, gv = (value.detach().to(vulkan_backend).requires_grad_()
                  for value in (x, w, g))
    params = dict(stride=(1, 2), padding=(0, 1), dilation=(1, 1), groups=2)
    y = torch.nn.functional.conv2d(x, w, None, **params)
    yv = torch.nn.functional.conv2d(xv, wv, None, **params)
    first = torch.autograd.grad(y, (x, w), g, create_graph=True, retain_graph=True)
    first_vk = torch.autograd.grad(yv, (xv, wv), gv,
                                   create_graph=True, retain_graph=True)
    assert [tuple(value.shape) for value in first_vk] == [
        tuple(value.shape) for value in first
    ] == [(1, 2, 5, 6), (4, 1, 2, 3)]
    assert all(value.requires_grad and value.grad_fn is not None
               for value in first + first_vk)
    assert [type(value.grad_fn).__name__ for value in first_vk] == [
        type(value.grad_fn).__name__ for value in first
    ]
    print(f"depthwise y/first shapes={tuple(y.shape)}/{[tuple(v.shape) for v in first]}; "
          f"history={[type(v.grad_fn).__name__ for v in first_vk]}")
    seeds = tuple(torch.randn(value.shape, generator=generator) for value in first)
    seeds_vk = tuple(value.to(vulkan_backend) for value in seeds)
    sx = torch.randn(x.shape, generator=generator)
    sx_vk = sx.to(vulkan_backend)
    gx_contraction = (first[0] * seeds[0]).sum()
    gw_contraction = (first[1] * seeds[1]).sum()
    gx_contraction_vk = (first_vk[0] * seeds_vk[0]).sum()
    gw_contraction_vk = (first_vk[1] * seeds_vk[1]).sum()
    cpu_xw = torch.autograd.grad(gx_contraction, w, retain_graph=True)[0]
    cpu_xg = torch.autograd.grad(gx_contraction, g, retain_graph=True)[0]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_xw = torch.autograd.grad(gx_contraction_vk, wv, retain_graph=True)[0]
    vk_xg = torch.autograd.grad(gx_contraction_vk, gv, retain_graph=True)[0]
    cpu_wx = torch.autograd.grad(gw_contraction, x, create_graph=True,
                                 retain_graph=True)[0]
    vk_wx = torch.autograd.grad(gw_contraction_vk, xv, create_graph=True,
                                retain_graph=True)[0]
    cpu_wg = torch.autograd.grad(gw_contraction, g, retain_graph=True)[0]
    vk_wg = torch.autograd.grad(gw_contraction_vk, gv, retain_graph=True)[0]
    cpu_third = torch.autograd.grad((cpu_wx * sx).sum(), g)[0]
    vk_third = torch.autograd.grad((vk_wx * sx_vk).sum(), gv)[0]
    oracle = torch.nn.functional.conv2d(sx, seeds[1], None, **params)
    assert all(torch.count_nonzero(value).item() > 0
               for value in (cpu_xw, cpu_xg, cpu_wg, cpu_third))
    torch.testing.assert_close(cpu_third, oracle, rtol=0, atol=0)

    pytorch_vulkan._C.synchronize()
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches > 0 and transfers == fallbacks == 0
    print(f"depthwise second/selected-third counters={(dispatches, copies, transfers, fallbacks)}")
    for actual, expected in zip(first_vk, first):
        torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                   rtol=7e-4, atol=7e-4)
    for actual, expected in ((vk_xw, cpu_xw), (vk_xg, cpu_xg), (vk_wx, cpu_wx),
                             (vk_wg, cpu_wg), (vk_third, cpu_third)):
        torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                   rtol=8e-4, atol=8e-4)
    vk_third_cpu = _synchronized_cpu(vk_third)
    assert torch.count_nonzero(vk_third_cpu).item() > 0
    torch.testing.assert_close(vk_third_cpu, oracle, rtol=8e-4, atol=8e-4)


def test_transposed_generated_convolution_second_order_boundary(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(4703)
    x = torch.randn((1, 2, 3, 3), generator=generator, requires_grad=True)
    w = torch.randn((2, 3, 2, 2), generator=generator, requires_grad=True)
    b = torch.randn((3,), generator=generator, requires_grad=True)
    g = torch.randn((1, 3, 7, 6), generator=generator, requires_grad=True)
    xv, wv, bv, gv = (value.detach().to(vulkan_backend).requires_grad_()
                      for value in (x, w, b, g))
    params = dict(stride=(2, 2), output_padding=(1, 0))
    y = torch.nn.functional.conv_transpose2d(x, w, b, **params)
    yv = torch.nn.functional.conv_transpose2d(xv, wv, bv, **params)
    first = torch.autograd.grad(y, (x, w, b), g, create_graph=True, retain_graph=True)
    first_vk = torch.autograd.grad(yv, (xv, wv, bv), gv,
                                   create_graph=True, retain_graph=True)
    assert [tuple(value.shape) for value in first_vk] == [
        tuple(value.shape) for value in first
    ] == [(1, 2, 3, 3), (2, 3, 2, 2), (3,)]
    assert all(value.requires_grad and value.grad_fn is not None
               for value in first + first_vk)
    assert [type(value.grad_fn).__name__ for value in first_vk] == [
        type(value.grad_fn).__name__ for value in first
    ]
    print(f"transposed y/first shapes={tuple(y.shape)}/{[tuple(v.shape) for v in first]}; "
          f"history={[type(v.grad_fn).__name__ for v in first_vk]}")
    seeds = tuple(torch.randn(value.shape, generator=generator) for value in first)
    seeds_vk = tuple(value.to(vulkan_backend) for value in seeds)
    sx = torch.randn(x.shape, generator=generator)
    sw = torch.randn(w.shape, generator=generator)
    sx_vk, sw_vk = sx.to(vulkan_backend), sw.to(vulkan_backend)

    cpu_ggi = torch.autograd.grad((first[0] * seeds[0]).sum(), (w, g),
                                  allow_unused=True, retain_graph=True)
    cpu_ggw = torch.autograd.grad((first[1] * seeds[1]).sum(), (x, g),
                                  allow_unused=True, retain_graph=True)
    cpu_ggb = torch.autograd.grad((first[2] * seeds[2]).sum(), (x, w, g),
                                  allow_unused=True, retain_graph=True)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_ggi = torch.autograd.grad((first_vk[0] * seeds_vk[0]).sum(), (wv, gv),
                                 allow_unused=True, retain_graph=True)
    vk_ggw = torch.autograd.grad((first_vk[1] * seeds_vk[1]).sum(), (xv, gv),
                                 allow_unused=True, retain_graph=True)
    vk_ggb = torch.autograd.grad((first_vk[2] * seeds_vk[2]).sum(),
                                 (xv, wv, gv), allow_unused=True, retain_graph=True)
    cpu_combined = torch.autograd.grad(
        first, (x, w, b, g), grad_outputs=seeds, allow_unused=True,
        retain_graph=True,
    )
    vk_combined = torch.autograd.grad(
        first_vk, (xv, wv, bv, gv), grad_outputs=seeds_vk,
        allow_unused=True, retain_graph=True,
    )
    for cpu_values, vk_values in ((cpu_ggi, vk_ggi), (cpu_ggw, vk_ggw),
                                  (cpu_ggb, vk_ggb),
                                  (cpu_combined, vk_combined)):
        for cpu_value, vk_value in zip(cpu_values, vk_values):
            assert (cpu_value is None) == (vk_value is None)
            if cpu_value is not None:
                assert torch.count_nonzero(cpu_value).item() > 0
    assert cpu_ggb[:2] == (None, None) and vk_ggb[:2] == (None, None)
    assert cpu_combined[2] is None and vk_combined[2] is None

    cpu_mixed = torch.autograd.grad((first[1] * sw).sum(), x,
                                    create_graph=True, retain_graph=True)[0]
    vk_mixed = torch.autograd.grad((first_vk[1] * sw_vk).sum(), xv,
                                   create_graph=True, retain_graph=True)[0]
    pytorch_vulkan._C.synchronize()
    successful_dispatches, successful_copies, successful_transfers, successful_fallbacks = \
        pytorch_vulkan._C.execution_counter_snapshot()
    assert successful_dispatches > 0
    assert successful_transfers == successful_fallbacks == 0
    print(f"transposed positive-second counters={(successful_dispatches, successful_copies, successful_transfers, successful_fallbacks)}")
    with pytest.raises(RuntimeError, match="output_padding is not supported for non-transposed convolutions") as cpu_boundary:
        torch.autograd.grad((cpu_mixed * sx).sum(), g)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_timing = pytorch_vulkan._C.timing_breakdown()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    with pytest.raises(RuntimeError, match="output_padding is not supported for non-transposed convolutions") as vk_boundary:
        torch.autograd.grad((vk_mixed * sx_vk).sum(), gv)
    pytorch_vulkan._C.synchronize()
    graph_boundary_counters = pytorch_vulkan._C.execution_counter_snapshot()
    print(f"transposed CPU boundary={type(cpu_boundary.value).__name__}: {cpu_boundary.value}")
    print(f"transposed Vulkan graph boundary={type(vk_boundary.value).__name__}: {vk_boundary.value}; counters={graph_boundary_counters}")
    assert graph_boundary_counters[2:] == (0, 0)

    # The native graph may perform valid composite work before it reaches the
    # offending ordinary backward. Qualify that leaf's preflight independently
    # through its public ATen schema with the retained nonzero metadata.
    pytorch_vulkan._C.reset_execution_counters()
    before_timing = pytorch_vulkan._C.timing_breakdown()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    with pytest.raises(RuntimeError, match="output_padding is not supported for non-transposed convolutions"):
        torch.ops.aten.convolution_backward.default(
            gv, xv, wv, None, [1, 1], [0, 0], [1, 1], False, [1, 0], 1,
            [True, True, False],
        )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    print("transposed Vulkan direct ordinary-backward boundary counters=(0, 0, 0, 0)")
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_timing["buffer_creations"]
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live

    for actual, expected in zip(first_vk, first):
        torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                   rtol=5e-4, atol=5e-4)
    for cpu_values, vk_values in ((cpu_ggi, vk_ggi), (cpu_ggw, vk_ggw),
                                  (cpu_ggb, vk_ggb),
                                  (cpu_combined, vk_combined)):
        for cpu_value, vk_value in zip(cpu_values, vk_values):
            if vk_value is not None:
                torch.testing.assert_close(_synchronized_cpu(vk_value), cpu_value,
                                           rtol=8e-4, atol=8e-4)


@pytest.mark.parametrize("bias,trainable", [
    (True, (False, False, False)), (True, (False, False, True)),
    (True, (False, True, False)), (True, (False, True, True)),
    (True, (True, False, False)), (True, (True, False, True)),
    (True, (True, True, False)), (True, (True, True, True)),
    (False, (False, False, False)), (False, (True, True, False)),
], ids=["mask-000", "mask-001", "mask-010", "mask-011", "mask-100",
       "mask-101", "mask-110", "mask-111", "biasfree-frozen", "biasfree-trainable"])
def test_stock_conv2d_trainability_mask_matrix_matches_generated_autograd(
    vulkan_backend, bias, trainable,
):
    generator = torch.Generator(device="cpu").manual_seed(4710 + sum(trainable) + int(bias))
    cpu_module = torch.nn.Conv2d(2, 3, (2, 3), bias=bias)
    cpu_module.weight.requires_grad_(trainable[1])
    if bias:
        cpu_module.bias.requires_grad_(trainable[2])
    cpu_input = torch.randn((1, 2, 4, 5), generator=generator)
    cpu_input.requires_grad_(trainable[0])
    cpu_module_vk = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(trainable[0])
    cpu_output = cpu_module(cpu_input)
    vk_output = cpu_module_vk(vk_input)
    cpu_seed = torch.randn(cpu_output.shape, generator=generator, requires_grad=True)
    vk_seed = cpu_seed.detach().to(vulkan_backend).requires_grad_()
    assert tuple(vk_output.shape) == tuple(cpu_output.shape)
    vk_targets_needed = (trainable[0], trainable[1], bias and trainable[2])
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    if any(vk_targets_needed):
        cpu_output.backward(cpu_seed, create_graph=True, retain_graph=True)
        vk_output.backward(vk_seed, create_graph=True, retain_graph=True)
        pytorch_vulkan._C.synchronize()
        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert counters[0] == sum(vk_targets_needed)
        assert counters[2:] == (0, 0)
    else:
        assert not cpu_output.requires_grad and not vk_output.requires_grad
        assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
        counters = pytorch_vulkan._C.execution_counter_snapshot()
    print(f"stock Conv2d bias={bias} trainable={trainable} counters={counters}")
    pairs = ((vk_input.grad, cpu_input.grad, trainable[0]),
             (cpu_module_vk.weight.grad, cpu_module.weight.grad, trainable[1]))
    if bias:
        pairs += ((cpu_module_vk.bias.grad, cpu_module.bias.grad, trainable[2]),)
    for actual, expected, selected in pairs:
        assert (actual is not None) is selected
        assert (expected is not None) is selected
        if selected:
            assert actual.requires_grad == expected.requires_grad
            assert (actual.grad_fn is not None) == (expected.grad_fn is not None)
            torch.testing.assert_close(_synchronized_cpu(actual), expected,
                                       rtol=4e-4, atol=4e-4)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(_synchronized_cpu(vk_output), cpu_output.detach(),
                               rtol=3e-4, atol=3e-4)
    cpu_input.grad = None
    vk_input.grad = None
    cpu_module.weight.grad = None
    cpu_module_vk.weight.grad = None
    if bias:
        cpu_module.bias.grad = None
        cpu_module_vk.bias.grad = None


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


def _transposed_witness_cpu():
    generator = torch.Generator(device="cpu").manual_seed(4821)
    x = torch.randn((2, 4, 3, 4), generator=generator)
    weight = torch.randn((4, 3, 2, 3), generator=generator)
    output = torch.nn.functional.conv_transpose2d(
        x, weight, None, stride=(2, 1), padding=(1, 1),
        output_padding=(2, 0), groups=2, dilation=(3, 2),
    )
    grad_output = torch.randn(output.shape, generator=generator)
    return x, weight, grad_output


def test_convolution_transposed_bias_free_forward_matches_cpu(vulkan_backend):
    x, weight, _ = _transposed_witness_cpu()
    vk_x, vk_weight = x.to(vulkan_backend), weight.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.convolution.default(
        vk_x, vk_weight, None,
        [2, 1], [1, 1], [3, 2], True, [2, 0], 2,
    )
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, None, stride=(2, 1), padding=(1, 1),
        output_padding=(2, 0), groups=2, dilation=(3, 2),
    )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    assert tuple(actual.shape) == (2, 6, 8, 6)
    torch.testing.assert_close(actual.cpu(), expected)


def test_transposed_op1_bias_is_guarded_and_indexes_target_ic():
    import re

    source = (Path(__file__).resolve().parents[2] /
              "src/vulkan/shaders/glsl/convolution.comp").read_text()
    op1 = source.split("} else if (params.operation == 1u) {", 1)[1].split(
        "} else if (params.operation == 2u) {", 1
    )[0]
    assert "if (params.has_bias != 0u)" in op1
    assert "bias_values[bm.storage_offset + ic * bm.strides[0]]" in op1
    reads = re.findall(r"\bbias_values\s*\[(?!\])", source)
    assert len(reads) == 2


def test_convolution_transposed_bias_present_forward_matches_cpu(vulkan_backend):
    x, weight, _ = _transposed_witness_cpu()
    generator = torch.Generator(device="cpu").manual_seed(4830)
    bias_base = torch.randn((12,), generator=generator)
    bias = bias_base[1::2]
    vk_x, vk_weight, vk_bias_base = (
        value.to(vulkan_backend) for value in (x, weight, bias_base)
    )
    vk_bias = vk_bias_base[1::2]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.convolution.default(
        vk_x, vk_weight, vk_bias,
        [2, 1], [1, 1], [3, 2], True, [2, 0], 2,
    )
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, bias, stride=(2, 1), padding=(1, 1),
        output_padding=(2, 0), groups=2, dilation=(3, 2),
    )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    assert tuple(actual.shape) == (2, 6, 8, 6)
    torch.testing.assert_close(actual.cpu(), expected)


def test_convolution_transposed_forward_does_not_bound_unused_op1_stride_product(
    vulkan_backend,
):
    int_max = 2**31 - 1
    cpu_x = torch.ones((1, 1, 2, 1), dtype=torch.float32)
    cpu_weight = torch.ones((1, 1, 2, 1), dtype=torch.float32)
    vk_x, vk_weight = cpu_x.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    expected = torch.nn.functional.conv_transpose2d(
        cpu_x, cpu_weight, None, stride=(int_max, 1),
        padding=(1073741823, 0), dilation=(1, 1), output_padding=(0, 0),
    )
    actual = torch.ops.aten.convolution.default(
        vk_x, vk_weight, None, [int_max, 1], [1073741823, 0], [1, 1],
        True, [0, 0], 1,
    )
    pytorch_vulkan._C.synchronize()
    assert tuple(actual.shape) == (1, 1, 3, 1)
    torch.testing.assert_close(actual.cpu(), expected)


def test_convolution_transposed_backward_result_slots_match_cpu(vulkan_backend):
    x, weight, grad_output = _transposed_witness_cpu()
    vk_x, vk_weight = x.to(vulkan_backend), weight.to(vulkan_backend)
    vk_grad = grad_output.to(vulkan_backend)
    with torch.backends.mkldnn.flags(enabled=False):
        expected_results = torch.ops.aten.convolution_backward.default(
            grad_output, x, weight, None, [2, 1], [1, 1], [3, 2], True,
            [2, 0], 2, [True, True, True],
        )
    for mask_bits in range(8):
        mask = [bool(mask_bits & (1 << (2 - slot))) for slot in range(3)]
        with torch.backends.mkldnn.flags(enabled=False):
            expected = torch.ops.aten.convolution_backward.default(
                grad_output, x, weight, None, [2, 1], [1, 1], [3, 2], True,
                [2, 0], 2, mask,
            )
        pytorch_vulkan._C.synchronize()
        actual = torch.ops.aten.convolution_backward.default(
            vk_grad, vk_x, vk_weight, None, [2, 1], [1, 1], [3, 2], True,
            [2, 0], 2, mask,
        )
        pytorch_vulkan._C.synchronize()
        for slot, (value, reference) in enumerate(zip(actual, expected)):
            assert (value is not None) is (reference is not None), (mask, slot)
            if value is not None:
                assert tuple(value.shape) == tuple(reference.shape)
                torch.testing.assert_close(value.cpu(), reference)
        if mask_bits in (4, 5, 6, 7):
            assert tuple(actual[0].shape) == tuple(x.shape)
            torch.testing.assert_close(actual[0].cpu(), expected_results[0])
    ordinary_input_gradient = torch.nn.functional.conv2d(
        grad_output, weight, None, (2, 1), (1, 1), (3, 2), 2,
    )[..., :3, :4]
    torch.testing.assert_close(expected_results[0], ordinary_input_gradient)


def test_convolution_transposed_backward_preflight_obeys_selected_shapes(vulkan_backend):
    x, weight, _ = _transposed_witness_cpu()
    mismatched_grad = torch.randn((2, 6, 7, 6), generator=torch.Generator().manual_seed(81))
    vk_x, vk_weight = x.to(vulkan_backend), weight.to(vulkan_backend)
    vk_grad = mismatched_grad.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    d_bias = torch.ops.aten.convolution_backward.default(
        vk_grad, vk_x, vk_weight, None, [2, 1], [1, 1], [3, 2], True,
        [2, 0], 2, [False, False, True],
    )
    pytorch_vulkan._C.synchronize()
    assert d_bias[0] is None and d_bias[1] is None
    assert tuple(d_bias[2].shape) == (6,)
    torch.testing.assert_close(d_bias[2].cpu(), mismatched_grad.sum((0, 2, 3)))

    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="grad_output shape expected"):
        torch.ops.aten.convolution_backward.default(
            vk_grad, vk_x, vk_weight, None, [2, 1], [1, 1], [3, 2], True,
            [2, 0], 2, [True, False, False],
        )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations

    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    no_results = torch.ops.aten.convolution_backward.default(
        vk_grad, vk_x, vk_weight, None, [2, 1], [1, 1], [3, 2], True,
        [2, 0], 2, [False, False, False],
    )
    pytorch_vulkan._C.synchronize()
    assert no_results == (None, None, None)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


@pytest.mark.parametrize("bias_present", [False, True], ids=["no-bias", "bias"])
def test_conv_transpose2d_function_and_module_forward_match_cpu(
    vulkan_backend, bias_present
):
    generator = torch.Generator(device="cpu").manual_seed(7319)
    x_cpu = torch.randn((2, 4, 3, 4), generator=generator)
    weight_cpu = torch.randn((4, 3, 2, 3), generator=generator)
    bias_cpu = torch.randn((6,), generator=generator) if bias_present else None
    x_vk, weight_vk = x_cpu.to(vulkan_backend), weight_cpu.to(vulkan_backend)
    bias_vk = bias_cpu.to(vulkan_backend) if bias_cpu is not None else None
    args = dict(stride=(2, 1), padding=(1, 1), output_padding=(2, 0),
                groups=2, dilation=(3, 2))

    expected = torch.nn.functional.conv_transpose2d(
        x_cpu, weight_cpu, bias_cpu, **args
    )
    actual = torch.nn.functional.conv_transpose2d(
        x_vk, weight_vk, bias_vk, **args
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)

    cpu_module = torch.nn.ConvTranspose2d(
        4, 6, (2, 3), stride=(2, 1), padding=(1, 1),
        output_padding=(2, 0), groups=2, dilation=(3, 2),
        bias=bias_present,
    )
    with torch.no_grad():
        cpu_module.weight.copy_(weight_cpu)
        if bias_present:
            cpu_module.bias.copy_(bias_cpu)
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    module_actual = vk_module(x_vk)
    module_expected = cpu_module(x_cpu)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(module_actual.cpu(), module_expected,
                               rtol=2e-4, atol=2e-4)


def test_conv_transpose2d_module_output_size_uses_stock_stride_policy(vulkan_backend):
    module = torch.nn.ConvTranspose2d(
        4, 6, (2, 3), stride=(2, 1), padding=(1, 1), groups=2,
        dilation=(3, 2),
    )
    cpu_input = torch.randn((2, 4, 3, 4), generator=torch.Generator().manual_seed(812))
    vk_module = copy.deepcopy(module).to(vulkan_backend)
    vk_input = cpu_input.to(vulkan_backend)
    with torch.backends.mkldnn.flags(enabled=False):
        expected = torch.nn.functional.conv_transpose2d(
            cpu_input, module.weight, module.bias, stride=(2, 1),
            padding=(1, 1), output_padding=(1, 0), groups=2, dilation=(3, 2),
        )
        module_expected = module(cpu_input, output_size=(2, 6, 7, 6))
    actual = vk_module(vk_input, output_size=(2, 6, 7, 6))
    pytorch_vulkan._C.synchronize()
    assert tuple(actual.shape) == (2, 6, 7, 6)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual.cpu(), module_expected, rtol=2e-4, atol=2e-4)
    with pytest.raises(ValueError, match="output size.*valid range|requested an output size"):
        vk_module(vk_input, output_size=(2, 6, 8, 6))


@pytest.mark.parametrize("trainable", [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (True, False, True), (True, True, True),
    (False, False, False),
], ids=["input", "weight", "bias", "input-weight", "input-bias", "all", "frozen"])
def test_conv_transpose2d_module_first_backward_matches_cpu_with_mixed_trainability(
    vulkan_backend, trainable
):
    generator = torch.Generator(device="cpu").manual_seed(941)
    cpu_module = torch.nn.ConvTranspose2d(
        4, 6, (2, 3), stride=(2, 1), padding=(1, 1),
        output_padding=(2, 0), groups=2, dilation=(3, 2), bias=True,
    )
    cpu_module.weight.requires_grad_(trainable[1])
    cpu_module.bias.requires_grad_(trainable[2])
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    x_cpu = torch.randn((2, 4, 3, 4), generator=generator)
    x_cpu.requires_grad_(trainable[0])
    # Upload first, then make the Vulkan input a new leaf.
    x_vk = x_cpu.detach().to(vulkan_backend).requires_grad_(trainable[0])
    cpu_output, vk_output = cpu_module(x_cpu), vk_module(x_vk)
    seed = torch.randn(cpu_output.shape, generator=generator)
    vk_seed = seed.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    if any(trainable):
        with torch.backends.mkldnn.flags(enabled=False):
            cpu_output.backward(seed)
        vk_output.backward(vk_seed)
    else:
        assert not cpu_output.requires_grad and not vk_output.requires_grad
        assert cpu_output.grad_fn is None and vk_output.grad_fn is None
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (sum(trainable), 0, 0, 0)
    for actual, expected, selected in (
        (x_vk.grad, x_cpu.grad, trainable[0]),
        (vk_module.weight.grad, cpu_module.weight.grad, trainable[1]),
        (vk_module.bias.grad, cpu_module.bias.grad, trainable[2]),
    ):
        assert (actual is not None) is selected
        assert (expected is not None) is selected
        if selected:
            torch.testing.assert_close(actual.cpu(), expected,
                                       rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("device", ["cpu", "vk:0"], ids=["cpu", "vulkan"])
@pytest.mark.parametrize("saved_role", ["input", "weight"])
def test_conv_transpose2d_module_saved_variable_mutation_checks_version(
    device, saved_role, vulkan_backend
):
    module = torch.nn.ConvTranspose2d(4, 6, (2, 3), groups=2).to(device)
    input = torch.randn((1, 4, 3, 4), generator=torch.Generator().manual_seed(451))
    input = input.to(device).requires_grad_(True)
    output = module(input)
    saved = input if saved_role == "input" else module.weight
    with torch.no_grad():
        saved.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        output.sum().backward()


@pytest.mark.parametrize("operand", ["offset-input", "transpose-input", "transpose-weight"])
def test_conv_transpose2d_reads_vulkan_views_against_cpu(operand, vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(1207)
    x_base = torch.randn((2, 4, 4, 5), generator=generator)
    weight_base = torch.randn((4, 3, 2, 3), generator=generator)
    vk_x_base, vk_weight_base = x_base.to(vulkan_backend), weight_base.to(vulkan_backend)
    if operand == "offset-input":
        cpu_x, vk_x = x_base[:, :, 1:, 1:], vk_x_base[:, :, 1:, 1:]
        cpu_weight, vk_weight = weight_base, vk_weight_base
    elif operand == "transpose-input":
        cpu_x, vk_x = x_base.transpose(2, 3), vk_x_base.transpose(2, 3)
        cpu_weight, vk_weight = weight_base, vk_weight_base
    else:
        cpu_x, vk_x = x_base[:, :, :3, :4], vk_x_base[:, :, :3, :4]
        cpu_weight, vk_weight = weight_base.transpose(2, 3), vk_weight_base.transpose(2, 3)
    args = dict(stride=(2, 1), padding=(1, 1), output_padding=(1, 0),
                groups=2, dilation=(2, 1))
    expected = torch.nn.functional.conv_transpose2d(cpu_x, cpu_weight, None, **args)
    actual = torch.nn.functional.conv_transpose2d(vk_x, vk_weight, None, **args)
    assert tuple(vk_x.stride()) == tuple(cpu_x.stride())
    assert tuple(vk_weight.stride()) == tuple(cpu_weight.stride())
    assert vk_x.storage_offset() == cpu_x.storage_offset()
    assert vk_weight.storage_offset() == cpu_weight.storage_offset()
    if operand == "offset-input":
        assert vk_x.storage_offset() > 0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


def test_conv_transpose2d_module_accepts_expanded_upstream_seed(vulkan_backend):
    generator = torch.Generator(device="cpu").manual_seed(1207)
    cpu_module = torch.nn.ConvTranspose2d(4, 6, (2, 3), groups=2)
    cpu_input = torch.randn((2, 4, 3, 4), generator=generator, requires_grad=True)
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(True)
    cpu_output, vk_output = cpu_module(cpu_input), vk_module(vk_input)
    cpu_seed_base = torch.randn((1, 6, 1, 1), generator=generator)
    vk_seed_base = cpu_seed_base.to(vulkan_backend)
    cpu_seed, vk_seed = cpu_seed_base.expand_as(cpu_output), vk_seed_base.expand(vk_output.shape)
    assert tuple(cpu_seed.stride()) == tuple(vk_seed.stride())
    assert tuple(vk_seed.stride()[-2:]) == (0, 0)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with torch.backends.mkldnn.flags(enabled=False):
        cpu_output.backward(cpu_seed)
    vk_output.backward(vk_seed)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (3, 0, 0, 0)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(vk_module.weight.grad.cpu(), cpu_module.weight.grad,
                               rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(vk_module.bias.grad.cpu(), cpu_module.bias.grad,
                               rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize(
    ("bias_present", "trainable"),
    [
        (False, (1, 0, 0)),
        (False, (0, 1, 0)),
        (False, (0, 0, 0)),
        (True, (1, 0, 1)),
        (True, (0, 1, 1)),
        (True, (0, 0, 1)),
        (True, (1, 1, 1)),
    ],
    ids=["input-only", "weight-only", "frozen", "input-bias", "weight-bias",
         "bias-only", "all-trainable"],
)
def test_conv_transpose2d_module_depthwise_multiplier_first_backward_matches_cpu(
    vulkan_backend,
    bias_present,
    trainable,
):
    generator = torch.Generator(device="cpu").manual_seed(1209)
    cpu_module = torch.nn.ConvTranspose2d(
        4, 8, (2, 3), stride=(2, 1), padding=(1, 1), groups=4,
        dilation=(2, 1), output_padding=(1, 0), bias=bias_present,
    )
    cpu_module.weight.requires_grad_(bool(trainable[1]))
    if bias_present:
        cpu_module.bias.requires_grad_(bool(trainable[2]))
    cpu_input = torch.randn((2, 4, 3, 4), generator=generator)
    cpu_input.requires_grad_(bool(trainable[0]))
    vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(bool(trainable[0]))
    assert cpu_module.groups == cpu_module.in_channels == cpu_module.weight.size(0) == 4
    assert cpu_module.weight.size(1) == 2
    assert cpu_module.out_channels == cpu_module.weight.size(1) * cpu_module.groups == 8
    cpu_output, vk_output = cpu_module(cpu_input), vk_module(vk_input)
    seed = torch.randn(cpu_output.shape, generator=generator)
    vk_seed = seed.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_output.cpu(), cpu_output, rtol=3e-4, atol=3e-4)
    pytorch_vulkan._C.reset_execution_counters()
    if any(trainable):
        with torch.backends.mkldnn.flags(enabled=False):
            cpu_output.backward(seed)
        vk_output.backward(vk_seed)
    else:
        assert not cpu_output.requires_grad and not vk_output.requires_grad
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (sum(trainable), 0, 0, 0)
    for actual, expected, selected in (
        (vk_input.grad, cpu_input.grad, trainable[0]),
        (vk_module.weight.grad, cpu_module.weight.grad, trainable[1]),
        (vk_module.bias.grad if bias_present else None,
         cpu_module.bias.grad if bias_present else None,
         bias_present and trainable[2]),
    ):
        selected = bool(selected)
        assert (actual is not None) is selected
        assert (expected is not None) is selected
        if selected:
            torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("bias_present", [False, True], ids=["no-bias", "bias"])
def test_conv_transpose2d_function_depthwise_multiplier_matches_cpu(
    vulkan_backend, bias_present
):
    generator = torch.Generator(device="cpu").manual_seed(1211)
    cpu_input = torch.randn((1, 4, 2, 3), generator=generator)
    cpu_weight = torch.randn((4, 2, 2, 3), generator=generator)
    cpu_bias = torch.randn((8,), generator=generator) if bias_present else None
    vk_input, vk_weight = cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    vk_bias = cpu_bias.to(vulkan_backend) if cpu_bias is not None else None
    kwargs = dict(stride=(2, 1), padding=(1, 1), output_padding=(1, 0),
                  groups=4, dilation=(2, 1))
    assert kwargs["groups"] == cpu_input.size(1) == cpu_weight.size(0) == 4
    assert cpu_weight.size(1) == 2
    assert cpu_weight.size(1) * kwargs["groups"] == 8
    expected = torch.nn.functional.conv_transpose2d(
        cpu_input, cpu_weight, cpu_bias, **kwargs
    )
    actual = torch.nn.functional.conv_transpose2d(vk_input, vk_weight, vk_bias, **kwargs)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("bias_sizes", [None, [], [999]], ids=["absent", "empty", "wrong-hint"])
def test_transposed_backward_bias_sizes_are_advisory(vulkan_backend, bias_sizes):
    generator = torch.Generator(device="cpu").manual_seed(1213)
    x = torch.randn((1, 4, 2, 3), generator=generator)
    weight = torch.randn((4, 2, 2, 3), generator=generator)
    output = torch.nn.functional.conv_transpose2d(
        x, weight, None, stride=(2, 1), padding=(1, 1),
        output_padding=(1, 0), groups=4, dilation=(2, 1),
    )
    grad = torch.randn(output.shape, generator=generator)
    vk_x, vk_weight, vk_grad = (value.to(vulkan_backend) for value in (x, weight, grad))
    actual = torch.ops.aten.convolution_backward.default(
        vk_grad, vk_x, vk_weight, bias_sizes, [2, 1], [1, 1], [2, 1],
        True, [1, 0], 4, [False, False, True],
    )
    pytorch_vulkan._C.synchronize()
    assert actual[0] is None and actual[1] is None
    assert tuple(actual[2].shape) == (8,)
    torch.testing.assert_close(actual[2].cpu(), grad.sum((0, 2, 3)))


def test_conv_transpose2d_accepts_canonical_dual_contiguous_singletons(vulkan_backend):
    cpu_input = torch.randn((1, 1, 1, 1), generator=torch.Generator().manual_seed(311))
    cpu_weight = torch.randn((1, 1, 1, 1), generator=torch.Generator().manual_seed(312))
    vk_input, vk_weight = cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend)
    assert cpu_input.is_contiguous()
    assert cpu_input.is_contiguous(memory_format=torch.channels_last)
    assert vk_input.is_contiguous()
    assert vk_input.is_contiguous(memory_format=torch.channels_last)
    expected = torch.nn.functional.conv_transpose2d(cpu_input, cpu_weight)
    actual = torch.nn.functional.conv_transpose2d(vk_input, vk_weight)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("route", ["functional", "module"])
def test_transposed_first_backward_matches_cpu_when_create_graph_is_false(
    vulkan_backend, route
):
    generator = torch.Generator(device="cpu").manual_seed(8054)
    cpu_input = torch.randn((1, 2, 2, 2), generator=generator, requires_grad=True)
    cpu_weight = torch.randn((2, 3, 2, 2), generator=generator, requires_grad=True)
    if route == "functional":
        cpu_output = torch.nn.functional.conv_transpose2d(cpu_input, cpu_weight)
        vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
        vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
        vk_output = torch.nn.functional.conv_transpose2d(vk_input, vk_weight)
        vk_targets = (vk_input, vk_weight)
    else:
        cpu_module = torch.nn.ConvTranspose2d(2, 3, 2, bias=False)
        with torch.no_grad():
            cpu_module.weight.copy_(cpu_weight)
        vk_module = copy.deepcopy(cpu_module).to(vulkan_backend)
        vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
        cpu_output = cpu_module(cpu_input)
        vk_output = vk_module(vk_input)
        cpu_weight = cpu_module.weight
        vk_targets = (vk_input, vk_module.weight)

    grad_output_cpu = torch.randn(cpu_output.shape, generator=generator)
    cpu_grads = torch.autograd.grad(
        cpu_output, (cpu_input, cpu_weight), grad_outputs=grad_output_cpu,
        create_graph=False,
    )
    vk_grads = torch.autograd.grad(
        vk_output, vk_targets, grad_outputs=grad_output_cpu.to(vulkan_backend),
        create_graph=False,
    )
    pytorch_vulkan._C.synchronize()
    for actual, expected in zip(vk_grads, cpu_grads):
        torch.testing.assert_close(actual.cpu(), expected)


def test_convolution_backward_op2_bounds_actual_reader_reduction_before_allocation(
    vulkan_backend,
):
    cpu_x = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    cpu_weight = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    cpu_grad_base = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    vk_x, vk_weight, vk_grad_base = (
        value.to(vulkan_backend) for value in (cpu_x, cpu_weight, cpu_grad_base)
    )
    vk_grad = vk_grad_base.expand((1, 1, 65535, 65537))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]

    with pytest.raises(RuntimeError, match="op2 reader reduction exceeds shader index range"):
        torch.ops.aten.convolution_backward.default(
            vk_grad, vk_x, vk_weight, None, [1, 1], [32767, 32768], [1, 1],
            False, [0, 0], 1, [False, True, False],
        )

    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_convolution_transposed_dinput_bounds_op0_sum_intermediate_before_allocation(
    vulkan_backend,
):
    int_max = 2**31 - 1
    cpu_x = torch.ones((1, 1, 2, 1), dtype=torch.float32)
    cpu_weight = torch.ones((1, 1, 2, 1), dtype=torch.float32)
    cpu_grad = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    vk_x, vk_weight, vk_grad = (
        value.to(vulkan_backend) for value in (cpu_x, cpu_weight, cpu_grad)
    )
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]

    with pytest.raises(RuntimeError, match="coordinate expression exceeds signed shader range"):
        torch.ops.aten.convolution_backward.default(
            vk_grad, vk_x, vk_weight, None, [int_max, 1], [int_max, 0],
            [int_max, 1], True, [0, 0], 1, [True, False, False],
        )

    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
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


def test_transposed_extent_outside_signed_shader_range_rejects_before_allocation(
    vulkan_backend,
):
    int_max = 2**31 - 1
    x = torch.ones((1, 1, 2, 1), dtype=torch.float32).to(vulkan_backend)
    weight = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="transposed convolution output extent exceeds shader range"):
        torch.ops.aten.convolution.default(
            x, weight, None, [int_max, 1], [0, 0], [1, 1], True, [0, 0], 1,
        )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_transposed_output_descriptor_size_rejects_before_allocation(vulkan_backend):
    x = torch.ones((1, 1, 1, 2), dtype=torch.float32).to(vulkan_backend)
    weight = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(ValueError, match="convolution has an invalid descriptor range"):
        torch.ops.aten.convolution.default(
            x, weight, None, [1, 1073741823], [0, 0], [1, 1], True, [0, 0], 1,
        )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


def test_transposed_dbias_bounds_maximal_expanded_grad_reduction_before_allocation(
    vulkan_backend,
):
    cpu_base = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    x = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(vulkan_backend)
    weight = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(vulkan_backend)
    vk_base = cpu_base.to(vulkan_backend)
    grad = vk_base.expand((1, 1, 65537, 65535))
    assert grad.numel() == 2**32 - 1
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    before_creations = pytorch_vulkan._C.timing_breakdown()["buffer_creations"]
    with pytest.raises(RuntimeError, match="grad_output logical numel/reduction exceeds shader index range"):
        torch.ops.aten.convolution_backward.default(
            grad, x, weight, None, [1, 1], [0, 0], [1, 1], True, [0, 0], 1,
            [False, False, True],
        )
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_creations


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
    with pytest.raises(RuntimeError, match="bias|shape|size|fixed|support|negative dimension|kernel span|geometry exceeds shader range"):
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


@pytest.mark.parametrize("output_padding", [(1,), (1, 0), (2**63 - 1, 0)])
def test_ordinary_forward_ignores_nonnegative_output_padding(vulkan_backend, output_padding):
    generator = torch.Generator(device="cpu").manual_seed(4601)
    value = torch.randn((1, 2, 4, 5), generator=generator)
    weight = torch.randn((3, 2, 2, 3), generator=generator)
    expected = torch.nn.functional.conv2d(value, weight)
    zero_vk = torch.nn.functional.conv2d(value.to(vulkan_backend), weight.to(vulkan_backend))
    actual = torch.ops.aten.convolution.default(
        value.to(vulkan_backend), weight.to(vulkan_backend), None,
        [1, 1], [0, 0], [1, 1], False, list(output_padding), 1,
    )
    pytorch_vulkan._C.synchronize()
    assert tuple(actual.shape) == tuple(expected.shape) == (1, 3, 3, 3)
    torch.testing.assert_close(actual.cpu(), zero_vk.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("output_padding", [[], [-1], [0, -1], [0, 0, 0]])
def test_convolution_rejects_malformed_output_padding(vulkan_backend, output_padding):
    value = torch.ones((1, 2, 4, 5))
    weight = torch.ones((3, 2, 2, 3))
    arguments = (None, [1, 1], [0, 0], [1, 1], False, output_padding, 1)
    with pytest.raises(RuntimeError):
        torch.ops.aten.convolution.default(value, weight, *arguments)
    with pytest.raises(RuntimeError):
        torch.ops.aten.convolution.default(
            value.to(vulkan_backend), weight.to(vulkan_backend), *arguments
        )


def test_ordinary_backward_nonzero_output_padding_rejects_before_allocation(vulkan_backend):
    value = torch.ones((1, 2, 4, 5))
    weight = torch.ones((3, 2, 2, 3))
    grad = torch.ones((1, 3, 3, 3))
    args = (None, [1, 1], [0, 0], [1, 1], False, [1, 0], 1, [False, False, False])
    with pytest.raises(RuntimeError, match="output_padding is not supported for non-transposed convolutions"):
        torch.ops.aten.convolution_backward.default(grad, value, weight, *args)
    vk_args = (grad.to(vulkan_backend), value.to(vulkan_backend), weight.to(vulkan_backend), *args)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_timing = pytorch_vulkan._C.timing_breakdown()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    with pytest.raises(RuntimeError, match="output_padding"):
        torch.ops.aten.convolution_backward.default(*vk_args)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.timing_breakdown()["buffer_creations"] == before_timing["buffer_creations"]
    assert pytorch_vulkan._C.live_resource_snapshot()[6] == before_live


@pytest.mark.parametrize("output_padding", [(0,), (0, 0)])
def test_ordinary_backward_accepts_zero_output_padding_normalization(vulkan_backend, output_padding):
    generator = torch.Generator(device="cpu").manual_seed(4603)
    value = torch.randn((1, 2, 4, 5), generator=generator)
    weight = torch.randn((3, 2, 2, 3), generator=generator)
    grad = torch.randn((1, 3, 3, 3), generator=generator)
    args = (None, [1, 1], [0, 0], [1, 1], False, list(output_padding), 1,
            [True, True, False])
    expected = torch.ops.aten.convolution_backward.default(
        grad, value, weight, None, [1, 1], [0, 0], [1, 1], False,
        [0, 0], 1, [True, True, False],
    )
    actual = torch.ops.aten.convolution_backward.default(
        grad.to(vulkan_backend), value.to(vulkan_backend), weight.to(vulkan_backend), *args
    )
    pytorch_vulkan._C.synchronize()
    for got, reference in zip(actual, expected):
        assert (got is None) == (reference is None)
        if got is not None:
            torch.testing.assert_close(got.cpu(), reference, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("output_padding", [(0, 0), (1, 0)])
def test_conv_transpose2d_defined_bias_forward_matches_cpu(
    vulkan_backend, output_padding
):
    generator = torch.Generator(device="cpu").manual_seed(6722 + output_padding[0])
    cpu_input = torch.randn((2, 4, 8, 8), generator=generator)
    cpu_weight = torch.randn((4, 1, 3, 3), generator=generator)
    cpu_bias = torch.randn((1,), generator=generator)
    expected = torch.nn.functional.conv_transpose2d(
        cpu_input, cpu_weight, cpu_bias, stride=2, padding=1,
        output_padding=output_padding,
    )
    actual = torch.nn.functional.conv_transpose2d(
        cpu_input.to(vulkan_backend), cpu_weight.to(vulkan_backend),
        cpu_bias.to(vulkan_backend), stride=2, padding=1,
        output_padding=output_padding,
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)
