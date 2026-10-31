"""avg_pool2d / max_pool2d_with_indices forward and backward on Vulkan.

Acceptance goes through nn.AvgPool2d, nn.MaxPool2d, F.avg_pool2d and
F.max_pool2d with a normal backward; direct aten tests supplement them. CPU
(PyTorch 2.4.0) is the reference. Out-of-range max-pool indices are undefined
behaviour on CPU, so they are NEVER given to CPU here: they are checked only
against the explicit Vulkan contract (they contribute nothing). Higher-order
derivatives, forward AD and torch.func are not claimed.
"""
import pytest
import torch
import torch.nn.functional as F

import pytorch_vulkan

aten = torch.ops.aten
AVG_TOL = dict(rtol=2e-5, atol=1e-6)
EXACT = dict(rtol=0, atol=0)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _data(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _zero_work():
    return pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _rejects(pattern, call):
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=pattern):
        call()
    assert _zero_work()


def _forward_backward(device, operation, x, upstream_seed=7):
    """Run `operation` and a backward pass; return (output, indices, grad) on CPU."""
    leaf = x.detach().to(device).requires_grad_()
    result = operation(leaf)
    indices = None
    if isinstance(result, tuple):
        result, indices = result
    upstream = _data(result.shape, upstream_seed)
    result.backward(upstream.to(device))
    if device != "cpu":
        pytorch_vulkan._C.synchronize()
    return (
        result.detach().cpu(),
        None if indices is None else indices.cpu(),
        leaf.grad.cpu(),
        None if indices is None else indices.stride(),
        result.stride(),
        leaf.grad.stride(),
    )


# --- geometry matrices ----------------------------------------------------------

AVG_CASES = [
    ((2, 3, 8, 8), dict(kernel_size=2)),
    ((2, 3, 7, 7), dict(kernel_size=3, stride=2, padding=1)),
    ((2, 3, 7, 7), dict(kernel_size=3, stride=2, padding=1, count_include_pad=False)),
    ((1, 2, 7, 7), dict(kernel_size=2, stride=2, ceil_mode=True)),
    ((1, 2, 7, 7), dict(kernel_size=2, stride=2, padding=1, ceil_mode=True)),
    ((1, 2, 7, 7), dict(kernel_size=2, stride=2, padding=1, ceil_mode=True, count_include_pad=False)),
    ((2, 2, 9, 6), dict(kernel_size=(3, 2), stride=(1, 3), padding=(1, 0), ceil_mode=True)),
    ((1, 1, 5, 5), dict(kernel_size=3, stride=2, padding=1, divisor_override=4)),
    ((3, 6, 6), dict(kernel_size=2)),
    ((2, 3, 6, 6), dict(kernel_size=1, stride=1)),
    ((2, 3, 6, 6), dict(kernel_size=3, stride=1, padding=1)),
    ((2, 3, 6, 5), dict(kernel_size=(2, 3), stride=(3, 1))),
    ((1, 1, 4, 4), dict(kernel_size=4)),
]
MAX_CASES = [
    ((2, 3, 8, 8), dict(kernel_size=2)),
    ((2, 3, 7, 7), dict(kernel_size=3, stride=2, padding=1)),
    ((2, 3, 6, 6), dict(kernel_size=3, stride=1, padding=1)),
    ((2, 2, 9, 9), dict(kernel_size=2, stride=1, dilation=2)),
    ((1, 2, 9, 7), dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2))),
    ((1, 2, 7, 7), dict(kernel_size=2, stride=2, ceil_mode=True)),
    ((1, 2, 7, 7), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True)),
    ((3, 6, 6), dict(kernel_size=2)),
    ((2, 3, 6, 6), dict(kernel_size=1, stride=1)),
    ((2, 3, 5, 5), dict(kernel_size=3, stride=3, padding=1)),
    ((1, 1, 4, 4), dict(kernel_size=4)),
    ((2, 3, 6, 5), dict(kernel_size=(2, 3), stride=(3, 1))),
]


def _ids(cases):
    return [f"{'x'.join(map(str, shape))}-{i}" for i, (shape, _) in enumerate(cases)]


@pytest.mark.parametrize("shape, params", AVG_CASES, ids=_ids(AVG_CASES))
def test_functional_avg_pool2d_forward_and_backward_match_cpu(vulkan_backend, shape, params):
    x = _data(shape, 11)
    operation = lambda t: F.avg_pool2d(t, **params)
    want = _forward_backward("cpu", operation, x)
    got = _forward_backward(vulkan_backend, operation, x)
    torch.testing.assert_close(got[0], want[0], **AVG_TOL)
    torch.testing.assert_close(got[2], want[2], **AVG_TOL)


@pytest.mark.parametrize("shape, params", MAX_CASES, ids=_ids(MAX_CASES))
def test_functional_max_pool2d_forward_indices_and_backward_match_cpu(vulkan_backend, shape, params):
    x = _data(shape, 12)
    operation = lambda t: F.max_pool2d(t, return_indices=True, **params)
    want = _forward_backward("cpu", operation, x)
    got = _forward_backward(vulkan_backend, operation, x)
    assert got[1].dtype == torch.int64
    torch.testing.assert_close(got[0], want[0], **EXACT)
    torch.testing.assert_close(got[1], want[1], **EXACT)
    # Same ascending accumulation order as CPU's scatter-add: bit-identical.
    torch.testing.assert_close(got[2], want[2], **EXACT)


@pytest.mark.parametrize("return_indices", [False, True])
def test_nn_max_pool2d_module_matches_cpu(vulkan_backend, return_indices):
    module = torch.nn.MaxPool2d(3, stride=2, padding=1, return_indices=return_indices)
    x = _data((2, 3, 7, 7), 13)
    operation = lambda t: module(t)
    want = _forward_backward("cpu", operation, x)
    got = _forward_backward(vulkan_backend, operation, x)
    torch.testing.assert_close(got[0], want[0], **EXACT)
    torch.testing.assert_close(got[2], want[2], **EXACT)
    if return_indices:
        torch.testing.assert_close(got[1], want[1], **EXACT)


@pytest.mark.parametrize("count_include_pad", [True, False])
def test_nn_avg_pool2d_module_matches_cpu(vulkan_backend, count_include_pad):
    module = torch.nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=count_include_pad)
    x = _data((2, 3, 7, 7), 14)
    want = _forward_backward("cpu", module, x)
    got = _forward_backward(vulkan_backend, module, x)
    torch.testing.assert_close(got[0], want[0], **AVG_TOL)
    torch.testing.assert_close(got[2], want[2], **AVG_TOL)


def test_pooling_backward_accumulates_across_repeated_calls(vulkan_backend):
    x = _data((2, 3, 6, 6), 15)
    leaf = x.to(vulkan_backend).requires_grad_()
    cpu = x.clone().requires_grad_()
    for _ in range(2):
        F.max_pool2d(leaf, 3, 1, 1).backward(_data((2, 3, 6, 6), 16).to(vulkan_backend))
        F.max_pool2d(cpu, 3, 1, 1).backward(_data((2, 3, 6, 6), 16))
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(leaf.grad.cpu(), cpu.grad, **EXACT)


def test_expanded_upstream_gradient_matches_cpu(vulkan_backend):
    # y.sum().backward() hands pooling backward a stride-0 grad_output.
    x = _data((2, 3, 6, 6), 17)
    for name, operation, tol in (
        ("avg", lambda t: F.avg_pool2d(t, 2), AVG_TOL),
        ("max", lambda t: F.max_pool2d(t, 3, 2, 1), EXACT),
    ):
        want_leaf = x.clone().requires_grad_()
        operation(want_leaf).backward(torch.tensor(1.5).expand(operation(x).shape))
        got_leaf = x.to(vulkan_backend).requires_grad_()
        operation(got_leaf).backward(torch.tensor(1.5).to(vulkan_backend).expand(operation(x).shape))
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(got_leaf.grad.cpu(), want_leaf.grad, **tol)


# --- max-pool index semantics ---------------------------------------------------

NEG_INF, NAN = float("-inf"), float("nan")


def _plane(values):
    side = int(len(values) ** 0.5)
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, side, side)


MAX_SEMANTIC = {
    "ties-first": (_plane([1.0, 5.0, 5.0, 2.0, 0.0, 5.0, 3.0, 9.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0]), dict(kernel_size=2)),
    "constant": (torch.full((1, 2, 4, 4), 3.0), dict(kernel_size=2)),
    "nan-last-wins": (_plane([NAN, NAN, 1.0, 2.0, 3.0, 4.0, NAN, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, NAN, 13.0, 14.0]), dict(kernel_size=2)),
    "signed-zero": (_plane([-0.0, 0.0, 0.0, -0.0] * 4), dict(kernel_size=2)),
    "all-neg-inf": (torch.full((1, 1, 4, 4), NEG_INF), dict(kernel_size=2)),
    "padding-never-wins": (-torch.arange(1.0, 26.0).reshape(1, 1, 5, 5), dict(kernel_size=3, stride=2, padding=1)),
    "dilation": (torch.arange(25.0).reshape(1, 1, 5, 5), dict(kernel_size=2, stride=1, dilation=2)),
    "multi-channel-plane-relative": (torch.arange(2 * 3 * 4 * 4.0).reshape(2, 3, 4, 4).flip(-1), dict(kernel_size=2)),
}


@pytest.mark.parametrize("name", sorted(MAX_SEMANTIC))
def test_max_pool_value_and_index_semantics_match_cpu_exactly(vulkan_backend, name):
    x, params = MAX_SEMANTIC[name]
    want_v, want_i = aten.max_pool2d_with_indices(
        x, params["kernel_size"] if isinstance(params["kernel_size"], (list, tuple)) else [params["kernel_size"]] * 2,
        [params.get("stride", params["kernel_size"])] * 2, [params.get("padding", 0)] * 2,
        [params.get("dilation", 1)] * 2, False,
    )
    got_v, got_i = F.max_pool2d(x.to(vulkan_backend), return_indices=True, **params)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got_v.cpu(), want_v, equal_nan=True, **EXACT)
    assert torch.equal(got_i.cpu(), want_i)


# --- layouts ---------------------------------------------------------------------


def _layout_inputs():
    base = _data((2, 3, 6, 6), 21)
    return {
        "channels-last": base.contiguous(memory_format=torch.channels_last),
        "permuted": _data((6, 6, 2, 3), 22).permute(2, 3, 0, 1),
        "offset-slice": _data((2, 3, 8, 8), 23)[:, :, 1:7, 1:7],
        "expanded": _data((1, 3, 6, 6), 24).expand(2, 3, 6, 6),
    }


@pytest.mark.parametrize("layout", ["channels-last", "permuted", "offset-slice", "expanded"])
@pytest.mark.parametrize("kind", ["avg", "max"])
def test_strided_inputs_match_cpu_values_and_output_strides(vulkan_backend, layout, kind):
    x = _layout_inputs()[layout]
    if kind == "avg":
        operation = lambda t: F.avg_pool2d(t, 3, 2, 1)
        tol = AVG_TOL
    else:
        operation = lambda t: F.max_pool2d(t, 3, 2, 1, return_indices=True)
        tol = EXACT
    leaf_cpu = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    leaf_vk = x.detach().to(vulkan_backend).requires_grad_()
    want = operation(leaf_cpu)
    got = operation(leaf_vk)
    want_t = want if kind == "avg" else want[0]
    got_t = got if kind == "avg" else got[0]
    upstream = _data(want_t.shape, 25)
    want_t.backward(upstream)
    got_t.backward(upstream.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got_t.detach().cpu(), want_t.detach(), **tol)
    assert got_t.stride() == want_t.stride()
    torch.testing.assert_close(leaf_vk.grad.cpu(), leaf_cpu.grad, **tol)
    assert leaf_vk.grad.stride() == leaf_cpu.grad.stride()
    if kind == "max":
        assert torch.equal(got[1].cpu(), want[1])
        assert got[1].stride() == want[1].stride()


def test_channels_last_is_required_for_outputs_indices_and_grad_input(vulkan_backend):
    x = _data((2, 3, 6, 6), 26).contiguous(memory_format=torch.channels_last)
    leaf = x.to(vulkan_backend).requires_grad_()
    out, idx = F.max_pool2d(leaf, 2, return_indices=True)
    out.backward(torch.ones_like(out))
    pytorch_vulkan._C.synchronize()
    for tensor in (out, idx, leaf.grad):
        assert tensor.is_contiguous(memory_format=torch.channels_last)
    avg = F.avg_pool2d(x.to(vulkan_backend), 2)
    assert avg.is_contiguous(memory_format=torch.channels_last)


# --- empty batch and rank 3 ------------------------------------------------------


def test_zero_batch_forward_and_backward_match_cpu(vulkan_backend):
    x = torch.zeros(0, 3, 6, 6)
    leaf = x.to(vulkan_backend).requires_grad_()
    avg = F.avg_pool2d(leaf, 2)
    mx, idx = F.max_pool2d(leaf, 2, return_indices=True)
    assert avg.shape == (0, 3, 3, 3) and mx.shape == (0, 3, 3, 3) and idx.shape == (0, 3, 3, 3)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    grad = aten.avg_pool2d_backward(
        torch.zeros(0, 3, 3, 3).to(vulkan_backend), x.to(vulkan_backend), [2, 2], [2, 2], [0, 0], False, True, None
    )
    grad_max = aten.max_pool2d_with_indices_backward(
        mx.detach(), x.to(vulkan_backend), [2, 2], [2, 2], [0, 0], [1, 1], False, idx
    )
    pytorch_vulkan._C.synchronize()
    assert grad.shape == (0, 3, 6, 6) and grad_max.shape == (0, 3, 6, 6)
    assert grad.device.type == "vk" and grad_max.device.type == "vk"
    dispatches, _, _, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches == 0 and fallbacks == 0


def test_rank_three_inputs_match_cpu_and_return_rank_three_results(vulkan_backend):
    x = _data((3, 7, 7), 27)
    want = _forward_backward("cpu", lambda t: F.max_pool2d(t, 3, 2, 1, return_indices=True), x)
    got = _forward_backward(vulkan_backend, lambda t: F.max_pool2d(t, 3, 2, 1, return_indices=True), x)
    assert got[0].dim() == 3 and got[1].dim() == 3 and got[2].shape == x.shape
    torch.testing.assert_close(got[0], want[0], **EXACT)
    assert torch.equal(got[1], want[1])
    torch.testing.assert_close(got[2], want[2], **EXACT)


# --- direct backward schemas -----------------------------------------------------


def test_avg_pool2d_backward_accepts_explicit_and_empty_stride_like_cpu(vulkan_backend):
    x = _data((2, 3, 6, 6), 31)
    for stride in ([2, 2], [], [2]):
        out = aten.avg_pool2d(x, [2, 2], stride, [0, 0], False, True, None)
        grad = _data(out.shape, 32)
        want = aten.avg_pool2d_backward(grad, x, [2, 2], stride, [0, 0], False, True, None)
        got = aten.avg_pool2d_backward(
            grad.to(vulkan_backend), x.to(vulkan_backend), [2, 2], stride, [0, 0], False, True, None
        )
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(got.cpu(), want, **AVG_TOL)


def _backward_inputs(seed=41, shape=(2, 3, 6, 6), params=(2, 2, 0, 1)):
    k, s, p, d = params
    x = _data(shape, seed)
    out, idx = aten.max_pool2d_with_indices(x, [k, k], [s, s], [p, p], [d, d], False)
    return x, out, idx, [k, k], [s, s], [p, p], [d, d]


def _max_backward(device, grad, x, geometry, idx):
    k, s, p, d = geometry
    return aten.max_pool2d_with_indices_backward(
        grad.to(device), x.to(device), k, s, p, d, False, idx.to(device)
    )


def test_max_backward_with_forward_produced_indices_is_exact(vulkan_backend):
    for params in ((2, 2, 0, 1), (3, 1, 1, 1), (2, 1, 0, 2)):
        x, out, idx, k, s, p, d = _backward_inputs(params=params, shape=(2, 3, 7, 7))
        grad = _data(out.shape, 42)
        want = aten.max_pool2d_with_indices_backward(grad, x, k, s, p, d, False, idx)
        got = _max_backward(vulkan_backend, grad, x, (k, s, p, d), idx)
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(got.cpu(), want, **EXACT)


def _channels_last(tensor, enabled):
    return tensor.contiguous(memory_format=torch.channels_last) if enabled else tensor


def _vk_indices(indices, channels_last, device):
    """Upload int64 indices. Vulkan int64 data can only arrive through the label
    upload (contiguous CPU data, values >= 0 or -100), so a channels-last index
    tensor is built by uploading NHWC-contiguous data and permuting it back to a
    logical NCHW view that has channels-last strides."""
    if not channels_last:
        return indices.contiguous().to(device)
    uploaded = indices.permute(0, 2, 3, 1).contiguous().to(device).permute(0, 3, 1, 2)
    assert uploaded.is_contiguous(memory_format=torch.channels_last)
    return uploaded


@pytest.mark.parametrize("cl", [False, True], ids=["nchw", "channels-last"])
@pytest.mark.parametrize(
    "case",
    ["all-to-one", "outside-own-window", "duplicates", "mixed-valid-and-crafted", "one-plane-crafted"],
)
def test_max_backward_with_crafted_in_range_indices_equals_cpu_scatter_exactly(vulkan_backend, case, cl):
    # Every index is inside [0, H*W): CPU scatters it wherever it points, even
    # outside its own window, accumulating in ascending output order.
    x = _data((2, 2, 4, 4), 43)
    out, idx = aten.max_pool2d_with_indices(x, [2, 2], [2, 2], [0, 0], [1, 1], False)
    grad = torch.arange(1.0, 1.0 + out.numel()).reshape(out.shape)
    crafted = idx.clone()
    if case == "all-to-one":
        crafted[:] = 15
    elif case == "outside-own-window":
        crafted[..., 0, 0], crafted[..., 1, 1] = 15, 0
    elif case == "duplicates":
        crafted[:] = torch.tensor([5, 5, 5, 5])[:out.shape[-1] * out.shape[-2]].reshape(out.shape[-2:])
    elif case == "mixed-valid-and-crafted":
        crafted[0, 0, 0, 1] = 12
        crafted[1, 1, 1, 0] = 3
    elif case == "one-plane-crafted":
        crafted[1, 0] = 0  # only plane (n=1, c=0) is crafted; the others stay valid
    want = aten.max_pool2d_with_indices_backward(
        _channels_last(grad, cl), _channels_last(x, cl), [2, 2], [2, 2], [0, 0], [1, 1], False,
        _channels_last(crafted, cl),
    )
    got = aten.max_pool2d_with_indices_backward(
        _channels_last(grad, cl).to(vulkan_backend), _channels_last(x, cl).to(vulkan_backend),
        [2, 2], [2, 2], [0, 0], [1, 1], False, _vk_indices(crafted, cl, vulkan_backend),
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **EXACT)
    assert got.stride() == want.stride()


@pytest.mark.parametrize("cl", [False, True], ids=["nchw", "channels-last"])
def test_max_backward_sums_in_ascending_output_order_like_cpu(vulkan_backend, cl):
    # float32 addition is order sensitive: ((1e8 + 1) - 1e8) + 1 == 1.0.
    x = torch.zeros(1, 1, 4, 4)
    grad = torch.tensor([[[[1e8, 1.0], [-1e8, 1.0]]]])
    idx = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
    want = aten.max_pool2d_with_indices_backward(
        _channels_last(grad, cl), _channels_last(x, cl), [2, 2], [2, 2], [0, 0], [1, 1], False, _channels_last(idx, cl)
    )
    got = aten.max_pool2d_with_indices_backward(
        _channels_last(grad, cl).to(vulkan_backend), _channels_last(x, cl).to(vulkan_backend),
        [2, 2], [2, 2], [0, 0], [1, 1], False, _vk_indices(idx, cl, vulkan_backend),
    )
    pytorch_vulkan._C.synchronize()
    assert want.flatten()[0].item() == 1.0
    torch.testing.assert_close(got.cpu(), want, **EXACT)


@pytest.mark.parametrize("cl", [False, True], ids=["nchw", "channels-last"])
def test_max_backward_out_of_range_indices_follow_the_explicit_vulkan_contract(vulkan_backend, cl):
    # CPU behaviour for out-of-range indices is undefined (memory corruption,
    # abort), so CPU never sees them. The Vulkan contract: such entries
    # contribute nothing and nothing faults. The expected value is computed on
    # CPU from a sanitized copy (out-of-range entries zeroed and pointed at 0).
    # Only values the backend can hold are used: int64 data reaches Vulkan only
    # through the label upload, which accepts >= 0 and -100 (so -1 and other
    # negatives are unreachable); the kernel still treats any negative as
    # out of range.
    x = _data((2, 2, 4, 4), 44)
    out, idx = aten.max_pool2d_with_indices(x, [2, 2], [2, 2], [0, 0], [1, 1], False)
    grad = torch.arange(1.0, 1.0 + out.numel()).reshape(out.shape)
    bad = idx.clone()
    bad[0, 0, 0, 0], bad[0, 0, 0, 1], bad[0, 0, 1, 0], bad[0, 0, 1, 1] = -100, 16, 2**40, 2**62
    bad[1, 1, 0, 0] = 99
    out_of_range = (bad < 0) | (bad >= 16)  # -100 is the only reachable negative
    sanitized_idx = torch.where(out_of_range, torch.zeros_like(bad), bad)
    sanitized_grad = torch.where(out_of_range, torch.zeros_like(grad), grad)
    assert out_of_range.sum() == 5
    want = aten.max_pool2d_with_indices_backward(
        _channels_last(sanitized_grad, cl), _channels_last(x, cl), [2, 2], [2, 2], [0, 0], [1, 1], False,
        _channels_last(sanitized_idx, cl),
    )
    got = aten.max_pool2d_with_indices_backward(
        _channels_last(grad, cl).to(vulkan_backend), _channels_last(x, cl).to(vulkan_backend),
        [2, 2], [2, 2], [0, 0], [1, 1], False, _vk_indices(bad, cl, vulkan_backend),
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **EXACT)


# --- rejections happen before any Vulkan work ----------------------------------


def test_pooling_geometry_errors_are_rejected_like_pytorch_with_zero_work(vulkan_backend):
    x = _data((1, 2, 6, 6), 51).to(vulkan_backend)
    call = lambda *a: aten.avg_pool2d(x, *a)
    _rejects("kernel size should be greater than zero", lambda: call([0, 2], [1, 1], [0, 0], False, True, None))
    _rejects("stride should not be zero", lambda: call([2, 2], [0, 1], [0, 0], False, True, None))
    _rejects("pad should be at most half of effective kernel size", lambda: call([2, 2], [1, 1], [2, 2], False, True, None))
    _rejects("Output size is too small", lambda: call([7, 7], [1, 1], [0, 0], False, True, None))
    _rejects("divisor must be not zero", lambda: call([2, 2], [1, 1], [0, 0], False, True, 0))
    _rejects("kernel_size must either be a single int", lambda: call([2, 2, 2], [1, 1], [0, 0], False, True, None))
    mp = lambda *a: aten.max_pool2d_with_indices(x, *a)
    _rejects("dilation should be greater than zero", lambda: mp([2, 2], [1, 1], [0, 0], [0, 1], False))
    _rejects("pad should be at most half of effective kernel size", lambda: mp([2, 2], [1, 1], [2, 2], [1, 1], False))
    _rejects("stride should not be zero", lambda: mp([2, 2], [0, 0], [0, 0], [1, 1], False))
    _rejects("Output size is too small", lambda: mp([9, 9], [1, 1], [0, 0], [1, 1], False))


@pytest.mark.parametrize("shape", [(2, 3, 0, 4), (2, 3, 4, 0), (2, 0, 4, 4), (0, 4, 4), (5, 5), (1, 1, 1, 5, 5)])
def test_unsupported_input_shapes_are_rejected_with_zero_work(vulkan_backend, shape):
    x = _vk_or_skip(shape, vulkan_backend)
    _rejects("Expected 3D or 4D", lambda: aten.avg_pool2d(x, [2, 2]))
    _rejects("Expected 3D or 4D", lambda: aten.max_pool2d_with_indices(x, [2, 2]))


def _vk_or_skip(shape, device):
    return torch.zeros(shape).to(device)


def test_dtype_and_device_mismatches_are_rejected_with_zero_work(vulkan_backend):
    x = _data((1, 2, 6, 6), 52).to(vulkan_backend)
    double = x.double()
    cpu = _data((1, 2, 6, 6), 52)
    _rejects("Vulkan avg_pool2d input requires a strided float32 vk:0 tensor",
             lambda: aten.avg_pool2d(double, [2, 2]))
    _rejects("Vulkan max_pool2d_with_indices input requires a strided float32 vk:0 tensor",
             lambda: aten.max_pool2d_with_indices(double, [2, 2]))
    _rejects("Vulkan avg_pool2d_backward grad_output requires a strided float32 vk:0 tensor",
             lambda: aten.avg_pool2d_backward(cpu[..., ::2, ::2], x, [2, 2], [2, 2], [0, 0], False, True, None))


def test_backward_argument_mismatches_are_rejected_with_zero_work(vulkan_backend):
    x = _data((1, 2, 6, 6), 53)
    out, idx = aten.max_pool2d_with_indices(x, [2, 2], [2, 2], [0, 0], [1, 1], False)
    vx, vgrad = x.to(vulkan_backend), out.to(vulkan_backend)
    vidx = idx.to(vulkan_backend)
    short_grad = _data((1, 2, 2, 3), 54).to(vulkan_backend)
    short_idx = idx[..., :2].contiguous().to(vulkan_backend)
    cpu_idx = idx
    geometry = ([2, 2], [2, 2], [0, 0], [1, 1], False)
    bwd = aten.max_pool2d_with_indices_backward
    _rejects("Expected a tensor of dimension 4 and tensor.size", lambda: bwd(short_grad, vx, *geometry, vidx))
    _rejects("Expected a tensor of dimension 4 and tensor.size", lambda: bwd(vgrad, vx, *geometry, short_idx))
    _rejects("Vulkan max_pool2d_with_indices_backward indices requires a strided int64 vk:0 tensor",
             lambda: bwd(vgrad, vx, *geometry, cpu_idx))
    _rejects("Vulkan max_pool2d_with_indices_backward indices requires a strided int64 vk:0 tensor",
             lambda: bwd(vgrad, vx, *geometry, vgrad))
    _rejects("Expected a tensor of dimension 4 and tensor.size", lambda: aten.avg_pool2d_backward(
        short_grad, vx, [2, 2], [2, 2], [0, 0], False, True, None))
    _rejects("divisor must be not zero", lambda: aten.avg_pool2d_backward(
        vgrad, vx, [2, 2], [2, 2], [0, 0], False, True, 0))


def test_out_variants_remain_unsupported(vulkan_backend):
    x = _data((1, 2, 6, 6), 55).to(vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError)):
        aten.avg_pool2d.out(x, [2, 2], out=torch.empty(0, device=vulkan_backend))


def test_pooling_results_stay_on_the_device_without_fallback(vulkan_backend):
    x = _data((2, 3, 6, 6), 56).to(vulkan_backend).requires_grad_()
    upstream = _data((2, 3, 3, 3), 57).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    F.max_pool2d(x, 2).backward(upstream)
    F.avg_pool2d(x, 2).backward(upstream)
    pytorch_vulkan._C.synchronize()
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches > 0 and transfers == 0 and fallbacks == 0
