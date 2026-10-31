"""2-D interpolation (nearest, nearest-exact, bilinear) forward and backward on Vulkan.

Acceptance goes through F.interpolate and nn.Upsample / UpsamplingNearest2d /
UpsamplingBilinear2d with a normal backward; direct aten tests supplement them.
CPU (PyTorch 2.4.0) is the reference.  Nearest and nearest-exact forward are index
exact copies, so they are compared bitwise with CPU.  Bilinear is compared with CPU
within tolerance (CPU's own paths contract differently) and bitwise with the
shader-model oracle in interpolation_reference.py, which also supplies the
contraction canaries.  Higher-order derivatives, forward AD, torch.func and vmap are
not claimed and are not asserted either way.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_vulkan
import interpolation_reference as ref

aten = torch.ops.aten
EXACT = dict(rtol=0, atol=0)
BIL = dict(rtol=1e-5, atol=1e-5)
BIL_LARGE = dict(rtol=1e-5, atol=1e-4)

GRAD_FN = {
    "nearest": "UpsampleNearest2DBackward0",
    "nearest-exact": "UpsampleNearestExact2DBackward0",
    "bilinear": "UpsampleBilinear2DBackward0",
}
FORWARD = {
    "nearest": lambda x, o, h, ac: aten.upsample_nearest2d(x, o, h[0], h[1]),
    "nearest-exact": lambda x, o, h, ac: aten._upsample_nearest_exact2d(x, o, h[0], h[1]),
    "bilinear": lambda x, o, h, ac: aten.upsample_bilinear2d(x, o, ac, h[0], h[1]),
}
BACKWARD = {
    "nearest": lambda g, o, i, h, ac: aten.upsample_nearest2d_backward(g, o, i, h[0], h[1]),
    "nearest-exact": lambda g, o, i, h, ac: aten._upsample_nearest_exact2d_backward(g, o, i, h[0], h[1]),
    "bilinear": lambda g, o, i, h, ac: aten.upsample_bilinear2d_backward(g, o, i, ac, h[0], h[1]),
}
MODE_KW = [
    ("nearest", {}),
    ("nearest-exact", {}),
    ("bilinear", {"align_corners": False}),
    ("bilinear", {"align_corners": True}),
]
MODE_IDS = ["nearest", "nearest-exact", "bilinear", "bilinear-ac"]


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _data(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _tol(mode, large=False):
    if mode != "bilinear":
        return EXACT if not large else dict(rtol=1e-5, atol=1e-5)
    return BIL_LARGE if large else BIL


def _zero_work():
    return pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _rejects(pattern, call):
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=pattern):
        call()
    assert _zero_work()


def _forward_backward(device, operation, x, upstream_seed=7):
    leaf = x.detach().to(device).requires_grad_()
    result = operation(leaf)
    upstream = _data(result.shape, upstream_seed)
    result.backward(upstream.to(device))
    if device != "cpu":
        pytorch_vulkan._C.synchronize()
    return result.detach().cpu(), leaf.grad.cpu(), result.stride(), leaf.grad.stride(), result


# --- ordinary APIs ----------------------------------------------------------------

INTERP_CASES = [
    ("size", dict(size=(5, 7))),
    ("size-int", dict(size=5)),
    ("sf2", dict(scale_factor=2)),
    ("sf-tuple", dict(scale_factor=(1.5, 2.5))),
    ("sf1.5-hint", dict(scale_factor=1.5, recompute_scale_factor=False)),
    ("sf1.5-recompute", dict(scale_factor=1.5, recompute_scale_factor=True)),
    ("sf1.7", dict(scale_factor=1.7)),
    ("down-sf0.5", dict(scale_factor=0.5)),
    ("down-size", dict(size=(2, 3))),
    ("same-size", dict(size=(4, 5))),
]


@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
@pytest.mark.parametrize("name,args", INTERP_CASES, ids=[c[0] for c in INTERP_CASES])
def test_interpolate_forward_and_backward_match_cpu(vulkan_backend, mode, kw, name, args):
    x = _data((2, 3, 4, 5), 11)
    operation = lambda t: F.interpolate(t, mode=mode, **args, **kw)
    want, want_grad, want_stride, want_grad_stride, _ = _forward_backward("cpu", operation, x)
    got, got_grad, got_stride, got_grad_stride, result = _forward_backward(
        vulkan_backend, operation, x
    )
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, **_tol(mode))
    torch.testing.assert_close(got_grad, want_grad, **_tol(mode, large=True))
    assert got_stride == want_stride
    assert got_grad_stride == want_grad_stride


@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_generated_autograd_node_attaches_without_fallback(vulkan_backend, mode, kw):
    leaf = _data((1, 2, 3, 4), 12).to(vulkan_backend).requires_grad_()
    pytorch_vulkan._C.synchronize()
    before = pytorch_vulkan._C.fallback_count()
    result = F.interpolate(leaf, scale_factor=2, mode=mode, **kw)
    assert type(result.grad_fn).__name__ == GRAD_FN[mode]
    result.sum().backward()
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.fallback_count() == before
    assert leaf.grad is not None and leaf.grad.device.type == "vk"


MODULE_CASES = [
    ("Upsample-nearest-sf", lambda: nn.Upsample(scale_factor=2, mode="nearest")),
    ("Upsample-nearest-size", lambda: nn.Upsample(size=(7, 3), mode="nearest")),
    ("Upsample-exact-sf", lambda: nn.Upsample(scale_factor=(1.5, 2), mode="nearest-exact")),
    ("Upsample-bilinear-sf", lambda: nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)),
    ("Upsample-bilinear-ac", lambda: nn.Upsample(size=(6, 9), mode="bilinear", align_corners=True)),
    ("UpsamplingNearest2d", lambda: nn.UpsamplingNearest2d(scale_factor=2)),
    ("UpsamplingBilinear2d", lambda: nn.UpsamplingBilinear2d(scale_factor=2)),
]


@pytest.mark.parametrize("name,build", MODULE_CASES, ids=[c[0] for c in MODULE_CASES])
def test_nn_modules_match_cpu(vulkan_backend, name, build):
    x = _data((2, 3, 4, 5), 13)
    module = build()
    want, want_grad, *_ = _forward_backward("cpu", module, x)
    got, got_grad, *_ = _forward_backward(vulkan_backend, module, x)
    exact = "nearest" in name.lower() and "bilinear" not in name.lower()
    torch.testing.assert_close(got, want, **(EXACT if exact else BIL))
    torch.testing.assert_close(got_grad, want_grad, **BIL_LARGE)


def test_input_without_grad_requirement_creates_no_history(vulkan_backend):
    x = _data((1, 2, 3, 4), 14).to(vulkan_backend)
    result = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert not result.requires_grad and result.grad_fn is None


# --- direct forward: hints, shortcuts, oracle --------------------------------------

DIRECT_CASES = [
    # (mode, input shape, output size, hints, align_corners)
    ("nearest", (1, 2, 3, 4), (5, 7), (None, None), False),
    ("nearest", (1, 2, 3, 4), (5, 7), (1.4, 0.6), False),  # hints disagree with size
    ("nearest", (1, 2, 3, 4), (6, 8), (1.4, 0.6), False),  # out == 2*in ignores hints
    ("nearest", (1, 2, 3, 4), (3, 4), (2.5, 2.5), False),  # out == in ignores hints
    ("nearest", (1, 2, 3, 4), (3, 9), (2.5, 2.5), False),  # one dimension identity
    ("nearest", (1, 2, 7, 7), (10, 10), (1.4, 1.4), False),
    ("nearest", (1, 2, 3, 4), (5, 7), (0.0, -1.0), False),  # non-positive == absent
    ("nearest", (1, 2, 3, 4), (5, 7), (float("nan"), float("inf")), False),
    ("nearest-exact", (1, 2, 3, 4), (5, 7), (None, None), False),
    ("nearest-exact", (1, 2, 3, 4), (6, 8), (1.4, 0.6), False),  # no 2x shortcut
    ("nearest-exact", (1, 2, 3, 4), (3, 4), (2.5, 2.5), False),
    ("nearest-exact", (1, 2, 3, 9), (4, 5), (0.0, float("inf")), False),
    ("nearest-exact", (1, 2, 7, 7), (10, 10), (1.4, 1.4), False),
    ("bilinear", (1, 2, 3, 4), (5, 7), (None, None), False),
    ("bilinear", (1, 2, 3, 4), (5, 7), (None, None), True),
    ("bilinear", (1, 2, 3, 4), (5, 7), (1.4, 0.6), False),
    ("bilinear", (1, 2, 3, 4), (5, 7), (1.4, 0.6), True),  # align_corners ignores hints
    ("bilinear", (1, 2, 3, 4), (3, 9), (2.5, 2.5), False),
    ("bilinear", (1, 2, 3, 4), (3, 9), (2.5, 2.5), True),
    ("bilinear", (1, 2, 3, 4), (1, 1), (None, None), True),  # out == 1: scale 0
    ("bilinear", (1, 2, 6, 5), (3, 2), (None, None), False),
    ("bilinear", (2, 3, 7, 7), (10, 10), (1.4, 1.4), False),
]
DIRECT_IDS = [f"{c[0]}-{c[1][2]}x{c[1][3]}-{c[2][0]}x{c[2][1]}-{i}" for i, c in enumerate(DIRECT_CASES)]


@pytest.mark.parametrize("case", DIRECT_CASES, ids=DIRECT_IDS)
def test_direct_forward_matches_cpu_and_oracle(vulkan_backend, case):
    mode, shape, out, hints, ac = case
    x = _data(shape, 15)
    want = FORWARD[mode](x, list(out), hints, ac)
    got = FORWARD[mode](x.to(vulkan_backend), list(out), hints, ac)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **_tol(mode))
    oracle = ref.forward(x.numpy(), out, mode, hints, ac)
    assert np.array_equal(got.cpu().numpy(), oracle), "Vulkan differs from the shader-model oracle"


def _grid_combos():
    combos = []
    for size_in in (1, 2, 3, 5, 8, 17):
        for size_out in (1, 2, 3, 4, 5, 6, 8, 10, 16, 17, 34, 35):
            ratio = size_out / size_in
            for hint in (None, 0.0, ratio, ratio * 1.1, 1.5):
                combos.append((size_in, size_out, hint))
    return combos


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_index_maps_equal_cpu_on_sampled_grid(vulkan_backend, mode):
    combos = _grid_combos()
    mismatches = []
    for k, (hi, ho, hh) in enumerate(combos):
        wi, wo, wh = combos[(k * 7 + 3) % len(combos)]
        # distinct value per source pixel, exactly representable
        x = (torch.arange(hi).reshape(hi, 1) * 64 + torch.arange(wi).reshape(1, wi)).float()
        x = x.reshape(1, 1, hi, wi)
        want = FORWARD[mode](x, [ho, wo], (hh, wh), False)
        got = FORWARD[mode](x.to(vulkan_backend), [ho, wo], (hh, wh), False)
        pytorch_vulkan._C.synchronize()
        if not torch.equal(got.cpu(), want):
            mismatches.append((hi, ho, hh, wi, wo, wh))
    assert not mismatches, f"{len(mismatches)} index-map mismatches, e.g. {mismatches[:3]}"


def test_contraction_canaries_are_bitwise_equal_to_the_unfused_model(vulkan_backend):
    canaries = ref.contraction_canaries()
    triples = sorted({(i, o, h) for i, o, h, _ in canaries}, key=lambda t: (t[0], t[1], str(t[2])))
    assert len(canaries) >= 100 and len(triples) >= 20, "canary list must not be vacuous"
    failures = []
    for size_in, size_out, hint in triples[:60]:
        # H carries the canary dimension; W is a 1->1 identity dimension.
        x = torch.arange(size_in, dtype=torch.float32).reshape(1, 1, size_in, 1)
        got = aten.upsample_bilinear2d(x.to(vulkan_backend), [size_out, 1], False, hint, None)
        pytorch_vulkan._C.synchronize()
        want = ref.forward(x.numpy(), (size_out, 1), "bilinear", (hint, None), False)
        if not np.array_equal(got.cpu().numpy(), want):
            failures.append((size_in, size_out, hint))
    assert not failures, f"device result differs from the unfused model (contraction?): {failures[:5]}"


def test_bilinear_random_data_is_bitwise_equal_to_the_model(vulkan_backend):
    for seed, (shape, out, hints, ac) in enumerate(
        [((2, 3, 9, 11), (14, 17), (None, None), False),
         ((1, 4, 16, 16), (29, 23), (1.8, 1.4), False),
         ((1, 4, 16, 16), (29, 23), (None, None), True),
         ((2, 2, 33, 21), (12, 40), (None, None), False)]
    ):
        x = _data(shape, 100 + seed)
        got = aten.upsample_bilinear2d(x.to(vulkan_backend), list(out), ac, hints[0], hints[1])
        pytorch_vulkan._C.synchronize()
        want = ref.forward(x.numpy(), out, "bilinear", hints, ac)
        assert np.array_equal(got.cpu().numpy(), want)


# --- direct backward ---------------------------------------------------------------

BACKWARD_CASES = [
    ("nearest", (2, 2, 3, 4), (7, 9), (None, None), False),
    ("nearest", (2, 2, 3, 4), (7, 9), (1.4, 0.6), False),
    ("nearest", (2, 2, 6, 8), (3, 4), (None, None), False),
    ("nearest", (1, 2, 4, 5), (8, 10), (None, None), False),
    ("nearest-exact", (2, 2, 3, 4), (7, 9), (None, None), False),
    ("nearest-exact", (2, 2, 6, 8), (3, 4), (2.0, 2.0), False),
    ("bilinear", (2, 2, 3, 4), (7, 9), (None, None), False),
    ("bilinear", (2, 2, 3, 4), (7, 9), (None, None), True),
    ("bilinear", (2, 2, 3, 4), (7, 9), (1.4, 0.6), False),
    ("bilinear", (2, 2, 6, 8), (3, 4), (None, None), False),
    ("bilinear", (1, 2, 4, 5), (8, 10), (None, None), False),
    ("bilinear", (1, 2, 4, 5), (1, 1), (None, None), True),
]
BACKWARD_IDS = [f"{c[0]}-{c[1][2]}x{c[1][3]}-{c[2][0]}x{c[2][1]}-{i}" for i, c in enumerate(BACKWARD_CASES)]


@pytest.mark.parametrize("case", BACKWARD_CASES, ids=BACKWARD_IDS)
def test_direct_backward_matches_cpu_oracle_and_counts(vulkan_backend, case):
    mode, input_size, out, hints, ac = case
    grad = _data((input_size[0], input_size[1], out[0], out[1]), 21)
    want = BACKWARD[mode](grad, list(out), list(input_size), hints, ac)
    got = BACKWARD[mode](grad.to(vulkan_backend), list(out), list(input_size), hints, ac)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **_tol(mode, large=True))
    oracle = ref.backward(grad.numpy(), input_size[2:], mode, hints, ac)
    assert np.array_equal(got.cpu().numpy(), oracle), "Vulkan backward differs from the model"
    ones = torch.ones_like(grad)
    counts = BACKWARD[mode](ones.to(vulkan_backend), list(out), list(input_size), hints, ac)
    pytorch_vulkan._C.synchronize()
    assert counts.sum().item() == pytest.approx(ones.numel(), rel=1e-6)
    torch.testing.assert_close(counts.cpu(), BACKWARD[mode](ones, list(out), list(input_size), hints, ac), **_tol(mode, large=True))


@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_gradients_accumulate_and_reset_like_cpu(vulkan_backend, mode, kw):
    x = _data((1, 2, 3, 4), 31)
    results = {}
    for device in ("cpu", vulkan_backend):
        leaf = x.detach().to(device).requires_grad_()
        for seed in (1, 2):
            out = F.interpolate(leaf, scale_factor=2, mode=mode, **kw)
            out.backward(_data(out.shape, seed).to(device))
        first = leaf.grad.clone()
        leaf.grad = None
        out = F.interpolate(leaf, scale_factor=2, mode=mode, **kw)
        out.backward(_data(out.shape, 3).to(device))
        if device != "cpu":
            pytorch_vulkan._C.synchronize()
        results[device] = (first.cpu(), leaf.grad.cpu())
    for got, want in zip(results[vulkan_backend], results["cpu"]):
        torch.testing.assert_close(got, want, **BIL_LARGE)


@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_repeat_execution_is_stable(vulkan_backend, mode, kw):
    x = _data((2, 3, 5, 6), 32).to(vulkan_backend)
    first = F.interpolate(x, scale_factor=2.5, mode=mode, **kw).cpu()
    for _ in range(3):
        again = F.interpolate(x, scale_factor=2.5, mode=mode, **kw).cpu()
        assert torch.equal(first, again)


# --- realistic sizes ---------------------------------------------------------------

REALISTIC = [
    ("nearest", (2, 8, 32, 32), (64, 64)),
    ("nearest-exact", (1, 4, 37, 53), (111, 79)),
    ("bilinear", (1, 8, 64, 64), (32, 32)),
    ("bilinear", (1, 3, 224, 224), (300, 300)),
]


@pytest.mark.parametrize("mode,shape,out", REALISTIC, ids=[f"{m}-{s[-1]}-{o[0]}" for m, s, o in REALISTIC])
def test_realistic_sizes_match_cpu_and_float64(vulkan_backend, mode, shape, out):
    x = _data(shape, 41)
    grad = _data((shape[0], shape[1], *out), 42)
    want = FORWARD[mode](x, list(out), (None, None), False)
    want64 = FORWARD[mode](x.double(), list(out), (None, None), False)
    got = FORWARD[mode](x.to(vulkan_backend), list(out), (None, None), False)
    want_g = BACKWARD[mode](grad, list(out), list(shape), (None, None), False)
    want_g64 = BACKWARD[mode](grad.double(), list(out), list(shape), (None, None), False)
    got_g = BACKWARD[mode](grad.to(vulkan_backend), list(out), list(shape), (None, None), False)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **_tol(mode, large=True))
    torch.testing.assert_close(got_g.cpu(), want_g, **BIL_LARGE)
    assert (got.cpu().double() - want64).abs().max().item() <= 1e-4
    assert (got_g.cpu().double() - want_g64).abs().max().item() <= 2e-4


# --- layouts -----------------------------------------------------------------------


def _layout_inputs():
    return {
        "channels-last": _data((2, 3, 4, 5), 51).contiguous(memory_format=torch.channels_last),
        "channels-last-n1": _data((1, 3, 4, 5), 52).contiguous(memory_format=torch.channels_last),
        "channels-last-c1": _data((2, 1, 4, 5), 53).contiguous(memory_format=torch.channels_last),
        "permuted": _data((5, 4, 2, 3), 54).permute(2, 3, 1, 0),
        "offset-slice": _data((2, 3, 8, 8), 55)[:, :, 1:5, 2:7],
        "expanded": _data((1, 3, 4, 5), 56).expand(2, 3, 4, 5),
    }


@pytest.mark.parametrize("layout", list(_layout_inputs()))
@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_strided_inputs_match_cpu_values_and_output_strides(vulkan_backend, layout, mode, kw):
    x = _layout_inputs()[layout]
    operation = lambda t: F.interpolate(t, scale_factor=(2, 1.5), mode=mode, **kw)
    leaf_cpu = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    leaf_vk = x.detach().to(vulkan_backend).requires_grad_()
    want = operation(leaf_cpu)
    got = operation(leaf_vk)
    upstream = _data(want.shape, 57)
    want.backward(upstream)
    got.backward(upstream.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.detach().cpu(), want.detach(), **_tol(mode))
    assert got.stride() == want.stride()
    torch.testing.assert_close(leaf_vk.grad.cpu(), leaf_cpu.grad, **BIL_LARGE)
    assert leaf_vk.grad.stride() == leaf_cpu.grad.stride()


GRAD_LAYOUTS = {
    "channels-last": lambda g: g.contiguous(memory_format=torch.channels_last),
    "permuted": lambda g: g.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2),
    "expanded": lambda g: g[:1, :1].expand_as(g),
    "offset-slice": lambda g: torch.cat([g, g], 3)[:, :, :, 3:3 + g.shape[3]],
}


@pytest.mark.parametrize("layout", list(GRAD_LAYOUTS))
@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_direct_backward_reads_strided_grad_output_and_follows_its_format(vulkan_backend, layout, mode, kw):
    ac = kw.get("align_corners", False)
    grad = GRAD_LAYOUTS[layout](_data((2, 3, 6, 8), 61))
    args = ([6, 8], [2, 3, 3, 4], (None, None), ac)
    want = BACKWARD[mode](grad, *args)
    got = BACKWARD[mode](grad.to(vulkan_backend), *args)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(got.cpu(), want, **_tol(mode, large=True))
    assert got.stride() == want.stride()


# --- empty batch, rejections -------------------------------------------------------


@pytest.mark.parametrize("mode,kw", MODE_KW, ids=MODE_IDS)
def test_zero_batch_forward_and_backward_match_cpu(vulkan_backend, mode, kw):
    x = torch.empty(0, 3, 4, 5)
    for device in ("cpu", vulkan_backend):
        leaf = x.detach().to(device).requires_grad_()
        out = F.interpolate(leaf, scale_factor=2, mode=mode, **kw)
        out.sum().backward()
        assert out.shape == (0, 3, 8, 10)
        assert leaf.grad.shape == (0, 3, 4, 5)
    pytorch_vulkan._C.synchronize()


def test_zero_batch_does_no_vulkan_dispatch(vulkan_backend):
    x = torch.empty(0, 3, 4, 5).to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    F.interpolate(x, scale_factor=2, mode="nearest")
    pytorch_vulkan._C.synchronize()
    assert _zero_work()


def test_cpu_error_messages_are_reproduced_with_zero_vulkan_work(vulkan_backend):
    x = _data((2, 3, 4, 5), 71).to(vulkan_backend)
    empty_channels = torch.empty(2, 0, 4, 5).to(vulkan_backend)
    _rejects("Non-empty 4D data tensor expected", lambda: aten.upsample_nearest2d(empty_channels, [8, 10]))
    _rejects("Input and output sizes should be greater than 0", lambda: aten.upsample_nearest2d(x, [0, 3]))
    _rejects("Input and output sizes should be greater than 0", lambda: aten.upsample_bilinear2d(x, [-5, 5], False))
    _rejects("It is expected output_size equals to 2", lambda: aten.upsample_nearest2d(x, [5]))
    _rejects("It is expected input_size equals to 4", lambda: aten.upsample_nearest2d(x[0], [5, 5]))
    _rejects("It is expected input_size equals to 4", lambda: aten.upsample_bilinear2d(x[0], [5, 5], False))
    grad = _data((2, 3, 8, 9), 72).to(vulkan_backend)
    _rejects("Expected grad_output to have the same shape as output",
             lambda: aten.upsample_nearest2d_backward(grad, [8, 10], [2, 3, 4, 5]))
    _rejects("It is expected input_size equals to 4",
             lambda: aten.upsample_nearest2d_backward(grad, [8, 9], [2, 3, 4]))
    _rejects("Input and output sizes should be greater than 0",
             lambda: aten._upsample_nearest_exact2d_backward(grad, [8, 9], [2, 3, 0, 5]))


@pytest.mark.parametrize("name", ["upsample_nearest2d", "_upsample_nearest_exact2d", "upsample_bilinear2d"])
def test_unsupported_dtypes_and_narrowed_limits_are_rejected_with_zero_work(vulkan_backend, name):
    extra = (False,) if name == "upsample_bilinear2d" else ()
    op = getattr(aten, name)
    f64 = _data((1, 2, 3, 4), 73).to(vulkan_backend).double()
    _rejects(f"Vulkan {name} requires a strided float32 vk:0 tensor", lambda: op(f64, [6, 8], *extra))
    tiny = _data((1, 1, 1, 1), 74).to(vulkan_backend)
    _rejects("must not exceed 8388608", lambda: op(tiny, [(1 << 23) + 1, 1], *extra))
    x = _data((1, 2, 3, 4), 75).to(vulkan_backend)
    _rejects("source scale", lambda: op(x, [5, 7], *extra, 1e-30, 1e-30))
    _rejects("source scale", lambda: op(x, [5, 7], *extra, 1e-39, 1.0))


def test_backward_dtype_is_rejected_with_zero_work(vulkan_backend):
    grad = _data((1, 2, 6, 8), 76).to(vulkan_backend).double()
    _rejects("Vulkan upsample_nearest2d_backward requires a strided float32 vk:0 tensor",
             lambda: aten.upsample_nearest2d_backward(grad, [6, 8], [1, 2, 3, 4]))


@pytest.mark.parametrize("name", ["upsample_nearest2d.out", "_upsample_nearest_exact2d.out", "upsample_bilinear2d.out"])
def test_out_forms_remain_unregistered(vulkan_backend, name):
    base, _ = name.split(".")
    x = _data((1, 2, 3, 4), 77).to(vulkan_backend)
    out = torch.empty(1, 2, 6, 8).to(vulkan_backend)
    extra = (False,) if base == "upsample_bilinear2d" else ()
    with pytest.raises((RuntimeError, NotImplementedError)):
        getattr(aten, base).out(x, [6, 8], *extra, out=out)


# --- composition -------------------------------------------------------------------


@pytest.mark.parametrize("mode,kw", MODE_KW[:1] + MODE_KW[2:3], ids=["nearest", "bilinear"])
def test_conv_upsample_loss_training_matches_cpu(vulkan_backend, mode, kw):
    torch.manual_seed(5)
    cpu_model = nn.Sequential(
        nn.Conv2d(2, 3, 3, padding=1), nn.Upsample(scale_factor=2, mode=mode, **kw)
    )
    vk_model = nn.Sequential(
        nn.Conv2d(2, 3, 3, padding=1), nn.Upsample(scale_factor=2, mode=mode, **kw)
    )
    vk_model.load_state_dict(cpu_model.state_dict())
    vk_model.to(vulkan_backend)
    x = _data((2, 2, 5, 5), 81)
    target = _data((2, 3, 10, 10), 82)
    cpu_opt = torch.optim.SGD(cpu_model.parameters(), lr=0.05)
    vk_opt = torch.optim.SGD(vk_model.parameters(), lr=0.05)
    losses = []
    for _ in range(3):
        cpu_opt.zero_grad()
        cpu_loss = F.mse_loss(cpu_model(x), target)
        cpu_loss.backward()
        cpu_opt.step()
        vk_opt.zero_grad()
        vk_loss = F.mse_loss(vk_model(x.to(vulkan_backend)), target.to(vulkan_backend))
        vk_loss.backward()
        vk_opt.step()
        pytorch_vulkan._C.synchronize()
        losses.append(vk_loss.item())
        assert vk_loss.item() == pytest.approx(cpu_loss.item(), rel=1e-4, abs=1e-5)
    assert losses[-1] < losses[0]
    for (name, want), (_, got) in zip(cpu_model.state_dict().items(), vk_model.state_dict().items()):
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-5)
