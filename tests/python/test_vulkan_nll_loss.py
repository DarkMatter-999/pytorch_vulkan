"""NLL / cross-entropy (2-D class-index targets) forward and first-order backward on Vulkan.

Acceptance goes through nn.CrossEntropyLoss / F.cross_entropy / nn.NLLLoss / F.nll_loss
with a normal backward; direct aten calls supplement them.  CPU (PyTorch 2.4.0) is the
reference.  The forward is a pure gather: ``none`` is bitwise equal to CPU.  ``sum`` and
``total_weight`` are bitwise equal to the fixed-order oracle in nll_loss_reference.py and
within PARITY_BOUND * u * sum|terms| of CPU (CPU's own order is unspecified).  Higher-order
derivatives, forward AD, torch.func, vmap, label smoothing and K-dim inputs are not
claimed and are not asserted either way.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_vulkan
import nll_loss_reference as ref

aten = torch.ops.aten
IGNORES = [-100, 2, 17, -1, 255, 2 ** 40, -2 ** 63]
REDUCTIONS = [0, 1, 2]
REDUCTION_NAMES = {0: "none", 1: "mean", 2: "sum"}


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _sync():
    pytorch_vulkan._C.synchronize()


def _zero_work():
    return pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _rejects(pattern, call, exc=(RuntimeError, ValueError)):
    _sync()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(exc, match=pattern):
        call()
    assert _zero_work()


def _case(n, c, ignore=-100, seed=0, ignored=2, weighted=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, generator=g) * 3
    t = torch.randint(0, c, (n,), generator=g)
    for i in range(min(ignored, max(n - 1, 0))):  # all-ignored has dedicated tests
        t[(i * 3) % n] = ignore
    w = (torch.rand(c, generator=g) * 3 + 0.05) if weighted else None
    return x, t, w


def _bits_equal(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a.view(np.uint32) == b.view(np.uint32))))


def _aten_forward(device, lp, target, weight, reduction, ignore):
    out, tw = aten.nll_loss_forward(
        lp.to(device), target.to(device), None if weight is None else weight.to(device),
        reduction, ignore)
    if device != "cpu":
        _sync()
    return out.cpu(), tw.cpu()


def _aten_backward(device, g, lp, target, weight, reduction, ignore, total):
    grad = aten.nll_loss_backward(
        g.to(device), lp.to(device), target.to(device),
        None if weight is None else weight.to(device), reduction, ignore, total.to(device))
    if device != "cpu":
        _sync()
    return grad.cpu()


def _upstream(shape, seed=11):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) + 1.5


# --- reference model self-checks (CPU only) -----------------------------------------

@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 2047, 2048, 2049, 100000])
def test_oracle_matches_float64_within_design_bound(n):
    rng = np.random.default_rng(n)
    terms = (rng.random(n) * 4 + 0.01).astype(np.float32)
    exact = terms.astype(np.float64).sum()
    error = abs(float(ref.reduce_oracle(terms)) - exact) / (ref.U * np.abs(terms).sum())
    assert error <= ref.ORACLE_DESIGN_BOUND


def test_oracle_forward_matches_cpu_formulas():
    x, t, w = _case(37, 6, weighted=True)
    lp = torch.log_softmax(x, 1)
    for reduction in REDUCTIONS:
        cpu_out, cpu_tw = aten.nll_loss_forward(lp, t, w, reduction, -100)
        out, tw = ref.oracle_forward(lp.numpy(), t.numpy(), w.numpy(), reduction, -100)
        sabs, wabs = ref.s_abs(lp.numpy(), t.numpy(), w.numpy(), -100)
        if reduction == 0:
            assert _bits_equal(out, cpu_out.numpy())
        else:
            assert ref.within_parity(out, cpu_out.item(), sabs if reduction == 2 else sabs / max(float(tw), 1))
            assert ref.within_parity(tw, cpu_tw.item(), wabs)


# --- finding A: forward is a gather, not a re-applied log-softmax -------------------

def test_nll_loss_does_not_reapply_log_softmax(vulkan_backend):
    values = torch.tensor([[2.0, -1.0, 0.5], [0.25, 4.0, -3.0]])
    target = torch.tensor([1, 2])
    for reduction in ("none", "mean", "sum"):
        expected = F.nll_loss(values, target, reduction=reduction)
        actual = F.nll_loss(values.to(vulkan_backend), target.to(vulkan_backend),
                            reduction=reduction)
        _sync()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-6)


# --- ordinary API: nll_loss on arbitrary values -------------------------------------

def _nll_api(device, x, t, w, reduction, ignore, module):
    leaf = x.detach().to(device).requires_grad_()
    wt = None if w is None else w.to(device)
    tt = t.to(device)
    if module:
        loss = nn.NLLLoss(weight=wt, ignore_index=ignore, reduction=REDUCTION_NAMES[reduction])(leaf, tt)
    else:
        loss = F.nll_loss(leaf, tt, weight=wt, ignore_index=ignore, reduction=REDUCTION_NAMES[reduction])
    upstream = _upstream(loss.shape)
    loss.backward(upstream.to(device))
    if device != "cpu":
        _sync()
    return loss, leaf.grad.cpu()


def _compare_nll(x, t, w, reduction, ignore, module=False, device="vk:0"):
    cpu_loss, cpu_grad = _nll_api("cpu", x, t, w, reduction, ignore, module)
    vk_loss, vk_grad = _nll_api(device, x, t, w, reduction, ignore, module)
    assert vk_loss.grad_fn.name() == "NllLossBackward0"
    assert vk_loss.device == torch.device(device)
    out = vk_loss.detach().cpu()
    lp = x.numpy()
    sabs, _ = ref.s_abs(lp, t.numpy(), None if w is None else w.numpy(), ignore)
    if reduction == 0:
        assert _bits_equal(out.numpy(), cpu_loss.detach().numpy())
    elif reduction == 2:
        assert ref.within_parity(out.item(), cpu_loss.item(), sabs)
    else:
        total = float(aten.nll_loss_forward(x, t, w, 1, ignore)[1])
        bound = sabs / total if total else 0.0
        assert ref.within_parity(out.item(), cpu_loss.item(), bound if total else 0.0)
    if reduction == 1 and w is not None and not torch.isfinite(cpu_grad).all():
        assert torch.isnan(vk_grad).equal(torch.isnan(cpu_grad))
        return
    if reduction == 1:
        assert ref.ulp_distance(vk_grad.numpy(), cpu_grad.numpy()).max() <= ref.BACKWARD_MEAN_ULPS
    else:
        assert _bits_equal(vk_grad.numpy(), cpu_grad.numpy())


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (4, 5), (7, 33), (257, 3)])
def test_nll_loss_shapes_match_cpu(vulkan_backend, reduction, weighted, shape):
    x, t, w = _case(*shape, weighted=weighted, seed=shape[0] * 7 + shape[1])
    _compare_nll(x, t, w, reduction, -100)


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
@pytest.mark.parametrize("ignore", IGNORES)
def test_nll_loss_ignore_index_full_int64_matches_cpu(vulkan_backend, reduction, weighted, ignore):
    x, t, w = _case(9, 5, ignore=ignore, weighted=weighted, seed=3)
    _compare_nll(x, t, w, reduction, ignore)


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
def test_nll_module_matches_functional(vulkan_backend, reduction):
    x, t, w = _case(6, 4, weighted=True, seed=21)
    _compare_nll(x, t, w, reduction, -100, module=True)


# --- ordinary API: cross entropy -----------------------------------------------------

def _ce(device, x, t, w, reduction, ignore, module):
    leaf = x.detach().to(device).requires_grad_()
    wt = None if w is None else w.to(device)
    tt = t.to(device)
    if module:
        loss = nn.CrossEntropyLoss(weight=wt, ignore_index=ignore, reduction=REDUCTION_NAMES[reduction])(leaf, tt)
    else:
        loss = F.cross_entropy(leaf, tt, weight=wt, ignore_index=ignore, reduction=REDUCTION_NAMES[reduction])
    loss.backward(_upstream(loss.shape).to(device))
    if device != "cpu":
        _sync()
    return loss, leaf.grad.cpu()


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (4, 5), (7, 33), (257, 3)])
@pytest.mark.parametrize("module", [False, True], ids=["functional", "module"])
def test_cross_entropy_matches_cpu(vulkan_backend, reduction, weighted, shape, module):
    x, t, w = _case(*shape, weighted=weighted, seed=shape[0] + 13 * shape[1])
    cpu_loss, cpu_grad = _ce("cpu", x, t, w, reduction, -100, module)
    vk_loss, vk_grad = _ce(vulkan_backend, x, t, w, reduction, -100, module)
    assert vk_loss.grad_fn.name() == "NllLossBackward0"
    torch.testing.assert_close(vk_loss.detach().cpu(), cpu_loss.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vk_grad, cpu_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ignore", IGNORES)
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
def test_cross_entropy_ignore_index_matches_cpu(vulkan_backend, ignore, reduction):
    x, t, w = _case(8, 5, ignore=ignore, weighted=True, seed=4)
    cpu_loss, cpu_grad = _ce("cpu", x, t, w, reduction, ignore, False)
    vk_loss, vk_grad = _ce(vulkan_backend, x, t, w, reduction, ignore, False)
    torch.testing.assert_close(vk_loss.detach().cpu(), cpu_loss.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vk_grad, cpu_grad, rtol=1e-5, atol=1e-5)


# --- unbatched 1-D input -------------------------------------------------------------

@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
def test_unbatched_input_matches_cpu(vulkan_backend, reduction, weighted):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(7, generator=g)
    w = torch.rand(7, generator=g) + 0.1 if weighted else None
    t = torch.tensor(3)
    for api in (F.nll_loss, F.cross_entropy):
        def run(device):
            leaf = x.detach().clone().to(device).requires_grad_()
            loss = api(leaf, t.to(device), weight=None if w is None else w.to(device),
                       reduction=REDUCTION_NAMES[reduction])
            assert loss.shape == ()
            loss.backward()
            if device != "cpu":
                _sync()
            return loss.detach().cpu(), leaf.grad.cpu()
        (cl, cg), (vl, vg) = run("cpu"), run(vulkan_backend)
        torch.testing.assert_close(vl, cl, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vg, cg, rtol=1e-5, atol=1e-5)


def test_unbatched_ignored_mean_is_nan_with_zero_gradient(vulkan_backend):
    x = torch.randn(5)
    leaf = x.detach().clone().to(vulkan_backend).requires_grad_()
    loss = F.cross_entropy(leaf, torch.tensor(2).to(vulkan_backend), ignore_index=2)
    loss.backward()
    _sync()
    assert torch.isnan(loss.cpu())
    assert leaf.grad.cpu().abs().sum() == 0


# --- total_weight and the edge table -------------------------------------------------

@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
def test_total_weight_returned_like_cpu(vulkan_backend, weighted):
    x, t, w = _case(13, 5, weighted=weighted, seed=8)
    lp = torch.log_softmax(x, 1)
    for reduction in REDUCTIONS:
        cpu_out, cpu_tw = _aten_forward("cpu", lp, t, w, reduction, -100)
        vk_out, vk_tw = _aten_forward(vulkan_backend, lp, t, w, reduction, -100)
        assert vk_tw.shape == ()
        if reduction == 0:
            assert vk_tw.item() == 0.0 and cpu_tw.item() == 0.0
        else:
            _, wabs = ref.s_abs(lp.numpy(), t.numpy(), None if w is None else w.numpy(), -100)
            assert ref.within_parity(vk_tw.item(), cpu_tw.item(), wabs)


@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
def test_all_ignored(vulkan_backend, weighted):
    x = torch.randn(4, 5)
    w = torch.rand(5) + 0.2 if weighted else None
    t = torch.full((4,), 2)
    lp = torch.log_softmax(x, 1)
    mean, tw = _aten_forward(vulkan_backend, lp, t, w, 1, 2)
    assert torch.isnan(mean) and tw.item() == 0.0
    total, tw = _aten_forward(vulkan_backend, lp, t, w, 2, 2)
    assert total.item() == 0.0 and tw.item() == 0.0
    none, tw = _aten_forward(vulkan_backend, lp, t, w, 0, 2)
    assert none.shape == (4,) and none.abs().sum() == 0 and tw.item() == 0.0
    for reduction, upstream in ((1, torch.tensor(1.5)), (2, torch.tensor(1.5)), (0, torch.ones(4))):
        grad = _aten_backward(vulkan_backend, upstream, lp, t, w, reduction, 2, torch.tensor(0.0))
        assert grad.shape == (4, 5) and torch.count_nonzero(grad) == 0 and not torch.isnan(grad).any()


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
def test_empty_batch(vulkan_backend, reduction):
    lp = torch.empty(0, 5)
    t = torch.empty(0, dtype=torch.int64)
    cpu_out, cpu_tw = aten.nll_loss_forward(lp, t, None, reduction, -100)
    out, tw = _aten_forward(vulkan_backend, lp, t, None, reduction, -100)
    assert out.shape == cpu_out.shape and tw.item() == 0.0
    if reduction == 1:
        assert torch.isnan(out)
    else:
        assert torch.equal(out, cpu_out)
    upstream = torch.ones(0) if reduction == 0 else torch.tensor(1.0)
    grad = _aten_backward(vulkan_backend, upstream, lp, t, None, reduction, -100, torch.tensor(0.0))
    assert grad.shape == (0, 5)


def test_zero_weight_class_mean_and_sum(vulkan_backend):
    x = torch.randn(4, 5)
    w = torch.rand(5) + 0.1
    w[2] = 0.0
    t = torch.full((4,), 2)
    lp = torch.log_softmax(x, 1)
    mean, tw = _aten_forward(vulkan_backend, lp, t, w, 1, -100)
    assert torch.isnan(mean) and tw.item() == 0.0
    total, _ = _aten_forward(vulkan_backend, lp, t, w, 2, -100)
    assert total.item() == 0.0
    grad = _aten_backward(vulkan_backend, torch.tensor(1.0), lp, t, w, 1, -100, torch.tensor(0.0))
    cpu_grad = aten.nll_loss_backward(torch.tensor(1.0), lp, t, w, 1, -100, torch.tensor(0.0))
    assert torch.equal(torch.isnan(grad), torch.isnan(cpu_grad))


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")], ids=["negative", "nan", "inf"])
def test_irregular_weights_follow_cpu(vulkan_backend, bad):
    x = torch.randn(6, 5)
    w = torch.rand(5) + 0.1
    w[1] = bad
    t = torch.tensor([0, 1, 2, 1, 4, 3])
    lp = torch.log_softmax(x, 1)
    for reduction in (1, 2):
        cpu_out, cpu_tw = _aten_forward("cpu", lp, t, w, reduction, -100)
        out, tw = _aten_forward(vulkan_backend, lp, t, w, reduction, -100)
        for a, b in ((out, cpu_out), (tw, cpu_tw)):
            if not torch.isfinite(b):
                assert torch.equal(torch.isnan(a), torch.isnan(b)) and (torch.isnan(a) or a.item() == b.item())
            else:
                sabs, wabs = ref.s_abs(lp.numpy(), t.numpy(), w.numpy(), -100)
                assert ref.within_parity(a.item(), b.item(), max(sabs, wabs))


def test_weights_never_index_the_ignored_label(vulkan_backend):
    x = torch.randn(4, 5)
    w = torch.rand(5) + 0.1
    t = torch.tensor([0, 17, 4, 17])  # ignore_index outside the class range
    lp = torch.log_softmax(x, 1)
    cpu = _aten_forward("cpu", lp, t, w, 1, 17)
    vk = _aten_forward(vulkan_backend, lp, t, w, 1, 17)
    assert abs(vk[0].item() - cpu[0].item()) < 1e-6 and abs(vk[1].item() - cpu[1].item()) < 1e-6


def test_negative_zero_sign_matches_cpu(vulkan_backend):
    lp = torch.zeros(2, 3)
    t = torch.tensor([0, 1])
    for w in (None, torch.ones(3)):
        out, _ = _aten_forward(vulkan_backend, lp, t, w, 0, -100)
        cpu, _ = _aten_forward("cpu", lp, t, w, 0, -100)
        assert torch.equal(torch.signbit(out), torch.signbit(cpu))


# --- backward through direct aten (bitwise sum/none, <=4 ulp mean) --------------------

@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
def test_aten_backward_matches_cpu(vulkan_backend, weighted, reduction):
    x, t, w = _case(11, 6, weighted=weighted, seed=19)
    lp = torch.log_softmax(x, 1)
    _, cpu_tw = _aten_forward("cpu", lp, t, w, reduction, -100)
    upstream = _upstream((11,) if reduction == 0 else ())
    cpu = _aten_backward("cpu", upstream, lp, t, w, reduction, -100, cpu_tw)
    vk = _aten_backward(vulkan_backend, upstream, lp, t, w, reduction, -100, cpu_tw)
    if reduction == 1:
        assert ref.ulp_distance(vk.numpy(), cpu.numpy()).max() <= ref.BACKWARD_MEAN_ULPS
    else:
        assert _bits_equal(vk.numpy(), cpu.numpy())
    assert _bits_equal(vk.numpy(), ref.oracle_backward(
        upstream.numpy(), lp.shape, t.numpy(), None if w is None else w.numpy(), reduction,
        -100, cpu_tw.item())) or reduction == 1


def test_aten_backward_accepts_expanded_grad_output(vulkan_backend):
    x, t, _ = _case(5, 4, seed=2)
    lp = torch.log_softmax(x, 1)
    ones = torch.ones(1).expand(5)
    cpu = _aten_backward("cpu", ones, lp, t, None, 0, -100, torch.tensor(0.0))
    vk = _aten_backward(vulkan_backend, ones, lp, t, None, 0, -100, torch.tensor(0.0))
    assert _bits_equal(vk.numpy(), cpu.numpy())


# --- oracle-bitwise reduction, determinism, accuracy ---------------------------------

@pytest.mark.parametrize("n", [1, 7, 255, 256, 257, 2047, 2048, 2049, 5000, 20000])
@pytest.mark.parametrize("weighted", [False, True], ids=["noweight", "weight"])
def test_reduction_bitwise_matches_oracle(vulkan_backend, n, weighted):
    x, t, w = _case(n, 7, weighted=weighted, seed=n)
    lp = torch.log_softmax(x, 1)
    for reduction in (1, 2):
        out, tw = _aten_forward(vulkan_backend, lp, t, w, reduction, -100)
        oo, otw = ref.oracle_forward(lp.numpy(), t.numpy(), None if w is None else w.numpy(), reduction, -100)
        assert _bits_equal(tw.numpy(), otw)
        if reduction == 2:
            assert _bits_equal(out.numpy(), oo)
        else:
            assert ref.ulp_distance(out.numpy(), oo).max() <= ref.MEAN_ULPS


def test_reduction_is_deterministic_across_runs(vulkan_backend):
    x, t, w = _case(30000, 9, weighted=True, seed=77)
    lp = torch.log_softmax(x, 1)
    first = [v.numpy().tobytes() for v in _aten_forward(vulkan_backend, lp, t, w, 1, -100)]
    for _ in range(4):
        assert [v.numpy().tobytes() for v in _aten_forward(vulkan_backend, lp, t, w, 1, -100)] == first


@pytest.mark.parametrize("n", [100000, 1000000])
def test_large_batches_stay_within_the_float32_bound(vulkan_backend, n):
    g = torch.Generator().manual_seed(n)
    lp = torch.log_softmax(torch.randn(n, 3, generator=g), 1)
    t = torch.randint(0, 3, (n,), generator=g)
    out, tw = _aten_forward(vulkan_backend, lp, t, None, 2, -100)
    cpu, _ = _aten_forward("cpu", lp, t, None, 2, -100)
    terms = (-lp.numpy()[np.arange(n), t.numpy()]).astype(np.float64)
    error = abs(float(out) - terms.sum()) / (ref.U * np.abs(terms).sum())
    assert error <= ref.ORACLE_DESIGN_BOUND
    assert ref.within_parity(out.item(), cpu.item(), np.abs(terms).sum())
    assert tw.item() == float(n)


# --- invalid targets: IndexError at the call, never a poisoned value -------------------

def _expect_index_error(call, value):
    _sync()
    with pytest.raises(IndexError, match=rf"Target {value} is out of bounds"):
        call()


@pytest.mark.parametrize("bad", [5, -1, 2 ** 40, -2 ** 63, 2 ** 63 - 1, 6])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_out_of_range_target_raises_like_cpu(vulkan_backend, bad, position):
    lp = torch.log_softmax(torch.randn(6, 5), 1)
    t = torch.tensor([0, 1, 2, 3, 4, 0])
    t[{"first": 0, "middle": 3, "last": 5}[position]] = bad
    with pytest.raises(IndexError, match=rf"Target {bad} is out of bounds"):
        aten.nll_loss_forward(lp, t, None, 1, -100)
    for reduction in REDUCTIONS:
        _expect_index_error(lambda: aten.nll_loss_forward(lp.to(vulkan_backend), t.to(vulkan_backend), None, reduction, -100), bad)
    good = torch.tensor([0, 1, 2, 3, 4, 0]).to(vulkan_backend)
    out, _ = aten.nll_loss_forward(lp.to(vulkan_backend), good, None, 1, -100)  # queue still healthy
    _sync()
    assert torch.isfinite(out.cpu())


def test_out_of_range_target_raises_through_cross_entropy(vulkan_backend):
    x = torch.randn(4, 5).to(vulkan_backend).requires_grad_()
    t = torch.tensor([0, 9, 2, 3]).to(vulkan_backend)
    _expect_index_error(lambda: F.cross_entropy(x, t), 9)


@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_NAMES.get)
def test_out_of_range_target_raises_in_direct_backward(vulkan_backend, reduction):
    lp = torch.log_softmax(torch.randn(4, 5), 1)
    t = torch.tensor([0, 7, 2, 3])
    upstream = torch.ones(4) if reduction == 0 else torch.tensor(1.0)
    if reduction != 0:
        with pytest.raises(IndexError, match="Target 7 is out of bounds"):
            aten.nll_loss_backward(upstream, lp, t, None, reduction, -100, torch.tensor(4.0))
    else:
        # CPU's nll_loss_backward does not validate targets for reduction=none (it returns a
        # zero row); ordinary APIs never reach this because forward always raises first.
        # Vulkan is deliberately stricter: an invalid label is never turned into a result.
        assert torch.count_nonzero(aten.nll_loss_backward(upstream, lp, t, None, 0, -100, torch.tensor(0.0))[1]) == 0
    _expect_index_error(lambda: aten.nll_loss_backward(
        upstream.to(vulkan_backend), lp.to(vulkan_backend), t.to(vulkan_backend), None, reduction,
        -100, torch.tensor(4.0).to(vulkan_backend)), 7)


def test_default_ignore_index_does_not_hide_minus_one(vulkan_backend):
    lp = torch.log_softmax(torch.randn(3, 4), 1)
    t = torch.tensor([0, -1, 2])
    _expect_index_error(lambda: aten.nll_loss_forward(lp.to(vulkan_backend), t.to(vulkan_backend), None, 1, -100), -1)
    out, tw = _aten_forward(vulkan_backend, lp, t, None, 1, -1)
    cpu = aten.nll_loss_forward(lp, t, None, 1, -1)
    assert abs(out.item() - cpu[0].item()) < 1e-6 and tw.item() == 2.0


def test_invalid_value_at_ignored_position_is_valid(vulkan_backend):
    lp = torch.log_softmax(torch.randn(3, 4), 1)
    t = torch.tensor([0, 99, 2])
    out, tw = _aten_forward(vulkan_backend, lp, t, None, 1, 99)
    assert tw.item() == 2.0 and torch.isfinite(out)


def test_lowest_invalid_index_is_reported_regardless_of_scheduling(vulkan_backend):
    n = 100000
    g = torch.Generator().manual_seed(1)
    lp = torch.log_softmax(torch.randn(n, 4, generator=g), 1)
    t = torch.randint(0, 4, (n,), generator=g)
    bad_positions = torch.randperm(n, generator=g)[: n // 2]
    t[bad_positions] = torch.randint(10, 10 ** 6, (n // 2,), generator=g)
    expected = ref.lowest_invalid(t.numpy(), 4, -100)
    lp_v, t_v = lp.to(vulkan_backend), t.to(vulkan_backend)
    for _ in range(25):
        _sync()
        with pytest.raises(IndexError, match=rf"Target {expected[1]} is out of bounds"):
            aten.nll_loss_forward(lp_v, t_v, None, 2, -100)


def test_status_is_not_stale_across_calls(vulkan_backend):
    lp = torch.log_softmax(torch.randn(4, 5), 1).to(vulkan_backend)
    good = torch.tensor([0, 1, 2, 3]).to(vulkan_backend)
    bad = torch.tensor([0, 1, 2, 8]).to(vulkan_backend)
    for _ in range(3):
        out, _ = aten.nll_loss_forward(lp, good, None, 1, -100)
        _sync()
        assert torch.isfinite(out.cpu())
        _expect_index_error(lambda: aten.nll_loss_forward(lp, bad, None, 1, -100), 8)


@pytest.mark.parametrize("value", [-1, 0, 255, 2 ** 31, 2 ** 40, -2 ** 63, 2 ** 63 - 1])
def test_any_int64_label_uploads_and_is_ignored_when_it_is_ignore_index(vulkan_backend, value):
    lp = torch.log_softmax(torch.randn(3, 4), 1)
    t = torch.tensor([1, value, 3])
    out, tw = _aten_forward(vulkan_backend, lp, t, None, 1, value)
    cpu = aten.nll_loss_forward(lp, t, None, 1, value)
    assert tw.item() == cpu[1].item()
    assert abs(out.item() - cpu[0].item()) < 1e-6


# --- rejections: documented Vulkan limitations, no Vulkan work -------------------------

def test_uint8_targets_are_a_documented_vulkan_limitation(vulkan_backend):
    target = torch.tensor([0, 1, 2, 0], dtype=torch.uint8)
    lp = torch.log_softmax(torch.randn(4, 3), 1)
    assert torch.isfinite(aten.nll_loss_forward(lp, target, None, 1, -100)[0])  # CPU accepts uint8
    lp_vk = lp.to(vulkan_backend)
    _rejects(".", lambda: target.to(vulkan_backend))
    _rejects(".", lambda: aten.nll_loss_forward(lp_vk, target, None, 1, -100))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8])
def test_other_integer_target_dtypes_cannot_be_uploaded(vulkan_backend, dtype):
    # Same documented limitation as uint8: only int64 labels exist on vk:0.
    target = torch.tensor([0, 1, 1, 0]).to(dtype)
    _rejects("supports only float32, bool, and traced int64", lambda: target.to(vulkan_backend))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_float_and_bool_targets_are_rejected_by_the_op(vulkan_backend, dtype):
    lp = torch.log_softmax(torch.randn(4, 3), 1).to(vulkan_backend)
    target = torch.tensor([0, 1, 1, 0]).to(dtype).to(vulkan_backend)
    _rejects("int64", lambda: aten.nll_loss_forward(lp, target, None, 1, -100))


def test_shape_and_weight_errors_match_cpu_wording(vulkan_backend):
    lp = torch.log_softmax(torch.randn(4, 5), 1).to(vulkan_backend)
    t = torch.tensor([0, 1, 2, 3]).to(vulkan_backend)
    short_weight = torch.rand(4).to(vulkan_backend)
    one_label = torch.tensor([1]).to(vulkan_backend)
    _rejects("size mismatch", lambda: aten.nll_loss_forward(lp, t[:3], None, 1, -100))
    _rejects("weight tensor should be defined either for all 5 classes",
             lambda: aten.nll_loss_forward(lp, t, short_weight, 1, -100))
    _rejects("size mismatch", lambda: aten.nll_loss_forward(lp[0], one_label, None, 1, -100))


def test_unsupported_shapes_and_reductions_are_rejected(vulkan_backend):
    t = torch.tensor([0, 1]).to(vulkan_backend)
    rank3 = torch.zeros(2, 3, 4).to(vulkan_backend)
    plain = torch.zeros(2, 3).to(vulkan_backend)
    no_classes = torch.zeros(2, 0).to(vulkan_backend)
    _rejects("1-D or 2-D input", lambda: aten.nll_loss_forward(rank3, t, None, 1, -100))
    _rejects("reduction", lambda: aten.nll_loss_forward(plain, t, None, 3, -100))
    _rejects("class", lambda: aten.nll_loss_forward(no_classes, t, None, 1, -100))


def test_reduction_larger_than_the_supported_element_count_is_rejected(vulkan_backend):
    n = ref.MAX_ELEMENTS + 1
    lp = torch.zeros(1, 1).to(vulkan_backend).expand(n, 1)
    t = torch.zeros(1, dtype=torch.int64).to(vulkan_backend).expand(n)
    _rejects("4194304|element", lambda: aten.nll_loss_forward(lp, t, None, 2, -100))


def test_float64_inputs_are_rejected(vulkan_backend):
    lp = torch.log_softmax(torch.randn(4, 3), 1).to(vulkan_backend)
    t = torch.tensor([0, 1, 1, 0]).to(vulkan_backend)
    wide_weight = torch.rand(3).to(vulkan_backend).double()
    _rejects(".", lambda: aten.nll_loss_forward(lp, t, wide_weight, 1, -100))


# --- strided operands ---------------------------------------------------------------

def test_noncontiguous_inputs_weights_and_slices_match_cpu(vulkan_backend):
    base = torch.randn(5, 8)
    lp_cpu = torch.log_softmax(base[:, ::2], 1)             # non-contiguous view
    t_cpu = torch.tensor([0, 3, 1, 2, 0])
    wbase = torch.rand(8) + 0.1
    w_cpu = wbase[::2]
    expected = _aten_forward("cpu", lp_cpu, t_cpu, w_cpu, 2, -100)
    lp = torch.log_softmax(base.to(vulkan_backend)[:, ::2], 1)
    out, tw = aten.nll_loss_forward(lp, t_cpu.to(vulkan_backend), wbase.to(vulkan_backend)[::2], 2, -100)
    _sync()
    sabs, wabs = ref.s_abs(lp_cpu.numpy(), t_cpu.numpy(), w_cpu.numpy(), -100)
    assert ref.within_parity(out.cpu().item(), expected[0].item(), sabs)
    assert ref.within_parity(tw.cpu().item(), expected[1].item(), wabs)
    sliced = torch.log_softmax(torch.randn(10, 4), 1)
    t10 = torch.randint(0, 4, (10,))
    sl_out, _ = aten.nll_loss_forward(sliced.to(vulkan_backend)[2:8], t10.to(vulkan_backend)[2:8], None, 0, -100)
    _sync()
    assert _bits_equal(sl_out.cpu().numpy(), aten.nll_loss_forward(sliced[2:8], t10[2:8], None, 0, -100)[0].numpy())


# --- training and composition -----------------------------------------------------------

def test_small_classifier_training_matches_cpu(vulkan_backend):
    torch.manual_seed(31)
    cpu_model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    vk_model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    vk_model.load_state_dict(cpu_model.state_dict())
    vk_model = vk_model.to(vulkan_backend)
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])
    x = torch.randn(10, 6)
    y = torch.randint(0, 4, (10,))
    y[3] = -7
    cpu_opt = torch.optim.SGD(cpu_model.parameters(), lr=0.05)
    vk_opt = torch.optim.SGD(vk_model.parameters(), lr=0.05)
    cpu_loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=-7)
    vk_loss_fn = nn.CrossEntropyLoss(weight=weight.to(vulkan_backend), ignore_index=-7)
    xv, yv = x.to(vulkan_backend), y.to(vulkan_backend)
    for _ in range(5):
        cpu_opt.zero_grad()
        cpu_loss = cpu_loss_fn(cpu_model(x), y)
        cpu_loss.backward()
        cpu_opt.step()
        vk_opt.zero_grad()
        vk_loss = vk_loss_fn(vk_model(xv), yv)
        vk_loss.backward()
        vk_opt.step()
        _sync()
        assert abs(vk_loss.item() - cpu_loss.item()) < 1e-4
    for a, b in zip(vk_model.parameters(), cpu_model.parameters()):
        torch.testing.assert_close(a.detach().cpu(), b.detach(), rtol=1e-4, atol=1e-4)


def test_soft_target_cross_entropy_by_composition(vulkan_backend):
    # Verify-only: soft targets reach only qualified float ops (no new kernel is claimed).
    g = torch.Generator().manual_seed(9)
    x = torch.randn(5, 4, generator=g)
    soft = torch.softmax(torch.randn(5, 4, generator=g), 1)
    cpu_leaf = x.clone().requires_grad_()
    cpu_loss = F.cross_entropy(cpu_leaf, soft)
    cpu_loss.backward()
    leaf = x.to(vulkan_backend).requires_grad_()
    loss = F.cross_entropy(leaf, soft.to(vulkan_backend))
    loss.backward()
    _sync()
    torch.testing.assert_close(loss.detach().cpu(), cpu_loss.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(leaf.grad.cpu(), cpu_leaf.grad, rtol=1e-5, atol=1e-5)


# --- log_softmax evidence beyond the (2,2) witness (CE depends on it) --------------------

@pytest.mark.parametrize("shape,dim", [((7,), 0), ((7,), -1), ((4, 5), 1), ((4, 5), -1),
                                       ((64, 10), 1), ((257, 33), 1), ((3, 1), 1)])
def test_log_softmax_widened_shapes_match_cpu(vulkan_backend, shape, dim):
    g = torch.Generator().manual_seed(sum(shape))
    x = torch.randn(shape, generator=g) * 4
    cpu_leaf = x.clone().requires_grad_()
    cpu = F.log_softmax(cpu_leaf, dim)
    up = torch.randn(shape, generator=g)
    cpu.backward(up)
    leaf = x.to(vulkan_backend).requires_grad_()
    out = F.log_softmax(leaf, dim)
    out.backward(up.to(vulkan_backend))
    _sync()
    torch.testing.assert_close(out.detach().cpu(), cpu.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(leaf.grad.cpu(), cpu_leaf.grad, rtol=1e-5, atol=1e-5)
