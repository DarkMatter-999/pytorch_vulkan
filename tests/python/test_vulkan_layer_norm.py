"""LayerNorm forward/backward on Vulkan against the PyTorch 2.4.0 CPU reference.

Acceptance goes through nn.LayerNorm and F.layer_norm with a normal backward;
the direct aten tests supplement them. Higher-order derivatives, forward AD and
torch.func transforms are not claimed here.
"""
import pytest
import torch
import torch.nn.functional as F

import pytorch_vulkan

RTOL, ATOL = 2e-5, 3e-6


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _zero_work():
    return pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _data(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _run(device, x, normalized_shape, weight, bias, eps, upstream, expand_upstream=False):
    """Forward + backward through F.layer_norm on `device`; returns CPU tensors."""
    x = x.detach().to(device).requires_grad_()
    w = None if weight is None else weight.detach().to(device).requires_grad_()
    b = None if bias is None else bias.detach().to(device).requires_grad_()
    y = F.layer_norm(x, normalized_shape, w, b, eps)
    gradient = upstream.to(device)
    if expand_upstream:
        gradient = upstream.to(device).reshape(()).expand(y.shape)
    y.backward(gradient)
    if device != "cpu":
        pytorch_vulkan._C.synchronize()
    grads = tuple(None if t is None else t.grad.cpu() for t in (x, w, b))
    return (y.detach().cpu(), *grads)


def _compare(vk, cpu, rtol=RTOL, atol=ATOL):
    for got, want in zip(vk, cpu):
        assert (got is None) == (want is None)
        if want is not None:
            torch.testing.assert_close(got, want, rtol=rtol, atol=atol, equal_nan=True)


def _params(normalized_shape, affine, seed):
    if affine in ("both", "weight"):
        weight = _data(normalized_shape, seed + 1) + 1.0
    else:
        weight = None
    bias = _data(normalized_shape, seed + 2) if affine in ("both", "bias") else None
    return weight, bias


SHAPES = [
    ((5, 8), (8,)),
    ((3, 4, 6), (6,)),
    ((3, 4, 6), (4, 6)),
    ((2, 3, 4, 5), (5,)),
    ((2, 3, 4, 5), (3, 4, 5)),
    ((2, 3, 4), (2, 3, 4)),
    ((1, 7), (7,)),
    ((4,), (4,)),
]


@pytest.mark.parametrize("affine", ["both", "weight", "bias", "none"])
@pytest.mark.parametrize("shape, normalized_shape", SHAPES)
def test_functional_layer_norm_forward_and_backward_match_cpu(
    vulkan_backend, shape, normalized_shape, affine
):
    x = _data(shape, 11)
    weight, bias = _params(normalized_shape, affine, 20)
    upstream = _data(shape, 31)
    cpu = _run("cpu", x, normalized_shape, weight, bias, 1e-5, upstream)
    vk = _run(vulkan_backend, x, normalized_shape, weight, bias, 1e-5, upstream)
    _compare(vk, cpu)


@pytest.mark.parametrize("affine", ["both", "none"])
def test_single_element_normalization_matches_the_float64_reference(vulkan_backend, affine):
    # N == 1 has zero variance, so dX and dW are analytically zero while rstd is
    # ~316; float32 CPU and Vulkan both return only cancellation noise (<~3e-5).
    x = _data((6, 1), 11)
    weight, bias = _params((1,), affine, 21)
    upstream = _data((6, 1), 31)
    reference = _run(
        "cpu", x.double(), (1,),
        None if weight is None else weight.double(), None if bias is None else bias.double(),
        1e-5, upstream.double(),
    )
    got = _run(vulkan_backend, x, (1,), weight, bias, 1e-5, upstream)
    for g, w in zip(got, reference):
        if w is not None:
            torch.testing.assert_close(g, w.float(), rtol=0, atol=1e-4)


def test_rank_eight_input_is_accepted_and_matches_cpu(vulkan_backend):
    shape = (2, 1, 2, 1, 2, 1, 2, 3)
    x = _data(shape, 141)
    weight, bias = _params((3,), "both", 142)
    upstream = _data(shape, 143)
    cpu = _run("cpu", x, (3,), weight, bias, 1e-5, upstream)
    vk = _run(vulkan_backend, x, (3,), weight, bias, 1e-5, upstream)
    _compare(vk, cpu)


@pytest.mark.parametrize("eps", [1e-5, 1e-2, 0.5])
@pytest.mark.parametrize("elementwise_affine, bias", [(True, True), (True, False), (False, True)])
def test_nn_layer_norm_module_matches_cpu(vulkan_backend, eps, elementwise_affine, bias):
    torch.manual_seed(5)
    cpu_module = torch.nn.LayerNorm((4, 6), eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    if elementwise_affine:
        with torch.no_grad():
            cpu_module.weight.copy_(_data((4, 6), 41) + 1.0)
            if bias:
                cpu_module.bias.copy_(_data((4, 6), 42))
    vk_module = torch.nn.LayerNorm((4, 6), eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    vk_module.load_state_dict(cpu_module.state_dict())
    vk_module.to(vulkan_backend)

    x = _data((3, 4, 6), 7)
    upstream = _data((3, 4, 6), 8)
    x_cpu = x.clone().requires_grad_()
    y_cpu = cpu_module(x_cpu)
    y_cpu.backward(upstream)
    x_vk = x.to(vulkan_backend).requires_grad_()
    y_vk = vk_module(x_vk)
    y_vk.backward(upstream.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()

    assert y_vk.device.type == "vk" and x_vk.grad.device.type == "vk"
    torch.testing.assert_close(y_vk.cpu(), y_cpu, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, rtol=RTOL, atol=ATOL)
    for name, parameter in cpu_module.named_parameters():
        torch.testing.assert_close(
            dict(vk_module.named_parameters())[name].grad.cpu(), parameter.grad,
            rtol=RTOL, atol=ATOL,
        )


def test_layer_norm_backward_accumulates_across_repeated_calls(vulkan_backend):
    weight = (_data((6,), 51) + 1.0).to(vulkan_backend).requires_grad_()
    x = _data((4, 6), 52).to(vulkan_backend)
    upstream = _data((4, 6), 53).to(vulkan_backend)
    for _ in range(2):
        F.layer_norm(x, (6,), weight, None).backward(upstream)
    pytorch_vulkan._C.synchronize()
    cpu_weight = (_data((6,), 51) + 1.0).requires_grad_()
    for _ in range(2):
        F.layer_norm(_data((4, 6), 52), (6,), cpu_weight, None).backward(_data((4, 6), 53))
    torch.testing.assert_close(weight.grad.cpu(), cpu_weight.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("affine", ["both", "none"])
def test_expanded_upstream_gradient_matches_cpu(vulkan_backend, affine):
    x = _data((5, 8), 61)
    weight, bias = _params((8,), affine, 62)
    upstream = torch.tensor(1.5)
    cpu = _run("cpu", x, (8,), weight, bias, 1e-5, upstream, expand_upstream=True)
    vk = _run(vulkan_backend, x, (8,), weight, bias, 1e-5, upstream, expand_upstream=True)
    _compare(vk, cpu)


# --- layouts: the kernels read dense storage, so strided operands are densified


def _vk(tensor, device):
    return tensor.to(device)


@pytest.mark.parametrize(
    "build",
    [
        lambda t: t.t(),
        lambda t: t[:, 1:7],
        lambda t: t[1:, :],
        lambda t: t.unsqueeze(0).expand(3, *t.shape),
    ],
    ids=["transposed", "column-slice", "row-offset", "expanded"],
)
def test_forward_accepts_strided_inputs_like_cpu(vulkan_backend, build):
    base = _data((8, 8), 71)
    view = build(base)
    normalized = (view.shape[-1],)
    weight = _data(normalized, 72) + 1.0
    bias = _data(normalized, 73)
    want = torch.native_layer_norm(view, normalized, weight, bias, 1e-5)
    vk_view = build(_vk(base, vulkan_backend))
    got = torch.native_layer_norm(
        vk_view, normalized, _vk(weight, vulkan_backend), _vk(bias, vulkan_backend), 1e-5
    )
    pytorch_vulkan._C.synchronize()
    for g, w in zip(got, want):
        assert g.device.type == "vk" and g.is_contiguous()
        torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL)


def test_backward_accepts_strided_operands_like_cpu(vulkan_backend):
    x_base = _data((6, 8), 81)
    x = x_base[:, :6]
    grad_base = _data((6, 8), 82)
    grad = grad_base.t().contiguous().t()[:, :6]
    weight = _data((6,), 83) + 1.0
    _, mean, rstd = torch.native_layer_norm(x, (6,), weight, None, 1e-5)
    mask = [True, True, False]
    want = torch.ops.aten.native_layer_norm_backward(grad, x, (6,), mean, rstd, weight, None, mask)
    got = torch.ops.aten.native_layer_norm_backward(
        _vk(grad_base.t().contiguous().t(), vulkan_backend)[:, :6],
        _vk(x_base, vulkan_backend)[:, :6], (6,), _vk(mean, vulkan_backend),
        _vk(rstd, vulkan_backend), _vk(weight, vulkan_backend), None, mask,
    )
    pytorch_vulkan._C.synchronize()
    assert got[2] is None
    for g, w in zip(got[:2], want[:2]):
        torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL)


# --- saved statistics and output masks (direct aten) ---------------------------


@pytest.mark.parametrize("shape, normalized_shape", SHAPES)
def test_native_layer_norm_statistics_shapes_and_values(vulkan_backend, shape, normalized_shape):
    x = _data(shape, 91)
    want = torch.native_layer_norm(x, normalized_shape, None, None, 1e-5)
    got = torch.native_layer_norm(_vk(x, vulkan_backend), normalized_shape, None, None, 1e-5)
    pytorch_vulkan._C.synchronize()
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.dtype == torch.float32 and g.device.type == "vk"
        torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "weight_on, bias_on, mask",
    [
        (False, False, [True, False, False]),
        (True, False, [True, False, False]),
        (True, False, [True, True, False]),
        (True, True, [True, True, True]),
        (False, True, [True, False, True]),
        (True, True, [False, True, True]),
        (True, True, [False, False, False]),
        (True, True, [True, False, False]),
    ],
)
def test_native_layer_norm_backward_masks(vulkan_backend, weight_on, bias_on, mask):
    x = _data((4, 3, 5), 101)
    weight = _data((5,), 102) + 1.0 if weight_on else None
    bias = _data((5,), 103) if bias_on else None
    grad = _data((4, 3, 5), 104)
    _, mean, rstd = torch.native_layer_norm(x, (5,), weight, bias, 1e-5)
    want = torch.ops.aten.native_layer_norm_backward(grad, x, (5,), mean, rstd, weight, bias, mask)
    to_vk = lambda t: None if t is None else _vk(t, vulkan_backend)
    got = torch.ops.aten.native_layer_norm_backward(
        to_vk(grad), to_vk(x), (5,), to_vk(mean), to_vk(rstd), to_vk(weight), to_vk(bias), mask
    )
    pytorch_vulkan._C.synchronize()
    for g, w, wanted in zip(got, want, mask):
        assert (g is None) == (not wanted) == (w is None)
        if wanted:
            assert g.device.type == "vk"
            torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL)


# --- empty and degenerate inputs ------------------------------------------------

EMPTY = [
    ((0, 4), (4,)),
    ((3, 0, 4), (4,)),
    ((2, 0), (0,)),
    ((3, 2, 0), (0,)),
    ((3, 0, 2), (0, 2)),
    ((0, 0), (0,)),
    ((0,), (0,)),
]


@pytest.mark.parametrize("eps", [1e-5, 0.0])
@pytest.mark.parametrize("shape, normalized_shape", EMPTY)
def test_empty_and_zero_width_inputs_match_cpu(vulkan_backend, shape, normalized_shape, eps):
    x = torch.randn(shape)
    weight, bias = torch.ones(normalized_shape), torch.zeros(normalized_shape)
    grad = torch.randn(shape)
    want = torch.native_layer_norm(x, normalized_shape, weight, bias, eps)
    got = torch.native_layer_norm(
        _vk(x, vulkan_backend), normalized_shape, _vk(weight, vulkan_backend),
        _vk(bias, vulkan_backend), eps,
    )
    pytorch_vulkan._C.synchronize()
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.device.type == "vk"
        torch.testing.assert_close(g.cpu(), w, rtol=0, atol=0, equal_nan=True)
    mask = [True, True, True]
    bwd_want = torch.ops.aten.native_layer_norm_backward(
        grad, x, normalized_shape, want[1], want[2], weight, bias, mask
    )
    bwd_got = torch.ops.aten.native_layer_norm_backward(
        _vk(grad, vulkan_backend), _vk(x, vulkan_backend), normalized_shape,
        got[1], got[2], _vk(weight, vulkan_backend), _vk(bias, vulkan_backend), mask,
    )
    pytorch_vulkan._C.synchronize()
    for g, w in zip(bwd_got, bwd_want):
        assert g.shape == w.shape and g.device.type == "vk"
        torch.testing.assert_close(g.cpu(), w, rtol=0, atol=0, equal_nan=True)


def test_empty_batch_backward_returns_zero_parameter_gradients(vulkan_backend):
    module = torch.nn.LayerNorm(4).to(vulkan_backend)
    x = torch.randn(0, 4).to(vulkan_backend).requires_grad_()
    y = module(x)
    y.backward(torch.randn(0, 4).to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    assert x.grad.shape == (0, 4)
    torch.testing.assert_close(module.weight.grad.cpu(), torch.zeros(4))
    torch.testing.assert_close(module.bias.grad.cpu(), torch.zeros(4))


def test_zero_width_module_matches_cpu(vulkan_backend):
    module = torch.nn.LayerNorm(0).to(vulkan_backend)
    x = torch.randn(2, 0).to(vulkan_backend).requires_grad_()
    y = module(x)
    y.backward(torch.randn(2, 0).to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    assert y.shape == (2, 0) and x.grad.shape == (2, 0)
    assert module.weight.grad.shape == (0,) and module.bias.grad.shape == (0,)


# --- numerical edge cases on eps and variance -----------------------------------

EPS_ROWS = torch.tensor([[1.0, 2.0, 3.0, 5.0], [2.0, 2.0, 2.0, 2.0], [-3.0, 0.5, 4.0, 1.0]])


@pytest.mark.parametrize(
    "eps", [0.0, -1.0, -10.0, 1e-50, 1e39, float("inf"), float("nan"), 1e-5]
)
def test_eps_edge_cases_match_cpu_including_non_finite_results(vulkan_backend, eps):
    weight = torch.tensor([1.0, 2.0, -1.0, 0.5])
    bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
    grad = _data(EPS_ROWS.shape, 111)
    want = torch.native_layer_norm(EPS_ROWS, (4,), weight, bias, eps)
    got = torch.native_layer_norm(
        _vk(EPS_ROWS, vulkan_backend), (4,), _vk(weight, vulkan_backend),
        _vk(bias, vulkan_backend), eps,
    )
    pytorch_vulkan._C.synchronize()
    for g, w in zip(got, want):
        torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL, equal_nan=True)
    mask = [True, True, True]
    bwd_want = torch.ops.aten.native_layer_norm_backward(
        grad, EPS_ROWS, (4,), want[1], want[2], weight, bias, mask
    )
    bwd_got = torch.ops.aten.native_layer_norm_backward(
        _vk(grad, vulkan_backend), _vk(EPS_ROWS, vulkan_backend), (4,), got[1], got[2],
        _vk(weight, vulkan_backend), _vk(bias, vulkan_backend), mask,
    )
    pytorch_vulkan._C.synchronize()
    for g, w in zip(bwd_got, bwd_want):
        torch.testing.assert_close(g.cpu(), w, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_constant_rows_with_default_eps_stay_finite(vulkan_backend):
    x = torch.full((3, 5), 2.5)
    want = _run("cpu", x, (5,), torch.ones(5), torch.zeros(5), 1e-5, _data((3, 5), 121))
    got = _run(vulkan_backend, x, (5,), torch.ones(5), torch.zeros(5), 1e-5, _data((3, 5), 121))
    _compare(got, want)
    assert torch.isfinite(got[0]).all() and torch.isfinite(got[1]).all()


# --- rejections happen before any Vulkan work ----------------------------------


def _rejects(pattern, call):
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=pattern):
        call()
    assert _zero_work()


def test_native_layer_norm_malformed_arguments_are_rejected_like_pytorch(vulkan_backend):
    x = _vk(torch.randn(2, 4), vulkan_backend)
    w3 = _vk(torch.randn(3), vulkan_backend)
    _rejects("at least 1-dimensional", lambda: torch.native_layer_norm(x, (), None, None, 1e-5))
    _rejects("expected input with shape", lambda: torch.native_layer_norm(x, (3,), None, None, 1e-5))
    _rejects("expected input with shape", lambda: torch.native_layer_norm(x, (2, 4, 5), None, None, 1e-5))
    _rejects("weight to be of same shape", lambda: torch.native_layer_norm(x, (4,), w3, None, 1e-5))
    _rejects("bias to be of same shape", lambda: torch.native_layer_norm(x, (4,), None, w3, 1e-5))


def test_native_layer_norm_rejects_float64_and_integer_tensors(vulkan_backend):
    good = _vk(torch.randn(2, 4), vulkan_backend)
    double = _vk(torch.randn(2, 4), vulkan_backend).double()
    integer = _vk(torch.arange(8).reshape(2, 4), vulkan_backend)
    double_w = _vk(torch.ones(4), vulkan_backend).double()
    for bad in (double, integer):
        _rejects("float32|dtype", lambda bad=bad: torch.native_layer_norm(bad, (4,), None, None, 1e-5))
    _rejects("float32|dtype", lambda: torch.native_layer_norm(good, (4,), double_w, None, 1e-5))
    _rejects("float32|dtype", lambda: torch.native_layer_norm(good, (4,), None, double_w, 1e-5))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_tensors_cannot_reach_layer_norm_on_vulkan(vulkan_backend, dtype):
    # Vulkan has no half-precision storage, so these never reach the kernel.
    with pytest.raises(RuntimeError):
        torch.randn(2, 4, dtype=dtype).to(vulkan_backend)


def test_native_layer_norm_rejects_cpu_operands_and_rank_nine(vulkan_backend):
    good = _vk(torch.randn(2, 4), vulkan_backend)
    _rejects("vk:0|PrivateUse1|Vulkan", lambda: torch.native_layer_norm(
        good, (4,), torch.ones(4), None, 1e-5))
    rank9 = _vk(torch.randn((1,) * 8 + (4,)), vulkan_backend)
    _rejects("ranks up to 8", lambda: torch.native_layer_norm(rank9, (4,), None, None, 1e-5))


def test_native_layer_norm_backward_validates_saved_values_before_dispatch(vulkan_backend):
    x = torch.randn(3, 4)
    weight = torch.ones(4)
    _, mean, rstd = torch.native_layer_norm(x, (4,), weight, None, 1e-5)
    # Every operand is built before the counters are reset: only the rejected
    # call itself must do zero Vulkan work.
    vx, vw = _vk(x, vulkan_backend), _vk(weight, vulkan_backend)
    vmean, vrstd = _vk(mean, vulkan_backend), _vk(rstd, vulkan_backend)
    short_mean, short_rstd = _vk(mean[:2].contiguous(), vulkan_backend), _vk(rstd[:2].contiguous(), vulkan_backend)
    grad = _vk(torch.randn(3, 4), vulkan_backend)
    wrong_grad = _vk(torch.randn(2, 4), vulkan_backend)
    grad64 = _vk(torch.randn(3, 4), vulkan_backend).double()
    full = [True, True, False]
    backward = torch.ops.aten.native_layer_norm_backward
    _rejects("grad_out shape", lambda: backward(wrong_grad, vx, (4,), vmean, vrstd, vw, None, full))
    _rejects("mean and rstd must have 3", lambda: backward(grad, vx, (4,), short_mean, vrstd, vw, None, full))
    _rejects("mean and rstd must have 3", lambda: backward(grad, vx, (4,), vmean, short_rstd, vw, None, full))
    _rejects("output_mask\\[1\\]", lambda: backward(grad, vx, (4,), vmean, vrstd, None, None, [True, True, False]))
    _rejects("output_mask\\[2\\]", lambda: backward(grad, vx, (4,), vmean, vrstd, vw, None, [True, False, True]))
    _rejects("float32|dtype", lambda: backward(grad64, vx, (4,), vmean, vrstd, vw, None, full))
    _rejects("vk:0|PrivateUse1|Vulkan", lambda: backward(grad, vx, (4,), mean, vrstd, vw, None, full))


def test_layer_norm_results_stay_on_the_device_without_fallback(vulkan_backend):
    x = _vk(_data((4, 6), 131), vulkan_backend).requires_grad_()
    module = torch.nn.LayerNorm(6).to(vulkan_backend)
    upstream = _vk(_data((4, 6), 132), vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    module(x).backward(upstream)
    pytorch_vulkan._C.synchronize()
    dispatches, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert dispatches > 0 and transfers == 0 and fallbacks == 0
