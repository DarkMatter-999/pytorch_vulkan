import pytest
import pytorch_vulkan
import torch


PAIRS = [
    ((), (3,)),
    ((2, 1), (1, 3)),
    ((2, 1, 3), (4, 3)),
    ((0, 1), (1, 3)),
    ((2, 0), (1, 0)),
    ((), ()),
]


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _random(shape, generator, requires_grad=False):
    return torch.randn(shape, generator=generator).requires_grad_(requires_grad)


def _sync_compare(actual, expected):
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("left_shape,right_shape", PAIRS)
def test_allocating_mul_broadcast_values_gradients_and_graph(
    vulkan_backend, left_shape, right_shape
):
    generator = torch.Generator().manual_seed(20261004)
    a = _random(left_shape, generator, requires_grad=True)
    b = _random(right_shape, generator, requires_grad=True)
    expected = a * b
    av = a.detach().to(vulkan_backend).requires_grad_()
    bv = b.detach().to(vulkan_backend).requires_grad_()

    actual = av * bv
    seed = _random(expected.shape, generator)
    cpu_grads = torch.autograd.grad(expected, (a, b), seed, create_graph=True)
    vk_grads = torch.autograd.grad(
        actual, (av, bv), seed.to(vulkan_backend), create_graph=True
    )

    assert actual.shape == expected.shape
    _sync_compare(actual, expected)
    for vk_grad, cpu_grad, operand in zip(vk_grads, cpu_grads, (a, b)):
        assert vk_grad.shape == operand.shape
        _sync_compare(vk_grad, cpu_grad)
        assert vk_grad.requires_grad == cpu_grad.requires_grad


def test_allocating_mul_broadcast_noncontiguous_read_views(vulkan_backend):
    base = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    vk_base = base.to(vulkan_backend)
    cases = [
        (base[1:4, 1:7:2], vk_base.as_strided((3, 3), (8, 2), 9), torch.ones(1, 3)),
        (base[:3, :4].t(), vk_base.as_strided((4, 3), (1, 8), 0), torch.ones(4, 1)),
        (
            torch.arange(3, dtype=torch.float32).expand(2, 3),
            vk_base.as_strided((2, 3), (0, 1), 0),
            torch.ones(1, 3),
        ),
        (
            base[:1, :3].expand(2, 3),
            vk_base.as_strided((2, 3), (0, 1), 0),
            torch.ones(2, 1),
        ),
    ]
    for lhs, vk_lhs, rhs in cases:
        vk_rhs = rhs.to(vulkan_backend)
        assert tuple(vk_lhs.stride()) == tuple(lhs.stride())
        actual = vk_lhs * vk_rhs
        _sync_compare(actual, lhs * rhs)


def test_allocating_mul_broadcast_has_mixed_reverse_dependency(vulkan_backend):
    generator = torch.Generator().manual_seed(811)
    a = _random((2, 1), generator, requires_grad=True)
    b = _random((1, 3), generator, requires_grad=True)
    seed = _random((2, 3), generator)
    probe = _random((2, 1), generator)
    av = a.detach().to(vulkan_backend).requires_grad_()
    bv = b.detach().to(vulkan_backend).requires_grad_()

    cpu_first = torch.autograd.grad((a * b), a, seed, create_graph=True)[0]
    vk_first = torch.autograd.grad(
        av * bv, av, seed.to(vulkan_backend), create_graph=True
    )[0]
    cpu_mixed = torch.autograd.grad(cpu_first, b, probe)[0]
    vk_mixed = torch.autograd.grad(
        vk_first, bv, probe.to(vulkan_backend)
    )[0]

    _sync_compare(vk_first, cpu_first)
    _sync_compare(vk_mixed, cpu_mixed)
    assert torch.count_nonzero(cpu_mixed).item() > 0


@pytest.mark.parametrize(
    "make_operands",
    [
        lambda vk: (torch.empty((2, 3), device=vk), torch.empty((4,), device=vk)),
        lambda vk: (torch.empty((0, 2), device=vk), torch.empty((3,), device=vk)),
        lambda vk: (torch.empty((2, 3), device=vk), torch.empty((2, 3))),
        lambda vk: (
            torch.empty((2, 3), device=vk),
            torch.empty((2, 3), dtype=torch.float64, device=vk),
        ),
    ],
    ids=["incompatible", "malformed-empty", "mixed-device", "mixed-dtype"],
)
def test_invalid_mul_broadcast_preflight_does_no_dispatch(vulkan_backend, make_operands):
    lhs, rhs = make_operands(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.mul(lhs, rhs)

    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_mul_broadcast_preserves_python_scalar_and_existing_alias_contracts(
    vulkan_backend,
):
    value = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    vk_value = value.detach().to(vulkan_backend).requires_grad_()
    actual = 2.5 * vk_value
    expected = 2.5 * value
    _sync_compare(actual, expected)
    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    _sync_compare(vk_value.grad, value.grad)

    lhs = torch.tensor([2.0, 3.0], device=vulkan_backend)
    rhs = torch.tensor([5.0, 7.0], device=vulkan_backend)
    assert lhs.mul_(rhs) is lhs
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(lhs.cpu(), torch.tensor([10.0, 21.0]))

    out = torch.empty((2,), device=vulkan_backend)
    assert torch.mul(rhs, rhs, out=out) is out
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor([25.0, 49.0]))


def test_stock_outer_and_ger_forward_and_first_gradients(vulkan_backend):
    generator = torch.Generator().manual_seed(713)
    a = _random((2,), generator, requires_grad=True)
    b = _random((3,), generator, requires_grad=True)
    av = a.detach().to(vulkan_backend).requires_grad_()
    bv = b.detach().to(vulkan_backend).requires_grad_()
    seed = _random((2, 3), generator)

    for operation in (torch.outer, torch.ger):
        cpu_result = operation(a, b)
        vk_result = operation(av, bv)
        _sync_compare(vk_result, cpu_result)
        cpu_grads = torch.autograd.grad(cpu_result, (a, b), seed)
        vk_grads = torch.autograd.grad(
            vk_result, (av, bv), seed.to(vulkan_backend)
        )
        for vk_grad, cpu_grad in zip(vk_grads, cpu_grads):
            _sync_compare(vk_grad, cpu_grad)
