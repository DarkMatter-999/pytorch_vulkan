import pytest
import torch

import pytorch_vulkan
from matrix_f32_policy_oracle import addmm_post_alpha, f32_class


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


def test_mm_reads_valid_offset_positive_stride_and_transpose_views(vulkan_backend):
    base = torch.arange(60, dtype=torch.float32).reshape(5, 12) / 10
    vb = base.to(vulkan_backend)
    pairs = [
        (base[1:4, 2:10:2], vb[1:4, 2:10:2]),
        (base[:2, :3].t(), vb[:2, :3].t()),
    ]
    for cpu_a, vk_a in pairs:
        b = torch.arange(cpu_a.shape[1] * 3, dtype=torch.float32).reshape(-1, 3) / 7
        result = torch.mm(vk_a, b.to(vulkan_backend))
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(
            result.cpu(), torch.mm(cpu_a, b), rtol=0.003, atol=0.003
        )
    torch.testing.assert_close(vb.cpu(), base)


def _matrix_operands(kind, device, requires_grad=(True, True, True)):
    if kind == "mm":
        values = [torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7 - 0.2,
                  torch.arange(12, dtype=torch.float32).reshape(3, 4) / 9 - 0.3]
        values = values[:2]
        return [x.to(device).requires_grad_(requires_grad[i]) for i, x in enumerate(values)]
    if kind == "addmm":
        values = [torch.arange(4, dtype=torch.float32) / 5 - 0.4,
                  torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7 - 0.2,
                  torch.arange(12, dtype=torch.float32).reshape(3, 4) / 9 - 0.3]
        return [x.to(device).requires_grad_(requires_grad[i]) for i, x in enumerate(values)]
    values = [torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 11 - 0.4,
              torch.arange(40, dtype=torch.float32).reshape(2, 4, 5) / 13 - 0.2]
    return [x.to(device).requires_grad_(requires_grad[i]) for i, x in enumerate(values)]


def _matrix_call(kind, operands):
    if kind == "mm":
        return torch.mm(*operands)
    if kind == "addmm":
        return torch.addmm(*operands, beta=-0.5, alpha=1.75)
    return torch.bmm(*operands)


def _assert_vulkan_cpu_pair(actual, expected, vulkan_backend):
    pytorch_vulkan._C.synchronize()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float32
    assert str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize(
    "a_value,b_value,alpha,cpu_class,expected_finite_value,cpu_policy_diff",
    [
        (1.0e20, 1.0e20, 1.0e-20, "finite", 2.0e20, True),
        (1.0e20, 1.0e-20, 1.0e20, "+inf", None, True),
        (-1.0e20, 1.0e20, 1.0e-20, "finite", -2.0e20, True),
        (1.0e20, 1.0e-20, -1.0e20, "-inf", None, True),
    ],
)
def test_addmm_finite_alpha_uses_policy_oracle_and_records_cpu_class(
    vulkan_backend, a_value, b_value, alpha, cpu_class, expected_finite_value,
    cpu_policy_diff
):
    a = torch.full((2, 2), a_value, dtype=torch.float32)
    b = torch.full((2, 2), b_value, dtype=torch.float32)
    self = torch.zeros((2, 2), dtype=torch.float32)
    expected = torch.addmm(self, a, b, alpha=alpha)
    assert f32_class(expected) == cpu_class
    if cpu_class == "finite":
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(
            expected, torch.full_like(expected, expected_finite_value), rtol=1e-6, atol=0
        )
    elif cpu_class == "+inf":
        assert torch.isposinf(expected).all()
    else:
        assert torch.isneginf(expected).all()

    policy = addmm_post_alpha(
        a.tolist(), b.tolist(), self.tolist(), alpha=alpha, beta=0.0
    )
    assert (f32_class(expected) != f32_class(policy)) is cpu_policy_diff

    vself, va, vb = (value.to(vulkan_backend) for value in (self, a, b))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.addmm(
        vself, va, vb, alpha=alpha
    )
    pytorch_vulkan._C.synchronize()
    compute, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert compute > 0 and transfers == fallbacks == 0
    assert str(actual.device) == vulkan_backend
    observed = actual.cpu()
    assert f32_class(observed) == f32_class(policy), (
        f"policy class mismatch: CPU={cpu_class}, "
        f"policy={f32_class(policy)}, Vulkan={f32_class(observed)}, "
        f"CPU/policy differ={cpu_class != f32_class(policy)}"
    )
    torch.testing.assert_close(observed, policy, rtol=0.003, atol=0.003)


@pytest.mark.parametrize(
    "kind,m,k,n,a_value,b_value,alpha,cpu_class,cpu_policy_diff",
    [
        ("transposed_rhs", 2, 2, 2, 1.0e20, 1.0e-20, 1.0e20, "finite", False),
        ("transposed_rhs", 2, 2, 2, 1.0e20, 1.0e20, 1.0e-20, "+inf", False),
        ("transposed_rhs", 2, 2, 2, -1.0e20, 1.0e-20, 1.0e20, "finite", False),
        ("single_column", 2, 2, 1, 1.0e20, 1.0e-20, 1.0e20, "finite", False),
        ("single_column", 2, 2, 1, 1.0e20, 1.0e20, 1.0e-20, "+inf", False),
        ("scalar_output", 1, 1, 1, 1.0e20, 1.0e20, 1.0e-20, "+inf", False),
        ("scalar_output", 1, 1, 1, 1.0e20, 1.0e-20, 1.0e20, "finite", False),
        ("scalar_output_k2", 1, 2, 1, 1.0e20, 1.0e20, 1.0e-20, "finite", True),
        ("scalar_output_k2", 1, 2, 1, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
        ("transposed_lhs", 2, 2, 2, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
        ("offset_rhs", 2, 2, 2, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
        ("zero_stride_rhs", 2, 2, 2, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
        ("rectangular", 3, 2, 4, 1.0e20, 1.0e20, 1.0e-20, "finite", True),
        ("k_one", 2, 1, 2, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
        ("m_one", 1, 2, 2, 1.0e20, 1.0e-20, 1.0e20, "+inf", True),
    ],
)
def test_addmm_finite_alpha_uses_policy_oracle_for_cpu_view_cases(
    vulkan_backend, kind, m, k, n, a_value, b_value, alpha, cpu_class,
    cpu_policy_diff
):
    a_base = torch.full((m, k), a_value, dtype=torch.float32)
    b_base = torch.full((k, n), b_value, dtype=torch.float32)
    if kind == "transposed_lhs":
        a_base = torch.full((k, m), a_value, dtype=torch.float32)
        a = a_base.t()
    else:
        a = a_base
    if kind == "transposed_rhs":
        b_base = torch.full((n, k), b_value, dtype=torch.float32)
        b = b_base.t()
    elif kind == "offset_rhs":
        b_base = torch.full((k + 1, n + 2), b_value, dtype=torch.float32)
        b = b_base[1:k + 1, 1:n + 1]
    elif kind == "zero_stride_rhs":
        b_base = torch.full((1, n), b_value, dtype=torch.float32)
        b = b_base.expand(k, n)
    else:
        b = b_base
    self = torch.full((a.shape[0], b.shape[1]), 1.0e20, dtype=torch.float32)
    expected = torch.addmm(self, a, b, beta=0.5, alpha=alpha)
    assert f32_class(expected) == cpu_class
    if cpu_class == "finite":
        assert torch.isfinite(expected).all()
    else:
        assert torch.isposinf(expected).all()
    policy = addmm_post_alpha(
        a.tolist(), b.tolist(), self.tolist(), alpha=alpha, beta=0.5
    )
    assert (f32_class(expected) != f32_class(policy)) is cpu_policy_diff

    va_base = a_base.to(vulkan_backend)
    va = va_base.t() if kind == "transposed_lhs" else va_base
    vb_base = b_base.to(vulkan_backend)
    if kind == "transposed_rhs":
        vb = vb_base.t()
    elif kind == "zero_stride_rhs":
        vb = vb_base.expand(k, n)
    elif kind == "offset_rhs":
        vb = vb_base[1:k + 1, 1:n + 1]
    else:
        vb = vb_base
    assert tuple(vb.stride()) == tuple(b.stride())
    assert tuple(va.stride()) == tuple(a.stride())
    assert vb.shape == b.shape
    vself = self.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.addmm(vself, va, vb, beta=0.5, alpha=alpha)
    pytorch_vulkan._C.synchronize()
    compute, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert compute > 0 and transfers == fallbacks == 0
    assert str(actual.device) == vulkan_backend
    observed = actual.cpu()
    assert f32_class(observed) == f32_class(policy), (
        f"policy class mismatch: CPU={cpu_class}, "
        f"policy={f32_class(policy)}, Vulkan={f32_class(observed)}, "
        f"CPU/policy differ={cpu_class != f32_class(policy)}"
    )
    torch.testing.assert_close(observed, policy, rtol=0.003, atol=0.003)


def test_mm_backward_preserves_create_graph_history(vulkan_backend):
    def probe(device):
        a = (torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7).to(device).requires_grad_()
        b = (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 9).to(device).requires_grad_()
        seed = torch.ones(2, 4).to(device).requires_grad_()
        ga = torch.autograd.grad(torch.mm(a, b), a, grad_outputs=seed,
                                 create_graph=True)[0]
        assert ga.requires_grad
        return ga, torch.autograd.grad(ga.sum(), seed)[0]
    expected = probe("cpu")
    actual = probe(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), ref, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("kind", ["mm", "addmm", "bmm"])
def test_matrix_generated_autograd_first_reverse_selective_and_seed(kind, vulkan_backend):
    cpu_inputs = _matrix_operands(kind, "cpu")
    vk_inputs = _matrix_operands(kind, vulkan_backend)
    vk_before = [value.cpu().clone() for value in vk_inputs]
    cpu_output, vk_output = _matrix_call(kind, cpu_inputs), _matrix_call(kind, vk_inputs)
    seed_cpu = torch.arange(cpu_output.numel(), dtype=torch.float32).reshape(cpu_output.shape) / 17 + 0.2
    seed_vk = seed_cpu.to(vulkan_backend).requires_grad_()
    seed_cpu.requires_grad_()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    requested_cpu = cpu_inputs[-1:]
    requested_vk = vk_inputs[-1:]
    cpu_grad = torch.autograd.grad(cpu_output, requested_cpu, grad_outputs=seed_cpu,
                                   allow_unused=True, create_graph=True)[0]
    vk_grad = torch.autograd.grad(vk_output, requested_vk, grad_outputs=seed_vk,
                                  allow_unused=True, create_graph=True)[0]
    assert (vk_grad is None) == (cpu_grad is None)
    if cpu_grad is not None:
        assert vk_grad.requires_grad == cpu_grad.requires_grad
        seed_deriv_cpu = torch.autograd.grad(cpu_grad.sum(), seed_cpu, allow_unused=True)[0]
        seed_deriv_vk = torch.autograd.grad(vk_grad.sum(), seed_vk, allow_unused=True)[0]
        assert (seed_deriv_vk is None) == (seed_deriv_cpu is None)
    # Request only the first operand; all other leaves are exact undefined slots.
    all_cpu = torch.autograd.grad(cpu_output, cpu_inputs, grad_outputs=seed_cpu,
                                  allow_unused=True, create_graph=True)
    all_vk = torch.autograd.grad(vk_output, vk_inputs, grad_outputs=seed_vk,
                                 allow_unused=True, create_graph=True)
    assert tuple(g is None for g in all_vk) == tuple(g is None for g in all_cpu)
    unused_cpu = torch.ones(2, dtype=torch.float32, requires_grad=True)
    unused_vk = torch.ones(2, dtype=torch.float32, device=vulkan_backend, requires_grad=True)
    cpu_none = torch.autograd.grad(cpu_output.sum(), (cpu_inputs[0], unused_cpu), allow_unused=True)
    vk_none = torch.autograd.grad(vk_output.sum(), (vk_inputs[0], unused_vk), allow_unused=True)
    assert cpu_none[1] is None and vk_none[1] is None
    pytorch_vulkan._C.synchronize()
    compute, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert compute > 0 and transfers == fallbacks == 0
    _assert_vulkan_cpu_pair(vk_output, cpu_output, vulkan_backend)
    if cpu_grad is not None:
        _assert_vulkan_cpu_pair(vk_grad, cpu_grad, vulkan_backend)
        if seed_deriv_cpu is not None:
            _assert_vulkan_cpu_pair(seed_deriv_vk, seed_deriv_cpu, vulkan_backend)
    for got, ref in zip(all_vk, all_cpu):
        if ref is not None:
            _assert_vulkan_cpu_pair(got, ref, vulkan_backend)
    for value, before in zip(vk_inputs, vk_before):
        torch.testing.assert_close(value.cpu(), before)


def _matrix_energy_gradients(kind, device, directions=None):
    inputs = _matrix_operands(kind, device, (True,) * (3 if kind == "addmm" else 2))
    if directions is None:
        directions = tuple(torch.full_like(x, 0.13 + 0.07 * i)
                           for i, x in enumerate(inputs))
    if device != "cpu":
        pytorch_vulkan._C.synchronize()
        pytorch_vulkan._C.reset_execution_counters()
    output = _matrix_call(kind, inputs)
    objective = (output * output).sum()
    gradients = torch.autograd.grad(objective, inputs, create_graph=True)
    if directions is None:
        return inputs, objective, gradients
    directional = sum((gradient * direction).sum()
                      for gradient, direction in zip(gradients, directions))
    return inputs, objective, gradients, torch.autograd.grad(directional, inputs)


@pytest.mark.parametrize("kind", ["mm", "addmm", "bmm"])
def test_matrix_generated_autograd_mixed_hvp_matches_cpu_and_centered_fd(kind, vulkan_backend):
    cpu_operands = _matrix_operands(kind, "cpu", (False,) * (3 if kind == "addmm" else 2))
    cpu_directions = tuple(torch.full_like(x, 0.13 + 0.07 * i)
                           for i, x in enumerate(cpu_operands))
    vk_directions = tuple(direction.to(vulkan_backend) for direction in cpu_directions)
    _, cpu_objective, cpu_first, cpu_hvp = _matrix_energy_gradients(
        kind, "cpu", cpu_directions
    )
    vk_inputs, vk_objective, vk_first, vk_hvp = _matrix_energy_gradients(
        kind, vulkan_backend, vk_directions
    )
    pytorch_vulkan._C.synchronize()
    compute, copies, transfers, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()
    assert compute > 0 and transfers == fallbacks == 0
    assert all(value.device == torch.device(vulkan_backend)
               for value in (*vk_inputs, *vk_directions, *vk_first, *vk_hvp))
    _assert_vulkan_cpu_pair(vk_objective, cpu_objective, vulkan_backend)
    for got, ref in zip(vk_first, cpu_first):
        _assert_vulkan_cpu_pair(got, ref, vulkan_backend)
    for got, ref in zip(vk_hvp, cpu_hvp):
        _assert_vulkan_cpu_pair(got, ref, vulkan_backend)

    base = _matrix_operands(kind, "cpu")
    directions = [torch.full_like(x, 0.13 + 0.07 * i) for i, x in enumerate(base)]
    def directional_first(step):
        shifted = [x.detach() + step * direction for x, direction in zip(base, directions)]
        for x in shifted:
            x.requires_grad_()
        value = (_matrix_call(kind, shifted) ** 2).sum()
        gradients = torch.autograd.grad(value, shifted)
        return sum((g * d).sum() for g, d in zip(gradients, directions))
    epsilon = 0.001
    fd = (directional_first(epsilon) - directional_first(-epsilon)) / (2 * epsilon)
    analytic = sum((g * d).sum() for g, d in zip(cpu_hvp, directions))
    torch.testing.assert_close(analytic, fd, rtol=0.008, atol=0.002)


def test_matrix_generated_autograd_saved_variable_version_check(vulkan_backend):
    def probe(device):
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(device).requires_grad_()
        b = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(device).requires_grad_()
        output = torch.mm(a, b)
        with torch.no_grad():
            b.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            torch.autograd.grad(output.sum(), a)
    probe("cpu")
    probe(vulkan_backend)


@pytest.mark.parametrize("layout", ["transpose", "offset", "zero-stride"])
def test_bmm_generated_autograd_readable_gradient_layouts(layout, vulkan_backend):
    if layout == "transpose":
        base_a = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 11
        base_b = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 13
        cpu_a, cpu_b = base_a.transpose(1, 2), base_b.transpose(1, 2)
    elif layout == "offset":
        base_a = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6) / 11
        base_b = torch.arange(70, dtype=torch.float32).reshape(2, 5, 7) / 13
        cpu_a, cpu_b = base_a[:, 1:4, 1:5], base_b[:, 1:5, 1:6]
    else:
        base_a = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 11
        base_b = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5) / 13
        cpu_a, cpu_b = base_a.expand(2, 3, 4), base_b.expand(2, 4, 5)
    vk_base_a, vk_base_b = base_a.to(vulkan_backend), base_b.to(vulkan_backend)
    vk_a, vk_b = vk_base_a, vk_base_b
    if layout == "transpose":
        vk_a, vk_b = vk_a.transpose(1, 2), vk_b.transpose(1, 2)
    elif layout == "offset":
        vk_a, vk_b = vk_a[:, 1:4, 1:5], vk_b[:, 1:5, 1:6]
    else:
        vk_a, vk_b = vk_a.expand(2, 3, 4), vk_b.expand(2, 4, 5)
    cpu_a, cpu_b = cpu_a.detach().requires_grad_(), cpu_b.detach().requires_grad_()
    vk_a, vk_b = vk_a.detach().requires_grad_(), vk_b.detach().requires_grad_()
    before_a, before_b = vk_base_a.cpu().clone(), vk_base_b.cpu().clone()
    cpu_output, vk_output = torch.bmm(cpu_a, cpu_b), torch.bmm(vk_a, vk_b)
    seed_cpu = torch.arange(cpu_output.numel(), dtype=torch.float32).reshape(cpu_output.shape) / 19 + 0.3
    seed_vk = seed_cpu.to(vulkan_backend)
    cpu_grad = torch.autograd.grad(cpu_output, (cpu_a, cpu_b), seed_cpu, create_graph=True)
    vk_grad = torch.autograd.grad(vk_output, (vk_a, vk_b), seed_vk, create_graph=True)
    _assert_vulkan_cpu_pair(vk_output, cpu_output, vulkan_backend)
    for actual, expected in zip(vk_grad, cpu_grad):
        _assert_vulkan_cpu_pair(actual, expected, vulkan_backend)
        assert actual.requires_grad == expected.requires_grad
    torch.testing.assert_close(vk_base_a.cpu(), before_a)
    torch.testing.assert_close(vk_base_b.cpu(), before_b)


@pytest.mark.parametrize("kind", ["mm", "addmm", "bmm"])
def test_matrix_generated_autograd_matches_graphless_zero_dependencies(kind, vulkan_backend):
    n = 3 if kind == "addmm" else 2
    cpu_inputs = _matrix_operands(kind, "cpu", (True,) * n)
    vk_inputs = _matrix_operands(kind, vulkan_backend, (True,) * n)
    if kind == "addmm":
        cpu_output = torch.addmm(cpu_inputs[0], cpu_inputs[1], cpu_inputs[2], beta=0.0)
        vk_output = torch.addmm(vk_inputs[0], vk_inputs[1], vk_inputs[2], beta=0.0)
        cpu_seed = torch.ones_like(cpu_output)
        vk_seed = torch.ones_like(vk_output.cpu()).to(vulkan_backend)
        cpu_self, vk_self = torch.autograd.grad(cpu_output, cpu_inputs[0], cpu_seed)[0], torch.autograd.grad(vk_output, vk_inputs[0], vk_seed)[0]
        assert not cpu_self.requires_grad and not vk_self.requires_grad
        _assert_vulkan_cpu_pair(vk_self, cpu_self, vulkan_backend)
    cpu_grad = torch.autograd.grad(_matrix_call(kind, cpu_inputs).sum(), cpu_inputs[0], allow_unused=True)[0]
    vk_grad = torch.autograd.grad(_matrix_call(kind, vk_inputs).sum(), vk_inputs[0], allow_unused=True)[0]
    assert (vk_grad is None) == (cpu_grad is None)


def test_mm_reads_zero_stride_and_overlapping_operands(vulkan_backend):
    expanded_base = torch.tensor([[1.0, -2.0, 3.0]])
    overlap_base = torch.tensor([1.0, 4.0, -2.0, 5.0])
    cpu_views = [
        expanded_base.expand(2, 3),
        overlap_base.as_strided((2, 3), (1, 1)),
    ]
    vk_bases = [expanded_base.to(vulkan_backend), overlap_base.to(vulkan_backend)]
    vk_views = [
        vk_bases[0].expand(2, 3),
        vk_bases[1].as_strided((2, 3), (1, 1)),
    ]
    rhs = torch.tensor([[1.0, 2.0], [-3.0, 0.5], [4.0, -1.0]])

    for cpu_a, vk_a, vk_base in zip(cpu_views, vk_views, vk_bases):
        before = vk_base.cpu().clone()
        result = torch.mm(vk_a, rhs.to(vulkan_backend))
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(
            result.cpu(), torch.mm(cpu_a, rhs), rtol=0.003, atol=0.003
        )
        torch.testing.assert_close(vk_base.cpu(), before)


def test_mm_materializes_singleton_stride_views_in_each_gemm_role(vulkan_backend):
    cases = [
        (
            torch.tensor([1.0, 2.0, -1.0]), (1, 3), (1, 1),
            torch.tensor([2.0, -1.0, 0.5, 3.0, -2.0, 4.0]), (3, 2), (2, 1),
        ),
        (
            torch.tensor([1.0, 2.0, -1.0]), (3, 1), (1, 0),
            torch.tensor([2.0, -1.0]), (1, 2), (2, 1),
        ),
        (
            torch.tensor([2.0, -1.0]), (2, 1), (1, 1),
            torch.tensor([1.0, 2.0, -1.0]), (1, 3), (1, 1),
        ),
        (
            torch.tensor([2.0, 1.0, -1.0, 0.5, -3.0, 4.0]), (2, 3), (3, 1),
            torch.tensor([1.0, 2.0, -1.0]), (3, 1), (1, 0),
        ),
    ]
    for base_a, shape_a, strides_a, base_b, shape_b, strides_b in cases:
        cpu_a = base_a.as_strided(shape_a, strides_a)
        cpu_b = base_b.as_strided(shape_b, strides_b)
        vk_base_a = base_a.to(vulkan_backend)
        vk_base_b = base_b.to(vulkan_backend)
        vk_a = vk_base_a.as_strided(shape_a, strides_a)
        vk_b = vk_base_b.as_strided(shape_b, strides_b)
        assert vk_a.stride() == strides_a and vk_a.is_contiguous()
        assert vk_b.stride() == strides_b and vk_b.is_contiguous()
        result = torch.mm(vk_a, vk_b)
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(
            result.cpu(), torch.mm(cpu_a, cpu_b), rtol=0.003, atol=0.003
        )
        torch.testing.assert_close(vk_a.cpu(), cpu_a)
        torch.testing.assert_close(vk_b.cpu(), cpu_b)


def test_matrix_invalid_inputs_fail_before_dispatch(vulkan_backend):
    valid = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(vulkan_backend)
    before = valid.cpu().clone()
    wrong_rank = torch.ones(2, 3, 4).to(vulkan_backend)

    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.mm(wrong_rank, valid)

    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    torch.testing.assert_close(valid.cpu(), before)


def test_invalid_second_operand_fails_before_materializing_first(vulkan_backend):
    cpu_base = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 3
    cpu_a = cpu_base[::2, ::2]
    valid_vk = cpu_base.to(vulkan_backend)
    vk_a = valid_vk[::2, ::2]
    cpu_b = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    vk_b = cpu_b.to(vulkan_backend)
    before = valid_vk.cpu().clone()

    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="matching 2-D"):
        torch.mm(vk_a, vk_b)

    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    torch.testing.assert_close(valid_vk.cpu(), before)


def test_mm_accepts_rectangular_offset_strided_and_transposed_inputs(vulkan_backend):
    base_a = torch.arange(80, dtype=torch.float32).reshape(8, 10) / 13
    base_b = torch.arange(90, dtype=torch.float32).reshape(10, 9) / 17
    cases = [
        (base_a[1:7:2, 1:9:2], base_b[1:9:2, 1:8:2]),
        (base_a[:4, :6].t(), base_b[:4, :6]),
    ]
    for cpu_a, cpu_b in cases:
        vk_a, vk_b = cpu_a.to(vulkan_backend), cpu_b.to(vulkan_backend)
        before_a, before_b = vk_a.cpu().clone(), vk_b.cpu().clone()
        pytorch_vulkan._C.synchronize()
        pytorch_vulkan._C.reset_execution_counters()
        actual = torch.mm(vk_a, vk_b)
        pytorch_vulkan._C.synchronize()
        expected = torch.mm(cpu_a, cpu_b)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.float32 and str(actual.device) == vulkan_backend
        assert pytorch_vulkan._C.compute_dispatch_count() > 0
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
        torch.testing.assert_close(vk_a.cpu(), before_a)
        torch.testing.assert_close(vk_b.cpu(), before_b)


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_mm_empty_dimensions_match_cpu(shape, vulkan_backend):
    m, k, n = shape
    cpu_a = torch.arange(m * k, dtype=torch.float32).reshape(m, k)
    cpu_b = torch.arange(k * n, dtype=torch.float32).reshape(k, n)
    actual = torch.mm(cpu_a.to(vulkan_backend), cpu_b.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    expected = torch.mm(cpu_a, cpu_b)
    assert actual.shape == expected.shape and str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("self_kind", ["scalar", "feature", "row", "matrix"])
@pytest.mark.parametrize("alpha,beta", [(2.0, 0.25), (-1.5, -2.0), (0.0, 1.0),
                                         (-0.0, -0.0), (-0.0, 2.0), (2.0, -0.0)])
def test_addmm_broadcast_self_and_independent_scalars(
    self_kind, alpha, beta, vulkan_backend
):
    a = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 7 - 1
    b = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 9 - 1
    self_shape = {"scalar": (), "feature": (4,), "row": (1, 4), "matrix": (3, 4)}[
        self_kind
    ]
    self_cpu = torch.arange(torch.tensor(self_shape).prod().item() if self_shape else 1,
                             dtype=torch.float32).reshape(self_shape) / 5 - 0.5
    vk_a, vk_b, vk_self = (x.to(vulkan_backend) for x in (a, b, self_cpu))
    before = [x.cpu().clone() for x in (vk_a, vk_b, vk_self)]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.addmm(vk_self, vk_a, vk_b, alpha=alpha, beta=beta)
    pytorch_vulkan._C.synchronize()
    expected = torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta)
    assert actual.shape == (3, 4) and actual.dtype == torch.float32
    assert str(actual.device) == vulkan_backend
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
    for tensor, original in zip((vk_a, vk_b, vk_self), before):
        torch.testing.assert_close(tensor.cpu(), original)


@pytest.mark.parametrize("self_shape", [(), (1,), (1, 4), (2, 1), (1, 1), (2, 4)])
def test_addmm_self_broadcasts_singleton_axes(self_shape, vulkan_backend):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 5 - 0.5
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7 - 0.25
    count = torch.empty(self_shape).numel()
    self_cpu = torch.arange(count, dtype=torch.float32).reshape(self_shape) - 0.5
    expected = torch.addmm(self_cpu, a, b, alpha=-0.5, beta=1.25)
    self_vk, a_vk, b_vk = (x.to(vulkan_backend) for x in (self_cpu, a, b))
    actual = torch.addmm(self_vk, a_vk, b_vk, alpha=-0.5, beta=1.25)
    pytorch_vulkan._C.synchronize()
    assert actual.shape == expected.shape == (2, 4)
    assert str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("self_shape", [(), (4,), (2, 1), (1, 1), (2, 4)])
@pytest.mark.parametrize("beta", [-0.5, 0.0])
def test_addmm_generated_autograd_reduces_broadcast_self(self_shape, beta, vulkan_backend):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7 - 0.2
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 9 - 0.3
    self_cpu = (torch.arange(torch.empty(self_shape).numel(), dtype=torch.float32)
                .reshape(self_shape) / 5 - 0.4)
    cpu_inputs = [self_cpu.clone().requires_grad_(), a.requires_grad_(), b.requires_grad_()]
    vk_inputs = [x.detach().to(vulkan_backend).requires_grad_() for x in cpu_inputs]
    cpu_seed = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11 + 0.2
    vk_seed = cpu_seed.to(vulkan_backend)
    cpu_out = torch.addmm(*cpu_inputs, beta=beta, alpha=1.75)
    vk_out = torch.addmm(*vk_inputs, beta=beta, alpha=1.75)
    cpu_grads = torch.autograd.grad(cpu_out, cpu_inputs, cpu_seed, create_graph=True)
    vk_grads = torch.autograd.grad(vk_out, vk_inputs, vk_seed, create_graph=True)
    for got, ref in zip(vk_grads, cpu_grads):
        _assert_vulkan_cpu_pair(got, ref, vulkan_backend)
        assert got.requires_grad == ref.requires_grad


@pytest.mark.parametrize("alpha,beta,role", [(0.0, 1.0, "product"), (1.0, 0.0, "self")])
def test_addmm_ignored_nan_term(alpha, beta, role, vulkan_backend):
    a, b, self_cpu = torch.ones(2, 3), torch.ones(3, 4), torch.ones(4)
    if role == "product":
        a.fill_(float("nan"))
    else:
        self_cpu.fill_(float("nan"))
    expected = torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta)
    actual = torch.addmm(
        self_cpu.to(vulkan_backend), a.to(vulkan_backend), b.to(vulkan_backend),
        alpha=alpha, beta=beta
    )
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003, equal_nan=True)


@pytest.mark.parametrize("alpha,beta", [(float("nan"), 1.0), (float("inf"), 1.0),
                                         (-float("inf"), 1.0), (1.0, float("nan")),
                                         (1.0, float("inf")), (1.0, -float("inf"))])
def test_addmm_nonfinite_scalars_match_cpu(alpha, beta, vulkan_backend):
    a = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    b = torch.tensor([[2.0, 1.0], [-1.0, 4.0]])
    self_cpu = torch.tensor([0.5, -3.0])
    expected = torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta)
    actual = torch.addmm(self_cpu.to(vulkan_backend), a.to(vulkan_backend),
                         b.to(vulkan_backend), alpha=alpha, beta=beta)
    pytorch_vulkan._C.synchronize()
    got = actual.cpu()
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    assert torch.equal(torch.signbit(got[torch.isinf(got)]),
                       torch.signbit(expected[torch.isinf(expected)]))
    torch.testing.assert_close(got, expected, rtol=0.003, atol=0.003, equal_nan=True)


def test_addmm_rejects_scalar_overflow_like_cpu(vulkan_backend):
    self_cpu, a, b = torch.ones(2, 4), torch.ones(2, 3), torch.ones(3, 4)
    with pytest.raises(RuntimeError, match="converted to type float"):
        torch.addmm(self_cpu, a, b, alpha=1e300)
    self_vk, a_vk, b_vk = (tensor.to(vulkan_backend) for tensor in (self_cpu, a, b))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.addmm(self_vk, a_vk, b_vk, alpha=1e300)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0


@pytest.mark.parametrize(
    "m,k,n,alpha,beta",
    [
        (0, 3, 4, 1e300, 1e300),
        (2, 3, 0, 1e300, 1e300),
        (2, 0, 4, 1e300, 1.0),
        (2, 0, 4, 1 + 1j, 0.0),
        (2, 0, 4, 1e300, 1e300),
    ],
)
def test_addmm_ignored_scalar_conversion_matches_cpu(
    m, k, n, alpha, beta, vulkan_backend
):
    a, b = torch.empty(m, k), torch.empty(k, n)
    self_cpu = torch.arange(n, dtype=torch.float32) + 1
    expected = torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta)
    actual = torch.addmm(
        self_cpu.to(vulkan_backend), a.to(vulkan_backend), b.to(vulkan_backend),
        alpha=alpha, beta=beta
    )
    pytorch_vulkan._C.synchronize()
    assert actual.shape == expected.shape and str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("beta", [1e-300, -1e-300])
def test_addmm_zero_k_underflowed_nonzero_beta_still_multiplies_self(
    beta, vulkan_backend
):
    self_cpu = torch.tensor([float("nan"), float("inf"), -2.0, 0.0])
    a, b = torch.empty(2, 0), torch.empty(0, 4)
    expected = torch.addmm(self_cpu, a, b, beta=beta)
    self_vk, a_vk, b_vk = (value.to(vulkan_backend) for value in (self_cpu, a, b))
    before = self_vk.cpu().clone()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.addmm(self_vk, a_vk, b_vk, beta=beta)
    pytorch_vulkan._C.synchronize()
    got = actual.cpu()
    assert actual.shape == expected.shape == (2, 4)
    assert str(actual.device) == vulkan_backend
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    finite = torch.isfinite(expected)
    assert torch.equal(torch.signbit(got[finite]), torch.signbit(expected[finite]))
    torch.testing.assert_close(got, expected, rtol=0.003, atol=0.003, equal_nan=True)
    torch.testing.assert_close(self_vk.cpu(), before, equal_nan=True)


@pytest.mark.parametrize("alpha,beta", [(1e300, 0.0), (1.0, 1e300)])
def test_addmm_participating_scalar_overflow_matches_cpu(alpha, beta, vulkan_backend):
    self_cpu, a, b = torch.ones(2, 4), torch.ones(2, 3), torch.ones(3, 4)
    with pytest.raises(RuntimeError, match="converted to type float"):
        torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta)
    with pytest.raises(RuntimeError):
        torch.addmm(self_cpu.to(vulkan_backend), a.to(vulkan_backend),
                    b.to(vulkan_backend), alpha=alpha, beta=beta)


@pytest.mark.parametrize("m,k,n", [(0, 3, 4), (2, 3, 0), (2, 0, 4)])
@pytest.mark.parametrize("beta,nan_self", [(0.0, True), (2.0, True), (-0.0, False)])
def test_addmm_empty_dimensions_match_cpu_and_stay_on_vulkan(
    m, k, n, beta, nan_self, vulkan_backend
):
    a = torch.ones(m, k)
    b = torch.ones(k, n)
    self_cpu = torch.full((n,), float("nan") if nan_self else 2.0)
    expected = torch.addmm(self_cpu, a, b, beta=beta, alpha=-3.0)
    actual = torch.addmm(self_cpu.to(vulkan_backend), a.to(vulkan_backend),
                         b.to(vulkan_backend), beta=beta, alpha=-3.0)
    pytorch_vulkan._C.synchronize()
    assert actual.shape == expected.shape and str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003,
                               equal_nan=True)


def test_addmm_participating_tensor_nonfinites_match_cpu(vulkan_backend):
    a = torch.tensor([[float("inf"), -1.0], [float("nan"), 2.0]])
    b = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    self_cpu = torch.tensor([1.0, -float("inf")])
    expected = torch.addmm(self_cpu, a, b, alpha=-2.0, beta=0.5)
    actual = torch.addmm(self_cpu.to(vulkan_backend), a.to(vulkan_backend),
                         b.to(vulkan_backend), alpha=-2.0, beta=0.5)
    pytorch_vulkan._C.synchronize()
    got = actual.cpu()
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    assert torch.equal(torch.signbit(got[torch.isinf(got)]),
                       torch.signbit(expected[torch.isinf(expected)]))


def test_bmm_rejects_unequal_batch_rank_and_inner_dimension(vulkan_backend):
    a = torch.ones(2, 3, 4, device=vulkan_backend)
    b_batch = torch.ones(3, 4, 5, device=vulkan_backend)
    b_inner = torch.ones(2, 5, 3, device=vulkan_backend)
    b_rank = torch.ones(4, 5, device=vulkan_backend)
    pytorch_vulkan._C.synchronize()
    for rhs in (b_batch, b_inner, b_rank):
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(RuntimeError):
            torch.bmm(a, rhs)
        assert pytorch_vulkan._C.compute_dispatch_count() == 0


@pytest.mark.parametrize("shape", [(0, 2, 3, 4), (2, 0, 3, 4),
                                    (2, 3, 0, 4), (2, 3, 4, 0)])
def test_bmm_empty_dimensions_match_cpu(shape, vulkan_backend):
    batch, m, k, n = shape
    cpu_a = torch.arange(batch * m * k, dtype=torch.float32).reshape(batch, m, k)
    cpu_b = torch.arange(batch * k * n, dtype=torch.float32).reshape(batch, k, n)
    actual = torch.bmm(cpu_a.to(vulkan_backend), cpu_b.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    expected = torch.bmm(cpu_a, cpu_b)
    assert actual.shape == expected.shape and str(actual.device) == vulkan_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


def test_bmm_batches_and_readable_layouts_match_cpu(vulkan_backend):
    cases = []

    dense_a = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 11
    dense_b = torch.arange(36, dtype=torch.float32).reshape(3, 3, 4) / 13
    cases.append((dense_a, dense_b, lambda x: x, lambda x: x))

    transpose_base = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) / 7
    rhs_base = torch.arange(36, dtype=torch.float32).reshape(3, 3, 4) / 5
    cases.append((transpose_base, rhs_base, lambda x: x.transpose(1, 2), lambda x: x))

    lhs_base = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 7
    rhs_transpose_base = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3) / 5
    cases.append((lhs_base, rhs_transpose_base, lambda x: x,
                  lambda x: x.transpose(1, 2)))

    offset_a_base = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6) / 17
    offset_b_base = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(3, 6, 8) / 19
    cases.append((offset_a_base, offset_b_base,
                  lambda x: x[:, ::2, 1::2], lambda x: x[:, ::2, ::2]))

    expanded_a_base = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) / 3
    expanded_b_base = dense_b.clone()
    cases.append((expanded_a_base, expanded_b_base,
                  lambda x: x.expand(3, 2, 3), lambda x: x))

    expanded_b_base = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 11
    cases.append((dense_a, expanded_b_base, lambda x: x,
                  lambda x: x.expand(3, 3, 4)))

    overlap_a_base = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 4, 4) / 23
    overlap_b_base = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2) / 29
    cases.append((overlap_a_base, overlap_b_base,
                  lambda x: x.as_strided((3, 2, 3), (16, 1, 1)), lambda x: x))

    overlap_lhs_base = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)
    overlap_rhs_base = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 4, 4) / 31
    cases.append((overlap_lhs_base, overlap_rhs_base, lambda x: x,
                  lambda x: x.as_strided((3, 3, 4), (16, 1, 1))))

    singleton_a_base = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1) / 5
    singleton_b_base = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) / 9
    cases.append((singleton_a_base, singleton_b_base, lambda x: x, lambda x: x))

    for base_a, base_b, view_a, view_b in cases:
        cpu_a, cpu_b = view_a(base_a), view_b(base_b)
        vk_base_a, vk_base_b = base_a.to(vulkan_backend), base_b.to(vulkan_backend)
        vk_a, vk_b = view_a(vk_base_a), view_b(vk_base_b)
        assert vk_a.shape == cpu_a.shape and vk_b.shape == cpu_b.shape
        assert vk_a.stride() == cpu_a.stride() and vk_b.stride() == cpu_b.stride()
        assert vk_a.storage_offset() == cpu_a.storage_offset()
        assert vk_b.storage_offset() == cpu_b.storage_offset()
        before_a, before_b = vk_base_a.cpu().clone(), vk_base_b.cpu().clone()
        pytorch_vulkan._C.synchronize()
        pytorch_vulkan._C.reset_execution_counters()
        actual = torch.bmm(vk_a, vk_b)
        pytorch_vulkan._C.synchronize()
        expected = torch.bmm(cpu_a, cpu_b)
        assert actual.shape == expected.shape and str(actual.device) == vulkan_backend
        assert pytorch_vulkan._C.compute_dispatch_count() > 0
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
        torch.testing.assert_close(vk_base_a.cpu(), before_a)
        torch.testing.assert_close(vk_base_b.cpu(), before_b)
        with pytest.raises(RuntimeError):
            torch.bmm(vk_a, vk_b[:2])
