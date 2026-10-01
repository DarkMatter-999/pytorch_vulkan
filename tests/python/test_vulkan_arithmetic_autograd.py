import pytest
import pytorch_vulkan
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cpu_pair():
    generator = torch.Generator().manual_seed(811)
    return tuple(
        torch.randn(2, 3, generator=generator).requires_grad_()
        for _ in range(2)
    )


def _vk_from_cpu(value, device):
    return value.detach().to(device).requires_grad_(value.requires_grad)


@pytest.fixture
def vulkan_backend():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_cpu_expanded_binary_hvp_matches_analytic_values():
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = torch.tensor([2., -1., 3.], requires_grad=True)
    ga, gb = torch.autograd.grad(
        (a * b.expand(2, 3) + a).sum(), (a, b), create_graph=True
    )
    assert torch.equal(ga, b.expand(2, 3) + 1)
    assert torch.equal(gb, torch.tensor([5., 7., 9.]))

    u = torch.tensor([[1., 2., 4.], [3., 5., 7.]])
    v = torch.tensor([2., 3., 5.])
    h_a, h_b = torch.autograd.grad((ga * u).sum() + (gb * v).sum(), (a, b))
    assert torch.equal(h_a, v.expand(2, 3))
    assert torch.equal(h_b, torch.tensor([4., 7., 11.]))


def test_cpu_equal_shape_and_trainable_scalar_upstream_hvp():
    a = torch.tensor([1., 2.], requires_grad=True)
    b = torch.tensor([3., 4.], requires_grad=True)
    loss = (a * b + a).sum()
    ga, gb = torch.autograd.grad(loss, (a, b), create_graph=True)
    u, v = torch.tensor([5., 6.]), torch.tensor([7., 8.])
    h_a, h_b = torch.autograd.grad((ga * u).sum() + (gb * v).sum(), (a, b))
    assert torch.equal(h_a, v)
    assert torch.equal(h_b, u)

    s = torch.tensor(2., requires_grad=True)
    ga_s, gb_s = torch.autograd.grad(loss, (a, b), grad_outputs=s, create_graph=True)
    h_s = torch.autograd.grad((ga_s * u).sum() + (gb_s * v).sum(), (a, b, s))
    assert torch.equal(h_s[0], torch.tensor([14., 16.]))
    assert torch.equal(h_s[1], torch.tensor([10., 12.]))
    assert torch.equal(h_s[2], torch.tensor(73.))


def test_cpu_sum_scalar_input_schema_normalization():
    x = torch.tensor(2., requires_grad=True)
    for dims in (None, [], [0]):
        y = torch.sum(x) if dims is None else torch.sum(x, dim=dims)
        assert y.shape == torch.Size([])
        assert y.item() == 2.
        assert torch.autograd.grad(y, x, create_graph=True)[0].item() == 1.


@pytest.mark.parametrize(
    "dims,keepdim,dtype",
    [(None, False, None), ([], False, torch.float32), ([0], False, None),
     ([-1], True, torch.float32), ([0, 2], False, None), ([0, 2], True, None)],
)
def test_vk_sum_generated_autograd_matches_cpu(vulkan_backend, dims, keepdim, dtype):
    generator = torch.Generator().manual_seed(991)
    cpu_x = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
    vk_x = _vk_from_cpu(cpu_x, vulkan_backend)
    upstream_shape = () if dims is None or not dims else tuple(
        1 if keepdim and i in [d % cpu_x.dim() for d in dims] else s
        for i, s in enumerate(cpu_x.shape)
        if keepdim or i not in [d % cpu_x.dim() for d in dims]
    )
    cpu_g = torch.randn(upstream_shape, requires_grad=True)
    vk_g = cpu_g.detach().to(vulkan_backend).requires_grad_()

    cpu_y = torch.sum(cpu_x, dim=dims, keepdim=keepdim, dtype=dtype)
    vk_y = torch.sum(vk_x, dim=dims, keepdim=keepdim, dtype=dtype)
    cpu_dx = torch.autograd.grad(cpu_y, cpu_x, cpu_g, create_graph=True)[0]
    vk_dx = torch.autograd.grad(vk_y, vk_x, vk_g, create_graph=True)[0]
    cpu_dg = torch.autograd.grad(cpu_dx, cpu_g, torch.ones_like(cpu_dx))[0]
    vk_dg = torch.autograd.grad(vk_dx, vk_g, torch.ones_like(vk_dx))[0]
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_y.cpu(), cpu_y)
    torch.testing.assert_close(vk_dx.cpu(), cpu_dx)
    torch.testing.assert_close(vk_dg.cpu(), cpu_dg)


@pytest.mark.parametrize("dims", [None, [], [0], [-1]])
@pytest.mark.parametrize("keepdim,dtype", [
    (False, None), (False, torch.float32), (True, None), (True, torch.float32)
])
def test_vk_sum_scalar_input_is_fresh_and_differentiable(
    vulkan_backend, dims, keepdim, dtype
):
    cpu_x = torch.tensor(2., requires_grad=True)
    vk_x = _vk_from_cpu(cpu_x, vulkan_backend)
    if dims is None and not keepdim:
        cpu_y = torch.sum(cpu_x, dtype=dtype)
        vk_y = torch.sum(vk_x, dtype=dtype)
    else:
        cpu_y = torch.sum(cpu_x, dim=dims, keepdim=keepdim, dtype=dtype)
        vk_y = torch.sum(vk_x, dim=dims, keepdim=keepdim, dtype=dtype)
    assert vk_y.shape == torch.Size([])
    assert vk_y.data_ptr() != vk_x.data_ptr()
    cpu_dx = torch.autograd.grad(cpu_y, cpu_x, create_graph=True)[0]
    vk_dx = torch.autograd.grad(vk_y, vk_x, create_graph=True)[0]
    torch.testing.assert_close(vk_y.cpu(), cpu_y)
    torch.testing.assert_close(vk_dx.cpu(), cpu_dx)


@pytest.mark.parametrize("scalar", [False, True])
def test_vk_expand_generated_backward_reduces_to_source_shape(vulkan_backend, scalar):
    if scalar:
        cpu_x = torch.tensor(2., requires_grad=True)
        cpu_y = cpu_x.expand(2, 3)
    else:
        cpu_x = torch.tensor([1., 2., 3.], requires_grad=True)
        cpu_y = cpu_x.expand(2, 3)
    vk_x = _vk_from_cpu(cpu_x, vulkan_backend)
    vk_y = vk_x.expand(2, 3)
    cpu_g = torch.tensor([[1., 2., 4.], [3., 5., 7.]], requires_grad=True)
    vk_g = cpu_g.detach().to(vulkan_backend).requires_grad_()
    cpu_dx = torch.autograd.grad(cpu_y, cpu_x, cpu_g, create_graph=True)[0]
    vk_dx = torch.autograd.grad(vk_y, vk_x, vk_g, create_graph=True)[0]
    cpu_ddx = torch.autograd.grad(cpu_dx.sum(), cpu_g)[0]
    vk_ddx = torch.autograd.grad(vk_dx.sum(), vk_g)[0]
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_dx.cpu(), cpu_dx)
    torch.testing.assert_close(vk_ddx.cpu(), cpu_ddx)


def test_vk_add_accepts_zero_stride_expanded_upstream(vulkan_backend):
    cpu_a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    cpu_b = torch.tensor([[2., 3., 4.], [5., 6., 7.]], requires_grad=True)
    vk_a, vk_b = _vk_from_cpu(cpu_a, vulkan_backend), _vk_from_cpu(cpu_b, vulkan_backend)
    cpu_base = torch.tensor([[1., 2., 4.]])
    cpu_g = cpu_base.expand(2, 3).requires_grad_()
    vk_g = cpu_base.to(vulkan_backend).expand(2, 3).requires_grad_()
    assert vk_g.stride(0) == 0
    cpu_grads = torch.autograd.grad(cpu_a + cpu_b, (cpu_a, cpu_b), cpu_g, create_graph=True)
    vk_grads = torch.autograd.grad(vk_a + vk_b, (vk_a, vk_b), vk_g, create_graph=True)
    cpu_dg = torch.autograd.grad(sum(g.sum() for g in cpu_grads), cpu_g)[0]
    vk_dg = torch.autograd.grad(sum(g.sum() for g in vk_grads), vk_g)[0]
    pytorch_vulkan._C.synchronize()
    for actual, expected in zip(vk_grads, cpu_grads):
        torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(vk_dg.cpu(), cpu_dg)


@pytest.mark.parametrize("operation,dim,seed_shape", [
    (torch.amax, 2, (1, 3)),
    (torch.softmax, 2, (2, 1, 4)),
])
def test_vk_reduction_backward_materializes_expanded_read_grad(
    vulkan_backend, operation, dim, seed_shape
):
    cpu_x = torch.tensor(
        [[[1., 3., 2., 4.], [4., 1., 2., 3.], [2., 5., 1., 3.]],
         [[2., 4., 1., 3.], [3., 2., 5., 1.], [5., 1., 3., 2.]]],
        requires_grad=True,
    )
    vk_x = _vk_from_cpu(cpu_x, vulkan_backend)
    cpu_y, vk_y = operation(cpu_x, dim=dim), operation(vk_x, dim=dim)
    seed = torch.arange(1., 1. + torch.tensor(seed_shape).prod().item()).reshape(seed_shape)
    cpu_seed = seed.expand(cpu_y.shape)
    vk_seed = seed.to(vulkan_backend).expand(vk_y.shape)
    assert any(stride == 0 for stride in vk_seed.stride())

    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_dx = torch.autograd.grad(cpu_y, cpu_x, cpu_seed)[0]
    vk_dx = torch.autograd.grad(vk_y, vk_x, vk_seed)[0]
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] > 0
    assert counters[1] > 0
    assert counters[2] == 0
    torch.testing.assert_close(vk_dx.cpu(), cpu_dx)


@pytest.mark.parametrize("dims", [[1], [0, 0], [2]])
def test_vk_sum_scalar_rejects_invalid_dimensions(vulkan_backend, dims):
    with pytest.raises((IndexError, RuntimeError)):
        torch.sum(torch.tensor(2.).to(vulkan_backend), dim=dims)


def test_vk_sum_rejects_mixed_expanded_and_overlapping_read_layout(vulkan_backend):
    base = torch.tensor([1., 2., 3.]).to(vulkan_backend)
    overlapping = base.as_strided((2, 2, 2), (0, 1, 1))
    with pytest.raises(RuntimeError, match="overlap|overlapping"):
        torch.sum(overlapping, dim=2)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
@pytest.mark.parametrize("schema,dims", [("default", None), ("dim", []), ("dim", [0])])
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_vk_scalar_sum_rejects_unsupported_input_dtype_before_copy(
    vulkan_backend, dtype, schema, output_dtype, dims
):
    vk_x = torch.tensor([1], dtype=dtype).to(vulkan_backend).select(0, 0)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="Vulkan sum supports only float32 tensors"):
        if schema == "default":
            torch.sum(vk_x, dtype=output_dtype)
        else:
            torch.sum(vk_x, dim=dims, dtype=output_dtype)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "name,operation,binary",
    [
        ("add_tensor_alpha1", lambda a, b: torch.add(a, b), True),
        ("add_tensor_alpha2", lambda a, b: torch.add(a, b, alpha=2), True),
        ("add_scalar_alpha2", lambda a: torch.add(a, 2.5, alpha=2), False),
        ("mul_tensor", torch.mul, True),
        ("mul_scalar", lambda a: torch.mul(a, 2.0), False),
    ],
)
def test_vk_add_mul_generated_autograd_matches_cpu_and_trainable_upstream(
    vulkan_backend, name, operation, binary
):
    def evaluate(device):
        a, b = _cpu_pair()
        a, b = _vk_from_cpu(a, device), _vk_from_cpu(b, device)
        g = torch.tensor([[1., 2., 4.], [3., 5., 7.]], device="cpu")
        g = g.to(device).requires_grad_()
        if str(device).startswith("vk"):
            pytorch_vulkan._C.synchronize()
            pytorch_vulkan._C.reset_execution_counters()
        y = operation(a, b) if binary else operation(a)
        leaves = (a, b) if binary else (a,)
        gradients = torch.autograd.grad(y, leaves, g, create_graph=True)
        return y, gradients, g, leaves

    cpu_y, cpu_gradients, cpu_g, cpu_leaves = evaluate("cpu")
    # Prepare all test tensors before measuring Vulkan execution.
    vk_y, vk_gradients, vk_g, vk_leaves = evaluate(vulkan_backend)
    vk_seeds = tuple(torch.ones_like(value) for value in vk_gradients)
    cpu_seeds = tuple(torch.ones_like(value) for value in cpu_gradients)

    assert all(value.requires_grad for value in cpu_gradients)
    assert all(value.requires_grad for value in vk_gradients)
    vk_dg = torch.autograd.grad(vk_gradients, vk_g, vk_seeds)[0]
    pytorch_vulkan._C.synchronize()
    vk_counts = pytorch_vulkan._C.execution_counter_snapshot()
    cpu_dg = torch.autograd.grad(cpu_gradients, cpu_g, cpu_seeds)[0]

    # Capture dispatch evidence before any tensor readback.
    assert vk_counts[0] > 0
    torch.testing.assert_close(vk_y.cpu(), cpu_y)
    for actual, expected in zip(vk_gradients, cpu_gradients):
        torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(vk_dg.cpu(), cpu_dg)
    a_values, b_values = _cpu_pair()
    g_values = torch.tensor([[1., 2., 4.], [3., 5., 7.]])
    if name == "add_tensor_alpha1":
        expected_gradients = (g_values, g_values)
        expected_dg = torch.full_like(g_values, 2.)
    elif name == "add_tensor_alpha2":
        expected_gradients = (g_values, 2 * g_values)
        expected_dg = torch.full_like(g_values, 3.)
    elif name == "add_scalar_alpha2":
        expected_gradients = (g_values,)
        expected_dg = torch.ones_like(g_values)
    elif name == "mul_tensor":
        expected_gradients = (g_values * b_values.detach(), g_values * a_values.detach())
        expected_dg = a_values.detach() + b_values.detach()
    else:
        expected_gradients = (2 * g_values,)
        expected_dg = torch.full_like(g_values, 2.)
    for actual, expected in zip(vk_gradients, expected_gradients):
        torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(vk_dg.cpu(), expected_dg)


def test_vk_add_scalar_left_alpha_matches_cpu_and_trainable_upstream(vulkan_backend):
    cpu_x = torch.tensor([1., 3.], requires_grad=True)
    vk_x = _vk_from_cpu(cpu_x, vulkan_backend)
    cpu_g = torch.tensor([2., 5.], requires_grad=True)
    vk_g = cpu_g.detach().to(vulkan_backend).requires_grad_()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()

    cpu_y = torch.add(2.0, cpu_x, alpha=2)
    vk_y = torch.add(2.0, vk_x, alpha=2)
    cpu_dx = torch.autograd.grad(cpu_y, cpu_x, cpu_g, create_graph=True)[0]
    vk_dx = torch.autograd.grad(vk_y, vk_x, vk_g, create_graph=True)[0]
    assert cpu_dx.requires_grad
    assert vk_dx.requires_grad
    cpu_dg = torch.autograd.grad(cpu_dx, cpu_g, torch.ones_like(cpu_dx))[0]
    vk_dg = torch.autograd.grad(vk_dx, vk_g, torch.ones_like(vk_dx))[0]

    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] > 0
    torch.testing.assert_close(cpu_y, torch.tensor([4., 8.]))
    torch.testing.assert_close(vk_y.cpu(), cpu_y)
    torch.testing.assert_close(vk_dx.cpu(), cpu_dx)
    torch.testing.assert_close(cpu_dx, 2 * cpu_g)
    torch.testing.assert_close(vk_dg.cpu(), cpu_dg)
    torch.testing.assert_close(vk_dg.cpu(), torch.full((2,), 2.))


@pytest.mark.parametrize("expanded", [False, True])
def test_vk_composed_arithmetic_hvp_matches_cpu_and_analytic(
    vulkan_backend, expanded
):
    generator = torch.Generator().manual_seed(2401)
    cpu_a = torch.randn(2, 3, generator=generator, requires_grad=True)
    cpu_b = (
        torch.randn(3, generator=generator, requires_grad=True)
        if expanded
        else torch.randn(2, 3, generator=generator, requires_grad=True)
    )
    cpu_s = torch.tensor(1.7, requires_grad=True)
    cpu_u = torch.randn(2, 3, generator=generator)
    cpu_v = (
        torch.randn(3, generator=generator)
        if expanded
        else torch.randn(2, 3, generator=generator)
    )

    def evaluate(device):
        a, b = _vk_from_cpu(cpu_a, device), _vk_from_cpu(cpu_b, device)
        s = _vk_from_cpu(cpu_s, device)
        u, v = cpu_u.to(device), cpu_v.to(device)
        operand = b.expand_as(a) if expanded else b
        loss = (a * operand + a).sum()
        ga, gb = torch.autograd.grad(loss, (a, b), grad_outputs=s, create_graph=True)
        hvp = torch.autograd.grad((ga * u).sum() + (gb * v).sum(), (a, b, s))
        return loss, ga, gb, hvp

    cpu_result = evaluate("cpu")
    vk_result = evaluate(vulkan_backend)
    a, b = cpu_a.detach(), cpu_b.detach()
    if expanded:
        analytic_hvp = (
            cpu_s.detach() * cpu_v.expand_as(a),
            cpu_s.detach() * cpu_u.sum(0),
            (cpu_u * (b.expand_as(a) + 1)).sum() + (cpu_v * a.sum(0)).sum(),
        )
        analytic_gb = cpu_s.detach() * a.sum(0)
    else:
        analytic_hvp = (
            cpu_s.detach() * cpu_v,
            cpu_s.detach() * cpu_u,
            (cpu_u * (b + 1)).sum() + (cpu_v * a).sum(),
        )
        analytic_gb = cpu_s.detach() * a
    analytic_ga = cpu_s.detach() * (b.expand_as(a) + 1)
    pytorch_vulkan._C.synchronize()
    for actual, expected in zip(vk_result, cpu_result):
        if isinstance(expected, tuple):
            for item, reference, formula in zip(actual, expected, analytic_hvp):
                torch.testing.assert_close(item.cpu(), reference)
                torch.testing.assert_close(item.cpu(), formula)
        else:
            torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(cpu_result[1], analytic_ga)
    torch.testing.assert_close(cpu_result[2], analytic_gb)
    assert all(x.requires_grad for x in cpu_result[1:3])
    assert all(x.requires_grad for x in vk_result[1:3])


def test_vk_alias_version_and_saved_mul_operand_match_cpu(vulkan_backend):
    def alias_version(device):
        base = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
        view = base[1:]
        before = base._version
        with torch.no_grad():
            view.add_(1.)
        return base._version - before, view, base

    cpu_bump, cpu_view, cpu_base = alias_version("cpu")
    vk_bump, vk_view, vk_base = alias_version(vulkan_backend)
    assert vk_bump == cpu_bump == 1
    torch.testing.assert_close(vk_view.cpu(), cpu_view)
    torch.testing.assert_close(vk_base.cpu(), cpu_base)

    def saved_operand_error(device):
        a = torch.tensor([2., 3.], device=device, requires_grad=True)
        b = torch.tensor([4., 5.], device=device, requires_grad=True)
        loss = (a * b).sum()
        with torch.no_grad():
            b.add_(1.)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            torch.autograd.grad(loss, (a, b))

    saved_operand_error("cpu")
    saved_operand_error(vulkan_backend)
    pytorch_vulkan._C.synchronize()


def test_mean_dim_sum_backward_matches_cpu_for_expanded_seed(vulkan_backend):
    cpu_input = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    cpu_loss = cpu_input.mean(dim=1).sum()
    vk_loss = vk_input.mean(dim=1).sum()
    cpu_loss.backward()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_loss.backward()
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)

    cpu_input2 = torch.arange(1.0, 7.0).reshape(2, 3).requires_grad_()
    vk_input2 = cpu_input2.detach().to(vulkan_backend).requires_grad_()
    cpu_output = cpu_input2.mean(dim=1)
    vk_output = vk_input2.mean(dim=1)
    upstream = torch.tensor([1.25, -2.5])
    cpu_output.backward(upstream)
    pytorch_vulkan._C.synchronize()
    vk_upstream = upstream.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_upstream)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(vk_input2.grad.cpu(), cpu_input2.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("module_api", ["functional", "module"])
@pytest.mark.parametrize("train_input,train_weight", [(True, True), (True, False), (False, True)])
def test_linear_sum_backward_handles_expanded_seed_and_matches_cpu(
    vulkan_backend, module_api, train_input, train_weight
):
    torch.manual_seed(951)
    cpu_input = torch.randn(2, 3, requires_grad=train_input)
    cpu_weight = torch.randn(4, 3, requires_grad=train_weight)
    cpu_bias = torch.randn(4)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_(train_input)
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_(train_weight)
    vk_bias = cpu_bias.to(vulkan_backend)
    if module_api == "functional":
        cpu_output = F.linear(cpu_input, cpu_weight, cpu_bias)
        vk_output = F.linear(vk_input, vk_weight, vk_bias)
    else:
        cpu_module = nn.Linear(3, 4)
        with torch.no_grad():
            cpu_module.weight.copy_(cpu_weight)
            cpu_module.bias.copy_(cpu_bias)
        cpu_module.weight.requires_grad_(train_weight)
        cpu_module.bias.requires_grad_(False)
        vk_module = nn.Linear(3, 4)
        with torch.no_grad():
            vk_module.weight.copy_(cpu_weight.detach())
            vk_module.bias.copy_(cpu_bias)
        vk_module.to(vulkan_backend)
        vk_module.weight.requires_grad_(train_weight)
        vk_module.bias.requires_grad_(False)
        cpu_weight = cpu_module.weight
        vk_weight = vk_module.weight
        cpu_output = cpu_module(cpu_input)
        vk_output = vk_module(vk_input)

    cpu_output.sum().backward(retain_graph=True)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.sum().backward(retain_graph=True)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    if train_input:
        torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=2e-3, atol=2e-3)
    if train_weight:
        torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad, rtol=2e-3, atol=2e-3)

    if train_input or train_weight:
        cpu_input.grad = None
        vk_input.grad = None
        if train_weight:
            cpu_weight.grad = None
            vk_weight.grad = None
        seed = torch.tensor([[1.25], [-2.5]])
        cpu_output.backward(seed.expand_as(cpu_output))
        pytorch_vulkan._C.synchronize()
        vk_seed = seed.to(vulkan_backend).expand_as(vk_output)
        pytorch_vulkan._C.reset_execution_counters()
        vk_output.backward(vk_seed)
        pytorch_vulkan._C.synchronize()
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
        assert pytorch_vulkan._C.fallback_count() == 0
        if train_input:
            torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=2e-3, atol=2e-3)
        if train_weight:
            torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad, rtol=2e-3, atol=2e-3)
