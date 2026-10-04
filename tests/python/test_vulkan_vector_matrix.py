import pytest
import torch

import pytorch_vulkan


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


def _assert_forward(actual, expected, device):
    pytorch_vulkan._C.synchronize()
    assert actual.device == torch.device(device)
    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)


def test_dot_and_mv_have_scalar_and_vector_outputs(vulkan_backend):
    a = torch.tensor([1.25, -2.0, 0.5])
    b = torch.tensor([-3.0, 4.0, 2.0])
    matrix = torch.tensor([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]])

    vk_a, vk_b, vk_matrix = (value.to(vulkan_backend) for value in (a, b, matrix))
    pytorch_vulkan._C.reset_execution_counters()
    dot = torch.dot(vk_a, vk_b)
    mv = torch.mv(vk_matrix, vk_b)
    assert pytorch_vulkan._C.compute_dispatch_count() == 2
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0

    _assert_forward(dot, torch.dot(a, b), vulkan_backend)
    _assert_forward(mv, torch.mv(matrix, b), vulkan_backend)
    assert dot.shape == torch.Size([])
    assert mv.shape == torch.Size([2])
    torch.testing.assert_close(dot.cpu(), torch.tensor(-10.75))
    torch.testing.assert_close(mv.cpu(), torch.tensor([3.0, -5.5]))


@pytest.mark.parametrize("length", [0, 1, 5])
def test_dot_empty_and_regular_contractions_match_cpu(vulkan_backend, length):
    a = torch.arange(length, dtype=torch.float32) - 1.5
    b = torch.arange(length, dtype=torch.float32) * 0.75 + 2.0
    result = torch.dot(a.to(vulkan_backend), b.to(vulkan_backend))
    _assert_forward(result, torch.dot(a, b), vulkan_backend)


@pytest.mark.parametrize("rows,columns", [(0, 4), (3, 0), (0, 0), (2, 4)])
def test_mv_empty_dimensions_and_regular_contraction_match_cpu(
    vulkan_backend, rows, columns
):
    matrix = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    vector = torch.arange(columns, dtype=torch.float32) - 0.5
    result = torch.mv(matrix.to(vulkan_backend), vector.to(vulkan_backend))
    _assert_forward(result, torch.mv(matrix, vector), vulkan_backend)


def test_dot_accepts_nonunit_offset_zero_stride_and_overlapping_reads(vulkan_backend):
    base = torch.tensor([2.0, -4.0, 1.5, 8.0, -3.0, 7.0])
    vk_base = base.to(vulkan_backend)
    cases = [
        (base[::2], vk_base[::2]),
        (base[1:5:2], vk_base[1:5:2]),
        (base.as_strided((4,), (0,)), vk_base.as_strided((4,), (0,))),
        (base.as_strided((4,), (1,)), vk_base.as_strided((4,), (1,))),
    ]
    other = torch.tensor([-1.0, 0.5, 3.0, 2.0])
    for cpu_vector, vk_vector in cases:
        rhs = other[: cpu_vector.numel()]
        actual = torch.dot(vk_vector, rhs.to(vulkan_backend))
        _assert_forward(actual, torch.dot(cpu_vector, rhs), vulkan_backend)
    torch.testing.assert_close(vk_base.cpu(), base)


def test_mv_accepts_readable_matrix_and_vector_views(vulkan_backend):
    matrix_storage = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 7 - 2
    vector_storage = torch.arange(12, dtype=torch.float32) / 5 - 1
    vk_matrix_storage = matrix_storage.to(vulkan_backend)
    vk_vector_storage = vector_storage.to(vulkan_backend)
    cases = [
        (matrix_storage[1:5, 1:7:2], vk_matrix_storage[1:5, 1:7:2],
         vector_storage[:3], vk_vector_storage[:3]),
        (matrix_storage[:3, :4].t(), vk_matrix_storage[:3, :4].t(),
         vector_storage[1:7:2], vk_vector_storage[1:7:2]),
        (matrix_storage[:1, :4].expand(3, 4),
         vk_matrix_storage[:1, :4].expand(3, 4),
         vector_storage.as_strided((4,), (0,)),
         vk_vector_storage.as_strided((4,), (0,))),
    ]
    for cpu_matrix, vk_matrix, cpu_vector, vk_vector in cases:
        assert cpu_matrix.size(1) == cpu_vector.numel()
        actual = torch.mv(vk_matrix, vk_vector)
        _assert_forward(actual, torch.mv(cpu_matrix, cpu_vector), vulkan_backend)
    # A repeated-row overlapping matrix read is legal and must not be mutated.
    cpu_matrix = matrix_storage[:1, :4].expand(3, 4)
    vk_matrix = vk_matrix_storage[:1, :4].expand(3, 4)
    cpu_vector = vector_storage[:4]
    actual = torch.mv(vk_matrix, cpu_vector.to(vulkan_backend))
    _assert_forward(actual, torch.mv(cpu_matrix, cpu_vector), vulkan_backend)
    torch.testing.assert_close(vk_matrix_storage.cpu(), matrix_storage)
    torch.testing.assert_close(vk_vector_storage.cpu(), vector_storage)


@pytest.mark.parametrize(
    "lhs_values,rhs_values,expected_class,expected_value",
    [
        ([1.0e20, 1.0e20], [1.0e-20, 1.0e-20], "finite", 2.0),
        ([1.0e20, 1.0e20], [1.0e20, -1.0e20], "nan", None),
        ([float("inf"), 2.0], [0.0, 1.0], "nan", None),
        ([float("nan"), 2.0], [1.0, 1.0], "nan", None),
    ],
)
def test_dot_and_mv_vector_extremes_follow_fixed_f32_contraction(
    vulkan_backend, lhs_values, rhs_values, expected_class, expected_value
):
    lhs = torch.tensor(lhs_values, dtype=torch.float32)
    rhs = torch.tensor(rhs_values, dtype=torch.float32)
    cpu_dot = torch.dot(lhs, rhs)
    cpu_mv = torch.mv(torch.stack((lhs, lhs)), rhs)
    assert ("finite" if torch.isfinite(cpu_dot) else "nan" if torch.isnan(cpu_dot)
            else "+inf" if torch.isposinf(cpu_dot) else "-inf") == expected_class
    if expected_class == "finite":
        torch.testing.assert_close(cpu_dot, torch.tensor(expected_value), rtol=0, atol=0)
        torch.testing.assert_close(cpu_mv, torch.full((2,), expected_value), rtol=0, atol=0)
    else:
        assert torch.isnan(cpu_dot) and torch.isnan(cpu_mv).all()

    vk_lhs = lhs.to(vulkan_backend)
    vk_rhs = rhs.to(vulkan_backend)
    actual_dot = torch.dot(vk_lhs, vk_rhs)
    actual_mv = torch.mv(torch.stack((lhs, lhs)).to(vulkan_backend), vk_rhs)
    pytorch_vulkan._C.synchronize()
    observed_dot = actual_dot.cpu()
    observed_mv = actual_mv.cpu()
    observed_class = (
        "finite" if torch.isfinite(observed_dot) else "nan" if torch.isnan(observed_dot)
        else "+inf" if torch.isposinf(observed_dot) else "-inf"
    )
    assert observed_class == expected_class
    if expected_class == "finite":
        torch.testing.assert_close(observed_dot, cpu_dot, rtol=0.003, atol=0.003)
        torch.testing.assert_close(observed_mv, cpu_mv, rtol=0.003, atol=0.003)
    else:
        assert torch.isnan(observed_mv).all()


@pytest.mark.parametrize("operation", ["dot", "mv"])
def test_vector_matrix_rejects_malformed_shapes_before_execution(
    vulkan_backend, operation
):
    if operation == "dot":
        cases = [
            (torch.ones(2, 2), torch.ones(4), RuntimeError),
            (torch.ones(3), torch.ones(4), RuntimeError),
            (torch.empty(0), torch.ones(1), RuntimeError),
        ]
        invoke = lambda a, b: torch.dot(a, b)
    else:
        cases = [
            (torch.ones(3), torch.ones(3), RuntimeError),
            (torch.ones(2, 4), torch.ones(3), RuntimeError),
            (torch.empty(0, 4), torch.ones(3), RuntimeError),
        ]
        invoke = lambda a, b: torch.mv(a, b)
    for cpu_a, cpu_b, error in cases:
        vk_a, vk_b = cpu_a.to(vulkan_backend), cpu_b.to(vulkan_backend)
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(error):
            invoke(vk_a, vk_b)
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
    vk_f32 = torch.ones(4).to(vulkan_backend)
    vk_f64 = vk_f32.to(torch.float64)
    if operation == "dot":
        dtype_operands = (vk_f64, vk_f64)
    else:
        # Preserve mv's 2-D matrix / 1-D vector contract so this reaches its
        # dtype validation rather than stopping at the earlier rank check.
        vk_f64_matrix = torch.ones(1, 4).to(vulkan_backend).to(torch.float64)
        dtype_operands = (vk_f64_matrix, vk_f64)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="strided float32"):
        invoke(*dtype_operands)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0


@pytest.mark.parametrize("operation", ["dot", "mv"])
@pytest.mark.parametrize("vulkan_operand", ["lhs", "rhs"])
def test_vector_matrix_rejects_mixed_devices_before_execution(
    vulkan_backend, operation, vulkan_operand
):
    if operation == "dot":
        lhs_cpu, rhs_cpu = torch.tensor([0.2, -0.4, 0.7]), torch.tensor([0.6, -0.2, 0.9])
        invoke = torch.dot
        expected_error = "Vulkan dot requires matching Vulkan device index 0 vectors"
    else:
        lhs_cpu, rhs_cpu = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]]), torch.tensor([0.6, -0.2, 0.9])
        invoke = torch.mv
        expected_error = "Vulkan mv requires matching Vulkan device index 0 operands"

    if vulkan_operand == "lhs":
        lhs, rhs = lhs_cpu.to(vulkan_backend), rhs_cpu
    else:
        lhs, rhs = lhs_cpu, rhs_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    counters_before = pytorch_vulkan._C.execution_counter_snapshot()
    with pytest.raises(RuntimeError, match=expected_error):
        invoke(lhs, rhs)
    assert pytorch_vulkan._C.execution_counter_snapshot() == counters_before == (0, 0, 0, 0)


def _vk_leaf(value, device, requires_grad=True):
    return value.detach().clone().to(device).requires_grad_(requires_grad)


def _assert_gradient(vk_value, cpu_value):
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_value.cpu(), cpu_value, rtol=0.003, atol=0.003)
    assert vk_value.requires_grad == cpu_value.requires_grad


@pytest.mark.parametrize("operation", ["dot", "mv"])
@pytest.mark.parametrize("selected", ["lhs", "rhs"])
def test_dot_mv_generated_gradients_history_and_selective_inputs(
    vulkan_backend, operation, selected
):
    lhs = torch.tensor([0.2, -0.4, 0.7], requires_grad=selected == "lhs")
    rhs = torch.tensor([0.6, -0.2, 0.9], requires_grad=selected == "rhs")
    if operation == "dot":
        cpu_result = torch.dot(lhs, rhs)
        vk_lhs, vk_rhs = _vk_leaf(lhs, vulkan_backend, lhs.requires_grad), _vk_leaf(
            rhs, vulkan_backend, rhs.requires_grad
        )
        vk_result = torch.dot(vk_lhs, vk_rhs)
        cpu_seed = torch.tensor(0.7)
        vk_seed = cpu_seed.to(vulkan_backend)
        selected_cpu = lhs if selected == "lhs" else rhs
        selected_vk = vk_lhs if selected == "lhs" else vk_rhs
    else:
        lhs = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]],
                           requires_grad=selected == "lhs")
        cpu_result = torch.mv(lhs, rhs)
        vk_lhs, vk_rhs = _vk_leaf(lhs, vulkan_backend, lhs.requires_grad), _vk_leaf(
            rhs, vulkan_backend, rhs.requires_grad
        )
        vk_result = torch.mv(vk_lhs, vk_rhs)
        cpu_seed = torch.tensor([0.7, -0.3])
        vk_seed = cpu_seed.to(vulkan_backend)
        selected_cpu = lhs if selected == "lhs" else rhs
        selected_vk = vk_lhs if selected == "lhs" else vk_rhs

    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_grad = torch.autograd.grad(cpu_result, selected_cpu, cpu_seed, create_graph=True)[0]
    vk_grad = torch.autograd.grad(vk_result, selected_vk, vk_seed, create_graph=True)[0]
    _assert_gradient(vk_grad, cpu_grad)
    assert vk_grad.requires_grad == cpu_grad.requires_grad
    frozen_cpu = rhs if selected == "lhs" else lhs
    frozen_vk = vk_rhs if selected == "lhs" else vk_lhs
    assert not frozen_cpu.requires_grad
    assert not frozen_vk.requires_grad
    unrelated_cpu = torch.ones_like(selected_cpu, requires_grad=True)
    unrelated_vk = torch.ones_like(selected_vk, requires_grad=True)
    assert torch.autograd.grad(
        cpu_result, unrelated_cpu, cpu_seed, allow_unused=True
    )[0] is None
    assert torch.autograd.grad(
        vk_result, unrelated_vk, vk_seed, allow_unused=True
    )[0] is None


@pytest.mark.parametrize("operation", ["dot", "mv"])
@pytest.mark.parametrize("first_operand", ["lhs", "rhs"])
def test_dot_mv_both_cross_mixed_directions_match_cpu_and_centered_fd(
    vulkan_backend, operation, first_operand
):
    if operation == "dot":
        a = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        b = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
        seed = torch.tensor(0.7)
        cotangent = torch.tensor([-0.3, 0.8, 0.4])
        direction = torch.tensor([0.1, 0.5, -0.2])
        first, second = (a, b) if first_operand == "lhs" else (b, a)
        vk_a, vk_b = _vk_leaf(a, vulkan_backend), _vk_leaf(b, vulkan_backend)
        vk_first, vk_second = (vk_a, vk_b) if first_operand == "lhs" else (vk_b, vk_a)

        def cpu_first_at(candidate_second):
            if first_operand == "lhs":
                target = a.detach().requires_grad_()
                result = torch.dot(target, candidate_second)
            else:
                target = b.detach().requires_grad_()
                result = torch.dot(candidate_second, target)
            return torch.autograd.grad(result, target, seed)[0]

        cpu_initial = torch.autograd.grad(torch.dot(a, b), first, seed, create_graph=True)[0]
        vk_initial = torch.autograd.grad(torch.dot(vk_a, vk_b), vk_first,
                                         seed.to(vulkan_backend), create_graph=True)[0]
    else:
        matrix = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]], requires_grad=True)
        vector = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
        seed = torch.tensor([0.7, -0.3])
        if first_operand == "rhs":
            cotangent = torch.tensor([0.3, -0.8, 0.4])
            direction = torch.tensor([[0.1, 0.5, -0.2], [-0.4, 0.3, 0.6]])
            first, second = vector, matrix
        else:
            cotangent = torch.tensor([[0.2, -0.4, 0.1], [0.3, 0.5, -0.6]])
            direction = torch.tensor([0.2, -0.7, 0.4])
            first, second = matrix, vector
        vk_matrix, vk_vector = _vk_leaf(matrix, vulkan_backend), _vk_leaf(vector, vulkan_backend)
        vk_first, vk_second = (vk_vector, vk_matrix) if first_operand == "rhs" else (vk_matrix, vk_vector)
        cpu_initial = torch.autograd.grad(torch.mv(matrix, vector), first, seed,
                                          create_graph=True)[0]
        vk_initial = torch.autograd.grad(
            torch.mv(vk_matrix, vk_vector), vk_first, seed.to(vulkan_backend),
            create_graph=True
        )[0]

        def cpu_first_at(candidate_second):
            if first_operand == "rhs":
                target = vector.detach().requires_grad_()
                result = torch.mv(candidate_second, target)
            else:
                target = matrix.detach().requires_grad_()
                result = torch.mv(target, candidate_second)
            return torch.autograd.grad(result, target, seed)[0]

    cpu_mixed = torch.autograd.grad(cpu_initial, second, cotangent)[0]
    vk_mixed = torch.autograd.grad(vk_initial, vk_second, cotangent.to(vulkan_backend))[0]
    signal = (cpu_mixed * direction).sum()
    assert signal.abs().item() > 0.01

    epsilon = 0.001
    cpu_primary = second.detach()
    plus = cpu_first_at(cpu_primary + epsilon * direction)
    minus = cpu_first_at(cpu_primary - epsilon * direction)
    finite_difference = ((plus - minus) * cotangent).sum() / (2 * epsilon)
    torch.testing.assert_close(signal, finite_difference, rtol=0.008, atol=0.002)
    _assert_gradient(vk_initial, cpu_initial)
    _assert_gradient(vk_mixed, cpu_mixed)


@pytest.mark.parametrize("api", ["matmul", "at"])
@pytest.mark.parametrize("kind", ["vv", "mv", "vm"])
def test_stock_low_rank_matmul_values_gradients_and_graph(
    vulkan_backend, api, kind
):
    if kind == "vv":
        a = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        b = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
        seed = torch.tensor(0.7)
    elif kind == "mv":
        a = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]], requires_grad=True)
        b = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
        seed = torch.tensor([0.7, -0.3])
    else:
        a = torch.tensor([0.2, -0.4], requires_grad=True)
        b = torch.tensor([[0.6, -0.2, 0.9], [0.1, 0.8, -0.3]], requires_grad=True)
        seed = torch.tensor([0.7, -0.3, 0.5])
    vk_a, vk_b = _vk_leaf(a, vulkan_backend), _vk_leaf(b, vulkan_backend)
    operation = torch.matmul if api == "matmul" else lambda x, y: x @ y
    cpu_result, vk_result = operation(a, b), operation(vk_a, vk_b)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_grads = torch.autograd.grad(cpu_result, (a, b), seed, create_graph=True)
    vk_grads = torch.autograd.grad(
        vk_result, (vk_a, vk_b), seed.to(vulkan_backend), create_graph=True
    )
    for vk_grad, cpu_grad in zip(vk_grads, cpu_grads):
        _assert_gradient(vk_grad, cpu_grad)


@pytest.mark.parametrize("kind", ["vv", "mv", "vm"])
def test_stock_low_rank_matmul_empty_contractions_and_gradients(vulkan_backend, kind):
    if kind == "vv":
        a, b = torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True)
        seed = torch.tensor(0.7)
    elif kind == "mv":
        a = torch.empty((2, 0), requires_grad=True)
        b = torch.empty(0, requires_grad=True)
        seed = torch.tensor([0.7, -0.3])
    else:
        a = torch.empty(0, requires_grad=True)
        b = torch.empty((0, 3), requires_grad=True)
        seed = torch.tensor([0.7, -0.3, 0.5])
    vk_a, vk_b = _vk_leaf(a, vulkan_backend), _vk_leaf(b, vulkan_backend)
    cpu_value, vk_value = torch.matmul(a, b), torch.matmul(vk_a, vk_b)
    cpu_grads = torch.autograd.grad(cpu_value, (a, b), seed, create_graph=True)
    vk_grads = torch.autograd.grad(
        vk_value, (vk_a, vk_b), seed.to(vulkan_backend), create_graph=True
    )
    torch.testing.assert_close(vk_value.cpu(), cpu_value)
    for vk_grad, cpu_grad in zip(vk_grads, cpu_grads):
        _assert_gradient(vk_grad, cpu_grad)


def test_low_rank_matmul_view_alias_offsets_empty_and_no_grad(vulkan_backend):
    base = torch.tensor([0.2, -0.4, 0.7, 0.8, 0.3, -0.5], requires_grad=True)
    matrix_base = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]],
                               requires_grad=True)
    vector_cpu = base[1:4]
    matrix_cpu = matrix_base[:, :]
    vk_base = _vk_leaf(base, vulkan_backend)
    vk_matrix_base = _vk_leaf(matrix_base, vulkan_backend)
    cpu_result = torch.mv(matrix_cpu, vector_cpu)
    vk_result = torch.mv(vk_matrix_base, vk_base[1:4])
    seed = torch.tensor([0.3, -0.8])
    cpu_base_grad = torch.autograd.grad(cpu_result, base, seed)[0]
    vk_base_grad = torch.autograd.grad(vk_result, vk_base, seed.to(vulkan_backend))[0]
    _assert_gradient(vk_base_grad, cpu_base_grad)

    empty_a, empty_b = torch.empty(0), torch.empty(0)
    assert torch.matmul(empty_a.to(vulkan_backend), empty_b.to(vulkan_backend)).shape == ()
    with torch.no_grad():
        result = torch.matmul(vk_matrix_base, vk_base[1:4])
    assert not result.requires_grad and result.grad_fn is None
    with torch.no_grad():
        dot_result = torch.dot(vk_base[1:4], vk_base[1:4])
        mv_result = torch.mv(vk_matrix_base, vk_base[1:4])
    assert not dot_result.requires_grad and dot_result.grad_fn is None
    assert not mv_result.requires_grad and mv_result.grad_fn is None


@pytest.mark.parametrize("kind", ["vv", "mv", "vm"])
def test_stock_matmul_noncontiguous_views_propagate_gradients_to_bases(
    vulkan_backend, kind
):
    if kind == "vv":
        lhs_base = torch.tensor([9.0, 0.2, -0.4, 0.7, 8.0], requires_grad=True)
        rhs_base = torch.tensor([0.6, 7.0, -0.2, 6.0, 0.9], requires_grad=True)
        lhs, rhs = lhs_base[1:4], rhs_base[::2]
        vk_lhs_base, vk_rhs_base = _vk_leaf(lhs_base, vulkan_backend), _vk_leaf(
            rhs_base, vulkan_backend
        )
        vk_lhs, vk_rhs = vk_lhs_base[1:4], vk_rhs_base[::2]
    elif kind == "mv":
        lhs_base = torch.tensor([[0.2, 0.8], [-0.4, 0.3], [0.7, -0.5]],
                                requires_grad=True)
        rhs_base = torch.tensor([8.0, 0.6, 7.0, -0.2, 6.0, 0.9], requires_grad=True)
        lhs, rhs = lhs_base.t(), rhs_base[1::2]
        vk_lhs_base, vk_rhs_base = _vk_leaf(lhs_base, vulkan_backend), _vk_leaf(
            rhs_base, vulkan_backend
        )
        vk_lhs, vk_rhs = vk_lhs_base.t(), vk_rhs_base[1::2]
    else:
        lhs_base = torch.tensor([0.2, 0.8, -0.4, 0.3], requires_grad=True)
        rhs_base = torch.tensor([[0.6, 0.1], [-0.2, 0.8], [0.9, -0.3]],
                                requires_grad=True)
        lhs, rhs = lhs_base[::2], rhs_base.t()
        vk_lhs_base, vk_rhs_base = _vk_leaf(lhs_base, vulkan_backend), _vk_leaf(
            rhs_base, vulkan_backend
        )
        vk_lhs, vk_rhs = vk_lhs_base[::2], vk_rhs_base.t()
    cpu_value, vk_value = torch.matmul(lhs, rhs), torch.matmul(vk_lhs, vk_rhs)
    seed = torch.arange(cpu_value.numel(), dtype=torch.float32).reshape(cpu_value.shape) / 5 + 0.3
    cpu_grads = torch.autograd.grad(cpu_value, (lhs_base, rhs_base), seed)
    vk_grads = torch.autograd.grad(
        vk_value, (vk_lhs_base, vk_rhs_base), seed.to(vulkan_backend)
    )
    torch.testing.assert_close(vk_value.cpu(), cpu_value, rtol=0.003, atol=0.003)
    for vk_grad, cpu_grad in zip(vk_grads, cpu_grads):
        _assert_gradient(vk_grad, cpu_grad)


@pytest.mark.parametrize("through_view", [False, True])
@pytest.mark.parametrize("operation", ["dot", "mv"])
def test_dot_mv_saved_operand_and_alias_view_mutations_raise_version_error(
    vulkan_backend, operation, through_view
):
    if operation == "dot":
        lhs_cpu = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        lhs_vk = _vk_leaf(lhs_cpu, vulkan_backend)
    else:
        lhs_cpu = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.3, -0.5]],
                               requires_grad=True)
        lhs_vk = _vk_leaf(lhs_cpu, vulkan_backend)
    if through_view:
        rhs_storage_cpu = torch.tensor([0.6, -0.2, 0.9, 0.8], requires_grad=True)
        rhs_cpu = rhs_storage_cpu[:3]
        rhs_storage_vk = _vk_leaf(rhs_storage_cpu, vulkan_backend)
        rhs_vk = rhs_storage_vk[:3]
    else:
        rhs_storage_cpu = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
        rhs_cpu = rhs_storage_cpu
        rhs_storage_vk = _vk_leaf(rhs_storage_cpu, vulkan_backend)
        rhs_vk = rhs_storage_vk
    cpu_output = (torch.dot(lhs_cpu, rhs_cpu) if operation == "dot"
                  else torch.mv(lhs_cpu, rhs_cpu))
    vk_output = (torch.dot(lhs_vk, rhs_vk) if operation == "dot"
                 else torch.mv(lhs_vk, rhs_vk))
    with torch.no_grad():
        rhs_storage_vk[0].add_(1)
    vk_seed = torch.ones_like(vk_output)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(vk_output, lhs_vk, vk_seed)
    # The CPU reference establishes the same generated saved-variable contract.
    with torch.no_grad():
        rhs_storage_cpu[0].add_(1)
    cpu_seed = torch.ones_like(cpu_output)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(cpu_output, lhs_cpu, cpu_seed)


@pytest.mark.parametrize("operation", [torch.outer, torch.ger])
def test_outer_ger_stock_reverse_mixed_and_expanded_base_gradients(vulkan_backend, operation):
    a = torch.tensor([0.2, -0.4], requires_grad=True)
    b = torch.tensor([0.6, -0.2, 0.9], requires_grad=True)
    seed = torch.tensor([[0.7, -0.3, 0.5], [-0.8, 0.4, 0.2]])
    probe = torch.tensor([0.3, -0.8])
    av, bv = _vk_leaf(a, vulkan_backend), _vk_leaf(b, vulkan_backend)
    cpu_first = torch.autograd.grad(operation(a, b), a, seed, create_graph=True)[0]
    vk_first = torch.autograd.grad(
        operation(av, bv), av, seed.to(vulkan_backend), create_graph=True
    )[0]
    cpu_mixed = torch.autograd.grad(cpu_first, b, probe)[0]
    vk_mixed = torch.autograd.grad(vk_first, bv, probe.to(vulkan_backend))[0]
    _assert_gradient(vk_first, cpu_first)
    _assert_gradient(vk_mixed, cpu_mixed)

    base = torch.tensor([0.2, -0.4, 0.7, 1.1], requires_grad=True)
    expanded_cpu = base[:1].expand(2)
    vk_base = _vk_leaf(base, vulkan_backend)
    expanded_vk = vk_base[:1].expand(2)
    cpu_value = torch.outer(expanded_cpu, b.detach())
    vk_value = torch.outer(expanded_vk, _vk_leaf(b, vulkan_backend, False))
    cpu_base_grad = torch.autograd.grad(cpu_value, base, seed)[0]
    vk_base_grad = torch.autograd.grad(vk_value, vk_base, seed.to(vulkan_backend))[0]
    _assert_gradient(vk_base_grad, cpu_base_grad)
