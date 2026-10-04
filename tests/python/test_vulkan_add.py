import pytest
import pytorch_vulkan
import torch
from vulkan_conformance import (
    SCALAR_OUT_CONTRACT_MATRIX,
    SCALAR_OUT_FUNCTIONAL_CASES,
    assert_scalar_out_counters,
    invoke_scalar_out_contract,
)


def test_cpu_only_add_remains_normal_pytorch_behavior():
    lhs = torch.tensor([1.0, -2.0], dtype=torch.float32)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = torch.add(lhs, rhs)

    torch.testing.assert_close(result, torch.tensor([4.0, 2.0]))


def test_cpu_only_sub_scalar_remains_normal_pytorch_behavior():
    values = torch.tensor([1.5, -2.0], dtype=torch.float32)

    torch.testing.assert_close(torch.sub(values, 0.5), torch.tensor([1.0, -2.5]))
    torch.testing.assert_close(torch.sub(0.5, values), torch.tensor([-1.0, 2.5]))


def test_cpu_only_mul_scalar_remains_normal_pytorch_behavior():
    values = torch.tensor([1.5, -2.0], dtype=torch.float32)

    torch.testing.assert_close(torch.mul(values, -2.0), torch.tensor([-3.0, 4.0]))
    torch.testing.assert_close(torch.mul(-2.0, values), torch.tensor([-3.0, 4.0]))


def _vulkan_scalar_operation(operation, tensor, scalar, scalar_left=False):
    if operation is torch.add:
        return (
            torch.add(scalar, tensor, alpha=1.0)
            if scalar_left
            else torch.add(tensor, scalar, alpha=1.0)
        )
    if operation is torch.sub:
        return torch.sub(scalar, tensor) if scalar_left else torch.sub(tensor, scalar)
    return torch.mul(scalar, tensor) if scalar_left else torch.mul(tensor, scalar)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


@pytest.mark.parametrize(
    ("shape", "left_values", "right_values", "expected"),
    [
        ((1,), [1.0], [3.0], [4.0]),
        ((3,), [1.0, -2.0, 0.5], [3.0, 4.0, -1.5], [4.0, 2.0, -1.0]),
        (
            (2, 2),
            [1.0, -2.0, 0.5, 8.0],
            [3.0, 4.0, -1.5, 2.0],
            [4.0, 2.0, -1.0, 10.0],
        ),
    ],
)
def test_add_equal_shape_tensors_preserves_metadata(
    vulkan_backend, shape, left_values, right_values, expected
):
    lhs = (
        torch.tensor(left_values, dtype=torch.float32).reshape(shape).to(vulkan_backend)
    )
    rhs = (
        torch.tensor(right_values, dtype=torch.float32)
        .reshape(shape)
        .to(vulkan_backend)
    )

    result = torch.add(lhs, rhs)

    assert result.device == lhs.device
    assert result.shape == lhs.shape
    assert result.dtype is torch.float32
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.cpu(),
        torch.tensor(expected, dtype=torch.float32).reshape(shape),
        rtol=0,
        atol=0,
    )


def test_add_operator_keeps_inputs_and_allocates_new_storage(vulkan_backend):
    lhs_cpu = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    rhs_cpu = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    lhs = lhs_cpu.to(vulkan_backend)
    rhs = rhs_cpu.to(vulkan_backend)

    result = lhs + rhs

    assert result is not lhs
    assert result.data_ptr() != lhs.data_ptr()
    assert result.data_ptr() != rhs.data_ptr()
    torch.testing.assert_close(lhs.cpu(), lhs_cpu, rtol=0, atol=0)
    torch.testing.assert_close(rhs.cpu(), rhs_cpu, rtol=0, atol=0)
    torch.testing.assert_close(result.cpu(), torch.tensor([[4.0, 2.0]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "operation, expected",
    [
        (torch.sub, [[-2.0, -6.0]]),
        (torch.mul, [[3.0, -8.0]]),
    ],
)
def test_tensor_tensor_sub_and_mul_keep_inputs_and_allocate_new_storage(
    vulkan_backend, operation, expected
):
    lhs_cpu = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    rhs_cpu = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    lhs = lhs_cpu.to(vulkan_backend)
    rhs = rhs_cpu.to(vulkan_backend)

    result = operation(lhs, rhs)

    assert result.device == lhs.device
    assert result is not lhs and result is not rhs
    assert result.data_ptr() != lhs.data_ptr()
    assert result.data_ptr() != rhs.data_ptr()
    torch.testing.assert_close(lhs.cpu(), lhs_cpu, rtol=0, atol=0)
    torch.testing.assert_close(rhs.cpu(), rhs_cpu, rtol=0, atol=0)
    torch.testing.assert_close(
        result.cpu(), torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar", [2.0, -2.0, 0.0, 0.25])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_pointwise_python_scalar_matches_cpu_and_preserves_inputs(
    vulkan_backend, operation, scalar, scalar_left
):
    values = torch.tensor([1.5, -2.0, 0.0, 4.25], dtype=torch.float32)
    tensor = values.to(vulkan_backend)
    before = tensor.cpu()
    operands = (scalar, tensor) if scalar_left else (tensor, scalar)

    result = _vulkan_scalar_operation(operation, tensor, scalar, scalar_left)
    expected = operation(*((scalar, values) if scalar_left else (values, scalar)))

    assert result.device.type == "vk"
    assert result.device.index == 0
    assert result.is_contiguous()
    assert result.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(tensor.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_scalar_functional_semantics_do_only_vulkan_compute(
    vulkan_backend, operation, scalar_left
):
    cpu_values = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32)
    tensor = cpu_values.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = _vulkan_scalar_operation(operation, tensor, 2.5, scalar_left)
    expected = operation(
        *((2.5, cpu_values) if scalar_left else (cpu_values, 2.5))
    )
    dispatches, copies, transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )

    assert result.device == torch.device("vk:0")
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert dispatches > 0
    assert copies == 0
    assert transfers == 0
    assert fallbacks == 0


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_empty_scalar_functional_semantics_do_zero_work(
    vulkan_backend, operation, scalar_left
):
    tensor = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = _vulkan_scalar_operation(operation, tensor, 2.5, scalar_left)
    dispatches, copies, transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )

    assert result.device == torch.device("vk:0")
    assert result.shape == tensor.shape
    assert dispatches == 0
    assert copies == 0
    assert transfers == 0
    assert fallbacks == 0


@pytest.mark.parametrize(
    "contract",
    SCALAR_OUT_FUNCTIONAL_CASES,
    ids=lambda case: case.case_name,
)
def test_scalar_contract_matrix_cases_are_named_and_vulkan_resident(
    vulkan_backend, contract
):
    values = torch.tensor([1.5, -2.0, 0.0], dtype=torch.float32)
    tensor = values.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = invoke_scalar_out_contract(contract, tensor)
    expected = contract.cpu_reference(values)

    assert result.device == torch.device("vk:0")
    assert contract.output_mode == "functional"
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "contract", SCALAR_OUT_FUNCTIONAL_CASES, ids=lambda case: case.case_name
)
def test_scalar_contract_matrix_gradients_follow_declaration(vulkan_backend, contract):
    cpu_values = torch.tensor([1.5, -2.0, 0.0], dtype=torch.float32, requires_grad=True)
    vk_values = cpu_values.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = contract.cpu_reference(cpu_values)
    vk_result = invoke_scalar_out_contract(contract, vk_values)
    cpu_result.sum().backward()
    vk_result.sum().backward()

    assert contract.check_gradients
    torch.testing.assert_close(vk_values.grad.cpu(), cpu_values.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    "contract", tuple(SCALAR_OUT_CONTRACT_MATRIX.values()), ids=lambda case: case.case_name
)
def test_scalar_contract_matrix_empty_counters_follow_declaration(vulkan_backend, contract):
    tensor = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    result = invoke_scalar_out_contract(contract, tensor)
    assert result.device == torch.device("vk:0")
    assert result.numel() == 0
    assert_scalar_out_counters(contract.empty_expected_counters)


@pytest.mark.parametrize(
    "contract", SCALAR_OUT_FUNCTIONAL_CASES, ids=lambda case: case.case_name
)
def test_scalar_contract_matrix_rejects_invalid_scalar_without_work(
    vulkan_backend, contract
):
    tensor = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    invalid = {"scalar": float("nan")}
    if contract.schema == "aten::rsub.Scalar":
        invalid = {"alpha": 2.0}

    with pytest.raises(RuntimeError, match=r"Vulkan|alpha|non-finite"):
        invoke_scalar_out_contract(contract, tensor, **invalid)

    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_scalar_add_nonunit_alpha_matches_cpu(vulkan_backend):
    contract = SCALAR_OUT_CONTRACT_MATRIX["aten::add.Scalar"]
    cpu_tensor = torch.tensor([1.0, -2.0], dtype=torch.float32)
    vk_tensor = cpu_tensor.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()

    cpu_result = torch.add(cpu_tensor, contract.scalar, alpha=2.0)
    vk_result = invoke_scalar_out_contract(contract, vk_tensor, alpha=2.0)
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] > 0
    assert counters[1:] == (0, 0, 0)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_result.cpu(), cpu_result)


@pytest.mark.parametrize(
    "tensor_value,scalar,alpha",
    [
        (-10000001024.0, 1e10, 1.00000006),
        (-11000000512.0, 1e10, 1.1),
    ],
)
@pytest.mark.parametrize("scalar_left", [False, True])
def test_scalar_add_nonunit_alpha_preserves_cpu_cancellation(
    vulkan_backend, tensor_value, scalar, alpha, scalar_left
):
    cpu_tensor = torch.tensor([tensor_value], dtype=torch.float32, requires_grad=True)
    vk_tensor = cpu_tensor.detach().to(vulkan_backend).requires_grad_()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = torch.add(scalar, cpu_tensor, alpha=alpha) if scalar_left else torch.add(
        cpu_tensor, scalar, alpha=alpha
    )
    vk_result = torch.add(scalar, vk_tensor, alpha=alpha) if scalar_left else torch.add(
        vk_tensor, scalar, alpha=alpha
    )
    forward_counters = pytorch_vulkan._C.execution_counter_snapshot()
    seed = torch.tensor([2.25], dtype=torch.float32)
    cpu_result.backward(seed)
    vk_seed = seed.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    vk_result.backward(vk_seed)
    pytorch_vulkan._C.synchronize()
    assert vk_result.device == torch.device("vk:0")
    assert forward_counters[0] > 0
    assert forward_counters[1:] == (0, 0, 0)
    torch.testing.assert_close(vk_result.cpu(), cpu_result.detach())
    torch.testing.assert_close(vk_tensor.grad.cpu(), cpu_tensor.grad, rtol=0, atol=0)


def test_scalar_left_add_invalid_alpha_operands_reject_before_work(vulkan_backend):
    tensor = torch.tensor([1.0, 3.0], device=vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="non-finite scalar"):
        torch.add(float("nan"), tensor, alpha=2.0)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "route",
    ["scalar-right-alpha", "scalar-left-alpha", "explicit-scalar-alpha", "tensor-alpha", "scalar-value", "mul-scalar"],
)
def test_integer_scalar_conversion_matches_torch_storage_rounding(vulkan_backend, route):
    integer = 2**60 + 2**36 + 1
    cpu_input = torch.tensor([-(2**60 + 2**37)], dtype=torch.float32)
    vk_input = cpu_input.to(vulkan_backend)

    if route == "scalar-right-alpha":
        expected = torch.add(cpu_input, 1.0, alpha=integer)
        actual = torch.add(vk_input, 1.0, alpha=integer)
    elif route == "scalar-left-alpha":
        expected = torch.add(1.0, cpu_input, alpha=integer)
        actual = torch.add(1.0, vk_input, alpha=integer)
    elif route == "explicit-scalar-alpha":
        expected = torch.ops.aten.add.Scalar(cpu_input, 1.0, integer)
        actual = torch.ops.aten.add.Scalar(vk_input, 1.0, integer)
    elif route == "tensor-alpha":
        cpu_other = torch.ones_like(cpu_input)
        vk_other = cpu_other.to(vulkan_backend)
        expected = torch.add(cpu_input, cpu_other, alpha=integer)
        actual = torch.add(vk_input, vk_other, alpha=integer)
    elif route == "scalar-value":
        expected = torch.add(cpu_input, integer)
        actual = torch.add(vk_input, integer)
    else:
        cpu_input = torch.ones(1, dtype=torch.float32)
        vk_input = cpu_input.to(vulkan_backend)
        expected = torch.mul(cpu_input, integer)
        actual = torch.mul(vk_input, integer)

    pytorch_vulkan._C.synchronize()
    assert actual.device == torch.device("vk:0")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_pointwise_python_scalar_repeated_operations_are_independent(
    vulkan_backend, operation, scalar_left
):
    tensor = torch.tensor([1.0, -2.0], dtype=torch.float32).to(vulkan_backend)
    for _ in range(8):
        tensor = _vulkan_scalar_operation(operation, tensor, 0.5, scalar_left)
    expected = torch.tensor([1.0, -2.0], dtype=torch.float32)
    for _ in range(8):
        expected = operation(*((0.5, expected) if scalar_left else (expected, 0.5)))
    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_zero_element_pointwise_scalar_returns_empty_without_changing_shape(
    vulkan_backend, operation, scalar_left
):
    tensor = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    operands = (1.25, tensor) if scalar_left else (tensor, 1.25)

    result = _vulkan_scalar_operation(operation, tensor, 1.25, scalar_left)

    assert result.device == tensor.device
    assert result.shape == tensor.shape
    assert result.dtype is torch.float32
    assert result.numel() == 0
    assert result.is_contiguous()


def test_repeated_add_operations_are_independent(vulkan_backend):
    value = torch.tensor([1.0, -2.0], dtype=torch.float32).to(vulkan_backend)

    for _ in range(32):
        value = torch.add(value, value)

    torch.testing.assert_close(
        value.cpu(), torch.tensor([2.0**32, -(2.0**33)]), rtol=0, atol=0
    )


def test_zero_element_add_preserves_shape_and_dtype(vulkan_backend):
    lhs = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)

    result = lhs + rhs

    assert result.device == lhs.device
    assert result.shape == (0, 3)
    assert result.dtype is torch.float32
    assert result.numel() == 0
    assert result.is_contiguous()


def _assert_zero_work():
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _assert_add_rejected(operation, message):
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        operation()
    _assert_zero_work()


def _scalar_rejection_pattern(operation, add_pattern):
    return (
        rf"({add_pattern}|scalar operand|float32|contiguous|broadcast|same device|out)"
    )


def test_mixed_device_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "same device")


@pytest.mark.parametrize("vulkan_first", [True, False])
def test_mixed_cpu_vulkan_add_does_not_fallback(vulkan_backend, vulkan_first):
    vulkan_tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    cpu_tensor = torch.empty((2,), dtype=torch.float32)
    operands = (
        (vulkan_tensor, cpu_tensor) if vulkan_first else (cpu_tensor, vulkan_tensor)
    )
    _assert_add_rejected(lambda: torch.add(*operands), "same device")


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_scalar_tensor_operand_is_rejected(vulkan_backend, operation, scalar_left):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    scalar_tensor = torch.tensor(1.0, dtype=torch.float32).to(vulkan_backend)
    operands = (scalar_tensor, tensor) if scalar_left else (tensor, scalar_tensor)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(
        RuntimeError,
        match=_scalar_rejection_pattern(operation, r"zero-dimensional|scalar operand"),
    ):
        operation(*operands)
    _assert_zero_work()


def test_zero_dim_tensor_add_operands_are_supported(vulkan_backend):
    lhs = torch.tensor(2.0, dtype=torch.float32, device=vulkan_backend)
    rhs = torch.tensor(3.0, dtype=torch.float32, device=vulkan_backend)
    result = torch.add(lhs, rhs)
    assert result.dim() == 0
    torch.testing.assert_close(result.cpu(), torch.tensor(5.0))


def test_broadcasting_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2, 3), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "broadcast")


def test_shape_mismatch_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "size")


def test_non_default_alpha_matches_cpu(vulkan_backend):
    cpu_lhs = torch.tensor([1.0, -2.0])
    cpu_rhs = torch.tensor([0.5, 3.0])
    lhs = cpu_lhs.to(vulkan_backend)
    rhs = cpu_rhs.to(vulkan_backend)
    expected = torch.add(cpu_lhs, cpu_rhs, alpha=2)
    torch.testing.assert_close(torch.add(lhs, rhs, alpha=2).cpu(), expected)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_non_contiguous_pointwise_operands_match_cpu(vulkan_backend, operation):
    lhs_cpu = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=torch.float32)
    rhs_cpu = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 2.0, 5.0]], dtype=torch.float32)
    lhs_base = lhs_cpu.to(vulkan_backend)
    rhs_base = rhs_cpu.to(vulkan_backend)
    cpu_views = [
        (lhs_cpu.transpose(0, 1), rhs_cpu.transpose(0, 1)),
        (lhs_cpu[:, 1:], rhs_cpu[:, 1:]),
        (
            torch.as_strided(lhs_cpu, (2, 3), (0, 1)),
            torch.as_strided(rhs_cpu, (2, 3), (0, 1)),
        ),
    ]

    vk_views = [
        (lhs_base.transpose(0, 1), rhs_base.transpose(0, 1)),
        (lhs_base[:, 1:], rhs_base[:, 1:]),
        (
            torch.as_strided(lhs_base, (2, 3), (0, 1)),
            torch.as_strided(rhs_base, (2, 3), (0, 1)),
        ),
    ]
    for (lhs_cpu_view, rhs_cpu_view), (lhs, rhs) in zip(cpu_views, vk_views):
        result = operation(lhs, rhs)
        torch.testing.assert_close(
            result.cpu(), operation(lhs_cpu_view, rhs_cpu_view), rtol=0, atol=0
        )


def test_pointwise_rejects_rank_above_eight_before_dispatch(vulkan_backend):
    lhs = torch.empty((1,) * 9, dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty_like(lhs)

    _assert_add_rejected(lambda: torch.add(lhs, rhs), "rank.*8")


def test_non_float32_add_operand_is_rejected(vulkan_backend):
    if pytorch_vulkan.formatter_double_supported():
        torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    else:
        with pytest.raises(RuntimeError, match="shaderFloat64"):
            torch.empty((2,), dtype=torch.float64, device=vulkan_backend)


def test_second_vulkan_device_add_is_rejected(vulkan_backend):
    _assert_add_rejected(
        lambda: torch.add(
            torch.empty((2,), dtype=torch.float32, device=vulkan_backend),
            torch.empty((2,), dtype=torch.float32, device="vk:1"),
        ),
        "only device index 0",
    )


def test_add_out_is_supported(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    output = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    assert torch.add(lhs, rhs, out=output) is output


def test_inplace_add_is_supported_outside_optimizer_step(vulkan_backend):
    lhs = torch.tensor([1.0, 2.0], dtype=torch.float32, device=vulkan_backend)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    assert lhs.add_(rhs) is lhs
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(
        lhs.detach().cpu(), torch.tensor([4.0, 6.0], dtype=torch.float32)
    )


@pytest.mark.parametrize("rank", range(1, 9))
def test_inplace_add_tensor_broadcasts_rhs_without_resizing_self(
    vulkan_backend, rank
):
    shape = (2,) * rank
    rhs_shape = (2,)
    self_cpu = torch.arange(1, 2 ** rank + 1, dtype=torch.float32).reshape(shape)
    rhs_cpu = torch.tensor([0.5, -1.0], dtype=torch.float32)
    expected = self_cpu.clone().add_(rhs_cpu, alpha=0.5)
    self_vk = self_cpu.to(vulkan_backend)
    rhs_vk = rhs_cpu.to(vulkan_backend)
    self_identity = id(self_vk)
    self_storage = self_vk.untyped_storage().data_ptr()
    self_offset = self_vk.storage_offset()
    self_strides = self_vk.stride()
    version = self_vk._version

    result = self_vk.add_(rhs_vk, alpha=0.5)

    assert result is self_vk and id(result) == self_identity
    assert result.untyped_storage().data_ptr() == self_storage
    assert result.shape == shape
    assert result.storage_offset() == self_offset
    assert result.stride() == self_strides
    assert result._version == version + 1
    assert rhs_vk.device.type == "vk"
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0.003, atol=0.003)
    torch.testing.assert_close(rhs_vk.cpu(), rhs_cpu, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("self_shape", "rhs_shape"),
    [((2, 3, 4), (4,)), ((2, 3, 4), (1, 4)), ((2, 3, 4), (2, 1, 4)), ((3, 4), ())],
)
def test_inplace_add_tensor_broadcast_shapes_match_cpu(
    vulkan_backend, self_shape, rhs_shape
):
    self_cpu = torch.arange(torch.tensor(self_shape).prod().item(), dtype=torch.float32)
    self_cpu = self_cpu.reshape(self_shape)
    rhs_cpu = torch.arange(max(1, torch.tensor(rhs_shape).prod().item()), dtype=torch.float32)
    rhs_cpu = rhs_cpu.reshape(rhs_shape)
    expected = self_cpu.clone().add_(rhs_cpu, alpha=-0.25)
    self_vk = self_cpu.to(vulkan_backend)
    rhs_vk = rhs_cpu.to(vulkan_backend)

    result = self_vk.add_(rhs_vk, alpha=-0.25)

    assert result is self_vk
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0.003, atol=0.003)


def test_inplace_add_tensor_broadcast_reads_offset_and_zero_stride_rhs(vulkan_backend):
    self_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs_base_cpu = torch.tensor([99.0, 7.0, 8.0, 9.0])
    rhs_offset_cpu = rhs_base_cpu[1:]
    rhs_broadcast_cpu = rhs_offset_cpu[:1].expand(2, 3)
    self_base_vk = self_cpu.to(vulkan_backend)
    rhs_base_vk = rhs_base_cpu.to(vulkan_backend)
    self_vk = self_base_vk
    rhs_offset_vk = rhs_base_vk[1:]
    rhs_vk = rhs_offset_vk[:1].expand(2, 3)
    expected = self_cpu.clone().add_(rhs_broadcast_cpu, alpha=0.5)

    result = self_vk.add_(rhs_vk, alpha=0.5)

    assert result is self_vk
    assert rhs_offset_vk.storage_offset() == 1
    assert rhs_vk.stride() == (0, 0)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0.003, atol=0.003)


def test_inplace_add_tensor_broadcast_reads_offset_noncontiguous_rhs(vulkan_backend):
    self_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs_base_cpu = torch.tensor(
        [[99.0, 7.0, 8.0, 9.0, 99.0], [99.0, 11.0, 12.0, 13.0, 99.0]]
    )
    rhs_cpu = rhs_base_cpu[:, 1:4]
    expected = self_cpu.clone().add_(rhs_cpu)
    self_vk = self_cpu.to(vulkan_backend)
    rhs_base_vk = rhs_base_cpu.to(vulkan_backend)
    rhs_vk = rhs_base_vk[:, 1:4]

    result = self_vk.add_(rhs_vk)

    assert rhs_vk.storage_offset() == 1
    assert rhs_vk.stride() == (5, 1)
    assert not rhs_vk.is_contiguous()
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "alias_case",
    ["full-overlap", "partial-overlap-forward", "transposed"],
)
def test_inplace_add_tensor_broadcast_rejects_overlap_dependent_alias_before_work(
    vulkan_backend, alias_case
):
    if alias_case == "transposed":
        base_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self_view = lambda tensor: tensor.transpose(0, 1)
        rhs_view = lambda tensor, self: self[:1, :].expand_as(self)
    else:
        base_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        slices = {
            "full-overlap": (slice(None), slice(0, 1)),
            "partial-overlap-forward": (slice(0, 3), slice(1, 2)),
        }
        self_slice, rhs_slice = slices[alias_case]
        self_view = lambda tensor: tensor[self_slice]
        rhs_view = lambda tensor, self: tensor[rhs_slice].expand_as(self)
    base_vk = base_cpu.to(vulkan_backend)
    self_vk = self_view(base_vk)
    rhs_vk = rhs_view(base_vk, self_vk)
    before = base_vk.cpu().clone()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError):
        self_vk.add_(rhs_vk)

    _assert_zero_work()
    assert self_vk._version == version
    torch.testing.assert_close(base_vk.cpu(), before, rtol=0, atol=0)


def test_inplace_add_broadcast_disjoint_same_storage_range_is_supported(vulkan_backend):
    base_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_base = base_cpu.clone()
    expected_base[1:].add_(expected_base[:1].expand_as(expected_base[1:]))
    base_vk = base_cpu.to(vulkan_backend)
    self_vk = base_vk[1:]
    rhs_vk = base_vk[:1].expand_as(self_vk)

    result = self_vk.add_(rhs_vk)

    pytorch_vulkan._C.synchronize()
    assert result is self_vk
    torch.testing.assert_close(base_vk.cpu(), expected_base, rtol=0, atol=0)


def test_inplace_add_exact_self_alias_remains_supported(vulkan_backend):
    self_cpu = torch.arange(1, 5, dtype=torch.float32)
    expected = self_cpu.clone().add_(self_cpu, alpha=0.5)
    self_vk = self_cpu.to(vulkan_backend)
    storage = self_vk.untyped_storage().data_ptr()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    result = self_vk.add_(self_vk, alpha=0.5)

    pytorch_vulkan._C.synchronize()
    assert result is self_vk
    assert result.untyped_storage().data_ptr() == storage
    assert result._version == version + 1
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_inplace_add_tensor_broadcast_preserves_generated_autograd_history(
    vulkan_backend,
):
    generator = torch.Generator().manual_seed(721)
    base_cpu = torch.randn((2, 3, 4), generator=generator, requires_grad=True)
    rhs_cpu = torch.randn((4,), generator=generator, requires_grad=True)
    seed_cpu = torch.randn((2, 3, 4), generator=generator, requires_grad=True)
    base_vk = base_cpu.detach().to(vulkan_backend).requires_grad_()
    rhs_vk = rhs_cpu.detach().to(vulkan_backend).requires_grad_()
    seed_vk = seed_cpu.detach().to(vulkan_backend).requires_grad_()

    def objective(base, rhs, seed):
        self = base * 1.5
        self.add_(rhs, alpha=0.25)
        return (self * seed).sum()

    cpu_output = objective(base_cpu, rhs_cpu, seed_cpu)
    vk_output = objective(base_vk, rhs_vk, seed_vk)
    cpu_grads = torch.autograd.grad(
        cpu_output, (base_cpu, rhs_cpu), create_graph=True
    )
    vk_grads = torch.autograd.grad(vk_output, (base_vk, rhs_vk), create_graph=True)
    direction_base = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4) / 13
    direction_rhs = torch.tensor([0.5, -0.25, 0.75, -1.0])
    direction_base_vk = direction_base.to(vulkan_backend)
    direction_rhs_vk = direction_rhs.to(vulkan_backend)
    cpu_second = torch.autograd.grad(
        (cpu_grads[0] * direction_base).sum()
        + (cpu_grads[1] * direction_rhs).sum(),
        seed_cpu,
    )[0]
    vk_second = torch.autograd.grad(
        (vk_grads[0] * direction_base_vk).sum()
        + (vk_grads[1] * direction_rhs_vk).sum(),
        seed_vk,
    )[0]

    assert all(grad.grad_fn is not None for grad in vk_grads)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_output.cpu(), cpu_output, rtol=0.003, atol=0.003)
    for actual, expected in zip(vk_grads, cpu_grads):
        assert actual.device.type == "vk"
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
    torch.testing.assert_close(vk_second.cpu(), cpu_second, rtol=0.003, atol=0.003)


def test_inplace_add_tensor_empty_broadcast_preserves_fixed_self(vulkan_backend):
    self_cpu = torch.empty((0, 3), dtype=torch.float32)
    rhs_cpu = torch.ones((1, 3), dtype=torch.float32)
    expected = self_cpu.clone().add_(rhs_cpu)
    self_vk = self_cpu.to(vulkan_backend)
    rhs_vk = rhs_cpu.to(vulkan_backend)
    version = self_vk._version

    result = self_vk.add_(rhs_vk)

    assert result is self_vk
    assert result.shape == (0, 3)
    assert result._version == version + 1
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_inplace_add_tensor_scalar_broadcast_zero_alpha_matches_cpu(vulkan_backend):
    self_cpu = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    rhs_cpu = torch.tensor(7.0)
    expected = self_cpu.clone().add_(rhs_cpu, alpha=0.0)
    self_vk = self_cpu.to(vulkan_backend)
    rhs_vk = rhs_cpu.to(vulkan_backend)
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    result = self_vk.add_(rhs_vk, alpha=0.0)

    pytorch_vulkan._C.synchronize()
    assert result is self_vk
    assert result._version == version + 1
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["not_broadcastable", "partial_overlap"])
def test_inplace_add_tensor_broadcast_rejects_invalid_inputs_before_work(
    vulkan_backend, case
):
    if case == "not_broadcastable":
        self_vk = torch.ones((2, 3), dtype=torch.float32, device=vulkan_backend)
        rhs_vk = torch.ones((4,), dtype=torch.float32, device=vulkan_backend)
    else:
        base = torch.arange(5, dtype=torch.float32).to(vulkan_backend)
        self_vk, rhs_vk = base[1:], base[:-1]
    before = self_vk.cpu().clone()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError):
        self_vk.add_(rhs_vk)

    _assert_zero_work()
    assert self_vk._version == version
    torch.testing.assert_close(self_vk.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["cpu_rhs", "dtype"])
def test_inplace_add_tensor_broadcast_rejects_device_and_dtype_before_work(
    vulkan_backend, case
):
    self_vk = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(vulkan_backend)
    if case == "cpu_rhs":
        rhs = torch.ones((3,), dtype=torch.float32)
    else:
        assert pytorch_vulkan.formatter_double_supported()
        rhs = torch.empty((3,), dtype=torch.float64, device=vulkan_backend)
    before = self_vk.cpu().clone()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError):
        self_vk.add_(rhs)

    _assert_zero_work()
    assert self_vk._version == version
    torch.testing.assert_close(self_vk.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_unsupported_python_scalar_conversion_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(TypeError, match=r"argument 'other'.*Tensor"):
        _vulkan_scalar_operation(operation, tensor, object())
    _assert_zero_work()


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_cpu_zero_dim_tensor_is_not_accepted_as_python_scalar(
    vulkan_backend, operation, scalar_left
):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    scalar_tensor = torch.tensor(1.0, dtype=torch.float32)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(
        (RuntimeError, NotImplementedError), match=r"scalar operand|Could not run"
    ):
        operands = (scalar_tensor, tensor) if scalar_left else (tensor, scalar_tensor)
        operation(*operands)
    _assert_zero_work()


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize(
    "scalar", [float("nan"), float("inf"), -float("inf"), 1e40, -1e40, 1e-50, -1e-50]
)
def test_unsupported_scalar_values_are_rejected(vulkan_backend, operation, scalar):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"non-finite|outside float32|underflows"):
        _vulkan_scalar_operation(operation, tensor, scalar)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_sub_scalar_nonunit_alpha_is_rejected_before_vulkan_work(vulkan_backend):
    tensor = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"alpha == 1"):
        torch.sub(tensor, 2.0, alpha=2.0)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_invalid_scalar_tensor_metadata_is_rejected(vulkan_backend, operation):
    wrong_dtype = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(
        RuntimeError,
        match=_scalar_rejection_pattern(operation, r"scalar operand"),
    ):
        operation(wrong_dtype, 1.0)
    _assert_zero_work()

    non_contiguous = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    result = operation(non_contiguous, 1.0)
    assert result.shape == non_contiguous.shape


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_vulkan_operand_on_wrong_device_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"only device index 0"):
        operation(tensor, 1.0) if tensor.device.index != 0 else operation(
            torch.empty((2,), dtype=torch.float32, device="vk:1"), 1.0
        )
    _assert_zero_work()


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_mixed_cpu_vulkan_tensor_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    cpu_tensor = torch.empty((2,), dtype=torch.float32)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"same device|scalar operand"):
        operation(tensor, cpu_tensor)
    _assert_zero_work()


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_tensor_broadcasting_is_rejected(vulkan_backend, operation):
    lhs = torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2, 3), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(
        RuntimeError, match=_scalar_rejection_pattern(operation, r"broadcast")
    ):
        operation(lhs, rhs)
    _assert_zero_work()


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_out_and_inplace_variant_are_both_supported(vulkan_backend, operation):
    tensor = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    output = torch.empty_like(tensor)
    assert operation(tensor, 1.0, out=output) is output
    pytorch_vulkan._C.reset_execution_counters()
    name = operation.__name__
    assert getattr(tensor, name + "_")(1.0) is tensor
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    pytorch_vulkan._C.synchronize()
    expected = {"add": [2.0, 2.0], "sub": [0.0, 0.0], "mul": [1.0, 1.0]}[name]
    torch.testing.assert_close(
        tensor.detach().cpu(), torch.tensor(expected, dtype=torch.float32)
    )


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_pointwise_has_compute_work_without_fallback(vulkan_backend, operation):
    tensor = torch.tensor([1.0, -2.0], dtype=torch.float32, device=vulkan_backend)
    output = torch.empty_like(tensor)
    pytorch_vulkan._C.reset_execution_counters()

    operation(tensor, 2.0, out=output)

    dispatches, copies, transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert dispatches > 0
    assert copies == 0
    assert transfers == 0
    assert fallbacks == 0


@pytest.mark.parametrize("operation", [torch.add, torch.sub])
def test_scalar_alpha_one_is_explicitly_supported(vulkan_backend, operation):
    values = torch.tensor([1.0, -2.0], dtype=torch.float32)
    tensor = values.to(vulkan_backend)
    result = operation(tensor, 2.0, alpha=1.0)

    torch.testing.assert_close(result.cpu(), operation(values, 2.0, alpha=1.0))


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_approved_bool_tensor_operation_matches_cpu(vulkan_backend, operation):
    lhs_cpu = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    rhs_cpu = torch.tensor([[True, True], [False, False]], dtype=torch.bool)
    lhs = lhs_cpu.to(vulkan_backend)
    rhs = rhs_cpu.to(vulkan_backend)

    result = operation(lhs, rhs)

    expected = operation(lhs_cpu, rhs_cpu)
    assert result.device == lhs.device
    assert result.dtype is torch.bool
    assert result.shape == lhs.shape
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_approved_bool_tensor_operation_out_matches_cpu(vulkan_backend, operation):
    lhs_cpu = torch.tensor([True, False], dtype=torch.bool)
    rhs_cpu = torch.tensor([False, True], dtype=torch.bool)
    lhs = lhs_cpu.to(vulkan_backend)
    rhs = rhs_cpu.to(vulkan_backend)
    output = torch.empty_like(lhs)

    assert operation(lhs, rhs, out=output) is output
    torch.testing.assert_close(output.cpu(), operation(lhs_cpu, rhs_cpu))


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_bool_tensor_operation_rejects_float32_out(vulkan_backend, operation):
    lhs = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)
    rhs = torch.tensor([False, True], dtype=torch.bool, device=vulkan_backend)
    output = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)

    with pytest.raises(RuntimeError, match="matching dtypes|dtype"):
        operation(lhs, rhs, out=output)


def test_bool_sub_tensor_operation_is_rejected(vulkan_backend):
    tensor = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)

    with pytest.raises(RuntimeError, match="bool|dtype|support"):
        torch.sub(tensor, tensor)


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_bool_scalar_operation_is_rejected(vulkan_backend, operation):
    tensor = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)

    with pytest.raises(RuntimeError, match="bool|dtype|support|scalar"):
        operation(tensor, 1)


def test_float_scalar_operation_rejects_bool_out(vulkan_backend):
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32, device=vulkan_backend)
    output = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)

    with pytest.raises(RuntimeError, match="matching.*dtype|dtype"):
        torch.add(tensor, 1.0, out=output)


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_bool_inplace_operation_is_rejected(vulkan_backend, operation):
    tensor = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)
    other = torch.tensor([False, True], dtype=torch.bool, device=vulkan_backend)

    with pytest.raises(RuntimeError, match="in-place|float32"):
        getattr(tensor, operation.__name__ + "_")(other)
