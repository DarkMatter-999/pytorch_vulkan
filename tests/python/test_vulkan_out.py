import pytest
import torch

import pytorch_vulkan
from vulkan_conformance import (
    SCALAR_OUT_CONTRACT_MATRIX,
    assert_scalar_out_counters,
    invoke_scalar_out_contract,
)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _call(operation, *args, out, **kwargs):
    return operation(*args, out=out, **kwargs)


def _assert_zero_work():
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


SCALAR_OUT_OUT_CASES = tuple(
    case for case in SCALAR_OUT_CONTRACT_MATRIX.values() if case.output_mode == "out"
)


@pytest.mark.parametrize(
    "operation, reference",
    [
        (torch.ops.aten.neg.out, torch.neg),
        (torch.ops.aten.abs.out, torch.abs),
        (torch.ops.aten.relu.out, torch.relu),
    ],
)
def test_unary_out_identity_values_and_inputs(vulkan_backend, operation, reference):
    input_cpu = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    input_vk = input_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    result = _call(operation, input_vk, out=out)
    assert result is out
    torch.testing.assert_close(out.cpu(), reference(input_cpu))
    torch.testing.assert_close(input_vk.cpu(), input_cpu)


@pytest.mark.parametrize(
    "operation, reference, kwargs",
    [
        (torch.add, lambda a, b: a + b, {}),
        (torch.sub, lambda a, b: a - b, {}),
        (torch.mul, lambda a, b: a * b, {}),
        (torch.add, lambda a, b: a + b, {"alpha": 1}),
        (torch.sub, lambda a, b: a - b, {"alpha": 1}),
    ],
)
def test_tensor_tensor_out_identity_values_and_inputs(
    vulkan_backend, operation, reference, kwargs
):
    lhs_cpu = torch.tensor([1.0, -2.0], dtype=torch.float32)
    rhs_cpu = torch.tensor([3.0, 4.0], dtype=torch.float32)
    lhs, rhs = lhs_cpu.to(vulkan_backend), rhs_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    result = _call(operation, lhs, rhs, out=out, **kwargs)
    assert result is out
    torch.testing.assert_close(out.cpu(), reference(lhs_cpu, rhs_cpu))
    torch.testing.assert_close(lhs.cpu(), lhs_cpu)
    torch.testing.assert_close(rhs.cpu(), rhs_cpu)


@pytest.mark.parametrize(
    "operation, args, reference, scalar_left",
    [
        (torch.add, (2.0,), lambda x: x + 2.0, False),
        (torch.sub, (2.0,), lambda x: x - 2.0, False),
        (torch.add, (2.0,), lambda x: x + 2.0, True),
        (torch.sub, (2.0,), lambda x: 2.0 - x, True),
        (torch.mul, (2.0,), lambda x: x * 2.0, False),
    ],
)
def test_scalar_out_identity_values_and_inputs(
    vulkan_backend, operation, args, reference, scalar_left
):
    input_cpu = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    input_vk = input_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    if scalar_left:
        result = _call(operation, *args, input_vk, out=out)
    else:
        result = _call(operation, input_vk, *args, out=out)
    assert result is out
    torch.testing.assert_close(out.cpu(), reference(input_cpu))
    torch.testing.assert_close(input_vk.cpu(), input_cpu)


@pytest.mark.parametrize(
    "contract",
    SCALAR_OUT_OUT_CASES,
    ids=lambda case: case.case_name,
)
def test_scalar_out_contract_matrix_cases_return_named_output_and_do_vulkan_work(
    vulkan_backend, contract
):
    lhs_cpu = torch.tensor([1.5, -2.0, 0.0], dtype=torch.float32)
    lhs = lhs_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    rhs_cpu = None
    rhs = None
    if contract.operand_order == "tensor-tensor":
        rhs_cpu = torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float32)
        rhs = rhs_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = invoke_scalar_out_contract(contract, lhs, rhs=rhs, out=out)
    expected = contract.cpu_reference(lhs_cpu, rhs=rhs_cpu)

    assert result is out
    assert contract.output_mode == "out"
    assert out.device == torch.device("vk:0")
    assert result.device == torch.device("vk:0")
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_scalar_out_contract_matrix_rejects_cpu_output_without_work(
    vulkan_backend, contract
):
    lhs = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = (
        torch.full_like(lhs, 2.0)
        if contract.operand_order == "tensor-tensor"
        else None
    )
    cpu_out = torch.empty((2,), dtype=torch.float32)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"Vulkan output tensor|Vulkan"):
        invoke_scalar_out_contract(contract, lhs, rhs=rhs, out=cpu_out)

    _assert_zero_work()


SCALAR_OUT_SCALAR_CASES = tuple(
    case for case in SCALAR_OUT_OUT_CASES if case.operand_order != "tensor-tensor"
)


def _contract_inputs(contract, device, shape=(2, 2)):
    lhs = torch.ones(shape, dtype=torch.float32, device=device)
    rhs = (
        torch.full_like(lhs, 2.0) if contract.operand_order == "tensor-tensor" else None
    )
    return lhs, rhs


def _invoke_contract(contract, lhs, rhs=None, out=None):
    return invoke_scalar_out_contract(contract, lhs, rhs=rhs, out=out)


@pytest.mark.parametrize(
    "contract", SCALAR_OUT_SCALAR_CASES, ids=lambda case: case.case_name
)
def test_scalar_out_contract_rejects_invalid_scalar_without_work(
    vulkan_backend, contract
):
    lhs = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    out = torch.empty_like(lhs)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"non-finite|outside float32|underflows"):
        invoke_scalar_out_contract(contract, lhs, out=out, scalar=float("nan"))

    _assert_zero_work()


@pytest.mark.parametrize(
    "contract",
    tuple(
        case
        for case in SCALAR_OUT_SCALAR_CASES
        if case.schema.split("::", 1)[1].split(".", 1)[0]
        in {"add", "sub", "rsub"}
    ),
    ids=lambda case: case.case_name,
)
def test_scalar_out_contract_rejects_invalid_alpha_without_work(
    vulkan_backend, contract
):
    lhs = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    out = torch.empty_like(lhs)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"alpha == 1"):
        invoke_scalar_out_contract(contract, lhs, out=out, alpha=2.0)

    _assert_zero_work()


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_out_contract_rejects_partial_overlap_for_every_schema(vulkan_backend, contract):
    base = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    lhs = base[:2]
    rhs = torch.ones_like(lhs) if contract.operand_order == "tensor-tensor" else None
    out = base[1:]
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="overlap"):
        _invoke_contract(contract, lhs, rhs=rhs, out=out)

    _assert_zero_work()


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_out_contract_rejects_internal_overlap_for_every_schema(vulkan_backend, contract):
    lhs, rhs = _contract_inputs(contract, vulkan_backend)
    out = torch.empty_strided(
        lhs.shape, (0, 1), dtype=torch.float32, device=vulkan_backend
    )
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="internal overlap"):
        _invoke_contract(contract, lhs, rhs=rhs, out=out)

    _assert_zero_work()


@pytest.mark.parametrize(
    "contract",
    tuple(case for case in SCALAR_OUT_OUT_CASES if case.operand_order == "tensor-tensor"),
    ids=lambda case: case.case_name,
)
def test_tensor_tensor_out_contract_exact_aliases(vulkan_backend, contract):
    lhs_cpu = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
    rhs_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    lhs, rhs = lhs_cpu.to(vulkan_backend), rhs_cpu.to(vulkan_backend)
    expected = contract.cpu_reference(lhs_cpu, rhs=rhs_cpu)
    pytorch_vulkan._C.reset_execution_counters()

    result = _invoke_contract(contract, lhs, rhs=rhs, out=lhs)

    assert result is lhs
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(lhs.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "contract",
    tuple(case for case in SCALAR_OUT_OUT_CASES if case.operand_order == "tensor-tensor"),
    ids=lambda case: case.case_name,
)
def test_tensor_tensor_out_contract_second_operand_exact_aliases(vulkan_backend, contract):
    lhs_cpu = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
    rhs_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    lhs, rhs = lhs_cpu.to(vulkan_backend), rhs_cpu.to(vulkan_backend)
    expected = contract.cpu_reference(lhs_cpu, rhs=rhs_cpu)
    pytorch_vulkan._C.reset_execution_counters()

    result = _invoke_contract(contract, lhs, rhs=rhs, out=rhs)

    assert result is rhs
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(rhs.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("contract", SCALAR_OUT_SCALAR_CASES, ids=lambda case: case.case_name)
def test_scalar_out_contract_exact_aliases(vulkan_backend, contract):
    value_cpu = torch.tensor([[-2.0, 1.0], [3.0, 4.0]], dtype=torch.float32)
    value = value_cpu.to(vulkan_backend)
    expected = contract.cpu_reference(value_cpu)
    pytorch_vulkan._C.reset_execution_counters()

    result = _invoke_contract(contract, value, out=value)

    assert result is value
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_out_contract_accepts_non_contiguous_output_for_every_schema(
    vulkan_backend, contract
):
    lhs, rhs = _contract_inputs(contract, vulkan_backend)
    out = torch.empty_strided(lhs.shape, (1, 2), dtype=torch.float32, device=vulkan_backend)
    assert not out.is_contiguous()
    expected = contract.cpu_reference(lhs.cpu(), rhs=rhs.cpu() if rhs is not None else None)
    pytorch_vulkan._C.reset_execution_counters()

    result = _invoke_contract(contract, lhs, rhs=rhs, out=out)

    assert result is out
    assert tuple(out.stride()) == (1, 2)
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_out_contract_resizes_non_contiguous_output_for_every_schema(
    vulkan_backend, contract
):
    lhs, rhs = _contract_inputs(contract, vulkan_backend, shape=(2, 3))
    out = torch.empty_strided((2, 2), (1, 2), dtype=torch.float32, device=vulkan_backend)
    assert not out.is_contiguous()
    expected = contract.cpu_reference(lhs.cpu(), rhs=rhs.cpu() if rhs is not None else None)
    pytorch_vulkan._C.reset_execution_counters()

    result = _invoke_contract(contract, lhs, rhs=rhs, out=out)

    assert result is out
    assert tuple(out.shape) == tuple(lhs.shape)
    assert_scalar_out_counters(contract.expected_counters)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("contract", SCALAR_OUT_SCALAR_CASES, ids=lambda case: case.case_name)
def test_scalar_out_contract_wrong_dtype_rejects_without_work(vulkan_backend, contract):
    lhs, _ = _contract_inputs(contract, vulkan_backend)
    out = torch.empty(lhs.shape, dtype=torch.float64, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"dtype Double|matching input and output"):
        _invoke_contract(contract, lhs, out=out)

    _assert_zero_work()


@pytest.mark.parametrize("contract", SCALAR_OUT_SCALAR_CASES, ids=lambda case: case.case_name)
def test_scalar_out_contract_cpu_output_rejects_without_work(vulkan_backend, contract):
    lhs, _ = _contract_inputs(contract, vulkan_backend)
    out = torch.empty(lhs.shape, dtype=torch.float32)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"Vulkan output tensor|Vulkan"):
        _invoke_contract(contract, lhs, out=out)

    _assert_zero_work()


@pytest.mark.parametrize(
    "contract",
    tuple(case for case in SCALAR_OUT_OUT_CASES if case.operand_order == "tensor-tensor"),
    ids=lambda case: case.case_name,
)
def test_tensor_tensor_out_contract_wrong_dtype_rejects_without_work(
    vulkan_backend, contract
):
    lhs, rhs = _contract_inputs(contract, vulkan_backend)
    out = torch.empty(lhs.shape, dtype=torch.float64, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"dtype Double|matching input and output"):
        _invoke_contract(contract, lhs, rhs=rhs, out=out)

    _assert_zero_work()


@pytest.mark.parametrize("contract", SCALAR_OUT_OUT_CASES, ids=lambda case: case.case_name)
def test_out_contract_wrong_vulkan_device_index_rejects_without_work(
    vulkan_backend, contract
):
    lhs, rhs = _contract_inputs(contract, vulkan_backend)
    try:
        out = torch.empty(lhs.shape, dtype=torch.float32, device="vk:1")
    except RuntimeError:
        pytest.skip("Vulkan device index 1 is not constructible in this environment")
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"device index 0|same Vulkan platform|Vulkan"):
        _invoke_contract(contract, lhs, rhs=rhs, out=out)

    _assert_zero_work()


def test_out_resizes_and_exact_aliases(vulkan_backend):
    input = torch.tensor([-2.0, 3.0], device=vulkan_backend)
    other = torch.tensor([4.0, 5.0], device=vulkan_backend)
    out = torch.empty((0,), device=vulkan_backend)
    assert torch.add(input, other, out=out) is out
    assert tuple(out.shape) == (2,)
    torch.testing.assert_close(out.cpu(), torch.tensor([2.0, 8.0]))

    alias = input
    assert torch.mul(alias, other, out=alias) is alias
    torch.testing.assert_close(alias.cpu(), torch.tensor([-8.0, 15.0]))


def test_rsub_scalar_out_routes_scalar_left(vulkan_backend):
    input_cpu = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    input_vk = input_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    result = torch.ops.aten.rsub.Scalar_out(input_vk, 2.0, out=out)
    assert result is out
    torch.testing.assert_close(out.cpu(), 2.0 - input_cpu)


@pytest.mark.parametrize(
    "operation, reference",
    [
        (torch.add, lambda values: 2.0 + values),
        (torch.mul, lambda values: 2.0 * values),
    ],
)
def test_scalar_left_out_exact_alias(vulkan_backend, operation, reference):
    values = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    assert operation(2.0, tensor, out=tensor) is tensor
    torch.testing.assert_close(tensor.cpu(), reference(values))


def test_rsub_scalar_out_exact_alias(vulkan_backend):
    values = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    assert torch.ops.aten.rsub.Scalar_out(tensor, 2.0, out=tensor) is tensor
    torch.testing.assert_close(tensor.cpu(), 2.0 - values)


def test_out_rejects_same_shaped_partial_overlap(vulkan_backend):
    base = torch.empty((3,), device=vulkan_backend)
    rhs = torch.empty((2,), device=vulkan_backend)
    lhs = base[:2]
    out = base[1:]
    with pytest.raises((RuntimeError, NotImplementedError), match="overlap"):
        torch.add(lhs, rhs, out=out)


def test_non_contiguous_out_preserves_layout_and_values(vulkan_backend):
    lhs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=vulkan_backend)
    rhs = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=vulkan_backend)
    out = torch.empty_strided((2, 2), (1, 2), device=vulkan_backend)
    assert not out.is_contiguous()

    assert torch.add(lhs, rhs, out=out) is out
    assert tuple(out.stride()) == (1, 2)
    torch.testing.assert_close(torch.neg(out).cpu(), -(lhs.cpu() + rhs.cpu()))


def test_out_rejects_internal_overlap(vulkan_backend):
    lhs = torch.empty((2, 2), device=vulkan_backend)
    rhs = torch.empty_like(lhs)
    out = torch.empty_strided((2, 2), (0, 1), device=vulkan_backend)

    with pytest.raises((RuntimeError, NotImplementedError), match="internal overlap"):
        torch.add(lhs, rhs, out=out)


def test_bool_out_exact_alias_is_safe(vulkan_backend):
    lhs = torch.tensor([True, False, True], dtype=torch.bool, device=vulkan_backend)
    rhs = torch.tensor([True, True, False], dtype=torch.bool, device=vulkan_backend)
    expected = torch.logical_and(lhs.cpu(), rhs.cpu())

    assert torch.bitwise_and(lhs, rhs, out=lhs) is lhs
    torch.testing.assert_close(lhs.cpu(), expected)


def test_bool_out_rejects_partial_and_internal_overlap(vulkan_backend):
    base = torch.empty((3,), dtype=torch.bool, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError), match="overlap"):
        torch.bitwise_and(base[:2], rhs, out=base[1:])

    internal = torch.empty_strided((2,), (0,), dtype=torch.bool, device=vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError), match="internal overlap"):
        torch.bitwise_and(rhs, rhs, out=internal)


def test_bool_out_rejects_rank_above_eight(vulkan_backend):
    lhs = torch.empty((1,) * 9, dtype=torch.bool, device=vulkan_backend)
    rhs = torch.empty_like(lhs)
    out = torch.empty_like(lhs)
    with pytest.raises((RuntimeError, NotImplementedError), match="rank.*8"):
        torch.bitwise_and(lhs, rhs, out=out)


@pytest.mark.parametrize(
    "operation, reference",
    [
        (torch.ops.aten.neg.out, torch.neg),
        (torch.ops.aten.abs.out, torch.abs),
        (torch.ops.aten.relu.out, torch.relu),
    ],
)
def test_unary_out_exact_alias(vulkan_backend, operation, reference):
    value = torch.tensor([-2.0, 3.0], device=vulkan_backend)
    expected = reference(value.cpu())
    assert operation(value, out=value) is value
    torch.testing.assert_close(value.cpu(), expected)


def test_out_rejects_invalid_metadata_overlap_and_parameters(vulkan_backend):
    lhs = torch.empty((2,), device=vulkan_backend)
    rhs = torch.empty((2,), device=vulkan_backend)
    with pytest.raises(
        (RuntimeError, NotImplementedError), match="Vulkan output tensor"
    ):
        torch.add(lhs, rhs, out=torch.empty(2))
    with pytest.raises((RuntimeError, NotImplementedError), match="dtype Double"):
        torch.add(
            lhs, rhs, out=torch.empty(2, dtype=torch.float64, device=vulkan_backend)
        )
    non_contiguous = torch.empty_strided((2, 2), (1, 2), device=vulkan_backend)
    assert torch.add(lhs, rhs, out=non_contiguous) is non_contiguous
    internally_overlapping = torch.empty_strided((2,), (0,), device=vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError), match="internal overlap"):
        torch.add(lhs, rhs, out=internally_overlapping)
    with pytest.raises((RuntimeError, NotImplementedError), match="alpha == 1"):
        torch.add(lhs, rhs, alpha=2, out=torch.empty_like(lhs))
    with pytest.raises((RuntimeError, NotImplementedError), match="non-finite"):
        torch.mul(lhs, float("nan"), out=torch.empty_like(lhs))


def test_out_rejects_unrelated_double_operators(vulkan_backend):
    if not pytorch_vulkan.formatter_double_supported():
        pytest.skip("formatter Double capability is unavailable")

    value = torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="Vulkan add does not support dtype Double"):
        torch.add(value, value)
    with pytest.raises(RuntimeError, match="Vulkan mul does not support dtype Double"):
        torch.mul(value, value)


def test_out_accepts_nonzero_storage_offset(vulkan_backend):
    input = torch.empty((2,), device=vulkan_backend)
    base = torch.empty((3,), device=vulkan_backend)
    out = torch.as_strided(base, (2,), (1,), storage_offset=1)
    assert torch.neg(input, out=out) is out


@pytest.mark.parametrize(
    "operation, reference",
    [
        (torch.add, lambda lhs, rhs: lhs + rhs),
        (torch.sub, lambda lhs, rhs: lhs - rhs),
        (torch.mul, lambda lhs, rhs: lhs * rhs),
    ],
)
def test_binary_out_empty_input_is_identity_parity_and_zero_work(
    vulkan_backend, operation, reference
):
    lhs_cpu = torch.empty((0, 3), dtype=torch.float32)
    rhs_cpu = torch.empty((0, 3), dtype=torch.float32)
    lhs, rhs = lhs_cpu.to(vulkan_backend), rhs_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = operation(lhs, rhs, out=out)

    assert result is out
    assert tuple(out.shape) == (0, 3)
    torch.testing.assert_close(out.cpu(), reference(lhs_cpu, rhs_cpu))
    _assert_zero_work()


@pytest.mark.parametrize(
    "operation, reference, scalar_left",
    [
        (torch.add, lambda value: value + 2.0, False),
        (torch.sub, lambda value: value - 2.0, False),
        (torch.add, lambda value: 2.0 + value, True),
        (torch.sub, lambda value: 2.0 - value, True),
        (torch.mul, lambda value: value * 2.0, False),
    ],
)
def test_scalar_out_empty_input_is_identity_parity_and_zero_work(
    vulkan_backend, operation, reference, scalar_left
):
    value_cpu = torch.empty((0, 3), dtype=torch.float32)
    value = value_cpu.to(vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    if scalar_left:
        result = operation(2.0, value, out=out)
    else:
        result = operation(value, 2.0, out=out)

    assert result is out
    assert tuple(out.shape) == (0, 3)
    torch.testing.assert_close(out.cpu(), reference(value_cpu))
    _assert_zero_work()


@pytest.mark.parametrize(
    "contract",
    tuple(case for case in SCALAR_OUT_OUT_CASES if case.operand_order == "tensor-tensor"),
    ids=lambda case: case.case_name,
)
def test_tensor_tensor_out_broadcast_rejected_before_work(vulkan_backend, contract):
    lhs = torch.ones((2, 1), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.ones((1, 2), dtype=torch.float32, device=vulkan_backend)
    out = torch.empty((1,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"equal tensor sizes|broadcast"):
        _invoke_contract(contract, lhs, rhs=rhs, out=out)

    _assert_zero_work()


def test_out_overlap_rejections_are_zero_work(vulkan_backend):
    lhs = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.ones_like(lhs)

    base = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="partially overlaps"):
        torch.add(base[:2], rhs, out=base[1:])
    _assert_zero_work()

    internally_overlapping = torch.empty_strided(
        (2,), (0,), dtype=torch.float32, device=vulkan_backend
    )
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="internal overlap"):
        torch.add(lhs, rhs, out=internally_overlapping)
    _assert_zero_work()


def test_out_dtype_and_device_rejections_are_zero_work(vulkan_backend):
    lhs = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.ones_like(lhs)

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="dtype Double"):
        torch.add(lhs, rhs, out=torch.empty(2, dtype=torch.float64, device=vulkan_backend))
    _assert_zero_work()

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"Vulkan output tensor"):
        torch.add(lhs, rhs, out=torch.empty(2, dtype=torch.float32))
    _assert_zero_work()
