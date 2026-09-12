import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _call(operation, *args, out, **kwargs):
    return operation(*args, out=out, **kwargs)


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
    try:
        lhs = base[:2]
        out = base[1:]
    except NotImplementedError:
        pytest.skip("Vulkan view/as_strided support is unavailable in this build")
    with pytest.raises((RuntimeError, NotImplementedError), match="overlap"):
        torch.add(lhs, rhs, out=out)


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
    with pytest.raises((RuntimeError, NotImplementedError), match="Vulkan output tensor"):
        torch.add(lhs, rhs, out=torch.empty(2))
    with pytest.raises((RuntimeError, NotImplementedError), match="float32 output"):
        torch.add(lhs, rhs, out=torch.empty(2, dtype=torch.float64, device=vulkan_backend))
    with pytest.raises((RuntimeError, NotImplementedError), match="contiguous"):
        torch.add(lhs, rhs, out=torch.empty_strided((2, 2), (1, 2), device=vulkan_backend))
    internally_overlapping = torch.empty_strided((2,), (0,), device=vulkan_backend)
    with pytest.raises((RuntimeError, NotImplementedError), match="internal overlap"):
        torch.add(lhs, rhs, out=internally_overlapping)
    with pytest.raises((RuntimeError, NotImplementedError), match="alpha == 1"):
        torch.add(lhs, rhs, alpha=2, out=torch.empty_like(lhs))
    with pytest.raises((RuntimeError, NotImplementedError), match="non-finite"):
        torch.mul(lhs, float("nan"), out=torch.empty_like(lhs))


def test_out_rejects_nonzero_storage_offset(vulkan_backend):
    input = torch.empty((2,), device=vulkan_backend)
    try:
        base = torch.empty((3,), device=vulkan_backend)
        out = torch.as_strided(base, (2,), (1,), storage_offset=1)
    except NotImplementedError:
        pytest.skip("Vulkan view/as_strided support is unavailable in this build")
    with pytest.raises((RuntimeError, NotImplementedError), match="storage_offset"):
        torch.neg(input, out=out)
