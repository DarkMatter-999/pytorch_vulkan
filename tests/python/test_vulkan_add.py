import pytest
import torch

import pytorch_vulkan


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
        return torch.add(scalar, tensor, alpha=1.0) if scalar_left else torch.add(
            tensor, scalar, alpha=1.0
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
    lhs = torch.tensor(left_values, dtype=torch.float32).reshape(shape).to(
        vulkan_backend
    )
    rhs = torch.tensor(right_values, dtype=torch.float32).reshape(shape).to(
        vulkan_backend
    )

    result = torch.add(lhs, rhs)

    assert result.device == lhs.device
    assert result.shape == lhs.shape
    assert result.dtype is torch.float32
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.cpu(), torch.tensor(expected, dtype=torch.float32).reshape(shape),
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
    torch.testing.assert_close(
        result.cpu(), torch.tensor([[4.0, 2.0]]), rtol=0, atol=0
    )


@pytest.mark.parametrize("operation, expected", [
    (torch.sub, [[-2.0, -6.0]]),
    (torch.mul, [[3.0, -8.0]]),
])
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
        value.cpu(), torch.tensor([2.0**32, -2.0**33]), rtol=0, atol=0
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


def _assert_add_rejected(operation, message):
    with pytest.raises(RuntimeError, match=message):
        operation()


def _scalar_rejection_pattern(operation, add_pattern):
    return rf"({add_pattern}|scalar operand|float32|contiguous|broadcast|same device|out)"


def test_mixed_device_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "same device")


@pytest.mark.parametrize("vulkan_first", [True, False])
def test_mixed_cpu_vulkan_add_does_not_fallback(vulkan_backend, vulkan_first):
    vulkan_tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    cpu_tensor = torch.empty((2,), dtype=torch.float32)
    operands = (vulkan_tensor, cpu_tensor) if vulkan_first else (cpu_tensor, vulkan_tensor)
    _assert_add_rejected(lambda: torch.add(*operands), "same device")


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_scalar_tensor_operand_is_rejected(vulkan_backend, operation, scalar_left):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    scalar_tensor = torch.tensor(1.0, dtype=torch.float32).to(vulkan_backend)
    operands = (scalar_tensor, tensor) if scalar_left else (tensor, scalar_tensor)
    with pytest.raises(
        RuntimeError,
        match=_scalar_rejection_pattern(operation, r"zero-dimensional|scalar operand"),
    ):
        operation(*operands)


def test_zero_dim_tensor_add_operand_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "zero-dimensional")


def test_broadcasting_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2, 3), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "broadcast")


def test_shape_mismatch_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "size")


def test_non_default_alpha_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs, alpha=2), "alpha")


def test_non_contiguous_add_operand_is_rejected(vulkan_backend):
    lhs = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    rhs = torch.empty((3, 2), dtype=torch.float32, device=vulkan_backend)
    assert not lhs.is_contiguous()
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "contiguous")


def test_non_float32_add_operand_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs), "float32")


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


def test_inplace_add_is_rejected(vulkan_backend):
    lhs = torch.tensor([1.0, 2.0], dtype=torch.float32, device=vulkan_backend)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32, device=vulkan_backend)
    before = lhs.cpu()
    _assert_add_rejected(lambda: lhs.add_(rhs), "in-place")
    torch.testing.assert_close(lhs.cpu(), before)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_unsupported_python_scalar_conversion_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(TypeError, match=r"argument 'other'.*Tensor"):
        _vulkan_scalar_operation(operation, tensor, object())


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_cpu_zero_dim_tensor_is_not_accepted_as_python_scalar(
    vulkan_backend, operation, scalar_left
):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    scalar_tensor = torch.tensor(1.0, dtype=torch.float32)
    with pytest.raises((RuntimeError, NotImplementedError), match=r"scalar operand|Could not run"):
        operands = (scalar_tensor, tensor) if scalar_left else (tensor, scalar_tensor)
        operation(*operands)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar", [float("nan"), float("inf"), -float("inf"), 1e40, -1e40, 1e-50, -1e-50])
def test_unsupported_scalar_values_are_rejected(vulkan_backend, operation, scalar):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(RuntimeError, match=r"non-finite|outside float32|underflows"):
        _vulkan_scalar_operation(operation, tensor, scalar)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_invalid_scalar_tensor_metadata_is_rejected(vulkan_backend, operation):
    wrong_dtype = torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    with pytest.raises(
        RuntimeError,
        match=_scalar_rejection_pattern(operation, r"scalar operand"),
    ):
        operation(wrong_dtype, 1.0)

    non_contiguous = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    with pytest.raises(
        RuntimeError, match=_scalar_rejection_pattern(operation, r"scalar operand")
    ):
        operation(non_contiguous, 1.0)

@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_vulkan_operand_on_wrong_device_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(RuntimeError, match=r"only device index 0"):
        operation(tensor, 1.0) if tensor.device.index != 0 else operation(
            torch.empty((2,), dtype=torch.float32, device="vk:1"), 1.0
        )


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_mixed_cpu_vulkan_tensor_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    cpu_tensor = torch.empty((2,), dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"same device|scalar operand"):
        operation(tensor, cpu_tensor)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_tensor_broadcasting_is_rejected(vulkan_backend, operation):
    lhs = torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2, 3), dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(
        RuntimeError, match=_scalar_rejection_pattern(operation, r"broadcast")
    ):
        operation(lhs, rhs)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_scalar_out_and_inplace_variants_are_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    output = torch.empty_like(tensor)
    assert operation(tensor, 1.0, out=output) is output
    before = tensor.cpu()
    with pytest.raises(
        (RuntimeError, NotImplementedError),
        match=_scalar_rejection_pattern(operation, r"in-place"),
    ):
        getattr(tensor, operation.__name__ + "_")(1.0)
    torch.testing.assert_close(tensor.cpu(), before)
