import pytest
import torch

import pytorch_vulkan


def test_cpu_only_add_remains_normal_pytorch_behavior():
    lhs = torch.tensor([1.0, -2.0], dtype=torch.float32)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = torch.add(lhs, rhs)

    torch.testing.assert_close(result, torch.tensor([4.0, 2.0]))


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


def test_python_scalar_add_operand_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, 1.0), "scalar operand")


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


def test_add_out_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    output = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: torch.add(lhs, rhs, out=output), "out")


def test_inplace_add_is_rejected(vulkan_backend):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    _assert_add_rejected(lambda: lhs.add_(rhs), r"Could not run 'aten::add\.out'")
