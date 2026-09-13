import pytest
import torch

import pytorch_vulkan


CPU_VALUES = torch.tensor([-3.5, 0.0, 2.25, 1_000_000.0], dtype=torch.float64)


@pytest.fixture
def double_tensor():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return CPU_VALUES.to(dtype=torch.float32).to("vk:0").to(dtype=torch.float64)


def assert_double_tensor(result, expected):
    assert result.device == torch.device("vk:0")
    assert result.dtype == torch.float64
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.numel() == expected.numel()
    actual = [value.item() for value in result.reshape(-1).unbind()]
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), expected.reshape(-1))


def test_formatter_double_abs(double_tensor):
    assert_double_tensor(torch.abs(double_tensor), CPU_VALUES.abs())


def test_formatter_double_min_and_max(double_tensor):
    minimum = double_tensor.min()
    maximum = double_tensor.max()
    assert minimum.device == torch.device("vk:0")
    assert maximum.device == torch.device("vk:0")
    assert minimum.dtype == maximum.dtype == torch.float64
    assert minimum.item() == pytest.approx(CPU_VALUES.min().item())
    assert maximum.item() == pytest.approx(CPU_VALUES.max().item())


def test_formatter_double_scalar_division(double_tensor):
    assert_double_tensor(double_tensor / double_tensor.abs().max(),
                         CPU_VALUES / CPU_VALUES.abs().max())


def test_formatter_double_scalar_comparisons(double_tensor):
    value = double_tensor.min()
    not_equal = value.ne(value.ceil())
    greater = value.gt(1.0e8)
    assert not_equal.device == double_tensor.device
    assert greater.device == double_tensor.device
    assert not_equal.dtype == torch.bool
    assert greater.dtype == torch.bool
    cpu_value = CPU_VALUES.min()
    torch.testing.assert_close(torch.tensor(not_equal.cpu().item()), cpu_value.ne(cpu_value.ceil()))
    torch.testing.assert_close(torch.tensor(greater.cpu().item()), cpu_value.gt(1.0e8))


def test_float_ne_tensor_nonzero_storage_offset_is_rejected(double_tensor):
    base = CPU_VALUES.to(dtype=torch.float32).to(double_tensor.device)
    lhs = torch.as_strided(base, (3,), (1,), storage_offset=1)
    rhs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(double_tensor.device)
    assert lhs.storage_offset() != 0

    with pytest.raises(RuntimeError, match="storage_offset"):
        torch.ne(lhs, rhs)


def test_formatter_double_ceil(double_tensor):
    assert_double_tensor(torch.ceil(double_tensor), CPU_VALUES.ceil())


def test_formatter_double_scalar_extraction(double_tensor):
    assert double_tensor.min().item() == pytest.approx(CPU_VALUES.min().item())


@pytest.mark.parametrize("operation", [lambda tensor: tensor + tensor, lambda tensor: tensor * tensor])
def test_formatter_double_rejects_unrelated_arithmetic(double_tensor, operation):
    with pytest.raises((RuntimeError, NotImplementedError), match="Double|unsupported|not exist"):
        operation(double_tensor)


def test_formatter_double_rejects_unregistered_comparison_forms(double_tensor):
    with pytest.raises((RuntimeError, NotImplementedError), match="Double|float32|not exist"):
        double_tensor.ne(0.0)
    with pytest.raises((RuntimeError, NotImplementedError), match="Double|float32|not exist"):
        torch.isfinite(double_tensor)
    with pytest.raises((RuntimeError, NotImplementedError), match="Double|float32|not exist"):
        torch.ne(double_tensor, 0.0, out=torch.empty(double_tensor.shape, dtype=torch.bool, device="vk:0"))
    with pytest.raises((RuntimeError, NotImplementedError), match="Double|float32|not exist|aten::lt"):
        torch.lt(double_tensor, 1.0e-4, out=torch.empty(double_tensor.shape, dtype=torch.bool, device="vk:0"))
