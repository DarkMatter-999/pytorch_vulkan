import re

import pytest
import torch

import pytorch_vulkan


FORMATTER_DOUBLE_ERROR = (
    "Vulkan formatter Double support requires the shaderFloat64 device feature"
)
FORMATTER_CONVERSION_ERROR = (
    "Vulkan formatter conversion supports only Vulkan float32 to Vulkan Double"
)
REQUESTED_DTYPE_ERROR = "Vulkan _to_copy requested dtype does not match source dtype"
DOUBLE_READBACK_ERROR = (
    "Vulkan formatter Double payload readback to CPU is unsupported"
)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_formatter_double_capability_is_explicit(vulkan_backend):
    supported = pytorch_vulkan.formatter_double_supported()
    assert isinstance(supported, bool)

    if supported:
        tensor = torch.empty((4,), dtype=torch.float64, device=vulkan_backend)
        assert tensor.device.type == "vk"
        assert tensor.device.index == 0
        assert tensor.dtype is torch.float64
        assert tensor.nbytes == 4 * torch.tensor([], dtype=torch.float64).element_size()
    else:
        with pytest.raises(RuntimeError, match=re.escape(FORMATTER_DOUBLE_ERROR)):
            torch.empty((4,), dtype=torch.float64, device=vulkan_backend)


def test_f32_to_double_conversion_is_vulkan_resident(vulkan_backend):
    source = torch.tensor([1.25, -2.5, 0.0, 7.0], dtype=torch.float32).to(vulkan_backend)

    if not pytorch_vulkan.formatter_double_supported():
        with pytest.raises(RuntimeError, match=re.escape(FORMATTER_DOUBLE_ERROR)):
            source.to(dtype=torch.float64)
        return

    converted = source.to(dtype=torch.float64)
    assert converted.device == source.device
    assert converted.dtype is torch.float64
    assert converted.is_contiguous()
    assert converted.storage_offset() == 0


def test_cpu_f32_to_vulkan_double_does_not_silently_preserve_f32(vulkan_backend):
    if not pytorch_vulkan.formatter_double_supported():
        pytest.skip("formatter Double capability is unavailable")

    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.escape(FORMATTER_CONVERSION_ERROR)):
        source.to(device=vulkan_backend, dtype=torch.float64)


@pytest.mark.parametrize("requested_dtype", [torch.float32, torch.float64])
def test_vulkan_double_cpu_transfer_does_not_change_requested_dtype(
    vulkan_backend, requested_dtype
):
    if not pytorch_vulkan.formatter_double_supported():
        pytest.skip("formatter Double capability is unavailable")

    source = torch.tensor([1.25, -2.5], dtype=torch.float32, device=vulkan_backend)
    double_source = source.to(dtype=torch.float64)
    with pytest.raises(RuntimeError, match=re.escape(DOUBLE_READBACK_ERROR)):
        double_source.to(device="cpu", dtype=requested_dtype)


def test_vulkan_double_nonzero_offset_cpu_readback_is_rejected(vulkan_backend):
    if not pytorch_vulkan.formatter_double_supported():
        pytest.skip("formatter Double capability is unavailable")

    source = torch.tensor([1.25, -2.5, 7.0], dtype=torch.float32, device=vulkan_backend)
    double_view = source.to(dtype=torch.float64)[1:]
    assert double_view.storage_offset() != 0
    with pytest.raises(RuntimeError, match=re.escape(DOUBLE_READBACK_ERROR)):
        double_view.to(device="cpu")
    with pytest.raises(RuntimeError, match=re.escape(DOUBLE_READBACK_ERROR)):
        double_view.tolist()


def test_vulkan_f32_to_cpu_double_is_not_silently_returned_as_f32(vulkan_backend):
    source = torch.tensor([1.25, -2.5], dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(RuntimeError, match=re.escape(REQUESTED_DTYPE_ERROR)):
        source.to(device="cpu", dtype=torch.float64)


def test_cpu_f32_to_vulkan_bool_is_not_silently_returned_as_f32(vulkan_backend):
    source = torch.tensor([1.0, 0.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.escape(REQUESTED_DTYPE_ERROR)):
        source.to(device=vulkan_backend, dtype=torch.bool)


def test_same_dtype_f32_cpu_vulkan_transfers_preserve_dtype(vulkan_backend):
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    device_tensor = source.to(device=vulkan_backend, dtype=torch.float32)
    result = device_tensor.to(device="cpu", dtype=torch.float32)
    assert device_tensor.dtype is torch.float32
    assert result.dtype is torch.float32


@pytest.mark.parametrize("operation", [torch.add, torch.mul, torch.neg])
def test_unrelated_double_operator_remains_rejected(vulkan_backend, operation):
    if not pytorch_vulkan.formatter_double_supported():
        pytest.skip("formatter Double capability is unavailable")

    tensor = torch.empty((4,), dtype=torch.float64, device=vulkan_backend)
    with pytest.raises(RuntimeError, match=r"Vulkan.*Double|Vulkan.*dtype"):
        if operation is torch.neg:
            operation(tensor)
        else:
            operation(tensor, tensor)
