import gc

import pytest
import torch

import pytorch_vulkan


assert torch._C._dispatch_key_for_device("vk") == "PrivateUse1"
pytestmark = pytest.mark.skipif(
    not pytorch_vulkan.is_available(),
    reason="no suitable Vulkan device is available",
)


def test_torch_empty_uses_vulkan_allocator():
    tensor = torch.empty((16,), dtype=torch.float32, device="vk")

    assert tensor.shape == (16,)
    assert tensor.dtype is torch.float32
    assert tensor.device.type == "vk"
    assert tensor.device.index == 0


def test_torch_empty_without_dtype_uses_default_dtype():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        tensor = torch.empty((16,), device="vk")

        assert tensor.dtype is torch.float64
    finally:
        torch.set_default_dtype(previous_dtype)


def test_multiple_vulkan_tensors_have_independent_lifetimes():
    first = torch.empty((16,), dtype=torch.float32, device="vk")
    second = torch.empty((16,), dtype=torch.float32, device="vk")

    del first
    gc.collect()
    assert second.shape == (16,)


def test_repeated_vulkan_tensor_creation_and_destruction():
    for _ in range(16):
        tensor = torch.empty((16,), dtype=torch.float32, device="vk")
        assert tensor.numel() == 16
        del tensor
        assert torch.vk.current_device() == 0
    gc.collect()
