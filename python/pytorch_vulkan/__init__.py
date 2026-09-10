import torch

from ._C import device_count, is_available


class _VulkanDeviceModule:
    @staticmethod
    def is_available():
        return is_available()

    @staticmethod
    def device_count():
        return device_count()

    @staticmethod
    def current_device():
        return 0


torch.utils.rename_privateuse1_backend("vulkan")
torch._register_device_module("vulkan", _VulkanDeviceModule)

__all__ = ["device_count", "is_available"]
