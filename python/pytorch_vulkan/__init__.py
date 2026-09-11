import torch

from ._C import current_device, device_count, is_available, set_device


class _VulkanDeviceModule:
    @staticmethod
    def is_available():
        return is_available()

    @staticmethod
    def device_count():
        return device_count()

    @staticmethod
    def current_device():
        return current_device()

    @staticmethod
    def set_device(index):
        return set_device(index)


torch.utils.rename_privateuse1_backend("vk")
torch._register_device_module("vk", _VulkanDeviceModule)

__all__ = ["device_count", "is_available"]
