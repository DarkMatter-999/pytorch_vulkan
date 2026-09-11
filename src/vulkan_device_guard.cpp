#include "vulkan_device_guard.h"

#include "vulkan_platform.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Exception.h>

#include <memory>

namespace pytorch_vulkan {
namespace {

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);
thread_local c10::Device current = kDevice;
thread_local c10::Stream stream(c10::Stream::DEFAULT, kDevice);

void check_device(c10::Device device) {
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                "Vulkan guard received device of type ", device.type());
    TORCH_CHECK(device.index() == 0,
                "Vulkan backend supports only device index 0, got ", device.index());
}

std::shared_ptr<VulkanPlatform> platform() {
    static std::shared_ptr<VulkanPlatform> instance = [] {
        if (!VulkanPlatform::is_available()) {
            return std::shared_ptr<VulkanPlatform>();
        }
        return std::make_shared<VulkanPlatform>();
    }();
    TORCH_CHECK(instance != nullptr, "Vulkan device is unavailable");
    return instance;
}

class VulkanPrivateUse1Hooks final : public at::PrivateUse1HooksInterface {
  public:
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        return device_index == 0 && VulkanPlatform::is_available();
    }
};

const bool hooks_registered = [] {
    at::RegisterPrivateUse1HooksInterface(new VulkanPrivateUse1Hooks());
    return true;
}();

} // namespace

c10::DeviceType VulkanDeviceGuard::type() const {
    return c10::DeviceType::PrivateUse1;
}

c10::Device VulkanDeviceGuard::exchangeDevice(c10::Device device) const {
    check_device(device);
    const c10::Device previous = current;
    current = device;
    return previous;
}

c10::Device VulkanDeviceGuard::getDevice() const {
    return current;
}

void VulkanDeviceGuard::setDevice(c10::Device device) const {
    check_device(device);
    current = device;
}

void VulkanDeviceGuard::uncheckedSetDevice(c10::Device device) const noexcept {
    if (device.type() == c10::DeviceType::PrivateUse1 && device.index() == 0) {
        current = device;
    }
}

c10::Stream VulkanDeviceGuard::getStream(c10::Device device) const noexcept {
    TORCH_INTERNAL_ASSERT(
        device.type() == c10::DeviceType::PrivateUse1 && device.index() == 0,
        "Vulkan backend supports only device index 0");
    return device == current ? stream : c10::Stream(c10::Stream::DEFAULT, device);
}

c10::Stream VulkanDeviceGuard::getDefaultStream(c10::Device device) const {
    check_device(device);
    return c10::Stream(c10::Stream::DEFAULT, device);
}

c10::Stream VulkanDeviceGuard::exchangeStream(c10::Stream next) const noexcept {
    TORCH_INTERNAL_ASSERT(next.device_type() == c10::DeviceType::PrivateUse1 &&
                              next.device_index() == 0,
                          "Vulkan backend supports only device index 0");
    const c10::Stream previous = stream;
    stream = next;
    return previous;
}

c10::DeviceIndex VulkanDeviceGuard::deviceCount() const noexcept {
    return VulkanPlatform::is_available() ? 1 : 0;
}

bool VulkanDeviceGuard::queryStream(const c10::Stream &stream) const {
    check_device(stream.device());
    const VkResult result = vkQueueWaitIdle(platform()->compute_queue());
    TORCH_CHECK(result == VK_SUCCESS, "Could not query Vulkan compute queue");
    return true;
}

void VulkanDeviceGuard::synchronizeStream(const c10::Stream &stream) const {
    check_device(stream.device());
    const VkResult result = vkQueueWaitIdle(platform()->compute_queue());
    TORCH_CHECK(result == VK_SUCCESS, "Could not synchronize Vulkan compute queue");
}

c10::Device current_device() {
    return current;
}

void set_device(c10::DeviceIndex index) {
    VulkanDeviceGuard guard;
    guard.setDevice(c10::Device(c10::DeviceType::PrivateUse1, index));
}

} // namespace pytorch_vulkan

C10_REGISTER_GUARD_IMPL(PrivateUse1, pytorch_vulkan::VulkanDeviceGuard)
