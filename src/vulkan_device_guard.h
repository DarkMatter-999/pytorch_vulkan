#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace pytorch_vulkan {

class VulkanLifecycleCoordinator;
// Host-only access; never constructs a Vulkan instance. Terminal/Python guards
// use this single coordinator rather than duplicating shutdown state.
VulkanLifecycleCoordinator& platform_lifecycle();

class VulkanDeviceGuard final : public c10::impl::DeviceGuardImplInterface {
  public:
    c10::DeviceType type() const override;
    c10::Device exchangeDevice(c10::Device device) const override;
    c10::Device getDevice() const override;
    void setDevice(c10::Device device) const override;
    void uncheckedSetDevice(c10::Device device) const noexcept override;
    c10::Stream getStream(c10::Device device) const noexcept override;
    c10::Stream getDefaultStream(c10::Device device) const override;
    c10::Stream exchangeStream(c10::Stream stream) const noexcept override;
    c10::DeviceIndex deviceCount() const noexcept override;
    bool queryStream(const c10::Stream &stream) const override;
    void synchronizeStream(const c10::Stream &stream) const override;
};

c10::Device current_device();
void set_device(c10::DeviceIndex index);
void shutdown_platform(); // nonterminal cache retirement; live allocations remain valid

} // namespace pytorch_vulkan
