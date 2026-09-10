#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

struct VulkanDeviceInfo {
    std::string name;
    uint32_t compute_queue_family = 0;
};

class VulkanPlatform {
  public:
    VulkanPlatform();
    ~VulkanPlatform();

    VulkanPlatform(const VulkanPlatform &) = delete;
    VulkanPlatform &operator=(const VulkanPlatform &) = delete;

    const VulkanDeviceInfo &device_info() const;
    uint32_t api_version() const;

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VulkanDeviceInfo device_info_;
    uint32_t api_version_ = VK_API_VERSION_1_1;
};
