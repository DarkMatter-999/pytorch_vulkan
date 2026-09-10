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
    VkPhysicalDevice physical_device() const;
    VkDevice device() const;
    VkQueue compute_queue() const;

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VulkanDeviceInfo device_info_;
    uint32_t api_version_ = VK_API_VERSION_1_1;
};
