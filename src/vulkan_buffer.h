#pragma once

#include "vulkan_platform.h"

class VulkanBuffer {
  public:
    VulkanBuffer(const VulkanPlatform &platform, VkDeviceSize size);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer &) = delete;
    VulkanBuffer &operator=(const VulkanBuffer &) = delete;

    VkBuffer buffer() const;
    VkDeviceMemory memory() const;
    VkDeviceSize size() const;

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
};
