#pragma once

#include "vulkan_platform.h"

class VulkanBuffer {
  public:
    VulkanBuffer(
        const VulkanPlatform &platform, VkDeviceSize size,
        VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer &) = delete;
    VulkanBuffer &operator=(const VulkanBuffer &) = delete;

    VkBuffer buffer() const;
    VkDeviceMemory memory() const;
    VkDeviceSize size() const;
    VkMemoryPropertyFlags memory_properties() const;
    void write(const void *data, VkDeviceSize size, VkDeviceSize offset = 0);
    void read(void *data, VkDeviceSize size, VkDeviceSize offset = 0) const;

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkMemoryPropertyFlags memory_properties_ = 0;
};
