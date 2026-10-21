#pragma once

#include "vulkan_platform.h"

struct VulkanMappedMemoryRange {
    VkDeviceSize offset;
    VkDeviceSize size;
};

class VulkanBuffer {
  public:
    VulkanBuffer(
        const VulkanPlatform &platform, VkDeviceSize size,
        VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer &) = delete;
    VulkanBuffer &operator=(const VulkanBuffer &) = delete;

    VkBuffer buffer() const;
    const VulkanPlatform *platform() const;
    VkDeviceMemory memory() const;
    VkDeviceSize size() const;
    VkMemoryPropertyFlags memory_properties() const;
    void write(const void *data, VkDeviceSize size, VkDeviceSize offset = 0);
    void read(void *data, VkDeviceSize size, VkDeviceSize offset = 0) const;
    static VulkanMappedMemoryRange mapped_range(VkDeviceSize offset,
                                                VkDeviceSize size,
                                                VkDeviceSize allocation_size,
                                                VkDeviceSize atom_size);

  private:
    const VulkanPlatform *platform_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkDeviceSize allocation_size_ = 0;
    VkMemoryPropertyFlags memory_properties_ = 0;
};
