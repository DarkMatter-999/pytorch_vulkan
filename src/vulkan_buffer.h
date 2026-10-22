#pragma once

#include "vulkan_platform.h"

#include <functional>

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
    // Per-instance fixture seam (production never sets it): invoked immediately
    // before the host maps this buffer in read()/write(), so a native test can
    // observe the host-map exclusion region. Scoped by construction; reset by
    // clearing the hook when the observation window ends.
    void set_before_map_for_testing(std::function<void()> hook) {
        before_map_for_testing_ = std::move(hook);
    }

  private:
    const VulkanPlatform *platform_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkDeviceSize allocation_size_ = 0;
    VkMemoryPropertyFlags memory_properties_ = 0;
    mutable std::function<void()> before_map_for_testing_;
};
