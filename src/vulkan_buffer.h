#pragma once

#include "vulkan_platform.h"

#include <functional>

struct VulkanMappedMemoryRange {
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct VulkanBufferMapOps {
    void *cookie = nullptr;
    VkResult (*map)(void *, VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize,
                    VkMemoryMapFlags, void **) = nullptr;
    void (*unmap)(void *, VkDevice, VkDeviceMemory) = nullptr;
    VkResult (*flush)(void *, VkDevice, uint32_t, const VkMappedMemoryRange *) = nullptr;
    VkResult (*invalidate)(void *, VkDevice, uint32_t, const VkMappedMemoryRange *) = nullptr;
    // Fixture-only coherency override. Actual allocation flags remain available
    // unchanged through memory_properties(); production leaves this disabled.
    bool override_coherent_for_testing = false;
    bool coherent_for_testing = false;
};

struct VulkanBufferResourceOps {
    void *cookie = nullptr;
    VkResult (*create_buffer)(void *, VkDevice, VkDeviceSize, VkBuffer *) = nullptr;
    void (*get_requirements)(void *, VkDevice, VkBuffer, VkMemoryRequirements *) = nullptr;
    void (*get_memory_properties)(void *, VkPhysicalDevice,
                                  VkPhysicalDeviceMemoryProperties *) = nullptr;
    VkResult (*allocate_memory)(void *, VkDevice, const VkMemoryAllocateInfo *,
                                VkDeviceMemory *) = nullptr;
    VkResult (*bind_buffer)(void *, VkDevice, VkBuffer, VkDeviceMemory,
                            VkDeviceSize) = nullptr;
    void (*destroy_buffer)(void *, VkDevice, VkBuffer) = nullptr;
    void (*free_memory)(void *, VkDevice, VkDeviceMemory) = nullptr;
};

class VulkanBuffer {
  public:
    VulkanBuffer(
        const VulkanPlatform &platform, VkDeviceSize size,
        VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        const VulkanBufferResourceOps *resource_ops_for_testing = nullptr);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer &) = delete;
    VulkanBuffer &operator=(const VulkanBuffer &) = delete;

    VkBuffer buffer() const;
    const VulkanPlatform *platform() const;
    VkDeviceMemory memory() const;
    VkDeviceSize size() const;
    VkMemoryPropertyFlags memory_properties() const;
    void *map_persistent();
    void *mapped_base() const;
    bool coherent() const;
    void flush_range(VkDeviceSize offset, VkDeviceSize size);
    void invalidate_range(VkDeviceSize offset, VkDeviceSize size);
    void write(const void *data, VkDeviceSize size, VkDeviceSize offset = 0);
    void read(void *data, VkDeviceSize size, VkDeviceSize offset = 0) const;
    static VulkanMappedMemoryRange mapped_range(VkDeviceSize offset,
                                                VkDeviceSize size,
                                                VkDeviceSize allocation_size,
                                                VkDeviceSize atom_size);
    // Per-instance fixture seam (production never sets it): invoked immediately
    // before each actual host map (map_persistent or a temporary read/write map),
    // not when read/write reuses the persistent mapping.
    void set_before_map_for_testing(std::function<void()> hook) {
        before_map_for_testing_ = std::move(hook);
    }
    void set_map_ops_for_testing(VulkanBufferMapOps ops) { map_ops_for_testing_ = ops; }

  private:
    const VulkanPlatform *platform_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkDeviceSize allocation_size_ = 0;
    VkMemoryPropertyFlags memory_properties_ = 0;
    mutable std::function<void()> before_map_for_testing_;
    VulkanBufferMapOps map_ops_for_testing_{};
    VulkanBufferResourceOps resource_ops_for_testing_{};
    void *mapped_base_ = nullptr;

    VkResult map_memory(void **mapped) const;
    void unmap_memory() const;
    VkResult flush_memory(const VkMappedMemoryRange *range) const;
    VkResult invalidate_memory(const VkMappedMemoryRange *range) const;
    VkMappedMemoryRange vk_range(VkDeviceSize offset, VkDeviceSize size) const;
};
