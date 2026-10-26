#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include "vulkan_buffer.h"
#include <vulkan/vulkan.h>

#include <memory>

struct VulkanPinnedHostRange {
    std::shared_ptr<void> lease;
    VulkanBuffer *buffer = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize bytes = 0;
};

class VulkanPlatform;

// Optional per-instance Vulkan entry points and lifecycle policy used by the
// focused native fixture. Null Vulkan entries call the real Vulkan API.
struct VulkanPinnedHostTestOps {
    VulkanBufferResourceOps buffer{};
    VulkanBufferMapOps map{};
    bool (*should_leak_for_testing)(void *, const VulkanPlatform &) noexcept = nullptr;
    void *cookie = nullptr;
};

class VulkanPinnedHostAllocator final : public at::Allocator {
  public:
    explicit VulkanPinnedHostAllocator(
        std::shared_ptr<VulkanPlatform> platform = {},
        VulkanPinnedHostTestOps ops = {});
    at::DataPtr allocate(size_t nbytes) override;
    void copy_data(void *dest, const void *src, std::size_t count) const override;

  private:
    std::shared_ptr<VulkanPlatform> platform_;
    VulkanPinnedHostTestOps ops_;
};

VulkanPinnedHostAllocator *vulkan_pinned_host_allocator_instance();
bool is_vulkan_pinned_storage(const c10::Storage &storage);
bool vulkan_pinned_host_range(const c10::Storage &storage, const void *data,
                              size_t bytes, VulkanPinnedHostRange *out);
VkMemoryPropertyFlags vulkan_pinned_host_memory_properties_for_testing(const at::DataPtr &data);
