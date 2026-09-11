#pragma once

#include <c10/core/Allocator.h>

class VulkanBuffer;
class VulkanPlatform;

namespace at {
class Tensor;
}

namespace pytorch_vulkan {

// These accessors borrow objects owned by the DataPtr.  The DataPtr must remain
// alive for the duration of every use of either returned object.
VulkanBuffer &allocation_buffer(const at::DataPtr &data);
const VulkanPlatform &allocation_platform(const at::DataPtr &data);

} // namespace pytorch_vulkan

class VulkanAllocator final : public at::Allocator {
  public:
    at::DataPtr allocate(size_t nbytes) override;
    void copy_data(void *dest, const void *src, std::size_t count) const override;
};

VulkanAllocator *vulkan_allocator_instance();
