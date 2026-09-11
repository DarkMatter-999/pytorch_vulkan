#pragma once

#include <c10/core/Allocator.h>

class VulkanAllocator final : public at::Allocator {
  public:
    at::DataPtr allocate(size_t nbytes) override;
    void copy_data(void *dest, const void *src, std::size_t count) const override;
};

VulkanAllocator *vulkan_allocator_instance();
