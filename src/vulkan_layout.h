#pragma once

#include <ATen/core/TensorBody.h>
#include <vulkan/vulkan.h>

namespace pytorch_vulkan {

struct VulkanTensorLayout {
    int64_t storage_offset;
    int64_t numel;
    VkDeviceSize byte_offset;
    VkDeviceSize byte_range;
    VkDeviceSize allocation_bytes;
};

VulkanTensorLayout inspect_vulkan_tensor_layout(const at::Tensor &tensor,
                                                 const char *label);
VulkanTensorLayout inspect_vulkan_view_layout(const at::Tensor &storage_owner,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides,
                                               int64_t storage_offset,
                                               const char *label);

} // namespace pytorch_vulkan
