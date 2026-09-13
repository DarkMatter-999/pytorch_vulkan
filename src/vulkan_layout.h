#pragma once

#include <ATen/core/TensorBody.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace pytorch_vulkan {

enum class VulkanOverlap : uint8_t {
    No,
    Yes,
    TooHard,
};

struct VulkanTensorLayout {
    int64_t rank;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    at::ScalarType scalar_type;
    uint64_t element_bytes;
    int64_t storage_offset;
    int64_t numel;
    VkDeviceSize byte_offset;
    VkDeviceSize byte_range;
    VkDeviceSize allocation_bytes;
    VulkanOverlap internal_overlap;
};

VulkanTensorLayout inspect_vulkan_tensor_layout(const at::Tensor &tensor,
                                                 const char *label);
VulkanTensorLayout inspect_vulkan_view_layout(const at::Tensor &storage_owner,
                                               at::IntArrayRef sizes,
                                                at::IntArrayRef strides,
                                                int64_t storage_offset,
                                                const char *label);

int64_t vulkan_storage_offset(const VulkanTensorLayout &layout,
                              at::IntArrayRef coordinate);
int64_t vulkan_storage_offset(const VulkanTensorLayout &layout, int64_t linear_index);

} // namespace pytorch_vulkan
