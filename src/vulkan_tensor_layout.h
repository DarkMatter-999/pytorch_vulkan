#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace pytorch_vulkan {

enum class VulkanOverlap : uint8_t { No, Yes, TooHard };

struct VulkanTensorLayout {
    int64_t rank;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int scalar_type;
    uint64_t element_bytes;
    int64_t storage_offset;
    int64_t numel;
    VkDeviceSize byte_offset;
    VkDeviceSize byte_range;
    VkDeviceSize allocation_bytes;
    VulkanOverlap internal_overlap;
};

} // namespace pytorch_vulkan
