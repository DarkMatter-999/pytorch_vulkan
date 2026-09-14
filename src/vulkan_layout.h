#pragma once

#include <ATen/core/TensorBody.h>
#include "vulkan_tensor_layout.h"

namespace pytorch_vulkan {

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
