#pragma once

#include <ATen/core/TensorBody.h>

#include <optional>

namespace pytorch_vulkan {

at::Tensor as_strided_tensor(const at::Tensor &self, at::IntArrayRef size,
                             at::IntArrayRef stride,
                             std::optional<int64_t> storage_offset);
at::Tensor view_tensor(const at::Tensor &self, at::IntArrayRef size);
at::Tensor reshape_alias_tensor(const at::Tensor &self, at::IntArrayRef size,
                                at::IntArrayRef stride);

} // namespace pytorch_vulkan
