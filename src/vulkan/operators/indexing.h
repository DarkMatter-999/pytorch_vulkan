#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
at::Tensor argmax_tensor(const at::Tensor &, c10::optional<int64_t>, bool);
at::Tensor &argmax_out(const at::Tensor &, c10::optional<int64_t>, bool, at::Tensor &);
} // namespace pytorch_vulkan
