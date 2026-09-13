#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
at::Tensor adaptive_avg_pool2d(const at::Tensor &, at::IntArrayRef);
at::Tensor adaptive_avg_pool2d_backward(const at::Tensor &, const at::Tensor &);
std::tuple<at::Tensor, at::Tensor>
max_pool2d_with_indices(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                        at::IntArrayRef, at::IntArrayRef, bool);
} // namespace pytorch_vulkan
