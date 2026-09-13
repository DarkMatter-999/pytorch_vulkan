#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                                                           at::IntArrayRef, at::IntArrayRef, bool);
}
