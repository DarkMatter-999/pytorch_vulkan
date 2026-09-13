#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
at::Tensor convolution_overrideable(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &,
                                    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t);
}
