#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor &ne_scalar_out(const at::Tensor &self, const at::Scalar &other,
                          at::Tensor &out);
at::Tensor &isfinite_out(const at::Tensor &self, at::Tensor &out);
at::Tensor &eq_tensor_out(const at::Tensor &self, const at::Tensor &other,
                          at::Tensor &out);
at::Tensor &bitwise_and_tensor_out(const at::Tensor &self, const at::Tensor &other,
                                   at::Tensor &out);
at::Tensor masked_select(const at::Tensor &self, const at::Tensor &mask);

} // namespace pytorch_vulkan
