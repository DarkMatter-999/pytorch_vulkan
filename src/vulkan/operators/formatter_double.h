#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor formatter_double_abs(const at::Tensor &input);
at::Tensor formatter_double_min(const at::Tensor &input);
at::Tensor formatter_double_max(const at::Tensor &input);
at::Tensor formatter_double_div(const at::Tensor &input, const at::Tensor &other);
at::Tensor formatter_double_ne(const at::Tensor &input, const at::Tensor &other);
at::Tensor formatter_double_gt(const at::Tensor &input, const at::Scalar &other);
at::Tensor formatter_double_lt(const at::Tensor &input, const at::Scalar &other);
at::Tensor formatter_double_ceil(const at::Tensor &input);
at::Scalar formatter_double_local_scalar_dense(const at::Tensor &input);

} // namespace pytorch_vulkan
