#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha);

} // namespace pytorch_vulkan
