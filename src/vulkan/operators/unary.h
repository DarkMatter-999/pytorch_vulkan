#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor neg_tensor(const at::Tensor &input);
at::Tensor abs_tensor(const at::Tensor &input);
at::Tensor relu_tensor(const at::Tensor &input);

} // namespace pytorch_vulkan
