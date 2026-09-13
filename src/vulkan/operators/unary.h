#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor neg_tensor(const at::Tensor &input);
at::Tensor abs_tensor(const at::Tensor &input);
at::Tensor relu_tensor(const at::Tensor &input);
at::Tensor ceil_tensor(const at::Tensor &input);
at::Tensor abs_backward_tensor(const at::Tensor &input, const at::Tensor &grad);
at::Tensor relu_backward_tensor(const at::Tensor &output, const at::Tensor &grad);

} // namespace pytorch_vulkan
