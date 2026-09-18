#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor neg_tensor(const at::Tensor &input);
at::Tensor abs_tensor(const at::Tensor &input);
at::Tensor relu_tensor(const at::Tensor &input);
at::Tensor sigmoid_tensor(const at::Tensor &input);
at::Tensor tanh_tensor(const at::Tensor &input);
at::Tensor gelu_tensor(const at::Tensor &input, c10::string_view approximate);
at::Tensor ceil_tensor(const at::Tensor &input);
at::Tensor abs_backward_tensor(const at::Tensor &input, const at::Tensor &grad);
at::Tensor relu_backward_tensor(const at::Tensor &output, const at::Tensor &grad);
at::Tensor sigmoid_backward_tensor(const at::Tensor &output, const at::Tensor &grad);
at::Tensor tanh_backward_tensor(const at::Tensor &output, const at::Tensor &grad);
at::Tensor gelu_backward_tensor(const at::Tensor &input, const at::Tensor &grad,
                               c10::string_view approximate);
at::Tensor &sigmoid_backward_out(const at::Tensor &grad_output, const at::Tensor &output,
                                 at::Tensor &grad_input);
at::Tensor &tanh_backward_out(const at::Tensor &grad_output, const at::Tensor &output,
                              at::Tensor &grad_input);
at::Tensor &gelu_backward_out(const at::Tensor &grad_output, const at::Tensor &input,
                              c10::string_view approximate, at::Tensor &grad_input);

} // namespace pytorch_vulkan
