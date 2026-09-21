#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace pytorch_vulkan {
at::Tensor mm(const at::Tensor &mat1, const at::Tensor &mat2);
at::Tensor bmm(const at::Tensor &mat1, const at::Tensor &mat2);
at::Tensor addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
                 const at::Scalar &beta, const at::Scalar &alpha);
at::Tensor transpose_contiguous_2d(const at::Tensor &input);
at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias);
at::Tensor linear_relu(const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor linear_relu_backward_input(const at::Tensor &, const at::Tensor &,
                                      const at::Tensor &);
at::Tensor linear_relu_backward_weight(const at::Tensor &, const at::Tensor &,
                                       const at::Tensor &);
at::Tensor linear_relu_backward_bias(const at::Tensor &, const at::Tensor &);
std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_relu_backward(const at::Tensor &,
                                                                    const at::Tensor &,
                                                                    const at::Tensor &,
                                                                    const at::Tensor &);
at::Tensor linear_backward_input(const at::Tensor &, const at::Tensor &,
                                 const at::Tensor &);
at::Tensor linear_backward_weight(const at::Tensor &, const at::Tensor &);
at::Tensor linear_backward_bias(const at::Tensor &);
} // namespace pytorch_vulkan
