#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace pytorch_vulkan {

enum class PointwiseOperation : uint32_t {
    Add = 0,
    Sub = 1,
    Mul = 2,
};

at::Tensor add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha);

at::Tensor add_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                      const at::Scalar &alpha);

at::Tensor pointwise_tensor_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                                   PointwiseOperation operation, bool scalar_left,
                                   const char *operation_name);

at::Tensor pointwise_tensor_operands(const at::Tensor &lhs, const at::Tensor &rhs,
                                      const at::Scalar &alpha, PointwiseOperation operation,
                                      const char *operation_name);


} // namespace pytorch_vulkan
