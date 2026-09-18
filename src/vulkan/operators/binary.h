#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace pytorch_vulkan {

enum class PointwiseOperation : uint32_t {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Neg = 3,
    Abs = 4,
    Relu = 5,
    AbsBackward = 6,
    ReluBackward = 7,
    Lerp = 12,
    Sqrt = 13,
    Div = 14,
    Sigmoid = 15,
    Tanh = 16,
    GeluTanh = 17,
    SigmoidBackward = 18,
    TanhBackward = 19,
    GeluTanhBackward = 20,
    Ceil = 10,
};

at::Tensor add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha);
at::Tensor div_tensor(const at::Tensor &lhs, const at::Tensor &rhs);

at::Tensor add_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                      const at::Scalar &alpha);

at::Tensor pointwise_tensor_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                                   PointwiseOperation operation, bool scalar_left,
                                   const char *operation_name);

at::Tensor pointwise_tensor_operands(const at::Tensor &lhs, const at::Tensor &rhs,
                                      const at::Scalar &alpha, PointwiseOperation operation,
                                      const char *operation_name);

at::Tensor &dispatch_unary_out(const at::Tensor &input, at::Tensor &out,
                               PointwiseOperation operation,
                               const char *operation_name);
at::Tensor &dispatch_tensor_tensor_out(const at::Tensor &lhs,
                                       const at::Tensor &rhs,
                                       const at::Scalar &alpha,
                                       at::Tensor &out,
                                       PointwiseOperation operation,
                                       const char *operation_name);
at::Tensor &dispatch_tensor_scalar_out(const at::Tensor &tensor,
                                        const at::Scalar &scalar,
                                        const at::Scalar &alpha,
                                        at::Tensor &out,
                                        PointwiseOperation operation,
                                        const char *operation_name,
                                        bool scalar_left = false);

} // namespace pytorch_vulkan
