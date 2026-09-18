#pragma once

#include "binary.h"

namespace pytorch_vulkan {

at::Tensor &dispatch_tensor_tensor_out(const at::Tensor &lhs, const at::Tensor &rhs,
                                       const at::Scalar &alpha, at::Tensor &out,
                                       PointwiseOperation operation, const char *name);

at::Tensor &dispatch_tensor_tensor_alias(at::Tensor &self, const at::Tensor &other,
                                         const at::Scalar &alpha,
                                         PointwiseOperation operation,
                                         const char *name);
at::Tensor &dispatch_tensor_scalar_alias(at::Tensor &self, const at::Scalar &scalar,
                                         PointwiseOperation operation,
                                         const char *name);
at::Tensor &dispatch_fill(at::Tensor &self, const at::Scalar &value);
at::Tensor &dispatch_zero(at::Tensor &self);

} // namespace pytorch_vulkan
