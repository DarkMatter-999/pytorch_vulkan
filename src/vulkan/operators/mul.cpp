#include "add.h"

#include <torch/library.h>

namespace {
at::Tensor mul_scalar(const at::Tensor &tensor, const at::Scalar &scalar) {
    return pytorch_vulkan::pointwise_tensor_scalar(
        tensor, scalar, pytorch_vulkan::PointwiseOperation::Mul, false, "mul");
}

at::Tensor mul_tensor(const at::Tensor &lhs, const at::Tensor &rhs) {
    return pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, at::Scalar(1), pytorch_vulkan::PointwiseOperation::Mul, "mul");
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mul.Scalar", &mul_scalar);
    m.impl("mul.Tensor", &mul_tensor);
}
