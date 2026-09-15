#include "binary.h"
#include "autograd.h"
#include "out.h"

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

at::Tensor &mul_out(const at::Tensor &lhs, const at::Tensor &rhs, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_tensor_out(
        lhs, rhs, at::Scalar(1), out, pytorch_vulkan::PointwiseOperation::Mul, "mul");
}

at::Tensor &mul_scalar_out(const at::Tensor &tensor, const at::Scalar &scalar,
                           at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_scalar_out(
        tensor, scalar, at::Scalar(1), out, pytorch_vulkan::PointwiseOperation::Mul, "mul");
}

at::Tensor &mul_inplace_tensor(at::Tensor &self, const at::Tensor &other) {
    return pytorch_vulkan::dispatch_tensor_tensor_alias(
        self, other, at::Scalar(1), pytorch_vulkan::PointwiseOperation::Mul, "mul_");
}

at::Tensor &mul_inplace_scalar(at::Tensor &self, const at::Scalar &other) {
    return pytorch_vulkan::dispatch_tensor_scalar_alias(
        self, other, pytorch_vulkan::PointwiseOperation::Mul, "mul_");
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mul.Scalar", &mul_scalar);
    m.impl("mul.Tensor", &mul_tensor);
    m.impl("mul.out", &mul_out);
    m.impl("mul.Scalar_out", &mul_scalar_out);
    m.impl("mul_.Tensor", &mul_inplace_tensor);
    m.impl("mul_.Scalar", &mul_inplace_scalar);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("mul.Tensor", &pytorch_vulkan::autograd_mul_tensor);
    m.impl("mul.Scalar", &pytorch_vulkan::autograd_mul_scalar);
}
