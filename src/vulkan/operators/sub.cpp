#include "add.h"

#include <torch/library.h>

namespace {
at::Tensor sub_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                      const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan sub supports only alpha == 1");
    return pytorch_vulkan::pointwise_tensor_scalar(
        tensor, scalar, pytorch_vulkan::PointwiseOperation::Sub, false, "sub");
}

at::Tensor sub_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha) {
    return pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, alpha, pytorch_vulkan::PointwiseOperation::Sub, "sub");
}

at::Tensor rsub_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                       const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan rsub supports only alpha == 1");
    return pytorch_vulkan::pointwise_tensor_scalar(
        tensor, scalar, pytorch_vulkan::PointwiseOperation::Sub, true, "sub");
}
} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sub.Scalar", &sub_scalar);
    m.impl("rsub.Scalar", &rsub_scalar);
    m.impl("sub.Tensor", &sub_tensor);
}
