#include "binary.h"
#include "out.h"

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

at::Tensor &sub_out(const at::Tensor &lhs, const at::Tensor &rhs,
                    const at::Scalar &alpha, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_tensor_out(
        lhs, rhs, alpha, out, pytorch_vulkan::PointwiseOperation::Sub, "sub");
}

at::Tensor &sub_scalar_out(const at::Tensor &tensor, const at::Scalar &scalar,
                           const at::Scalar &alpha, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_scalar_out(
        tensor, scalar, alpha, out, pytorch_vulkan::PointwiseOperation::Sub, "sub");
}

at::Tensor &rsub_scalar_out(const at::Tensor &tensor, const at::Scalar &scalar,
                            const at::Scalar &alpha, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_scalar_out(
        tensor, scalar, alpha, out, pytorch_vulkan::PointwiseOperation::Sub, "rsub", true);
}

at::Tensor &reject_sub_inplace_tensor(at::Tensor &self, const at::Tensor &other,
                                      const at::Scalar &alpha) {
    TORCH_CHECK(false, "Vulkan sub in-place variants are unsupported");
    return self;
}

at::Tensor &reject_sub_inplace_scalar(at::Tensor &self, const at::Scalar &other,
                                      const at::Scalar &alpha) {
    TORCH_CHECK(false, "Vulkan sub in-place variants are unsupported");
    return self;
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
    m.impl("sub.out", &sub_out);
    m.impl("sub.Scalar_out", &sub_scalar_out);
    m.impl("rsub.Scalar_out", &rsub_scalar_out);
    m.impl("sub_.Tensor", &reject_sub_inplace_tensor);
    m.impl("sub_.Scalar", &reject_sub_inplace_scalar);
}
