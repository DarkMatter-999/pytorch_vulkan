#include "autograd.h"

#include "binary.h"
#include "unary.h"

#include <torch/library.h>

namespace pytorch_vulkan {

at::Tensor autograd_neg(const at::Tensor &input) {
    return autograd_unary_no_save<&neg_tensor, &neg_backward>(input);
}

at::Tensor neg_backward(const at::Tensor &grad) {
    return neg_tensor(grad);
}

namespace {

at::Tensor add_tensor_raw(const at::Tensor &lhs, const at::Tensor &rhs,
                          const at::Scalar &alpha) {
    return add_tensor(lhs, rhs, alpha);
}

at::Tensor sub_tensor_raw(const at::Tensor &lhs, const at::Tensor &rhs,
                          const at::Scalar &alpha) {
    return pointwise_tensor_operands(lhs, rhs, alpha, PointwiseOperation::Sub, "sub");
}

at::Tensor mul_tensor_raw(const at::Tensor &lhs, const at::Tensor &rhs,
                          const at::Scalar &alpha) {
    return pointwise_tensor_operands(lhs, rhs, alpha, PointwiseOperation::Mul, "mul");
}

at::Tensor add_backward_raw(const at::Tensor &, const at::Tensor &, const at::Tensor &grad) {
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor sub_backward_raw(const at::Tensor &, const at::Tensor &, const at::Tensor &grad) {
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor neg_sub_backward_raw(const at::Tensor &, const at::Tensor &, const at::Tensor &grad) {
    return neg_tensor(grad);
}

at::Tensor mul_backward_raw(const at::Tensor &, const at::Tensor &other,
                            const at::Tensor &grad) {
    return pointwise_tensor_operands(grad, other, at::Scalar(1),
                                     PointwiseOperation::Mul, "mul");
}

at::Tensor add_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                          const at::Scalar &alpha) {
    return add_scalar(tensor, scalar, alpha);
}

at::Tensor sub_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                          const at::Scalar &alpha) {
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Sub, false, "sub");
}

at::Tensor rsub_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                           const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan rsub supports only alpha == 1");
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Sub, true, "rsub");
}

at::Tensor mul_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                          const at::Scalar &) {
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Mul, false, "mul");
}

at::Tensor scalar_add_backward(const at::Tensor &, const at::Scalar &scalar,
                               const at::Tensor &grad) {
    (void)scalar;
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor scalar_sub_backward(const at::Tensor &, const at::Scalar &scalar,
                               const at::Tensor &grad) {
    (void)scalar;
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor scalar_rsub_backward(const at::Tensor &, const at::Scalar &scalar,
                                const at::Tensor &grad) {
    (void)scalar;
    return neg_tensor(grad);
}

at::Tensor scalar_mul_backward(const at::Tensor &, const at::Scalar &scalar,
                               const at::Tensor &grad) {
    return pointwise_tensor_scalar(grad, scalar, PointwiseOperation::Mul, false, "mul backward");
}

} // namespace

at::Tensor autograd_add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                               const at::Scalar &alpha) {
    return autograd_binary_tensor<&add_tensor_raw, &add_backward_raw, &add_backward_raw>(lhs, rhs, alpha);
}

at::Tensor autograd_sub_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                               const at::Scalar &alpha) {
    return autograd_binary_tensor<&sub_tensor_raw, &sub_backward_raw, &neg_sub_backward_raw>(lhs, rhs, alpha);
}

at::Tensor autograd_mul_tensor(const at::Tensor &lhs, const at::Tensor &rhs) {
    return autograd_binary_tensor<&mul_tensor_raw, &mul_backward_raw, &mul_backward_raw, true>(
        lhs, rhs, at::Scalar(1));
}

at::Tensor autograd_add_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                               const at::Scalar &alpha) {
    return autograd_binary_scalar<&add_scalar_raw, &scalar_add_backward, false>(
        tensor, scalar, alpha);
}

at::Tensor autograd_sub_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                               const at::Scalar &alpha) {
    return autograd_binary_scalar<&sub_scalar_raw, &scalar_sub_backward, false>(
        tensor, scalar, alpha);
}

at::Tensor autograd_rsub_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                                const at::Scalar &alpha) {
    return autograd_binary_scalar<&rsub_scalar_raw, &scalar_rsub_backward, true>(
        tensor, scalar, alpha);
}

at::Tensor autograd_mul_scalar(const at::Tensor &tensor, const at::Scalar &scalar) {
    return autograd_binary_scalar<&mul_scalar_raw, &scalar_mul_backward, false>(
        tensor, scalar, at::Scalar(1));
}

at::Tensor autograd_abs(const at::Tensor &input) {
    return autograd_unary_saved_input<&abs_tensor, &abs_backward_tensor>(input);
}

at::Tensor autograd_relu(const at::Tensor &input) {
    return autograd_unary_saved_output<&relu_tensor, &relu_backward_tensor>(input);
}

} // namespace pytorch_vulkan
