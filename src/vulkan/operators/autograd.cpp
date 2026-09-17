#include "autograd.h"
#include "formatter_double.h"

#include "binary.h"
#include "convolution.h"
#include "linear.h"
#include "pooling.h"
#include "unary.h"

#include <torch/library.h>

#include <cstdlib>
#include <cstring>

namespace pytorch_vulkan {

namespace {
bool disable_multi_output_backward() {
    const char *value = std::getenv("PYTORCH_VULKAN_DISABLE_MULTI_OUTPUT_BACKWARD");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool supports_multi_output_backward(const at::Tensor &tensor) {
    return tensor.scalar_type() == at::kFloat && tensor.dim() == 2 &&
           tensor.is_contiguous();
}

class ConvolutionAutogradFunction final
    : public torch::autograd::Function<ConvolutionAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, const at::Tensor &weight,
                              const c10::optional<at::Tensor> &bias,
                              at::IntArrayRef stride, at::IntArrayRef padding,
                              at::IntArrayRef dilation, bool transposed,
                              at::IntArrayRef output_padding, int64_t groups) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({input, weight});
        ctx->saved_data["has_bias"] = bias.has_value();
        return pytorch_vulkan::convolution(input, weight, bias, stride, padding,
                                           dilation, transposed, output_padding,
                                           groups);
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()};
        auto saved = ctx->get_saved_variables();
        return {convolution_backward_input(grads[0], saved[1]),
                convolution_backward_weight(grads[0], saved[0]),
                ctx->saved_data["has_bias"].toBool()
                    ? convolution_backward_bias(grads[0])
                    : at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor()};
    }
};
class LinearAutogradFunction final
    : public torch::autograd::Function<LinearAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, const at::Tensor &weight,
                              const c10::optional<at::Tensor> &bias) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({input, weight});
        ctx->saved_data["has_bias"] = bias.has_value();
        return pytorch_vulkan::linear(input, weight, bias);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        const at::Tensor &grad = grad_outputs[0];
        return {linear_backward_input(grad, saved[1], saved[0]),
                linear_backward_weight(grad, saved[0]),
                ctx->saved_data["has_bias"].toBool() ? linear_backward_bias(grad)
                                                     : at::Tensor()};
    }
};
class LinearReluAutogradFunction final
    : public torch::autograd::Function<LinearReluAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, const at::Tensor &weight,
                              const at::Tensor &bias) {
        at::AutoDispatchBelowAutograd guard;
        auto output = pytorch_vulkan::linear_relu(input, weight, bias);
        ctx->save_for_backward({input, weight, output});
        return output;
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined()) return {at::Tensor(), at::Tensor(), at::Tensor()};
        auto saved = ctx->get_saved_variables();
        if (disable_multi_output_backward() ||
            !supports_multi_output_backward(grads[0]) ||
            !supports_multi_output_backward(saved[0]) ||
            !supports_multi_output_backward(saved[1]) ||
            !supports_multi_output_backward(saved[2]))
            return {linear_relu_backward_input(grads[0], saved[1], saved[2]),
                    linear_relu_backward_weight(grads[0], saved[0], saved[2]),
                    linear_relu_backward_bias(grads[0], saved[2])};
        auto gradients = linear_relu_backward(grads[0], saved[0], saved[1], saved[2]);
        return {std::get<0>(gradients), std::get<1>(gradients),
                std::get<2>(gradients)};
    }
};
class AdaptiveAvgPoolAutogradFunction final
    : public torch::autograd::Function<AdaptiveAvgPoolAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, at::IntArrayRef output_size) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({input});
        return pytorch_vulkan::adaptive_avg_pool2d(input, output_size);
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor()};
        return {pytorch_vulkan::adaptive_avg_pool2d_backward(
                    grads[0], ctx->get_saved_variables()[0]),
                at::Tensor()};
    }
};
} // namespace

at::Tensor autograd_linear(const at::Tensor &input, const at::Tensor &weight,
                           const c10::optional<at::Tensor> &bias) {
    return LinearAutogradFunction::apply(input, weight, bias);
}

at::Tensor autograd_linear_relu(const at::Tensor &input, const at::Tensor &weight,
                                const at::Tensor &bias) {
    return LinearReluAutogradFunction::apply(input, weight, bias);
}

at::Tensor autograd_convolution(const at::Tensor &input, const at::Tensor &weight,
                                const c10::optional<at::Tensor> &bias,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool transposed,
                                at::IntArrayRef output_padding, int64_t groups) {
    return ConvolutionAutogradFunction::apply(input, weight, bias, stride, padding,
                                              dilation, transposed, output_padding,
                                              groups);
}

at::Tensor autograd_adaptive_avg_pool2d(const at::Tensor &input,
                                        at::IntArrayRef output_size) {
    return AdaptiveAvgPoolAutogradFunction::apply(input, output_size);
}

at::Tensor autograd_neg(const at::Tensor &input) {
    return autograd_unary_no_save<&neg_tensor, &neg_backward>(input);
}

at::Tensor neg_backward(const at::Tensor &grad) { return neg_tensor(grad); }

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

at::Tensor add_backward_raw(const at::Tensor &, const at::Tensor &,
                            const at::Tensor &grad) {
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor sub_backward_raw(const at::Tensor &, const at::Tensor &,
                            const at::Tensor &grad) {
    return add_scalar(grad, at::Scalar(0), at::Scalar(1));
}

at::Tensor neg_sub_backward_raw(const at::Tensor &, const at::Tensor &,
                                const at::Tensor &grad) {
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
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Sub, false,
                                   "sub");
}

at::Tensor rsub_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                           const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan rsub supports only alpha == 1");
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Sub, true,
                                   "rsub");
}

at::Tensor mul_scalar_raw(const at::Tensor &tensor, const at::Scalar &scalar,
                          const at::Scalar &) {
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Mul, false,
                                   "mul");
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
    return pointwise_tensor_scalar(grad, scalar, PointwiseOperation::Mul, false,
                                   "mul backward");
}

} // namespace

at::Tensor autograd_add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                               const at::Scalar &alpha) {
    return autograd_binary_tensor<&add_tensor_raw, &add_backward_raw,
                                  &add_backward_raw>(lhs, rhs, alpha);
}

at::Tensor autograd_sub_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                               const at::Scalar &alpha) {
    return autograd_binary_tensor<&sub_tensor_raw, &sub_backward_raw,
                                  &neg_sub_backward_raw>(lhs, rhs, alpha);
}

at::Tensor autograd_mul_tensor(const at::Tensor &lhs, const at::Tensor &rhs) {
    return autograd_binary_tensor<&mul_tensor_raw, &mul_backward_raw, &mul_backward_raw,
                                  true>(lhs, rhs, at::Scalar(1));
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
    if (input.scalar_type() == at::kDouble)
        return pytorch_vulkan::formatter_double_abs(input);
    return autograd_unary_saved_input<&abs_tensor, &abs_backward_tensor>(input);
}

at::Tensor autograd_relu(const at::Tensor &input) {
    return autograd_unary_saved_output<&relu_tensor, &relu_backward_tensor>(input);
}

} // namespace pytorch_vulkan
