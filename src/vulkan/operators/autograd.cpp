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
#include <vector>

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
        const at::Tensor &raw_grad = grad_outputs[0];

        const auto input_2d = saved[0].dim() == 2
            ? saved[0]
            : saved[0].reshape({saved[0].numel() / saved[0].size(-1), saved[0].size(-1)});
        at::Tensor grad = raw_grad.dim() == 2
            ? raw_grad
            : raw_grad.reshape({input_2d.size(0), input_2d.size(1)});
        auto grad_input = linear_backward_input(grad, saved[1], input_2d);
        if (saved[0].dim() != 2)
            grad_input = grad_input.reshape(saved[0].sizes());
        return {grad_input,
                linear_backward_weight(grad, input_2d),
                ctx->saved_data["has_bias"].toBool() ? linear_backward_bias(grad)
                                                     : at::Tensor()};
    }
};

class MmAutogradFunction final : public torch::autograd::Function<MmAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &mat1, const at::Tensor &mat2) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({mat1, mat2});
        return pytorch_vulkan::mm(mat1, mat2);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined())
            return {at::Tensor(), at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        const at::Tensor &grad = grad_outputs[0];
        return {ctx->needs_input_grad(0)
                    ? pytorch_vulkan::mm(
                          grad, pytorch_vulkan::transpose_contiguous_2d(saved[1]))
                    : at::Tensor(),
                ctx->needs_input_grad(1)
                    ? pytorch_vulkan::mm(
                          pytorch_vulkan::transpose_contiguous_2d(saved[0]), grad)
                    : at::Tensor()};
    }
};

class BmmAutogradFunction final : public torch::autograd::Function<BmmAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &mat1, const at::Tensor &mat2) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({mat1, mat2});
        return pytorch_vulkan::bmm(mat1, mat2);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined())
            return {at::Tensor(), at::Tensor()};
        auto saved = ctx->get_saved_variables();
        auto grad = grad_outputs[0];
        auto grad_a = ctx->needs_input_grad(0)
            ? pytorch_vulkan::bmm(grad, saved[1].transpose(1, 2))
            : at::Tensor();
        auto grad_b = ctx->needs_input_grad(1)
            ? pytorch_vulkan::bmm(saved[0].transpose(1, 2), grad)
            : at::Tensor();
        return {grad_a, grad_b};
    }
};

class AddmmAutogradFunction final
    : public torch::autograd::Function<AddmmAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &self, const at::Tensor &mat1,
                              const at::Tensor &mat2, const at::Scalar &beta,
                              const at::Scalar &alpha) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({self, mat1, mat2});
        ctx->saved_data["beta"] = beta;
        ctx->saved_data["alpha"] = alpha;
        return pytorch_vulkan::addmm(self, mat1, mat2, beta, alpha);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        const at::Tensor &grad = grad_outputs[0];
        const at::Scalar beta = ctx->saved_data["beta"].toScalar();
        const at::Scalar alpha = ctx->saved_data["alpha"].toScalar();
        auto scaled = [](const at::Tensor &tensor, const at::Scalar &scalar) {
            return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Mul,
                                           false, "addmm backward scaling");
        };
        at::Tensor grad_mat1;
        at::Tensor grad_mat2;
        if (ctx->needs_input_grad(1)) {
            grad_mat1 =
                scaled(pytorch_vulkan::mm(
                           grad, pytorch_vulkan::transpose_contiguous_2d(saved[2])),
                       alpha);
        }
        if (ctx->needs_input_grad(2)) {
            grad_mat2 =
                scaled(pytorch_vulkan::mm(
                           pytorch_vulkan::transpose_contiguous_2d(saved[1]), grad),
                       alpha);
        }
        at::Tensor grad_self;
        if (ctx->needs_input_grad(0)) {
            if (grad.numel() == 0) {
                grad_self = at::zeros(saved[0].sizes(), grad.options());
            } else {
                grad_self = scaled(grad, beta);
                if (saved[0].dim() == 1)
                    grad_self = at::sum(grad_self, {0});
            }
        }
        return {grad_self, ctx->needs_input_grad(1) ? grad_mat1 : at::Tensor(),
                ctx->needs_input_grad(2) ? grad_mat2 : at::Tensor(), at::Tensor(),
                at::Tensor()};
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
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor()};
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
        return {std::get<0>(gradients), std::get<1>(gradients), std::get<2>(gradients)};
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

at::Tensor autograd_mm(const at::Tensor &mat1, const at::Tensor &mat2) {
    return MmAutogradFunction::apply(mat1, mat2);
}

at::Tensor autograd_bmm(const at::Tensor &mat1, const at::Tensor &mat2) {
    return BmmAutogradFunction::apply(mat1, mat2);
}

at::Tensor autograd_addmm(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, const at::Scalar &beta,
                          const at::Scalar &alpha) {
    return AddmmAutogradFunction::apply(self, mat1, mat2, beta, alpha);
}

namespace {
at::Tensor stack_raw(at::TensorList tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "Vulkan stack requires at least one tensor");
    TORCH_CHECK(dim == 0 || dim == 1, "Vulkan stack supports only dim=0 or dim=1");
    const auto &first = tensors[0];
    TORCH_CHECK(first.device().type() == c10::DeviceType::PrivateUse1 &&
                    first.device().index() == 0 && first.scalar_type() == at::kFloat &&
                    first.dim() == 2 && first.is_contiguous(),
                "Vulkan stack requires contiguous float32 Vulkan matrices");
    for (const auto &tensor : tensors) {
        TORCH_CHECK(tensor.device() == first.device() &&
                        tensor.scalar_type() == at::kFloat && tensor.dim() == 2 &&
                        tensor.sizes().equals(first.sizes()) && tensor.is_contiguous(),
                    "Vulkan stack requires matching contiguous matrices");
    }
    at::Tensor output = dim == 0
        ? at::empty({static_cast<int64_t>(tensors.size()), first.size(0), first.size(1)},
                    first.options())
        : at::empty({first.size(0), static_cast<int64_t>(tensors.size()), first.size(1)},
                    first.options());
    for (int64_t index = 0; index < static_cast<int64_t>(tensors.size()); ++index)
        output.select(dim, index).copy_(tensors[index]);
    return output;
}

class StackAutogradFunction final
    : public torch::autograd::Function<StackAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              at::TensorList tensors, int64_t dim) {
        at::AutoDispatchBelowAutograd guard;
        std::vector<at::Tensor> saved(tensors.begin(), tensors.end());
        ctx->save_for_backward(saved);
        ctx->saved_data["dim"] = dim;
        return stack_raw(tensors, dim);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        const int64_t dim = ctx->saved_data["dim"].toInt();
        torch::autograd::variable_list result;
        result.reserve(saved.size() + 1);
        for (int64_t index = 0; index < static_cast<int64_t>(saved.size()); ++index)
            result.push_back(grads[0].select(dim, index));
        result.push_back(at::Tensor());
        return result;
    }
};
} // namespace

at::Tensor autograd_stack(at::TensorList tensors, int64_t dim) {
    return StackAutogradFunction::apply(tensors, dim);
}

at::Tensor stack(at::TensorList tensors, int64_t dim) { return stack_raw(tensors, dim); }

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
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan sub supports only alpha == 1");
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

at::Tensor autograd_sigmoid(const at::Tensor &input) {
    return autograd_unary_saved_output<&sigmoid_tensor, &sigmoid_backward_tensor>(
        input);
}

at::Tensor autograd_tanh(const at::Tensor &input) {
    return autograd_unary_saved_output<&tanh_tensor, &tanh_backward_tensor>(input);
}

namespace {
class GeluAutogradFunction final
    : public torch::autograd::Function<GeluAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, c10::string_view approximate) {
        at::AutoDispatchBelowAutograd guard;
        TORCH_CHECK(approximate == "tanh",
                    "Vulkan gelu supports only approximate=\"tanh\"");
        ctx->save_for_backward({input});
        return gelu_tensor(input, approximate);
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        TORCH_CHECK(!c10::GradMode::is_enabled(),
                    "Vulkan gelu does not support higher-order gradients");
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor()};
        return {gelu_backward_tensor(ctx->get_saved_variables()[0], grads[0], "tanh"),
                at::Tensor()};
    }
};
} // namespace

at::Tensor autograd_gelu(const at::Tensor &input, c10::string_view approximate) {
    return GeluAutogradFunction::apply(input, approximate);
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("stack", &pytorch_vulkan::stack);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("stack", &pytorch_vulkan::autograd_stack);
}
