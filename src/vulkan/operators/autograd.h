#pragma once

#include <ATen/ATen.h>
#include <torch/autograd.h>

namespace pytorch_vulkan {

using RawUnaryForward = at::Tensor (*)(const at::Tensor &);
using RawUnaryBackward = at::Tensor (*)(const at::Tensor &, const at::Tensor &);
using RawUnaryNoSaveBackward = at::Tensor (*)(const at::Tensor &);
using RawBinaryTensorForward = at::Tensor (*)(const at::Tensor &, const at::Tensor &,
                                              const at::Scalar &);
using RawBinaryTensorBackward = at::Tensor (*)(const at::Tensor &, const at::Tensor &,
                                               const at::Tensor &);
using RawBinaryScalarForward = at::Tensor (*)(const at::Tensor &, const at::Scalar &,
                                              const at::Scalar &);
using RawBinaryScalarBackward = at::Tensor (*)(const at::Tensor &, const at::Scalar &,
                                               const at::Tensor &);

template <RawUnaryForward Forward, RawUnaryBackward Backward>
class UnaryAutogradFunctionSaved
    : public torch::autograd::Function<UnaryAutogradFunctionSaved<Forward, Backward>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input) {
        at::AutoDispatchBelowAutograd guard;
        at::Tensor output = Forward(input);
        ctx->save_for_backward({input});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined()) {
            return {at::Tensor()};
        }
        return {Backward(ctx->get_saved_variables()[0], grad_outputs[0])};
    }
};

template <RawUnaryForward Forward, RawUnaryBackward Backward>
at::Tensor autograd_unary_saved_input(const at::Tensor &input) {
    return UnaryAutogradFunctionSaved<Forward, Backward>::apply(input);
}

template <RawUnaryForward Forward, RawUnaryBackward Backward>
class UnaryAutogradFunctionSavedOutput
    : public torch::autograd::Function<UnaryAutogradFunctionSavedOutput<Forward, Backward>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input) {
        at::AutoDispatchBelowAutograd guard;
        at::Tensor output = Forward(input);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined()) {
            return {at::Tensor()};
        }
        return {Backward(ctx->get_saved_variables()[0], grad_outputs[0])};
    }
};

template <RawUnaryForward Forward, RawUnaryBackward Backward>
at::Tensor autograd_unary_saved_output(const at::Tensor &input) {
    return UnaryAutogradFunctionSavedOutput<Forward, Backward>::apply(input);
}

template <RawUnaryForward Forward, RawUnaryNoSaveBackward Backward>
class UnaryAutogradFunctionNoSave
    : public torch::autograd::Function<UnaryAutogradFunctionNoSave<Forward, Backward>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input) {
        (void)ctx;
        at::AutoDispatchBelowAutograd guard;
        return Forward(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        (void)ctx;
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined()) {
            return {at::Tensor()};
        }
        return {Backward(grad_outputs[0])};
    }
};

template <RawUnaryForward Forward, RawUnaryNoSaveBackward Backward>
at::Tensor autograd_unary_no_save(const at::Tensor &input) {
    return UnaryAutogradFunctionNoSave<Forward, Backward>::apply(input);
}

at::Tensor autograd_neg(const at::Tensor &input);
at::Tensor autograd_abs(const at::Tensor &input);
at::Tensor autograd_relu(const at::Tensor &input);
at::Tensor neg_backward(const at::Tensor &grad);

template <RawBinaryTensorForward Forward, RawBinaryTensorBackward BackwardLhs,
          RawBinaryTensorBackward BackwardRhs,
          bool SaveOperands>
class BinaryTensorAutogradFunction
    : public torch::autograd::Function<
          BinaryTensorAutogradFunction<Forward, BackwardLhs, BackwardRhs, SaveOperands>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &lhs, const at::Tensor &rhs,
                              const at::Scalar &alpha) {
        at::AutoDispatchBelowAutograd guard;
        at::Tensor output = Forward(lhs, rhs, alpha);
        if constexpr (SaveOperands) {
            ctx->save_for_backward({lhs, rhs});
        }
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined()) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }
        at::Tensor lhs;
        at::Tensor rhs;
        if constexpr (SaveOperands) {
            auto saved = ctx->get_saved_variables();
            lhs = saved[0];
            rhs = saved[1];
        }
        return {BackwardLhs(lhs, rhs, grad_outputs[0]),
                BackwardRhs(rhs, lhs, grad_outputs[0]), at::Tensor()};
    }
};

template <RawBinaryTensorForward Forward, RawBinaryTensorBackward BackwardLhs,
          RawBinaryTensorBackward BackwardRhs,
          bool SaveOperands = false>
at::Tensor autograd_binary_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                                  const at::Scalar &alpha) {
    return BinaryTensorAutogradFunction<Forward, BackwardLhs, BackwardRhs, SaveOperands>::apply(
        lhs, rhs, alpha);
}

template <RawBinaryScalarForward Forward, RawBinaryScalarBackward Backward,
          bool ScalarLeft>
class BinaryScalarAutogradFunction
    : public torch::autograd::Function<
          BinaryScalarAutogradFunction<Forward, Backward, ScalarLeft>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &tensor, const at::Scalar &scalar,
                              const at::Scalar &alpha) {
        (void)ctx;
        at::AutoDispatchBelowAutograd guard;
        ctx->saved_data["scalar"] = scalar;
        return Forward(tensor, scalar, alpha);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        (void)ctx;
        at::AutoDispatchBelowAutograd guard;
        if (!grad_outputs[0].defined()) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }
        const at::Scalar scalar = ctx->saved_data["scalar"].toScalar();
        return {Backward(at::Tensor(), scalar, grad_outputs[0]),
                at::Tensor(), at::Tensor()};
    }
};

template <RawBinaryScalarForward Forward, RawBinaryScalarBackward Backward,
          bool ScalarLeft>
at::Tensor autograd_binary_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                                  const at::Scalar &alpha) {
    return BinaryScalarAutogradFunction<Forward, Backward, ScalarLeft>::apply(
        tensor, scalar, alpha);
}

at::Tensor autograd_add_tensor(const at::Tensor &, const at::Tensor &, const at::Scalar &);
at::Tensor autograd_sub_tensor(const at::Tensor &, const at::Tensor &, const at::Scalar &);
at::Tensor autograd_mul_tensor(const at::Tensor &, const at::Tensor &);
at::Tensor autograd_add_scalar(const at::Tensor &, const at::Scalar &, const at::Scalar &);
at::Tensor autograd_sub_scalar(const at::Tensor &, const at::Scalar &, const at::Scalar &);
at::Tensor autograd_rsub_scalar(const at::Tensor &, const at::Scalar &, const at::Scalar &);
at::Tensor autograd_mul_scalar(const at::Tensor &, const at::Scalar &);

} // namespace pytorch_vulkan
