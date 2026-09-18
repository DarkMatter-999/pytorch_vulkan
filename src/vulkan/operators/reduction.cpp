#include "reduction.h"
#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace pytorch_vulkan {
at::Tensor softmax_tensor(const at::Tensor &, int64_t, bool);
at::Tensor log_softmax_tensor(const at::Tensor &, int64_t, bool);
} // namespace pytorch_vulkan

namespace {
uint64_t checked_product(uint64_t lhs, uint64_t rhs, const char *name) {
    TORCH_CHECK(rhs == 0 || lhs <= std::numeric_limits<uint64_t>::max() / rhs,
                "Vulkan ", name, " size arithmetic overflow");
    return lhs * rhs;
}

void validate_input(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.device().index() == 0,
                "Vulkan ", name, " requires Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided, "Vulkan ", name,
                " requires a strided tensor");
}

std::vector<int64_t> normalized_dims(const at::Tensor &, c10::OptionalArrayRef<int64_t>,
                                     const char *);
at::Tensor dispatch(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool, uint32_t,
                    const char *, at::Tensor * = nullptr);

template <bool Mean>
class ReductionAutograd final
    : public torch::autograd::Function<ReductionAutograd<Mean>> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input,
                              c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                              c10::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto normalized = normalized_dims(input, dims, Mean ? "mean" : "sum");
        uint64_t mask = 0, count = 1;
        for (auto dim : normalized) {
            mask |= 1ULL << dim;
            count = checked_product(count, static_cast<uint64_t>(input.size(dim)),
                                    "reduction");
        }
        ctx->save_for_backward({input});
        ctx->saved_data["mask"] = static_cast<int64_t>(mask);
        ctx->saved_data["count"] = static_cast<int64_t>(count);
        ctx->saved_data["keepdim"] = keepdim;
        return dispatch(input, dims, keepdim, Mean ? 1u : 0u, Mean ? "mean" : "sum");
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        auto input = ctx->get_saved_variables()[0];
        pytorch_vulkan::validate_operator_rank(input.dim(),
                                               Mean ? "mean backward" : "sum backward");
        auto mask = static_cast<uint64_t>(ctx->saved_data["mask"].toInt());
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        std::vector<int64_t> compact_shape, full_shape;
        for (int64_t d = 0; d < input.dim(); ++d) {
            const bool reduced = mask & (1ULL << d);
            full_shape.push_back(reduced ? 1 : input.size(d));
            if (keepdim || !reduced)
                compact_shape.push_back(reduced ? 1 : input.size(d));
        }
        auto expanded = grads[0].reshape(compact_shape).reshape(full_shape);
        auto result = at::empty(input.sizes(), input.options());
        const auto expanded_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(
            expanded, "reduction gradient");
        const auto result_layout =
            pytorch_vulkan::inspect_vulkan_tensor_layout(result, "reduction gradient");
        TORCH_CHECK(expanded_layout.internal_overlap ==
                        pytorch_vulkan::VulkanOverlap::No,
                    "Vulkan reduction gradient rejects overlapping input views");
        const auto &source_data = expanded.storage().data_ptr();
        const auto &result_data = result.storage().data_ptr();
        const auto &platform = pytorch_vulkan::allocation_platform(source_data);
        pytorch_vulkan::validate_allocation(
            source_data, expanded_layout.allocation_bytes, "reduction gradient");
        pytorch_vulkan::validate_allocation(result_data, result_layout.allocation_bytes,
                                            "reduction gradient");
        platform.compute().broadcast(
            pytorch_vulkan::allocation_buffer(source_data).buffer(), expanded_layout,
            pytorch_vulkan::allocation_buffer(result_data).buffer(), result_layout,
            static_cast<uint32_t>(input.numel()),
            Mean ? static_cast<float>(
                       1.0 / static_cast<double>(ctx->saved_data["count"].toInt()))
                 : 1.0F);
        return {result, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

std::vector<int64_t> normalized_dims(const at::Tensor &input,
                                     c10::OptionalArrayRef<int64_t> dims,
                                     const char *name) {
    std::vector<int64_t> result;
    if (!dims || dims->empty()) {
        for (int64_t d = 0; d < input.dim(); ++d)
            result.push_back(d);
        return result;
    }
    for (int64_t dim : *dims) {
        if (dim < 0)
            dim += input.dim();
        TORCH_CHECK(dim >= 0 && dim < input.dim(), "Vulkan ", name,
                    " dimension out of range");
        TORCH_CHECK(std::find(result.begin(), result.end(), dim) == result.end(),
                    "Vulkan ", name, " received duplicate dimensions");
        result.push_back(dim);
    }
    return result;
}

std::vector<int64_t> output_sizes(const at::Tensor &input,
                                  const std::vector<int64_t> &dims, bool keepdim) {
    std::vector<bool> reduced(input.dim(), false);
    for (auto dim : dims)
        reduced[dim] = true;
    std::vector<int64_t> sizes;
    for (int64_t d = 0; d < input.dim(); ++d)
        if (keepdim || !reduced[d])
            sizes.push_back(reduced[d] ? 1 : input.size(d));
    return sizes;
}

at::Tensor dispatch(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                    bool keepdim, uint32_t operation, const char *name,
                    at::Tensor *provided_output) {
    validate_input(input, name);
    TORCH_CHECK(input.dim() > 0, "Vulkan ", name,
                " does not support zero-dimensional inputs");
    pytorch_vulkan::validate_operator_rank(input.dim(), name);
    pytorch_vulkan::validate_reduction_dtype(input.scalar_type(), operation == 1u,
                                             name);
    auto reduced_dims = normalized_dims(input, dims, name);
    TORCH_CHECK(!reduced_dims.empty(), "Vulkan ", name,
                " requires at least one dimension");
    std::vector<uint32_t> sizes(input.dim());
    uint64_t reduce_numel = 1, output_numel = 1;
    std::vector<bool> reduced(input.dim(), false);
    for (auto dim : reduced_dims)
        reduced[dim] = true;
    for (int64_t d = 0; d < input.dim(); ++d) {
        TORCH_CHECK(input.size(d) <= std::numeric_limits<uint32_t>::max(), "Vulkan ",
                    name, " dimension is too large");
        sizes[d] = static_cast<uint32_t>(input.size(d));
        if (reduced[d])
            reduce_numel = checked_product(reduce_numel, sizes[d], name);
    }
    for (auto size : output_sizes(input, reduced_dims, keepdim))
        output_numel = checked_product(output_numel, static_cast<uint64_t>(size), name);
    TORCH_CHECK(reduce_numel <= std::numeric_limits<uint32_t>::max() &&
                    output_numel <= std::numeric_limits<uint32_t>::max(),
                "Vulkan ", name, " exceeds dispatch limits");
    auto expected_sizes = output_sizes(input, reduced_dims, keepdim);
    at::Tensor output =
        provided_output ? *provided_output : at::empty(expected_sizes, input.options());
    if (provided_output) {
        TORCH_CHECK(
            output.device() == input.device() && output.scalar_type() == at::kFloat &&
                output.layout() == at::kStrided && output.is_contiguous() &&
                output.sizes().equals(expected_sizes),
            "Vulkan ", name,
            " out requires a contiguous float32 output with the expected shape");
    }
    auto input_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(input, name);
    if (output_numel == 0)
        return output;
    if (reduce_numel == 0) {
        // Reduction over an empty dimension has a defined result even though
        // the empty Vulkan input has no backing buffer to dispatch against.
        // Initialize the already allocated Vulkan output directly.
        auto output_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(output, name);
        const auto &output_data = output.storage().data_ptr();
        pytorch_vulkan::validate_allocation(output_data, output_layout.allocation_bytes,
                                            name);
        TORCH_CHECK(operation == 0u || operation == 1u || operation == 4u, "Vulkan ",
                    name, " cannot reduce an empty dimension");
        const uint32_t pattern = operation == 1u   ? 0x7fc00000U
                                 : operation == 4u ? 0x3f800000U
                                                   : 0U;
        pytorch_vulkan::allocation_platform(output_data)
            .fill_buffer_sync(pytorch_vulkan::allocation_buffer(output_data).buffer(),
                              output_layout.byte_offset, output_layout.byte_range,
                              pattern);
        return output;
    }
    TORCH_CHECK(input_layout.internal_overlap == pytorch_vulkan::VulkanOverlap::No,
                "Vulkan ", name, " rejects overlapping input views");
    auto output_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(output, name);
    TORCH_CHECK(output_layout.internal_overlap == pytorch_vulkan::VulkanOverlap::No,
                "Vulkan ", name, " rejects overlapping output views");
    const auto &in_data = input.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    input_layout.allocation_bytes = pytorch_vulkan::allocation_buffer(in_data).size();
    output_layout.allocation_bytes = pytorch_vulkan::allocation_buffer(out_data).size();
    pytorch_vulkan::validate_allocation(in_data, input_layout.allocation_bytes,
                                        "input");
    pytorch_vulkan::validate_allocation(out_data, output_layout.allocation_bytes,
                                        "output");
    const auto &platform = pytorch_vulkan::allocation_platform(in_data);
    TORCH_CHECK(&platform == &pytorch_vulkan::allocation_platform(out_data), "Vulkan ",
                name, " requires one Vulkan platform");
    uint32_t mask = 0;
    for (auto dim : reduced_dims)
        mask |= 1u << dim;
    platform.compute().reduction(
        pytorch_vulkan::allocation_buffer(in_data).buffer(), input_layout,
        pytorch_vulkan::allocation_buffer(out_data).buffer(), output_layout, mask,
        static_cast<uint32_t>(reduce_numel), static_cast<uint32_t>(output_numel),
        operation);
    return output;
}

at::Tensor reduction_backward_dispatch(const at::Tensor &input,
                                       const at::Tensor &forward,
                                       const at::Tensor &grad_output, int64_t dim,
                                       uint32_t operation, bool keepdim) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && input.is_contiguous() &&
                    forward.is_contiguous() && grad_output.is_contiguous(),
                "Vulkan reduction backward requires contiguous float32 tensors");
    TORCH_CHECK(dim >= 0 && dim < input.dim(),
                "Vulkan reduction backward dimension out of range");
    TORCH_CHECK(input.size(dim) > 0,
                "Vulkan reduction backward rejects empty dimensions");
    auto grad_input = at::empty_like(input);
    auto input_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(input, "reduction backward input");
    auto forward_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(
        forward, "reduction backward output");
    auto grad_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(
        grad_output, "reduction backward grad");
    auto result_layout = pytorch_vulkan::inspect_vulkan_tensor_layout(
        grad_input, "reduction backward result");
    const auto &input_data = input.storage().data_ptr();
    const auto &forward_data = forward.storage().data_ptr();
    const auto &grad_data = grad_output.storage().data_ptr();
    const auto &result_data = grad_input.storage().data_ptr();
    pytorch_vulkan::validate_allocation(input_data, input_layout.allocation_bytes,
                                        "reduction backward");
    pytorch_vulkan::validate_allocation(forward_data, forward_layout.allocation_bytes,
                                        "reduction backward");
    pytorch_vulkan::validate_allocation(grad_data, grad_layout.allocation_bytes,
                                        "reduction backward");
    pytorch_vulkan::validate_allocation(result_data, result_layout.allocation_bytes,
                                        "reduction backward");
    const auto &platform = pytorch_vulkan::allocation_platform(input_data);
    TORCH_CHECK(&platform == &pytorch_vulkan::allocation_platform(forward_data) &&
                    &platform == &pytorch_vulkan::allocation_platform(grad_data) &&
                    &platform == &pytorch_vulkan::allocation_platform(result_data),
                "Vulkan reduction backward requires one Vulkan platform");
    platform.compute().reduction_backward(
        &pytorch_vulkan::allocation_buffer(input_data), input_layout,
        &pytorch_vulkan::allocation_buffer(forward_data), forward_layout,
        &pytorch_vulkan::allocation_buffer(grad_data), grad_layout,
        &pytorch_vulkan::allocation_buffer(result_data), result_layout, 1u << dim,
        static_cast<uint32_t>(input.size(dim)), static_cast<uint32_t>(dim), operation,
        keepdim);
    return grad_input;
}

template <uint32_t Operation>
class ExtendedReductionAutograd final
    : public torch::autograd::Function<ExtendedReductionAutograd<Operation>> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input,
                              c10::OptionalArrayRef<int64_t> dims, bool keepdim) {
        at::AutoDispatchBelowAutograd guard;
        auto normalized = normalized_dims(input, dims, "reduction");
        TORCH_CHECK(normalized.size() == 1,
                    "Vulkan reduction backward supports one dimension");
        auto output = dispatch(input, dims, keepdim, Operation,
                               Operation == 2   ? "amax"
                               : Operation == 3 ? "amin"
                                                : "prod");
        ctx->save_for_backward({input, output});
        ctx->saved_data["dim"] = normalized[0];
        ctx->saved_data["keepdim"] = keepdim;
        return output;
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        auto saved = ctx->get_saved_variables();
        return {reduction_backward_dispatch(saved[0], saved[1], grads[0],
                                            ctx->saved_data["dim"].toInt(),
                                            Operation == 4   ? 2u
                                            : Operation == 2 ? 0u
                                                             : 1u,
                                            ctx->saved_data["keepdim"].toBool()),
                at::Tensor(), at::Tensor()};
    }
};

class SoftmaxAutograd final : public torch::autograd::Function<SoftmaxAutograd> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, int64_t dim, bool half_to_float,
                              bool logarithmic) {
        at::AutoDispatchBelowAutograd guard;
        auto output =
            logarithmic ? pytorch_vulkan::log_softmax_tensor(input, dim, half_to_float)
                        : pytorch_vulkan::softmax_tensor(input, dim, half_to_float);
        ctx->save_for_backward({input, output});
        ctx->saved_data["dim"] = dim < 0 ? dim + input.dim() : dim;
        ctx->saved_data["logarithmic"] = logarithmic;
        return output;
    }
    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        auto saved = ctx->get_saved_variables();
        return {reduction_backward_dispatch(
                    saved[0], saved[1], grads[0], ctx->saved_data["dim"].toInt(),
                    ctx->saved_data["logarithmic"].toBool() ? 4u : 3u, true),
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};
} // namespace

namespace pytorch_vulkan {
at::Tensor sum_tensor(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                      bool keepdim, c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan sum supports only float32 output");
    return dispatch(input, dims, keepdim, 0u, "sum");
}
at::Tensor mean_tensor(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                       bool keepdim, c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan mean supports only float32 output");
    return dispatch(input, dims, keepdim, 1u, "mean");
}

at::Tensor amax_tensor(const at::Tensor &input, c10::ArrayRef<int64_t> dims,
                       bool keepdim) {
    return dispatch(input, dims, keepdim, 2u, "amax");
}
at::Tensor amin_tensor(const at::Tensor &input, c10::ArrayRef<int64_t> dims,
                       bool keepdim) {
    return dispatch(input, dims, keepdim, 3u, "amin");
}
at::Tensor &amax_out(const at::Tensor &input, c10::ArrayRef<int64_t> dims, bool keepdim,
                     at::Tensor &out) {
    dispatch(input, dims, keepdim, 2u, "amax", &out);
    return out;
}
at::Tensor &amin_out(const at::Tensor &input, c10::ArrayRef<int64_t> dims, bool keepdim,
                     at::Tensor &out) {
    dispatch(input, dims, keepdim, 3u, "amin", &out);
    return out;
}
at::Tensor prod_tensor(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                       bool keepdim, c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan prod supports only float32 output");
    return dispatch(input, dims, keepdim, 4u, "prod");
}

at::Tensor prod_dim_tensor(const at::Tensor &input, int64_t dim, bool keepdim,
                           c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan prod supports only float32 output");
    return dispatch(input, c10::ArrayRef<int64_t>(&dim, 1), keepdim, 4u, "prod");
}

at::Tensor &prod_int_out(const at::Tensor &input, int64_t dim, bool keepdim,
                         c10::optional<at::ScalarType> dtype, at::Tensor &out) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan prod supports only float32 output");
    dispatch(input, c10::ArrayRef<int64_t>(&dim, 1), keepdim, 4u, "prod", &out);
    return out;
}

at::Tensor softmax_tensor(const at::Tensor &input, int64_t dim, bool half_to_float) {
    TORCH_CHECK(!half_to_float, "Vulkan softmax does not support half_to_float");
    validate_input(input, "softmax");
    validate_operator_rank(input.dim(), "softmax");
    validate_reduction_dtype(input.scalar_type(), false, "softmax");
    auto dims = normalized_dims(input, c10::ArrayRef<int64_t>(&dim, 1), "softmax");
    TORCH_CHECK(dims.size() == 1 && input.size(dims[0]) > 0,
                "Vulkan softmax requires a non-empty dimension");
    auto output = at::empty_like(input);
    if (input.numel() == 0)
        return output;
    auto in_layout = inspect_vulkan_tensor_layout(input, "softmax");
    auto out_layout = inspect_vulkan_tensor_layout(output, "softmax");
    TORCH_CHECK(in_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan softmax rejects overlapping input views");
    const auto &in_data = input.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    validate_allocation(in_data, in_layout.allocation_bytes, "softmax input");
    validate_allocation(out_data, out_layout.allocation_bytes, "softmax output");
    TORCH_CHECK(&allocation_platform(in_data) == &allocation_platform(out_data),
                "Vulkan softmax requires one Vulkan platform");
    uint64_t count = 1;
    for (auto size : input.sizes())
        count = checked_product(count, static_cast<uint64_t>(size), "softmax");
    uint64_t reduce = static_cast<uint64_t>(input.size(dims[0]));
    TORCH_CHECK(count <= std::numeric_limits<uint32_t>::max() &&
                    reduce <= std::numeric_limits<uint32_t>::max(),
                "Vulkan softmax exceeds dispatch limits");
    allocation_platform(in_data).compute().reduction(
        allocation_buffer(in_data).buffer(), in_layout,
        allocation_buffer(out_data).buffer(), out_layout, 0u,
        static_cast<uint32_t>(reduce), static_cast<uint32_t>(count),
        input.dim() > 0 ? 5u : 5u, static_cast<uint32_t>(dims[0]));
    return output;
}

at::Tensor &softmax_out(const at::Tensor &input, int64_t dim, bool half_to_float,
                        at::Tensor &out) {
    TORCH_CHECK(
        out.device() == input.device() && out.scalar_type() == at::kFloat &&
            out.layout() == at::kStrided && out.is_contiguous() &&
            out.sizes().equals(input.sizes()),
        "Vulkan softmax out requires a contiguous float32 output with the input shape");
    auto result = softmax_tensor(input, dim, half_to_float);
    out.copy_(result);
    return out;
}

at::Tensor log_softmax_tensor(const at::Tensor &input, int64_t dim,
                              bool half_to_float) {
    TORCH_CHECK(!half_to_float, "Vulkan log_softmax does not support half_to_float");
    // Keep the same validation and dispatch contract as softmax; the shader selects log
    // mode.
    validate_input(input, "log_softmax");
    validate_operator_rank(input.dim(), "log_softmax");
    validate_reduction_dtype(input.scalar_type(), false, "log_softmax");
    TORCH_CHECK(dim >= -input.dim() && dim < input.dim(),
                "Vulkan log_softmax dimension out of range");
    if (dim < 0)
        dim += input.dim();
    TORCH_CHECK(input.size(dim) > 0,
                "Vulkan log_softmax requires a non-empty dimension");
    auto output = at::empty_like(input);
    if (input.numel() == 0)
        return output;
    auto in_layout = inspect_vulkan_tensor_layout(input, "log_softmax");
    auto out_layout = inspect_vulkan_tensor_layout(output, "log_softmax");
    TORCH_CHECK(in_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan log_softmax rejects overlapping input views");
    const auto &in_data = input.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    validate_allocation(in_data, in_layout.allocation_bytes, "log_softmax input");
    validate_allocation(out_data, out_layout.allocation_bytes, "log_softmax output");
    TORCH_CHECK(&allocation_platform(in_data) == &allocation_platform(out_data),
                "Vulkan log_softmax requires one Vulkan platform");
    uint64_t count = static_cast<uint64_t>(input.numel());
    TORCH_CHECK(count <= std::numeric_limits<uint32_t>::max(),
                "Vulkan log_softmax exceeds dispatch limits");
    allocation_platform(in_data).compute().reduction(
        allocation_buffer(in_data).buffer(), in_layout,
        allocation_buffer(out_data).buffer(), out_layout, 0u,
        static_cast<uint32_t>(input.size(dim)), static_cast<uint32_t>(count), 6u,
        static_cast<uint32_t>(dim));
    return output;
}

at::Tensor &log_softmax_out(const at::Tensor &input, int64_t dim, bool half_to_float,
                            at::Tensor &out) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat &&
                    out.layout() == at::kStrided && out.is_contiguous() &&
                    out.sizes().equals(input.sizes()),
                "Vulkan log_softmax out requires a contiguous float32 output with the "
                "input shape");
    auto result = log_softmax_tensor(input, dim, half_to_float);
    out.copy_(result);
    return out;
}
at::Tensor &sum_out(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                    bool keepdim, c10::optional<at::ScalarType> dtype,
                    at::Tensor &out) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat &&
                    out.is_contiguous(),
                "Vulkan sum out requires contiguous float32 Vulkan output");
    dispatch(input, dims, keepdim, 0u, "sum", &out);
    return out;
}
at::Tensor &mean_out(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                     bool keepdim, c10::optional<at::ScalarType> dtype,
                     at::Tensor &out) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat &&
                    out.is_contiguous(),
                "Vulkan mean out requires contiguous float32 Vulkan output");
    dispatch(input, dims, keepdim, 1u, "mean", &out);
    return out;
}
at::Tensor autograd_sum(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                        bool keepdim, c10::optional<at::ScalarType> dtype) {
    return ReductionAutograd<false>::apply(input, dims, keepdim, dtype);
}
at::Tensor autograd_mean(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                         bool keepdim, c10::optional<at::ScalarType> dtype) {
    return ReductionAutograd<true>::apply(input, dims, keepdim, dtype);
}
at::Tensor sum_default(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    return autograd_sum(input, c10::nullopt, false, dtype);
}
at::Tensor mean_default(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    return autograd_mean(input, c10::nullopt, false, dtype);
}
at::Tensor autograd_amax(const at::Tensor &input, c10::ArrayRef<int64_t> dims,
                         bool keepdim) {
    return ExtendedReductionAutograd<2>::apply(
        input, c10::OptionalArrayRef<int64_t>(dims), keepdim);
}
at::Tensor autograd_amin(const at::Tensor &input, c10::ArrayRef<int64_t> dims,
                         bool keepdim) {
    return ExtendedReductionAutograd<3>::apply(
        input, c10::OptionalArrayRef<int64_t>(dims), keepdim);
}
at::Tensor autograd_prod_dim(const at::Tensor &input, int64_t dim, bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "Vulkan prod supports only float32 output");
    return ExtendedReductionAutograd<4>::apply(input, c10::ArrayRef<int64_t>(&dim, 1),
                                               keepdim);
}
at::Tensor autograd_softmax(const at::Tensor &input, int64_t dim, bool half_to_float) {
    return SoftmaxAutograd::apply(input, dim, half_to_float, false);
}
at::Tensor autograd_log_softmax(const at::Tensor &input, int64_t dim,
                                bool half_to_float) {
    return SoftmaxAutograd::apply(input, dim, half_to_float, true);
}

at::Tensor &softmax_backward_out(const at::Tensor &grad_output,
                                 const at::Tensor &output, int64_t dim,
                                 c10::ScalarType input_dtype, at::Tensor &grad_input) {
    TORCH_CHECK(input_dtype == at::kFloat,
                "Vulkan softmax backward supports only float32");
    auto result =
        reduction_backward_dispatch(output, output, grad_output, dim, 3u, true);
    grad_input.copy_(result);
    return grad_input;
}
at::Tensor &log_softmax_backward_out(const at::Tensor &grad_output,
                                     const at::Tensor &output, int64_t dim,
                                     c10::ScalarType input_dtype, at::Tensor &out) {
    TORCH_CHECK(input_dtype == at::kFloat,
                "Vulkan log_softmax backward supports only float32");
    auto result =
        reduction_backward_dispatch(output, output, grad_output, dim, 4u, true);
    out.copy_(result);
    return out;
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("sum", &pytorch_vulkan::sum_default);
    m.impl("sum.dim_IntList", &pytorch_vulkan::autograd_sum);
    m.impl("mean", &pytorch_vulkan::mean_default);
    m.impl("mean.dim", &pytorch_vulkan::autograd_mean);
    m.impl("amax", &pytorch_vulkan::autograd_amax);
    m.impl("amin", &pytorch_vulkan::autograd_amin);
    m.impl("prod.dim_int", &pytorch_vulkan::autograd_prod_dim);
    m.impl("_softmax", &pytorch_vulkan::autograd_softmax);
    m.impl("_log_softmax", &pytorch_vulkan::autograd_log_softmax);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sum", &pytorch_vulkan::sum_default);
    m.impl("sum.dim_IntList", &pytorch_vulkan::sum_tensor);
    m.impl("sum.IntList_out", &pytorch_vulkan::sum_out);
    m.impl("mean", &pytorch_vulkan::mean_default);
    m.impl("mean.dim", &pytorch_vulkan::mean_tensor);
    m.impl("mean.out", &pytorch_vulkan::mean_out);
    m.impl("amax", &pytorch_vulkan::amax_tensor);
    m.impl("amin", &pytorch_vulkan::amin_tensor);
    m.impl("amax.out", &pytorch_vulkan::amax_out);
    m.impl("amin.out", &pytorch_vulkan::amin_out);
    m.impl("prod.dim_int", &pytorch_vulkan::prod_dim_tensor);
    m.impl("prod.int_out", &pytorch_vulkan::prod_int_out);
    m.impl("_softmax", &pytorch_vulkan::softmax_tensor);
    m.impl("_softmax.out", &pytorch_vulkan::softmax_out);
    m.impl("_log_softmax", &pytorch_vulkan::log_softmax_tensor);
    m.impl("_log_softmax.out", &pytorch_vulkan::log_softmax_out);
    m.impl("_softmax_backward_data.out", &pytorch_vulkan::softmax_backward_out);
    m.impl("_log_softmax_backward_data.out", &pytorch_vulkan::log_softmax_backward_out);
}
