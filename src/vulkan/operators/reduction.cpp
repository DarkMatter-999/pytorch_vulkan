#include "reduction.h"
#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <torch/autograd.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace {
void validate_input(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.device().index() == 0,
                "Vulkan ", name, " requires Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided && input.is_contiguous(),
                "Vulkan ", name, " requires a contiguous strided tensor");
    TORCH_CHECK(input.storage_offset() == 0, "Vulkan ", name, " does not support non-zero storage_offset()");
}

std::vector<int64_t> normalized_dims(const at::Tensor &, c10::OptionalArrayRef<int64_t>, const char *);
at::Tensor dispatch(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool, bool, const char *, at::Tensor * = nullptr);

template <bool Mean>
class ReductionAutograd final : public torch::autograd::Function<ReductionAutograd<Mean>> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input,
                              c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                              c10::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto normalized = normalized_dims(input, dims, Mean ? "mean" : "sum");
        uint64_t mask = 0, count = 1;
        for (auto dim : normalized) { mask |= 1ULL << dim; count *= input.size(dim); }
        ctx->save_for_backward({input});
        ctx->saved_data["mask"] = static_cast<int64_t>(mask);
        ctx->saved_data["count"] = static_cast<int64_t>(count);
        ctx->saved_data["keepdim"] = keepdim;
        return dispatch(input, dims, keepdim, Mean, Mean ? "mean" : "sum");
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined()) return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        auto input = ctx->get_saved_variables()[0];
        pytorch_vulkan::validate_operator_rank(input.dim(), Mean ? "mean backward" : "sum backward");
        auto mask = static_cast<uint64_t>(ctx->saved_data["mask"].toInt());
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        std::vector<int64_t> compact_shape, full_shape;
        std::vector<uint32_t> input_sizes(input.dim()), output_sizes(input.dim());
        for (int64_t d = 0; d < input.dim(); ++d) {
            const bool reduced = mask & (1ULL << d);
            full_shape.push_back(reduced ? 1 : input.size(d));
            input_sizes[d] = reduced ? 1U : static_cast<uint32_t>(input.size(d));
            output_sizes[d] = static_cast<uint32_t>(input.size(d));
            if (keepdim || !reduced) compact_shape.push_back(reduced ? 1 : input.size(d));
        }
        auto expanded = grads[0].reshape(compact_shape).reshape(full_shape);
        auto result = at::empty(input.sizes(), input.options());
        const auto &source_data = expanded.storage().data_ptr();
        const auto &result_data = result.storage().data_ptr();
        const auto &platform = pytorch_vulkan::allocation_platform(source_data);
        pytorch_vulkan::validate_allocation(source_data, expanded.numel() * sizeof(float), "reduction gradient");
        pytorch_vulkan::validate_allocation(result_data, result.numel() * sizeof(float), "reduction gradient");
        platform.compute().broadcast(pytorch_vulkan::allocation_buffer(source_data).buffer(),
                                     pytorch_vulkan::allocation_buffer(result_data).buffer(),
                                     static_cast<uint32_t>(input.dim()), input_sizes.data(), output_sizes.data(),
                                     static_cast<uint32_t>(input.numel()),
                                     Mean ? static_cast<float>(1.0 / static_cast<double>(ctx->saved_data["count"].toInt())) : 1.0F);
        return {result, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

std::vector<int64_t> normalized_dims(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims,
                                     const char *name) {
    std::vector<int64_t> result;
    if (!dims || dims->empty()) {
        for (int64_t d = 0; d < input.dim(); ++d) result.push_back(d);
        return result;
    }
    for (int64_t dim : *dims) {
        if (dim < 0) dim += input.dim();
        TORCH_CHECK(dim >= 0 && dim < input.dim(), "Vulkan ", name, " dimension out of range");
        TORCH_CHECK(std::find(result.begin(), result.end(), dim) == result.end(),
                    "Vulkan ", name, " received duplicate dimensions");
        result.push_back(dim);
    }
    return result;
}

std::vector<int64_t> output_sizes(const at::Tensor &input, const std::vector<int64_t> &dims, bool keepdim) {
    std::vector<bool> reduced(input.dim(), false);
    for (auto dim : dims) reduced[dim] = true;
    std::vector<int64_t> sizes;
    for (int64_t d = 0; d < input.dim(); ++d)
        if (keepdim || !reduced[d]) sizes.push_back(reduced[d] ? 1 : input.size(d));
    return sizes;
}

at::Tensor dispatch(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                    bool mean, const char *name, at::Tensor *provided_output) {
    validate_input(input, name);
    TORCH_CHECK(input.dim() > 0, "Vulkan ", name, " does not support zero-dimensional inputs");
    pytorch_vulkan::validate_operator_rank(input.dim(), name);
    TORCH_CHECK(input.numel() != 0, "Vulkan ", name,
                " does not support empty inputs in the current capability matrix");
    pytorch_vulkan::validate_reduction_dtype(input.scalar_type(), mean, name);
    auto reduced_dims = normalized_dims(input, dims, name);
    TORCH_CHECK(!reduced_dims.empty(), "Vulkan ", name, " requires at least one dimension");
    std::vector<uint32_t> sizes(input.dim());
    uint64_t reduce_numel = 1, output_numel = 1;
    std::vector<bool> reduced(input.dim(), false);
    for (auto dim : reduced_dims) reduced[dim] = true;
    for (int64_t d = 0; d < input.dim(); ++d) {
        TORCH_CHECK(input.size(d) <= std::numeric_limits<uint32_t>::max(), "Vulkan ", name, " dimension is too large");
        sizes[d] = static_cast<uint32_t>(input.size(d));
        if (reduced[d]) reduce_numel *= sizes[d];
    }
    for (auto size : output_sizes(input, reduced_dims, keepdim)) output_numel *= static_cast<uint64_t>(size);
    TORCH_CHECK(reduce_numel <= std::numeric_limits<uint32_t>::max() && output_numel <= std::numeric_limits<uint32_t>::max(),
                "Vulkan ", name, " exceeds dispatch limits");
    auto expected_sizes = output_sizes(input, reduced_dims, keepdim);
    at::Tensor output = provided_output ? *provided_output : at::empty(expected_sizes, input.options());
    if (provided_output) {
        TORCH_CHECK(output.device() == input.device() && output.scalar_type() == at::kFloat &&
                        output.layout() == at::kStrided && output.is_contiguous() &&
                        output.sizes().equals(expected_sizes),
                    "Vulkan ", name, " out requires a contiguous float32 output with the expected shape");
    }
    if (output_numel == 0) return output;
    const auto &in_data = input.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    pytorch_vulkan::validate_allocation(in_data, input.numel() * sizeof(float), "input");
    pytorch_vulkan::validate_allocation(out_data, output.numel() * sizeof(float), "output");
    const auto &platform = pytorch_vulkan::allocation_platform(in_data);
    TORCH_CHECK(&platform == &pytorch_vulkan::allocation_platform(out_data), "Vulkan ", name,
                " requires one Vulkan platform");
    uint32_t mask = 0;
    for (auto dim : reduced_dims) mask |= 1u << dim;
    platform.compute().reduction(pytorch_vulkan::allocation_buffer(in_data).buffer(),
                                 pytorch_vulkan::allocation_buffer(out_data).buffer(),
                                 static_cast<VkDeviceSize>(input.numel() * sizeof(float)),
                                 static_cast<uint32_t>(input.dim()), sizes.data(), mask,
                                 static_cast<uint32_t>(reduce_numel), static_cast<uint32_t>(output_numel), mean);
    return output;
}
}

namespace pytorch_vulkan {
at::Tensor sum_tensor(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                      c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat, "Vulkan sum supports only float32 output");
    return dispatch(input, dims, keepdim, false, "sum");
}
at::Tensor mean_tensor(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                       c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype || *dtype == at::kFloat, "Vulkan mean supports only float32 output");
    return dispatch(input, dims, keepdim, true, "mean");
}
at::Tensor &sum_out(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                    c10::optional<at::ScalarType> dtype, at::Tensor &out) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat && out.is_contiguous(),
                "Vulkan sum out requires contiguous float32 Vulkan output");
    dispatch(input, dims, keepdim, false, "sum", &out); return out;
}
at::Tensor &mean_out(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                     c10::optional<at::ScalarType> dtype, at::Tensor &out) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat && out.is_contiguous(),
                "Vulkan mean out requires contiguous float32 Vulkan output");
    dispatch(input, dims, keepdim, true, "mean", &out); return out;
}
at::Tensor autograd_sum(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                         c10::optional<at::ScalarType> dtype) { return ReductionAutograd<false>::apply(input, dims, keepdim, dtype); }
at::Tensor autograd_mean(const at::Tensor &input, c10::OptionalArrayRef<int64_t> dims, bool keepdim,
                         c10::optional<at::ScalarType> dtype) { return ReductionAutograd<true>::apply(input, dims, keepdim, dtype); }
at::Tensor sum_default(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    return autograd_sum(input, c10::nullopt, false, dtype);
}
at::Tensor mean_default(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    return autograd_mean(input, c10::nullopt, false, dtype);
}
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("sum", &pytorch_vulkan::sum_default);
    m.impl("sum.dim_IntList", &pytorch_vulkan::autograd_sum);
    m.impl("mean", &pytorch_vulkan::mean_default);
    m.impl("mean.dim", &pytorch_vulkan::autograd_mean);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sum", &pytorch_vulkan::sum_default);
    m.impl("sum.dim_IntList", &pytorch_vulkan::sum_tensor);
    m.impl("sum.IntList_out", &pytorch_vulkan::sum_out);
    m.impl("mean", &pytorch_vulkan::mean_default);
    m.impl("mean.dim", &pytorch_vulkan::mean_tensor);
    m.impl("mean.out", &pytorch_vulkan::mean_out);
}
