#include "indexing.h"
#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"
#include "vulkan_layout.h"
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <limits>
#include <memory>
#include <vector>

namespace pytorch_vulkan {
namespace {
at::Tensor argmax_tensor_impl(const at::Tensor &input, c10::optional<int64_t> dim, bool keepdim,
                              at::Tensor *provided_output) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.device().index() == 0,
                "Vulkan argmax requires Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided, "Vulkan argmax requires a strided tensor");
    validate_index_input_dtype(input.scalar_type(), "argmax");
    TORCH_CHECK(input.dim() > 0, "Vulkan argmax does not support zero-dimensional inputs");
    validate_operator_rank(input.dim(), "argmax");
    if (!dim) {
        const auto original_layout = inspect_vulkan_tensor_layout(input, "argmax");
        TORCH_CHECK(original_layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan argmax rejects overlapping input views");
        auto flattened = input.reshape({input.numel()});
        if (provided_output) {
            auto scalar_output = provided_output->reshape({});
            argmax_tensor_impl(flattened, 0, false, &scalar_output);
            return *provided_output;
        }
        auto result = argmax_tensor_impl(flattened, 0, false, nullptr);
        if (!keepdim) return result;
        std::vector<int64_t> shape(input.dim(), 1);
        return result.reshape(shape);
    }
    int64_t reduce_dim = *dim;
    if (reduce_dim < 0) reduce_dim += input.dim();
    TORCH_CHECK(reduce_dim >= 0 && reduce_dim < input.dim(), "Vulkan argmax dimension out of range");
    TORCH_CHECK(input.size(reduce_dim) > 0, "Vulkan argmax cannot reduce an empty dimension");
    std::vector<int64_t> shape;
    for (int64_t d = 0; d < input.dim(); ++d)
        if (keepdim || d != reduce_dim) shape.push_back(d == reduce_dim ? 1 : input.size(d));
    std::unique_ptr<VulkanIndexOutputAllocationGuard> index_guard;
    if (!provided_output) index_guard = std::make_unique<VulkanIndexOutputAllocationGuard>();
    at::Tensor output = provided_output ? *provided_output
                                        : at::empty(shape, input.options().dtype(at::kLong));
    if (provided_output) {
        TORCH_CHECK(output.device() == input.device() && output.scalar_type() == at::kLong &&
                        output.layout() == at::kStrided && output.is_contiguous() &&
                        output.storage_offset() == 0 && output.sizes().equals(shape),
                    "Vulkan argmax out has invalid metadata");
    }
    uint64_t output_numel = 1;
    for (auto size : shape) {
        TORCH_CHECK(size == 0 || output_numel <= std::numeric_limits<uint64_t>::max() /
                                      static_cast<uint64_t>(size),
                    "Vulkan argmax output size arithmetic overflow");
        output_numel *= static_cast<uint64_t>(size);
    }
    TORCH_CHECK(output_numel <= std::numeric_limits<uint32_t>::max(),
                "Vulkan argmax output exceeds dispatch limits");
    if (output_numel == 0) return output;
    std::vector<uint32_t> sizes(input.dim());
    for (int64_t d = 0; d < input.dim(); ++d) {
        TORCH_CHECK(input.size(d) <= std::numeric_limits<uint32_t>::max(), "Vulkan argmax dimension is too large");
        sizes[d] = static_cast<uint32_t>(input.size(d));
    }
    const auto &in_data = input.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    const auto input_layout = inspect_vulkan_tensor_layout(input, "argmax");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan argmax rejects overlapping input views");
    const auto output_layout = inspect_vulkan_tensor_layout(output, "argmax");
    validate_allocation(in_data, input_layout.allocation_bytes, "input");
    validate_allocation(out_data, output_layout.allocation_bytes, "output");
    const auto &platform = allocation_platform(in_data);
    TORCH_CHECK(&platform == &allocation_platform(out_data), "Vulkan argmax requires one Vulkan platform");
    platform.compute().argmax(allocation_buffer(in_data).buffer(), input_layout,
                               allocation_buffer(out_data).buffer(), output_layout,
                               static_cast<uint32_t>(reduce_dim), static_cast<uint32_t>(input.size(reduce_dim)),
                               static_cast<uint32_t>(output_numel));
    return output;
}

} // namespace

at::Tensor argmax_tensor(const at::Tensor &input, c10::optional<int64_t> dim, bool keepdim) {
    return argmax_tensor_impl(input, dim, keepdim, nullptr);
}

at::Tensor &argmax_out(const at::Tensor &input, c10::optional<int64_t> dim, bool keepdim, at::Tensor &out) {
    TORCH_CHECK(input.device() == out.device(), "Vulkan argmax out requires the same device");
    validate_index_output_dtype(out.scalar_type(), "argmax");
    TORCH_CHECK(out.layout() == at::kStrided && out.is_contiguous() && out.storage_offset() == 0,
                "Vulkan argmax out requires contiguous strided output with zero storage offset");
    validate_operator_rank(input.dim(), "argmax");
    std::vector<int64_t> expected;
    if (dim) {
        int64_t d = *dim < 0 ? *dim + input.dim() : *dim;
        TORCH_CHECK(d >= 0 && d < input.dim(), "Vulkan argmax dimension out of range");
        for (int64_t i = 0; i < input.dim(); ++i)
            if (keepdim || i != d) expected.push_back(i == d ? 1 : input.size(i));
    }
    TORCH_CHECK(!dim || out.sizes().equals(expected), "Vulkan argmax out has the wrong shape");
    if (!dim) {
        std::vector<int64_t> all_ones(input.dim(), 1);
        TORCH_CHECK(out.sizes().equals(all_ones), "Vulkan argmax out has the wrong shape");
    }
    argmax_tensor_impl(input, dim, keepdim, &out);
    return out;
}
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("argmax", &pytorch_vulkan::argmax_tensor);
    m.impl("argmax.out", &pytorch_vulkan::argmax_out);
}
