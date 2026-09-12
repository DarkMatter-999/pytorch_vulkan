#include "view.h"

#include "vulkan_layout.h"

#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorUtils.h>
#include <ATen/ops/alias.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace {

int64_t requested_numel(at::IntArrayRef sizes, const char *name) {
    int64_t result = 1;
    for (const int64_t size : sizes) {
        TORCH_CHECK(size >= 0 &&
                        (size == 0 || result <= std::numeric_limits<int64_t>::max() / size),
                    "Vulkan ", name, " view size is invalid");
        result *= size;
    }
    return result;
}

at::Tensor metadata_only_view(const at::Tensor &self, at::IntArrayRef sizes,
                              at::IntArrayRef strides,
                              std::optional<int64_t> storage_offset,
                              const char *name, bool require_view_compatibility) {
    const auto source_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(self, name);
    TORCH_CHECK(sizes.size() == strides.size(), "Vulkan ", name,
                " sizes and strides must have matching ranks");
    const int64_t numel = requested_numel(sizes, name);
    const int64_t requested_offset = storage_offset.value_or(self.storage_offset());
    (void)source_layout;
    TORCH_CHECK(!require_view_compatibility ||
                    (self.storage_offset() == 0 && self.is_contiguous() &&
                     at::has_internal_overlap(self) == at::MemOverlap::No),
                "Vulkan ", name,
                " requires a contiguous source with storage_offset == 0 and no internal overlap");
    TORCH_CHECK(!require_view_compatibility || source_layout.numel == numel, "Vulkan ", name,
                " view size must preserve the number of elements");
    (void)pytorch_vulkan::inspect_vulkan_view_layout(
        self, sizes, strides, requested_offset, name);
    if (require_view_compatibility) {
        TORCH_CHECK(at::geometry_is_contiguous(sizes, strides), "Vulkan ", name,
                    " requires contiguous view metadata");
    }

    at::Tensor result = at::_ops::alias::call(self);
    result.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    result.unsafeGetTensorImpl()->set_storage_offset(requested_offset);
    return result;
}

} // namespace

namespace pytorch_vulkan {

at::Tensor as_strided_tensor(const at::Tensor &self, at::IntArrayRef size,
                             at::IntArrayRef stride,
                             std::optional<int64_t> storage_offset) {
    return metadata_only_view(self, size, stride, storage_offset, "as_strided", false);
}

at::Tensor view_tensor(const at::Tensor &self, at::IntArrayRef size) {
    const auto inferred_size = at::infer_size_dv(size, self.numel());
    const auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(stride.has_value(),
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead.");
    return metadata_only_view(self, inferred_size, *stride, std::nullopt, "view", true);
}

at::Tensor reshape_alias_tensor(const at::Tensor &self, at::IntArrayRef size,
                                at::IntArrayRef stride) {
    return metadata_only_view(self, size, stride, std::nullopt, "_reshape_alias", true);
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("as_strided", &pytorch_vulkan::as_strided_tensor);
    m.impl("view", &pytorch_vulkan::view_tensor);
    m.impl("_reshape_alias", &pytorch_vulkan::reshape_alias_tensor);
}
