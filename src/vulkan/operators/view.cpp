#include "view.h"
#include "fake_tensor.h"

#include "vulkan_layout.h"

#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorUtils.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/as_strided.h>
#include <c10/util/Exception.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <limits>

namespace pytorch_vulkan {
at::Tensor vulkan_contiguous_copy(const at::Tensor &source);
}

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
    if (pytorch_vulkan::is_fake_tensor(self)) {
        std::vector<c10::SymInt> symbolic_sizes;
        std::vector<c10::SymInt> symbolic_strides;
        symbolic_sizes.reserve(sizes.size());
        symbolic_strides.reserve(strides.size());
        for (const auto size : sizes)
            symbolic_sizes.emplace_back(size);
        for (const auto stride : strides)
            symbolic_strides.emplace_back(stride);
        return at::_ops::as_strided::redispatch(
            c10::DispatchKeySet(c10::DispatchKey::Meta), self, symbolic_sizes,
            symbolic_strides,
            storage_offset.has_value() ? std::optional<c10::SymInt>(storage_offset.value())
                                       : std::nullopt);
    }
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

class VulkanReshapeCopyAutogradFunction final
    : public torch::autograd::Function<VulkanReshapeCopyAutogradFunction> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &self,
                              std::vector<int64_t> sizes) {
        at::AutoDispatchBelowAutograd guard;
        ctx->save_for_backward({self});
        return pytorch_vulkan::vulkan_contiguous_copy(self).view(sizes).detach();
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grads) {
        at::AutoDispatchBelowAutograd guard;
        if (!grads[0].defined())
            return {at::Tensor(), at::Tensor()};
        const auto input = ctx->get_saved_variables()[0];
        auto grad = grads[0].reshape(input.sizes());
        // The copy's input is already a logical view. Return that logical
        // gradient and let the preceding view node replay its own mapping.
        return {grad, at::Tensor()};
    }
};

} // namespace

namespace pytorch_vulkan {

at::Tensor as_strided_tensor(const at::Tensor &self, at::IntArrayRef size,
                             at::IntArrayRef stride,
                             std::optional<int64_t> storage_offset) {
    return metadata_only_view(self, size, stride, storage_offset, "as_strided", false)
        .detach();
}

at::Tensor view_tensor(const at::Tensor &self, at::IntArrayRef size) {
    const auto inferred_size = at::infer_size_dv(size, self.numel());
    const auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(stride.has_value(),
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead.");
    return as_strided_tensor(self, inferred_size, *stride, std::nullopt);
}

at::Tensor reshape_alias_tensor(const at::Tensor &self, at::IntArrayRef size,
                                at::IntArrayRef stride) {
    return as_strided_tensor(self, size, stride, std::nullopt);
}

at::Tensor reshape_tensor(const at::Tensor &self, at::IntArrayRef size) {
    const auto inferred_size = at::infer_size_dv(size, self.numel());
    const auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    if (stride.has_value()) {
        // computeStride is reshape's compatibility contract; unlike view, it
        // also accepts compatible non-contiguous source layouts.
        return as_strided_tensor(self, inferred_size, *stride, std::nullopt);
    }
    return VulkanReshapeCopyAutogradFunction::apply(
        self, std::vector<int64_t>(inferred_size.begin(), inferred_size.end()));
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("as_strided", &pytorch_vulkan::as_strided_tensor);
    m.impl("view", &pytorch_vulkan::view_tensor);
    m.impl("_reshape_alias", &pytorch_vulkan::reshape_alias_tensor);
    m.impl("reshape", &pytorch_vulkan::reshape_tensor);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("reshape", &pytorch_vulkan::reshape_tensor);
}
