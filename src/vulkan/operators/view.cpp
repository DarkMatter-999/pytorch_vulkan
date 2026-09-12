#include "view.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

namespace pytorch_vulkan {

at::Tensor formatter_view(const at::Tensor &tensor, c10::SymIntArrayRef size) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Vulkan view requires a Vulkan tensor");
    TORCH_CHECK(tensor.device().index() == 0,
                "Vulkan view supports only Vulkan device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided,
                "Vulkan view requires a strided tensor");
    TORCH_CHECK(tensor.is_contiguous(),
                "Vulkan view requires a contiguous tensor");
    TORCH_CHECK(tensor.storage_offset() == 0,
                "Vulkan view does not support tensors with non-zero storage_offset()");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "Vulkan view supports only float32 tensors");
    const int64_t requested_size = size.size() == 1
        ? size[0].guard_int(__FILE__, __LINE__)
        : -1;
    TORCH_CHECK(size.size() == 1 &&
                    (requested_size == -1 || requested_size == tensor.numel()),
                "Vulkan view supports only one-dimensional flattening");

    at::Tensor result = at::alias(tensor);
    result.unsafeGetTensorImpl()->set_sizes_contiguous(
        std::vector<int64_t>{tensor.numel()});
    return result;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("view", &pytorch_vulkan::formatter_view);
}
