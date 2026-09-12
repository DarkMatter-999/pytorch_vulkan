#include "abs.h"

#include "add.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace {

void validate_tensor(const at::Tensor &tensor, const char *label) {
    TORCH_CHECK(tensor.defined(), "Vulkan abs.out requires a defined ", label, " tensor");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Vulkan abs.out requires ", label, " on Vulkan");
    TORCH_CHECK(tensor.device().index() == 0,
                "Vulkan abs.out supports only Vulkan device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided,
                "Vulkan abs.out requires strided ", label);
    TORCH_CHECK(tensor.is_contiguous(),
                "Vulkan abs.out requires contiguous ", label);
    TORCH_CHECK(tensor.storage_offset() == 0,
                "Vulkan abs.out does not support ", label,
                " tensors with non-zero storage_offset()");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "Vulkan abs.out supports only float32 tensors");
    TORCH_CHECK(!tensor.requires_grad(),
                "Vulkan abs.out does not support autograd tensors");
}

std::size_t checked_bytes(const at::Tensor &tensor) {
    const int64_t elements = tensor.numel();
    TORCH_CHECK(elements >= 0, "Vulkan abs.out has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / sizeof(float),
                "Vulkan abs.out byte count does not fit size_t");
    const auto bytes = count * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan abs.out byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

} // namespace

namespace pytorch_vulkan {

at::Tensor &abs_out(const at::Tensor &self, at::Tensor &out) {
    validate_tensor(self, "input");
    validate_tensor(out, "output");
    TORCH_CHECK(self.device() == out.device(),
                "Vulkan abs.out requires input and output on the same device");
    // Formatter out buffers start empty; standard out semantics resize them once.
    if (out.numel() == 0 && self.numel() != 0 && !self.sizes().equals(out.sizes())) {
        out.set_data(at::empty(self.sizes(), self.options().device(self.device())));
    }
    TORCH_CHECK(self.sizes().equals(out.sizes()),
                "Vulkan abs.out requires input and output to have matching sizes");

    const std::size_t bytes = checked_bytes(self);
    TORCH_CHECK(bytes == checked_bytes(out),
                "Vulkan abs.out requires input and output to have matching sizes");
    if (bytes == 0) {
        return out;
    }

    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    const at::DataPtr &input_data = self.storage().data_ptr();
    const at::DataPtr &output_data = out.storage().data_ptr();
    validate_allocation(input_data, size, "input");
    validate_allocation(output_data, size, "output");
    VulkanBuffer &input_buffer = allocation_buffer(input_data);
    VulkanBuffer &output_buffer = allocation_buffer(output_data);
    const VulkanPlatform &input_platform = allocation_platform(input_data);
    const VulkanPlatform &output_platform = allocation_platform(output_data);
    TORCH_CHECK(&input_platform == &output_platform &&
                    input_platform.device() == output_platform.device(),
                "Vulkan abs.out requires all tensors to use the same Vulkan platform/device");
    input_platform.compute().tensor_scalar(
        input_buffer.buffer(), output_buffer.buffer(), size, 0.0F,
        static_cast<uint32_t>(PointwiseOperation::Abs));
    return out;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("abs.out", &pytorch_vulkan::abs_out);
}
