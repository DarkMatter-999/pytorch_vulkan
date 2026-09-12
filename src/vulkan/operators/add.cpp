#include "add.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>
#include <string>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1;
}

std::size_t checked_bytes(const at::Tensor &tensor) {
    const int64_t elements = tensor.numel();
    TORCH_CHECK(elements >= 0, "Vulkan add has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / sizeof(float),
                "Vulkan add byte count does not fit size_t");
    const auto bytes = count * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan add byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

void validate(const at::Tensor &lhs, const at::Tensor &rhs, const at::Scalar &alpha) {
    TORCH_CHECK(!(lhs.device().is_cpu() && rhs.device().is_cpu()),
                "Vulkan add requires a Vulkan operand");
    TORCH_CHECK(!(lhs.device().is_cpu() && lhs.dim() == 0) &&
                    !(rhs.device().is_cpu() && rhs.dim() == 0),
                "Vulkan add does not support a scalar operand");
    TORCH_CHECK(is_vulkan_device(lhs.device()) && is_vulkan_device(rhs.device()),
                "Vulkan add requires operands on the same device; got ", lhs.device(),
                " and ", rhs.device());
    TORCH_CHECK(lhs.device() == rhs.device(),
                "Vulkan add requires operands on the same device; got ", lhs.device(),
                " and ", rhs.device());
    TORCH_CHECK(lhs.device().index() == 0 && rhs.device().index() == 0,
                "Vulkan add supports only Vulkan device index 0");
    TORCH_CHECK(lhs.layout() == at::kStrided && rhs.layout() == at::kStrided,
                "Vulkan add requires strided tensors");
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(),
                "Vulkan add requires contiguous tensors");
    TORCH_CHECK(lhs.storage_offset() == 0 && rhs.storage_offset() == 0,
                "Vulkan add does not support tensors with non-zero storage_offset()");
    TORCH_CHECK(lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kFloat,
                "Vulkan add supports only float32 tensors");
    TORCH_CHECK(lhs.dim() != 0 && rhs.dim() != 0,
                "Vulkan add does not support zero-dimensional tensor operands");
    TORCH_CHECK(lhs.sizes().equals(rhs.sizes()),
                "Vulkan add requires equal tensor sizes; broadcasting is unsupported");
    TORCH_CHECK(alpha.toDouble() == 1.0,
                "Vulkan add supports only alpha == 1");
}

} // namespace

namespace pytorch_vulkan {

at::Tensor add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha) {
    validate(lhs, rhs, alpha);
    at::Tensor output = at::empty(lhs.sizes(), lhs.options().device(lhs.device()));
    const std::size_t bytes = checked_bytes(lhs);
    if (bytes == 0) {
        return output;
    }

    const at::DataPtr &lhs_data = lhs.storage().data_ptr();
    const at::DataPtr &rhs_data = rhs.storage().data_ptr();
    const at::DataPtr &output_data = output.storage().data_ptr();
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    validate_allocation(lhs_data, size, "lhs");
    validate_allocation(rhs_data, size, "rhs");
    validate_allocation(output_data, size, "output");
    VulkanBuffer &lhs_buffer = allocation_buffer(lhs_data);
    VulkanBuffer &rhs_buffer = allocation_buffer(rhs_data);
    VulkanBuffer &output_buffer = allocation_buffer(output_data);
    const VulkanPlatform &lhs_platform = allocation_platform(lhs_data);
    const VulkanPlatform &rhs_platform = allocation_platform(rhs_data);
    const VulkanPlatform &output_platform = allocation_platform(output_data);
    TORCH_CHECK(&lhs_platform == &rhs_platform && &lhs_platform == &output_platform &&
                    lhs_platform.device() == rhs_platform.device() &&
                    lhs_platform.device() == output_platform.device(),
                "Vulkan add requires all tensors to use the same Vulkan platform/device");

    lhs_platform.compute().add(lhs_buffer.buffer(), rhs_buffer.buffer(),
                               output_buffer.buffer(), size);
    return output;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", &pytorch_vulkan::add_tensor);
}
