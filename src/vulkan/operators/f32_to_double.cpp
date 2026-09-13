#include "capability.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace {

at::Tensor f32_to_double(const at::Tensor &input,
                         c10::optional<c10::ScalarType> dtype,
                         c10::optional<at::Layout> layout,
                         c10::optional<c10::Device> device,
                         c10::optional<bool> pin_memory, bool non_blocking,
                         c10::optional<at::MemoryFormat> memory_format) {
    const auto output_device = device.value_or(input.device());
    const auto requested_dtype = dtype.value_or(input.scalar_type());
    if (output_device.is_cpu()) {
        TORCH_CHECK(input.device().type() != c10::DeviceType::PrivateUse1 ||
                        input.scalar_type() != at::kDouble,
                    "Vulkan formatter Double payload readback to CPU is unsupported");
        TORCH_CHECK(requested_dtype == input.scalar_type(),
                    "Vulkan _to_copy requested dtype does not match source dtype");
        auto output = at::empty(input.sizes(),
                                input.options().dtype(requested_dtype).device(output_device));
        pytorch_vulkan::copy_tensor(output, input, non_blocking);
        return output;
    }
    if (input.device().is_cpu()) {
        TORCH_CHECK(requested_dtype != at::kDouble,
                    "Vulkan formatter conversion supports only Vulkan float32 to Vulkan Double");
        TORCH_CHECK(requested_dtype == input.scalar_type(),
                    "Vulkan _to_copy requested dtype does not match source dtype");
        auto output = at::empty(input.sizes(),
                                input.options().dtype(requested_dtype).device(output_device));
        pytorch_vulkan::copy_tensor(output, input, non_blocking);
        return output;
    }
    if (requested_dtype != at::kDouble) {
        TORCH_CHECK(requested_dtype == input.scalar_type(),
                    "Vulkan _to_copy requested dtype does not match source dtype");
        TORCH_CHECK(false, "Vulkan-to-Vulkan transfer is unsupported");
    }
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Vulkan formatter conversion supports only Vulkan float32 to Vulkan Double");
    TORCH_CHECK(!device || !device->is_cpu(),
                "Vulkan formatter conversion requires a Vulkan output device");
    TORCH_CHECK(!layout || *layout == at::kStrided,
                "Vulkan formatter conversion requires strided tensors");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "Vulkan formatter conversion does not support pinned memory");
    TORCH_CHECK(!memory_format || *memory_format == at::MemoryFormat::Contiguous,
                "Vulkan formatter conversion requires contiguous output");
    TORCH_CHECK(!non_blocking, "Vulkan formatter conversion does not support non_blocking=True");
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.device().index() == 0,
                "Vulkan formatter conversion requires a Vulkan device index 0 input");
    TORCH_CHECK(input.layout() == at::kStrided && input.is_contiguous(),
                "Vulkan formatter conversion requires a contiguous input");
    TORCH_CHECK(input.storage_offset() == 0,
                "Vulkan formatter conversion does not support tensors with non-zero storage_offset()");
    TORCH_CHECK(pytorch_vulkan::formatter_double_supported(),
                "Vulkan formatter Double support requires the shaderFloat64 device feature");
    TORCH_CHECK(input.numel() >= 0,
                "Vulkan formatter conversion has a negative element count");
    TORCH_CHECK(static_cast<uint64_t>(input.numel()) <=
                    std::numeric_limits<uint32_t>::max(),
                "Vulkan formatter conversion element count exceeds uint32 range");

    const auto count = static_cast<uint64_t>(input.numel());
    TORCH_CHECK(count <= std::numeric_limits<size_t>::max() / sizeof(float) &&
                    count <= std::numeric_limits<size_t>::max() / sizeof(double),
                "Vulkan formatter conversion byte count does not fit size_t");
    const size_t input_bytes = static_cast<size_t>(count * sizeof(float));
    const size_t output_bytes = static_cast<size_t>(count * sizeof(double));
    TORCH_CHECK(input_bytes <= std::numeric_limits<VkDeviceSize>::max() &&
                    output_bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan formatter conversion byte count does not fit VkDeviceSize");

    TORCH_CHECK(output_device == input.device(),
                "Vulkan formatter conversion requires the same Vulkan device");
    auto output = at::empty(input.sizes(), input.options().dtype(at::kDouble));
    if (count == 0)
        return output;

    const auto &input_data = input.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    pytorch_vulkan::validate_allocation(input_data, input_bytes, "conversion input");
    pytorch_vulkan::validate_allocation(output_data, output_bytes, "conversion output");
    const auto &input_platform = pytorch_vulkan::allocation_platform(input_data);
    const auto &output_platform = pytorch_vulkan::allocation_platform(output_data);
    TORCH_CHECK(&input_platform == &output_platform,
                "Vulkan formatter conversion requires the same Vulkan platform");
    input_platform.compute().f32_to_double(
        pytorch_vulkan::allocation_buffer(input_data).buffer(),
        pytorch_vulkan::allocation_buffer(output_data).buffer(),
        static_cast<VkDeviceSize>(input_bytes), static_cast<VkDeviceSize>(output_bytes),
        static_cast<uint32_t>(count));
    return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_to_copy", &f32_to_double);
}
