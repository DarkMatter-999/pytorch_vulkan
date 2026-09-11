#include "vulkan_transfer.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1 ||
           device.type() == c10::DeviceType::Vulkan;
}

std::string direction(const at::Tensor &destination, const at::Tensor &source) {
    if (source.device().is_cpu() && !destination.device().is_cpu()) {
        return "CPU->Vulkan";
    }
    if (!source.device().is_cpu() && destination.device().is_cpu()) {
        return "Vulkan->CPU";
    }
    if (!source.device().is_cpu() && !destination.device().is_cpu()) {
        return "Vulkan-to-Vulkan";
    }
    return "unsupported device pair";
}

std::size_t checked_bytes(const at::Tensor &destination, const at::Tensor &source) {
    const int64_t elements = destination.numel();
    TORCH_CHECK(elements >= 0, "Vulkan copy has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / sizeof(float),
                "Vulkan copy byte count does not fit size_t");
    const auto bytes = count * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan copy byte count does not fit VkDeviceSize");
    (void)source;
    return static_cast<std::size_t>(bytes);
}

void validate(const at::Tensor &destination, const at::Tensor &source,
              bool non_blocking) {
    TORCH_CHECK(!non_blocking, "Vulkan copy does not support non_blocking=True");
    TORCH_CHECK(destination.layout() == at::kStrided && source.layout() == at::kStrided,
                "Vulkan copy requires strided tensors");
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(),
                "Vulkan copy requires contiguous tensors");
    TORCH_CHECK(destination.scalar_type() == at::kFloat && source.scalar_type() == at::kFloat,
                "Vulkan copy supports only float32 tensors");
    TORCH_CHECK(destination.numel() == source.numel(),
                "Vulkan copy requires matching sizes");

    const bool destination_cpu = destination.device().is_cpu();
    const bool source_cpu = source.device().is_cpu();
    TORCH_CHECK(destination_cpu != source_cpu,
                "Vulkan copy ", direction(destination, source),
                " requires exactly one CPU and one Vulkan device; destination ",
                destination.device(), ", source ", source.device());
    TORCH_CHECK(destination_cpu ? is_vulkan_device(source.device())
                                : is_vulkan_device(destination.device()),
                "Vulkan copy requires a CPU and PrivateUse1/Vulkan device; destination ",
                destination.device(), ", source ", source.device());
    const c10::Device &vulkan_device = destination_cpu ? source.device() : destination.device();
    const at::Tensor &vulkan_tensor = destination_cpu ? source : destination;
    TORCH_CHECK(vulkan_tensor.storage_offset() == 0,
                "Vulkan copy does not support tensors with non-zero storage_offset()",
                "; offset is ", vulkan_tensor.storage_offset());
    TORCH_CHECK(vulkan_device.index() == 0,
                "Vulkan copy ", direction(destination, source),
                " supports only Vulkan device index 0, got ", vulkan_device.index());
    (void)checked_bytes(destination, source);
}

} // namespace

namespace pytorch_vulkan {

at::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,
                        bool non_blocking) {
    validate(destination, source, non_blocking);
    const std::size_t bytes = checked_bytes(destination, source);
    if (bytes == 0) {
        return destination;
    }

    const bool cpu_to_vulkan = source.device().is_cpu();
    const at::Tensor &vulkan_tensor = cpu_to_vulkan ? destination : source;
    const at::DataPtr &vulkan_data = vulkan_tensor.storage().data_ptr();
    VulkanBuffer &vulkan_buffer = allocation_buffer(vulkan_data);
    const VulkanPlatform &platform = allocation_platform(vulkan_data);
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    TORCH_CHECK(size <= vulkan_buffer.size(), "Vulkan copy ", direction(destination, source),
                " exceeds Vulkan allocation (", bytes, " bytes, allocation is ",
                vulkan_buffer.size(), " bytes)");

    float *cpu_destination = cpu_to_vulkan ? nullptr : destination.data_ptr<float>();
    const float *cpu_source = cpu_to_vulkan ? source.data_ptr<float>() : nullptr;
    const bool host_visible = (vulkan_buffer.memory_properties() &
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (host_visible) {
        platform.wait_for_transfer();
        if (cpu_to_vulkan) {
            vulkan_buffer.write(cpu_source, size);
        } else {
            vulkan_buffer.read(cpu_destination, size);
        }
        return destination;
    }

    std::unique_ptr<VulkanBuffer> staging;
    try {
        staging = std::make_unique<VulkanBuffer>(
            platform, size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("Vulkan copy staging allocation failed for ") +
                                 direction(destination, source) + ": " + error.what());
    }
    if (cpu_to_vulkan) {
        staging->write(cpu_source, size);
        platform.copy_buffer_sync(staging->buffer(), vulkan_buffer.buffer(), size);
    } else {
        platform.copy_buffer_sync(vulkan_buffer.buffer(), staging->buffer(), size);
        staging->read(cpu_destination, size);
    }
    return destination;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("copy_", &pytorch_vulkan::copy_tensor);
}
