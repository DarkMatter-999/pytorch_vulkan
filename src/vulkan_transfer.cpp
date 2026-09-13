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
    const std::size_t element_bytes =
        pytorch_vulkan::vulkan_storage_bytes(destination.scalar_type());
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / element_bytes,
                "Vulkan copy byte count does not fit size_t");
    const auto bytes = count * element_bytes;
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan copy byte count does not fit VkDeviceSize");
    (void)source;
    return static_cast<std::size_t>(bytes);
}

std::size_t checked_range_bytes(const at::Tensor &source, std::size_t bytes) {
    TORCH_CHECK(source.storage_offset() >= 0, "Vulkan formatter presentation has a negative storage offset");
    const auto offset = static_cast<uint64_t>(source.storage_offset());
    const auto element_bytes = pytorch_vulkan::vulkan_storage_bytes(source.scalar_type());
    TORCH_CHECK(offset <= std::numeric_limits<uint64_t>::max() / element_bytes,
                "Vulkan formatter presentation offset overflows");
    const auto byte_offset = offset * element_bytes;
    TORCH_CHECK(bytes <= std::numeric_limits<std::size_t>::max() - byte_offset,
                "Vulkan formatter presentation byte range overflows");
    return static_cast<std::size_t>(byte_offset) + bytes;
}

void validate(const at::Tensor &destination, const at::Tensor &source,
              bool non_blocking) {
    TORCH_CHECK(!non_blocking, "Vulkan copy does not support non_blocking=True");
    TORCH_CHECK(destination.layout() == at::kStrided && source.layout() == at::kStrided,
                "Vulkan copy requires strided tensors");
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(),
                "Vulkan copy requires contiguous tensors");
    (void)pytorch_vulkan::vulkan_storage_bytes(destination.scalar_type());
    (void)pytorch_vulkan::vulkan_storage_bytes(source.scalar_type());
    if (destination.scalar_type() != source.scalar_type() &&
        (destination.scalar_type() == at::kDouble || source.scalar_type() == at::kDouble)) {
        TORCH_CHECK(false,
                    "Vulkan copy supports only float32 and bool tensors for this transfer");
    }
    TORCH_CHECK(destination.scalar_type() == source.scalar_type(),
                "Vulkan copy requires matching dtypes");
    TORCH_CHECK(destination.numel() == source.numel(),
                "Vulkan copy requires matching sizes");

    const bool destination_cpu = destination.device().is_cpu();
    const bool source_cpu = source.device().is_cpu();
    TORCH_CHECK(!(destination_cpu && !source_cpu && source.scalar_type() == at::kDouble),
                "Vulkan formatter Double payload readback to CPU is unsupported");
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
    if (destination.device().is_cpu() && !source.device().is_cpu() &&
        source.scalar_type() == at::kFloat &&
        destination.scalar_type() == source.scalar_type()) {
        TORCH_CHECK(!non_blocking,
                    "Vulkan formatter presentation does not support non_blocking=True");
        return formatter_presentation_copy(destination, source);
    }
    validate(destination, source, non_blocking);
    if (!destination.device().is_cpu() || !source.device().is_cpu()) {
        ensure_process_local_vulkan();
    }
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

    void *cpu_destination = cpu_to_vulkan ? nullptr : destination.data_ptr();
    const void *cpu_source = cpu_to_vulkan ? source.data_ptr() : nullptr;
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

at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source) {
    TORCH_CHECK(destination.device().is_cpu() && !source.device().is_cpu(),
                "Vulkan formatter presentation requires Vulkan source and CPU destination");
    TORCH_CHECK(destination.layout() == at::kStrided && source.layout() == at::kStrided &&
                    destination.is_contiguous() && source.is_contiguous(),
                "Vulkan formatter presentation requires contiguous tensors");
    TORCH_CHECK(destination.scalar_type() == source.scalar_type(),
                "Vulkan formatter presentation requires matching dtypes");
    TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1 &&
                    source.device().index() == 0,
                "Vulkan formatter presentation requires vk:0 source");
    TORCH_CHECK(source.numel() == destination.numel(),
                "Vulkan formatter presentation requires matching sizes");
    const std::size_t bytes = checked_bytes(destination, source);
    if (bytes == 0)
        return destination;

    const at::DataPtr &data = source.storage().data_ptr();
    validate_allocation(data, checked_range_bytes(source, bytes),
                        "formatter presentation source");
    const VulkanBuffer &buffer = allocation_buffer(data);
    const VulkanPlatform &platform = allocation_platform(data);
    const VkDeviceSize offset = static_cast<VkDeviceSize>(
        static_cast<uint64_t>(source.storage_offset()) *
        pytorch_vulkan::vulkan_storage_bytes(source.scalar_type()));
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    TORCH_CHECK(offset <= buffer.size() && size <= buffer.size() - offset,
                "Vulkan formatter presentation source range exceeds allocation");

    if ((buffer.memory_properties() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        platform.wait_for_transfer();
        buffer.read(destination.data_ptr(), size, offset);
        return destination;
    }

    VulkanBuffer staging(platform, size,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    platform.copy_buffer_sync(buffer.buffer(), staging.buffer(), size, offset);
    staging.read(destination.data_ptr(), size);
    return destination;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("copy_", &pytorch_vulkan::copy_tensor);
}
