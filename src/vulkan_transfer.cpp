#include "vulkan_transfer.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_execution.h"
#include "vulkan_layout.h"

#include <ATen/MemoryOverlap.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    TORCH_CHECK(source.storage_offset() >= 0,
                "Vulkan formatter presentation has a negative storage offset");
    const auto offset = static_cast<uint64_t>(source.storage_offset());
    const auto element_bytes =
        pytorch_vulkan::vulkan_storage_bytes(source.scalar_type());
    TORCH_CHECK(offset <= std::numeric_limits<uint64_t>::max() / element_bytes,
                "Vulkan formatter presentation offset overflows");
    const auto byte_offset = offset * element_bytes;
    TORCH_CHECK(bytes <= std::numeric_limits<std::size_t>::max() - byte_offset,
                "Vulkan formatter presentation byte range overflows");
    return static_cast<std::size_t>(byte_offset) + bytes;
}

uint64_t checked_byte_offset(int64_t element_offset, uint64_t element_bytes,
                             const char *label) {
    TORCH_CHECK(element_offset >= 0, "Vulkan ", label,
                " has a negative element offset");
    const auto offset = static_cast<uint64_t>(element_offset);
    TORCH_CHECK(offset <= std::numeric_limits<uint64_t>::max() / element_bytes,
                "Vulkan ", label, " byte offset overflows");
    return offset * element_bytes;
}

struct TransferLayout {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t numel = 0;
    std::size_t element_bytes = 0;
    bool vulkan = false;
    pytorch_vulkan::VulkanTensorLayout vulkan_layout{};
};

TransferLayout transfer_layout(const at::Tensor &tensor, const char *label) {
    TORCH_CHECK(tensor.layout() == at::kStrided,
                "Vulkan copy requires strided tensors");
    TransferLayout result;
    result.sizes = tensor.sizes().vec();
    result.strides = tensor.strides().vec();
    result.numel = tensor.numel();
    result.element_bytes = pytorch_vulkan::vulkan_storage_bytes(tensor.scalar_type());
    for (const auto stride : result.strides) {
        TORCH_CHECK(stride >= 0, "Vulkan copy ", label, " has a negative stride");
    }
    TORCH_CHECK(result.numel >= 0, "Vulkan copy has a negative element count");
    if (is_vulkan_device(tensor.device())) {
        result.vulkan = true;
        result.vulkan_layout =
            pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, label);
    }
    return result;
}

uint64_t logical_offset(const TransferLayout &layout, int64_t linear_index) {
    if (layout.vulkan)
        return static_cast<uint64_t>(
            pytorch_vulkan::vulkan_storage_offset(layout.vulkan_layout, linear_index));
    TORCH_CHECK(linear_index >= 0 && linear_index < layout.numel,
                "Vulkan copy linear index is out of range");
    uint64_t offset = 0;
    if (!layout.sizes.empty()) {
        uint64_t remaining = static_cast<uint64_t>(linear_index);
        for (int64_t dim = static_cast<int64_t>(layout.sizes.size()) - 1; dim >= 0;
             --dim) {
            const auto coordinate =
                remaining % static_cast<uint64_t>(layout.sizes[dim]);
            remaining /= static_cast<uint64_t>(layout.sizes[dim]);
            if (layout.strides[dim] == 0)
                continue;
            TORCH_CHECK(coordinate <= std::numeric_limits<uint64_t>::max() /
                                          static_cast<uint64_t>(layout.strides[dim]),
                        "Vulkan copy address arithmetic overflow");
            TORCH_CHECK(offset <=
                            std::numeric_limits<uint64_t>::max() -
                                coordinate * static_cast<uint64_t>(layout.strides[dim]),
                        "Vulkan copy address arithmetic overflow");
            offset += coordinate * static_cast<uint64_t>(layout.strides[dim]);
        }
    }
    return offset;
}

bool same_allocation(const at::Tensor &lhs, const at::Tensor &rhs) {
    return lhs.storage().data_ptr().get_context() ==
           rhs.storage().data_ptr().get_context();
}

bool same_layout(const TransferLayout &lhs, const TransferLayout &rhs) {
    return lhs.sizes == rhs.sizes && lhs.strides == rhs.strides &&
           lhs.vulkan_layout.storage_offset == rhs.vulkan_layout.storage_offset;
}

void validate(const at::Tensor &destination, const at::Tensor &source,
              bool non_blocking, TransferLayout &destination_layout,
              TransferLayout &source_layout) {
    TORCH_CHECK(!non_blocking, "Vulkan copy does not support non_blocking=True");
    destination_layout = transfer_layout(destination, "destination");
    source_layout = transfer_layout(source, "source");
    if (destination.scalar_type() != source.scalar_type() &&
        (destination.scalar_type() == at::kDouble ||
         source.scalar_type() == at::kDouble)) {
        TORCH_CHECK(
            false,
            "Vulkan copy supports only float32 and bool tensors for this transfer");
    }
    TORCH_CHECK(destination.scalar_type() == source.scalar_type(),
                "Vulkan copy requires matching dtypes");
    TORCH_CHECK(destination.numel() == source.numel(),
                "Vulkan copy requires matching sizes");

    const bool destination_cpu = destination.device().is_cpu();
    const bool source_cpu = source.device().is_cpu();
    if (destination_cpu) {
        TORCH_CHECK(at::has_internal_overlap(destination) == at::MemOverlap::No,
                    "Vulkan copy destination has internal overlap");
    }
    TORCH_CHECK(
        !(destination_cpu && !source_cpu && source.scalar_type() == at::kDouble),
        "Vulkan formatter Double payload readback to CPU is unsupported");
    TORCH_CHECK(!(destination_cpu && source_cpu), "Vulkan copy ",
                direction(destination, source),
                " requires exactly one CPU and one Vulkan device, or two Vulkan "
                "tensors; destination ",
                destination.device(), ", source ", source.device());
    TORCH_CHECK(
        destination_cpu ? is_vulkan_device(source.device())
                        : is_vulkan_device(destination.device()),
        "Vulkan copy requires a CPU and PrivateUse1/Vulkan device; destination ",
        destination.device(), ", source ", source.device());
    const c10::Device &vulkan_device =
        destination_cpu ? source.device() : destination.device();
    TORCH_CHECK(vulkan_device.index() == 0, "Vulkan copy ",
                direction(destination, source),
                " supports only Vulkan device index 0, got ", vulkan_device.index());
    if (!destination_cpu && !source_cpu) {
        TORCH_CHECK(destination.device() == source.device(),
                    "Vulkan copy requires tensors on the same Vulkan device");
        TORCH_CHECK(destination_layout.vulkan_layout.internal_overlap ==
                        pytorch_vulkan::VulkanOverlap::No,
                    "Vulkan copy destination has internal overlap");
        if (same_allocation(destination, source)) {
            const auto &dl = destination_layout.vulkan_layout;
            const auto &sl = source_layout.vulkan_layout;
            const bool ranges_overlap =
                dl.byte_offset < sl.byte_offset + sl.byte_range &&
                sl.byte_offset < dl.byte_offset + dl.byte_range;
            TORCH_CHECK(!ranges_overlap ||
                            same_layout(destination_layout, source_layout),
                        "Vulkan copy source and destination partially overlap");
        }
    }
    (void)checked_bytes(destination, source);
}

void record_vulkan_copy(const VulkanPlatform &platform, VkBuffer source,
                        VkBuffer destination, VkDeviceSize size,
                        VkDeviceSize source_offset, VkDeviceSize destination_offset) {
    VkCommandBuffer command_buffer = platform.execution_context().command_buffer();
    const VkMemoryBarrier before_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                      VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before_copy, 0, nullptr,
                         0, nullptr);
    const VkBufferCopy copy_region{source_offset, destination_offset, size};
    vkCmdCopyBuffer(command_buffer, source, destination, 1, &copy_region);
    platform.record_copy_command();
    const VkMemoryBarrier after_copy{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after_copy, 0,
                         nullptr, 0, nullptr);
}

void record_vulkan_copies(const VulkanPlatform &platform, VkBuffer source,
                          VkBuffer destination,
                          const std::vector<VkBufferCopy> &regions) {
    VkCommandBuffer command_buffer = platform.execution_context().command_buffer();
    const VkMemoryBarrier before_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                      VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before_copy, 0,
                         nullptr, 0, nullptr);
    vkCmdCopyBuffer(command_buffer, source, destination,
                    static_cast<uint32_t>(regions.size()), regions.data());
    platform.record_copy_command();
    const VkMemoryBarrier after_copy{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after_copy, 0,
                         nullptr, 0, nullptr);
}

} // namespace

namespace pytorch_vulkan {

at::Tensor vulkan_contiguous_copy(const at::Tensor &source) {
    const auto source_layout = inspect_vulkan_tensor_layout(source, "reshape source");
    TORCH_CHECK(source_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan reshape copy requires a non-overlapping source layout");

    at::Tensor destination =
        at::empty(source.sizes(), source.options().device(source.device()));
    const auto destination_layout =
        inspect_vulkan_tensor_layout(destination, "reshape destination");
    TORCH_CHECK(destination_layout.internal_overlap == VulkanOverlap::No &&
                    destination_layout.storage_offset == 0 &&
                    at::geometry_is_contiguous(destination_layout.sizes,
                                               destination_layout.strides),
                "Vulkan reshape copy requires a contiguous destination layout");
    TORCH_CHECK(source_layout.scalar_type == destination_layout.scalar_type,
                "Vulkan reshape copy requires matching dtypes");
    TORCH_CHECK(source_layout.numel == destination_layout.numel,
                "Vulkan reshape copy requires matching element counts");

    if (source_layout.numel == 0) {
        return destination;
    }

    const at::DataPtr &source_data = source.storage().data_ptr();
    const at::DataPtr &destination_data = destination.storage().data_ptr();
    validate_allocation(source_data,
                        source_layout.byte_offset + source_layout.byte_range,
                        "reshape source");
    validate_allocation(destination_data,
                        destination_layout.byte_offset + destination_layout.byte_range,
                        "reshape destination");
    const VulkanBuffer &source_buffer = allocation_buffer(source_data);
    const VulkanBuffer &destination_buffer = allocation_buffer(destination_data);
    const VulkanPlatform &platform = allocation_platform(source_data);
    TORCH_CHECK(&platform == &allocation_platform(destination_data),
                "Vulkan reshape copy requires tensors on the same Vulkan device");
    platform.record_transfer_operation();
    platform.record_vulkan_copy();

    const bool source_contiguous =
        at::geometry_is_contiguous(source_layout.sizes, source_layout.strides);
    if (source_contiguous) {
        if (platform.execution_context().recording()) {
            record_vulkan_copy(
                platform, source_buffer.buffer(), destination_buffer.buffer(),
                static_cast<VkDeviceSize>(source_layout.byte_range),
                source_layout.byte_offset, destination_layout.byte_offset);
        } else {
            platform.copy_buffer_sync(
                source_buffer.buffer(), destination_buffer.buffer(),
                static_cast<VkDeviceSize>(source_layout.byte_range),
                source_layout.byte_offset, destination_layout.byte_offset);
        }
        return destination;
    }

    std::vector<VkBufferCopy> regions;
    regions.reserve(static_cast<std::size_t>(source_layout.numel));
    for (int64_t index = 0; index < source_layout.numel; ++index) {
        const auto source_element = vulkan_storage_offset(source_layout, index);
        const auto destination_element =
            vulkan_storage_offset(destination_layout, index);
        const auto source_offset = static_cast<VkDeviceSize>(checked_byte_offset(
            source_element, source_layout.element_bytes, "reshape source"));
        const auto destination_offset = static_cast<VkDeviceSize>(
            checked_byte_offset(destination_element, destination_layout.element_bytes,
                                "reshape destination"));
        regions.push_back({source_offset, destination_offset,
                           static_cast<VkDeviceSize>(source_layout.element_bytes)});
    }
    if (platform.execution_context().recording()) {
        record_vulkan_copies(platform, source_buffer.buffer(), destination_buffer.buffer(),
                             regions);
    } else {
        std::vector<VulkanBufferCopy> copies;
        copies.reserve(regions.size());
        for (const auto &region : regions)
            copies.push_back({source_buffer.buffer(), destination_buffer.buffer(),
                              region.srcOffset, region.dstOffset, region.size});
        platform.copy_buffers_sync(copies);
    }
    return destination;
}

at::Tensor vulkan_stack_copy(at::TensorList tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "Vulkan stack requires at least one tensor");
    const auto &first = tensors[0];
    at::Tensor output = dim == 0
        ? at::empty({static_cast<int64_t>(tensors.size()), first.size(0), first.size(1)},
                    first.options())
        : at::empty({first.size(0), static_cast<int64_t>(tensors.size()), first.size(1)},
                    first.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "stack destination");
    const auto &first_data = first.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    VulkanBuffer &output_buffer = allocation_buffer(output_data);
    const VulkanPlatform &platform = allocation_platform(output_data);
    TORCH_CHECK(&platform == &allocation_platform(first_data),
                "Vulkan stack requires tensors on the same Vulkan device");

    std::vector<VulkanBufferCopy> copies;
    for (int64_t index = 0; index < static_cast<int64_t>(tensors.size()); ++index) {
        const auto &tensor = tensors[index];
        const auto &data = tensor.storage().data_ptr();
        const auto &buffer = allocation_buffer(data);
        const auto layout = inspect_vulkan_tensor_layout(tensor, "stack source");
        const auto element_bytes = static_cast<VkDeviceSize>(layout.element_bytes);
        if (dim == 0) {
            copies.push_back({buffer.buffer(), output_buffer.buffer(),
                              static_cast<VkDeviceSize>(layout.byte_offset),
                              static_cast<VkDeviceSize>(output_layout.byte_offset) +
                                  static_cast<VkDeviceSize>(index * first.numel()) *
                                      element_bytes,
                              static_cast<VkDeviceSize>(first.numel()) * element_bytes});
        } else {
            for (int64_t row = 0; row < first.size(0); ++row) {
                copies.push_back({
                    buffer.buffer(), output_buffer.buffer(),
                    static_cast<VkDeviceSize>(layout.byte_offset + row * first.size(1) *
                                              layout.element_bytes),
                    static_cast<VkDeviceSize>(output_layout.byte_offset +
                                              (row * tensors.size() + index) * first.size(1) *
                                                  layout.element_bytes),
                    static_cast<VkDeviceSize>(first.size(1)) * element_bytes});
            }
        }
    }
    if (copies.empty())
        return output;
    platform.record_vulkan_copy();
    platform.record_transfer_operation();
    if (platform.execution_context().recording()) {
        for (std::size_t index = 0; index < copies.size();) {
            const auto source = copies[index].source;
            const auto destination = copies[index].destination;
            std::vector<VkBufferCopy> regions;
            while (index < copies.size() && copies[index].source == source &&
                   copies[index].destination == destination) {
                const auto &copy = copies[index++];
                regions.push_back(
                    {copy.source_offset, copy.destination_offset, copy.size});
            }
            record_vulkan_copies(platform, source, destination, regions);
        }
    } else {
        platform.copy_buffers_sync(copies);
    }
    return output;
}

at::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,
                        bool non_blocking) {
    if (destination.device().is_cpu() && !source.device().is_cpu() &&
        source.scalar_type() == at::kFloat &&
        destination.scalar_type() == source.scalar_type() &&
        destination.is_contiguous() && source.is_contiguous()) {
        TORCH_CHECK(!non_blocking,
                    "Vulkan formatter presentation does not support non_blocking=True");
        return formatter_presentation_copy(destination, source);
    }
    TransferLayout destination_layout;
    TransferLayout source_layout;
    validate(destination, source, non_blocking, destination_layout, source_layout);
    if (!destination.device().is_cpu() || !source.device().is_cpu()) {
        ensure_process_local_vulkan();
    }
    const std::size_t bytes = checked_bytes(destination, source);
    if (bytes == 0) {
        return destination;
    }
    if (!destination.device().is_cpu() && !source.device().is_cpu() &&
        same_allocation(destination, source) &&
        same_layout(destination_layout, source_layout)) {
        return destination;
    }

    const bool cpu_to_vulkan = source.device().is_cpu();
    const bool vulkan_to_vulkan = !destination.device().is_cpu() && !cpu_to_vulkan;
    const bool label_copy = vulkan_to_vulkan && source.scalar_type() == at::kLong &&
                            destination.scalar_type() == at::kLong;
    if (label_copy) {
        TORCH_CHECK(is_validated_label_allocation(source.storage().data_ptr()),
                    "Vulkan int64 copy has no validated label provenance");
    }
    const auto mark_label_copy = [&] {
        if (label_copy)
            mark_validated_label_allocation(destination.storage().data_ptr());
    };
    const at::Tensor &vulkan_tensor = cpu_to_vulkan ? destination : source;
    const at::DataPtr &vulkan_data = vulkan_tensor.storage().data_ptr();
    VulkanBuffer &vulkan_buffer = allocation_buffer(vulkan_data);
    const VulkanPlatform &platform = allocation_platform(vulkan_data);
    platform.record_transfer_operation();
    const bool host_visible =
        (vulkan_buffer.memory_properties() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    std::unique_lock<std::mutex> transfer_lock;
    if (!vulkan_to_vulkan && !host_visible)
        transfer_lock = std::unique_lock<std::mutex>(platform.transfer_mutex());
    VulkanBuffer *staging = nullptr;
    if (!vulkan_to_vulkan && !host_visible) {
        staging = &platform.staging_buffer(bytes);
    }
    if (host_visible || vulkan_to_vulkan)
        platform.wait_for_transfer();
    if (vulkan_to_vulkan)
        platform.record_vulkan_copy();

    const auto *cpu_source = source.device().is_cpu()
                                 ? static_cast<const char *>(source.data_ptr())
                                 : nullptr;
    auto *cpu_destination = destination.device().is_cpu()
                                ? static_cast<char *>(destination.data_ptr())
                                : nullptr;
    const auto element_size =
        static_cast<VkDeviceSize>(destination_layout.element_bytes);
    if (!vulkan_to_vulkan)
        platform.record_explicit_transfer();

    const bool bulk_contiguous = source.scalar_type() == at::kFloat &&
                                 destination.scalar_type() == at::kFloat &&
                                 source.is_contiguous() && destination.is_contiguous();
    if (bulk_contiguous) {
        const VkDeviceSize source_offset = static_cast<VkDeviceSize>(
            checked_byte_offset(static_cast<int64_t>(logical_offset(source_layout, 0)),
                                source_layout.element_bytes, "source"));
        const VkDeviceSize destination_offset =
            static_cast<VkDeviceSize>(checked_byte_offset(
                static_cast<int64_t>(logical_offset(destination_layout, 0)),
                destination_layout.element_bytes, "destination"));
        if (cpu_to_vulkan) {
            if (host_visible) {
                vulkan_buffer.write(cpu_source, bytes, destination_offset);
            } else {
                staging->write(cpu_source, bytes);
                platform.copy_buffer_sync(staging->buffer(), vulkan_buffer.buffer(),
                                          bytes, 0, destination_offset);
            }
        } else if (vulkan_to_vulkan) {
            const auto &source_buffer = allocation_buffer(source.storage().data_ptr());
            auto &destination_buffer =
                allocation_buffer(destination.storage().data_ptr());
            if (platform.execution_context().recording()) {
                record_vulkan_copy(platform, source_buffer.buffer(),
                                   destination_buffer.buffer(), bytes, source_offset,
                                   destination_offset);
            } else {
                platform.copy_buffer_sync(source_buffer.buffer(),
                                          destination_buffer.buffer(), bytes,
                                          source_offset, destination_offset);
            }
        } else {
            if (host_visible) {
                vulkan_buffer.read(cpu_destination, bytes, source_offset);
            } else {
                platform.copy_buffer_sync(vulkan_buffer.buffer(), staging->buffer(),
                                          bytes, source_offset, 0);
                staging->read(cpu_destination, bytes);
            }
        }
        mark_label_copy();
        return destination;
    }
    if (vulkan_to_vulkan && platform.execution_context().recording()) {
        const auto &source_buffer = allocation_buffer(source.storage().data_ptr());
        auto &destination_buffer =
            allocation_buffer(destination.storage().data_ptr());
        std::vector<VkBufferCopy> regions;
        regions.reserve(static_cast<std::size_t>(destination_layout.numel));
        for (int64_t index = 0; index < destination_layout.numel; ++index) {
            const VkDeviceSize destination_offset =
                static_cast<VkDeviceSize>(checked_byte_offset(
                    static_cast<int64_t>(logical_offset(destination_layout, index)),
                    destination_layout.element_bytes, "destination"));
            const VkDeviceSize source_offset =
                static_cast<VkDeviceSize>(checked_byte_offset(
                    static_cast<int64_t>(logical_offset(source_layout, index)),
                    source_layout.element_bytes, "source"));
            regions.push_back({source_offset, destination_offset, element_size});
        }
        record_vulkan_copies(platform, source_buffer.buffer(),
                             destination_buffer.buffer(), regions);
        mark_label_copy();
        return destination;
    }
    for (int64_t index = 0; index < destination_layout.numel; ++index) {
        const VkDeviceSize destination_offset =
            static_cast<VkDeviceSize>(checked_byte_offset(
                static_cast<int64_t>(logical_offset(destination_layout, index)),
                destination_layout.element_bytes, "destination"));
        const VkDeviceSize source_offset =
            static_cast<VkDeviceSize>(checked_byte_offset(
                static_cast<int64_t>(logical_offset(source_layout, index)),
                source_layout.element_bytes, "source"));
        if (cpu_to_vulkan) {
            if (host_visible) {
                vulkan_buffer.write(cpu_source + source_offset, element_size,
                                    destination_offset);
            } else {
                staging->write(cpu_source + source_offset, element_size);
                platform.copy_buffer_sync(staging->buffer(), vulkan_buffer.buffer(),
                                          element_size, 0, destination_offset);
            }
        } else if (vulkan_to_vulkan) {
            const auto &source_buffer = allocation_buffer(source.storage().data_ptr());
            auto &destination_buffer =
                allocation_buffer(destination.storage().data_ptr());
            platform.copy_buffer_sync(source_buffer.buffer(),
                                      destination_buffer.buffer(), element_size,
                                      source_offset, destination_offset);
        } else {
            if (host_visible) {
                vulkan_buffer.read(cpu_destination + destination_offset, element_size,
                                   source_offset);
            } else {
                platform.copy_buffer_sync(vulkan_buffer.buffer(), staging->buffer(),
                                          element_size, source_offset, 0);
                staging->read(cpu_destination + destination_offset, element_size);
            }
        }
    }
    mark_label_copy();
    return destination;
}

at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source) {
    TORCH_CHECK(
        destination.device().is_cpu() && !source.device().is_cpu(),
        "Vulkan formatter presentation requires Vulkan source and CPU destination");
    TORCH_CHECK(destination.layout() == at::kStrided && source.layout() == at::kStrided,
                "Vulkan formatter presentation requires strided tensors");
    if (!destination.is_contiguous() || !source.is_contiguous())
        return copy_tensor(destination, source, false);
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
    platform.record_transfer_operation();
    const VkDeviceSize offset = static_cast<VkDeviceSize>(
        static_cast<uint64_t>(source.storage_offset()) *
        pytorch_vulkan::vulkan_storage_bytes(source.scalar_type()));
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    TORCH_CHECK(offset <= buffer.size() && size <= buffer.size() - offset,
                "Vulkan formatter presentation source range exceeds allocation");
    platform.record_explicit_transfer();

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
