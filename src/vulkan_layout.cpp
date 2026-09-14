#include "vulkan_layout.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"

#include <ATen/MemoryOverlap.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>

namespace {

uint64_t checked_add(uint64_t lhs, uint64_t rhs, const char *label) {
    TORCH_CHECK(rhs <= std::numeric_limits<uint64_t>::max() - lhs,
                "Vulkan ", label, " layout arithmetic overflow");
    return lhs + rhs;
}

uint64_t checked_multiply(uint64_t lhs, uint64_t rhs, const char *label) {
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs,
                "Vulkan ", label, " layout arithmetic overflow");
    return lhs * rhs;
}

void validate_storage_owner(const at::Tensor &tensor, const char *label) {
    const auto device = tensor.device();
    TORCH_CHECK((device.type() == c10::DeviceType::PrivateUse1 ||
                 device.type() == c10::DeviceType::Vulkan) &&
                    device.index() == 0,
                "Vulkan ", label, " must be a Vulkan device index 0 tensor");
    TORCH_CHECK(tensor.layout() == at::kStrided,
                "Vulkan ", label, " must be strided");
    (void)pytorch_vulkan::vulkan_storage_bytes(tensor.scalar_type());
    TORCH_CHECK(pytorch_vulkan::is_vulkan_allocation(tensor.storage().data_ptr()),
                "Vulkan ", label, " has an invalid allocation payload");
}

pytorch_vulkan::VulkanOverlap classify_strided_overlap(at::IntArrayRef sizes,
                                                        at::IntArrayRef strides,
                                                        const char *label) {
    // Validated nonnegative strides make this sorted-span check exact: a stride
    // below the addresses covered by earlier dimensions necessarily aliases.
    // Tensor descriptors use ATen's three-state classification below instead.
    std::vector<std::pair<uint64_t, uint64_t>> dimensions;
    dimensions.reserve(sizes.size());
    for (size_t dim = 0; dim < sizes.size(); ++dim) {
        if (sizes[dim] <= 1) {
            continue;
        }
        dimensions.emplace_back(static_cast<uint64_t>(strides[dim]),
                                static_cast<uint64_t>(sizes[dim]));
    }
    std::sort(dimensions.begin(), dimensions.end());
    uint64_t covered = 1;
    for (const auto &[stride, size] : dimensions) {
        if (stride < covered) {
            return pytorch_vulkan::VulkanOverlap::Yes;
        }
        covered = checked_multiply(covered, size, label);
    }
    return pytorch_vulkan::VulkanOverlap::No;
}

pytorch_vulkan::VulkanOverlap classify_tensor_overlap(const at::Tensor &tensor) {
    const auto overlap = at::has_internal_overlap(tensor);
    switch (overlap) {
        case at::MemOverlap::No:
            return pytorch_vulkan::VulkanOverlap::No;
        case at::MemOverlap::Yes:
            return pytorch_vulkan::VulkanOverlap::Yes;
        case at::MemOverlap::TooHard:
            return pytorch_vulkan::VulkanOverlap::TooHard;
    }
    TORCH_CHECK(false, "Vulkan tensor overlap classification is invalid");
}

pytorch_vulkan::VulkanTensorLayout inspect_layout(const at::Tensor &storage_owner,
                                                   at::IntArrayRef sizes,
                                                   at::IntArrayRef strides,
                                                   int64_t storage_offset,
                                                   const char *label,
                                                   const at::Tensor *overlap_tensor) {
    validate_storage_owner(storage_owner, label);
    TORCH_CHECK(sizes.size() == strides.size(),
                "Vulkan ", label, " sizes and strides must have matching ranks");
    TORCH_CHECK(storage_offset >= 0,
                "Vulkan ", label, " has a negative storage offset");

    uint64_t numel = 1;
    bool empty = false;
    for (const int64_t size : sizes) {
        TORCH_CHECK(size >= 0, "Vulkan ", label, " has a negative size");
        if (size == 0) {
            empty = true;
        }
        numel = checked_multiply(numel, static_cast<uint64_t>(size), label);
    }
    TORCH_CHECK(numel <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Vulkan ", label, " numel exceeds int64 range");

    uint64_t max_element = static_cast<uint64_t>(storage_offset);
    if (!empty) {
        for (size_t dim = 0; dim < sizes.size(); ++dim) {
            TORCH_CHECK(strides[dim] >= 0,
                        "Vulkan ", label, " has a negative stride");
            const uint64_t reach = checked_multiply(
                static_cast<uint64_t>(sizes[dim] - 1),
                static_cast<uint64_t>(strides[dim]), label);
            max_element = checked_add(max_element, reach, label);
        }
    } else {
        for (const int64_t stride : strides) {
            TORCH_CHECK(stride >= 0, "Vulkan ", label, " has a negative stride");
        }
    }

    const uint64_t kElementBytes =
        static_cast<uint64_t>(pytorch_vulkan::vulkan_storage_bytes(storage_owner.scalar_type()));
    const uint64_t byte_offset = checked_multiply(
        static_cast<uint64_t>(storage_offset), kElementBytes, label);
    const uint64_t byte_range = empty
        ? 0
        : checked_multiply(
              checked_add(max_element - static_cast<uint64_t>(storage_offset), 1, label),
              kElementBytes, label);
    const uint64_t end_byte = checked_add(byte_offset, byte_range, label);
    TORCH_CHECK(byte_offset <= std::numeric_limits<VkDeviceSize>::max() &&
                    byte_range <= std::numeric_limits<VkDeviceSize>::max() &&
                    end_byte <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan ", label, " layout exceeds VkDeviceSize range");

    const at::DataPtr &data = storage_owner.storage().data_ptr();
    VkDeviceSize allocation_bytes = 0;
    if (!empty) {
        // Validate the allocator's platform and buffer payload before borrowing it.
        pytorch_vulkan::validate_allocation(data, 0, label);
        allocation_bytes = pytorch_vulkan::allocation_buffer(data).size();
        TORCH_CHECK(end_byte <= allocation_bytes,
                    "Vulkan ", label, " reaches outside its Vulkan allocation");
    }

    const auto overlap_classification = overlap_tensor
        ? classify_tensor_overlap(*overlap_tensor)
        : classify_strided_overlap(sizes, strides, label);
    return {static_cast<int64_t>(sizes.size()), sizes.vec(), strides.vec(),
            static_cast<int>(storage_owner.scalar_type()), kElementBytes, storage_offset,
            static_cast<int64_t>(numel),
            static_cast<VkDeviceSize>(byte_offset), static_cast<VkDeviceSize>(byte_range),
            allocation_bytes, overlap_classification};
}

} // namespace

namespace pytorch_vulkan {

VulkanTensorLayout inspect_vulkan_tensor_layout(const at::Tensor &tensor,
                                                 const char *label) {
    return inspect_layout(tensor, tensor.sizes(), tensor.strides(), tensor.storage_offset(),
                           label, &tensor);
}

VulkanTensorLayout inspect_vulkan_view_layout(const at::Tensor &storage_owner,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides,
                                               int64_t storage_offset,
                                               const char *label) {
    return inspect_layout(storage_owner, sizes, strides, storage_offset, label, nullptr);
}

int64_t vulkan_storage_offset(const VulkanTensorLayout &layout,
                              at::IntArrayRef coordinate) {
    TORCH_CHECK(coordinate.size() == static_cast<size_t>(layout.rank),
                "Vulkan layout coordinate rank does not match descriptor rank");
    uint64_t element_offset = static_cast<uint64_t>(layout.storage_offset);
    for (size_t dim = 0; dim < coordinate.size(); ++dim) {
        TORCH_CHECK(coordinate[dim] >= 0 && coordinate[dim] < layout.sizes[dim],
                    "Vulkan layout coordinate is out of range");
        const uint64_t contribution = checked_multiply(
            static_cast<uint64_t>(coordinate[dim]),
            static_cast<uint64_t>(layout.strides[dim]), "address");
        element_offset = checked_add(element_offset, contribution, "address");
    }
    TORCH_CHECK(element_offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Vulkan layout address exceeds int64 range");
    return static_cast<int64_t>(element_offset);
}

int64_t vulkan_storage_offset(const VulkanTensorLayout &layout, int64_t linear_index) {
    TORCH_CHECK(linear_index >= 0 && linear_index < layout.numel,
                "Vulkan layout linear index is out of range");
    if (layout.rank == 0) {
        return layout.storage_offset;
    }
    std::vector<int64_t> coordinate(static_cast<size_t>(layout.rank));
    uint64_t remaining = static_cast<uint64_t>(linear_index);
    for (int64_t dim = layout.rank - 1; dim >= 0; --dim) {
        const uint64_t size = static_cast<uint64_t>(layout.sizes[dim]);
        coordinate[dim] = static_cast<int64_t>(remaining % size);
        remaining /= size;
    }
    return vulkan_storage_offset(layout, coordinate);
}

} // namespace pytorch_vulkan
