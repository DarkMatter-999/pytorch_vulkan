#include "vulkan_layout.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"

#include <c10/util/Exception.h>

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
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "Vulkan ", label, " must be float32");
    TORCH_CHECK(pytorch_vulkan::is_vulkan_allocation(tensor.storage().data_ptr()),
                "Vulkan ", label, " has an invalid allocation payload");
}

pytorch_vulkan::VulkanTensorLayout inspect_layout(const at::Tensor &storage_owner,
                                                   at::IntArrayRef sizes,
                                                   at::IntArrayRef strides,
                                                   int64_t storage_offset,
                                                   const char *label) {
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

    constexpr uint64_t kElementBytes = sizeof(float);
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

    return {storage_offset, static_cast<int64_t>(numel),
            static_cast<VkDeviceSize>(byte_offset), static_cast<VkDeviceSize>(byte_range),
            allocation_bytes};
}

} // namespace

namespace pytorch_vulkan {

VulkanTensorLayout inspect_vulkan_tensor_layout(const at::Tensor &tensor,
                                                 const char *label) {
    return inspect_layout(tensor, tensor.sizes(), tensor.strides(), tensor.storage_offset(),
                          label);
}

VulkanTensorLayout inspect_vulkan_view_layout(const at::Tensor &storage_owner,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides,
                                               int64_t storage_offset,
                                               const char *label) {
    return inspect_layout(storage_owner, sizes, strides, storage_offset, label);
}

} // namespace pytorch_vulkan
