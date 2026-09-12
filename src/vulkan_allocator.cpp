#include "vulkan_allocator.h"

#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

std::string device_type_name(c10::DeviceType type) {
    if (type == c10::DeviceType::PrivateUse1) {
        return "PrivateUse1";
    }
    if (type == c10::DeviceType::Vulkan) {
        return "Vulkan";
    }
    return c10::DeviceTypeName(type).c_str();
}

struct VulkanAllocation {
    std::shared_ptr<VulkanPlatform> platform;
    std::unique_ptr<VulkanBuffer> buffer;
};

std::string allocation_context(c10::Device device, size_t nbytes) {
    std::ostringstream message;
    message << "requested " << nbytes << " bytes for "
            << device_type_name(device.type())
            << " device index "
            << static_cast<int>(device.index());
    return message.str();
}

void check_device(c10::Device device) {
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 ||
                    device.type() == c10::DeviceType::Vulkan,
                "Vulkan allocator received device of type ", device.type());
    TORCH_CHECK(device.index() == 0,
                "Vulkan backend supports only device index 0, got ", device.index());

}

void delete_allocation(void *context) noexcept {
    delete static_cast<VulkanAllocation *>(context);
}

VulkanAllocator allocator;
thread_local c10::optional<c10::Device> allocation_device_override;

struct AllocationDeviceOverrideGuard {
    const c10::optional<c10::Device> previous;

    explicit AllocationDeviceOverrideGuard(c10::Device device)
        : previous(allocation_device_override) {
        allocation_device_override = device;
    }

    ~AllocationDeviceOverrideGuard() { allocation_device_override = previous; }
};

} // namespace

namespace pytorch_vulkan {

void validate_allocation(const at::DataPtr &data, VkDeviceSize required_bytes,
                         const char *label) {
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "Vulkan add ", label, " has an invalid allocation payload");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->platform != nullptr && allocation->buffer != nullptr,
                "Vulkan add ", label, " allocation payload is incomplete");
    TORCH_CHECK(required_bytes <= allocation->buffer->size(),
                "Vulkan add ", label, " allocation is undersized (required ",
                required_bytes, " bytes, allocation is ", allocation->buffer->size(),
                " bytes)");
}

bool is_vulkan_allocation(const at::DataPtr &data) {
    return data.get_context() != nullptr && data.get_deleter() == &delete_allocation;
}

VulkanBuffer &allocation_buffer(const at::DataPtr &data) {
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->buffer != nullptr,
                "Vulkan allocation payload has no buffer");
    return *allocation->buffer;
}

const VulkanPlatform &allocation_platform(const at::DataPtr &data) {
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->platform != nullptr,
                "Vulkan allocation payload has no platform");
    return *allocation->platform;
}

} // namespace pytorch_vulkan

at::DataPtr VulkanAllocator::allocate(size_t nbytes) {
    const c10::Device device = allocation_device_override.value_or(
        pytorch_vulkan::current_device());
    const std::string context = allocation_context(device, nbytes);
    constexpr size_t kMaximumAllocationSize = size_t{1} << 40;
    if (nbytes > kMaximumAllocationSize) {
        throw std::invalid_argument("Vulkan allocation rejected: " + context +
                                    " exceeds the supported maximum");
    }

    auto allocation = std::make_unique<VulkanAllocation>();
    try {
        check_device(device);
        if (nbytes == 0) {
            VulkanAllocation *payload = allocation.release();
            return at::DataPtr(payload, payload, delete_allocation, device);
        }
        allocation->platform = pytorch_vulkan::platform();
        allocation->buffer = std::make_unique<VulkanBuffer>(*allocation->platform, nbytes);
    } catch (const VulkanUnavailable &error) {
        throw VulkanUnavailable("Vulkan allocation failed (" + context + "): " +
                                error.what());
    } catch (const std::exception &error) {
        throw std::runtime_error("Vulkan allocation failed (" + context + "): " +
                                 error.what());
    }
    VulkanAllocation *payload = allocation.release();
    return at::DataPtr(payload, payload, delete_allocation, device);
}

void VulkanAllocator::copy_data(void *, const void *, std::size_t) const {
    throw std::runtime_error("Vulkan allocator opaque-pointer copies are unsupported");
}

VulkanAllocator *vulkan_allocator_instance() { return &allocator; }

namespace {

at::Tensor vulkan_empty(c10::SymIntArrayRef size,
                        c10::optional<at::ScalarType> dtype,
                        c10::optional<at::Layout> layout,
                        c10::optional<c10::Device> device,
                        c10::optional<bool> pin_memory,
                        c10::optional<c10::MemoryFormat> memory_format) {
    TORCH_CHECK(!layout || *layout == at::Layout::Strided,
                "Vulkan allocator supports only strided tensors");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "Vulkan allocator does not support pinned memory");
    const c10::Device requested = device.value_or(pytorch_vulkan::current_device());
    const c10::Device target(
        requested.type(), requested.index() == c10::DeviceIndex(-1)
                              ? c10::DeviceIndex(0)
                              : requested.index());
    check_device(target);
    AllocationDeviceOverrideGuard override(target);
    return at::detail::empty_generic_symint(
        size, vulkan_allocator_instance(),
        c10::DispatchKeySet(target.type() == c10::DeviceType::Vulkan
                                ? c10::DispatchKey::Vulkan
                                : c10::DispatchKey::PrivateUse1),
        dtype.value_or(c10::get_default_dtype_as_scalartype()), memory_format);
}

at::Tensor vulkan_empty_strided(c10::SymIntArrayRef size,
                                c10::SymIntArrayRef stride,
                                c10::optional<at::ScalarType> dtype,
                                c10::optional<c10::Layout> layout,
                                c10::optional<c10::Device> device,
                                c10::optional<bool> pin_memory) {
    TORCH_CHECK(!layout || *layout == at::Layout::Strided,
                "Vulkan allocator supports only strided tensors");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "Vulkan allocator does not support pinned memory");
    const c10::Device requested = device.value_or(pytorch_vulkan::current_device());
    const c10::Device target(
        requested.type(), requested.index() == c10::DeviceIndex(-1)
                              ? c10::DeviceIndex(0)
                              : requested.index());
    check_device(target);
    AllocationDeviceOverrideGuard override(target);
    return at::detail::empty_strided_symint_generic(
        size, stride, vulkan_allocator_instance(),
        c10::DispatchKeySet(target.type() == c10::DeviceType::Vulkan
                                ? c10::DispatchKey::Vulkan
                                : c10::DispatchKey::PrivateUse1),
        dtype.value_or(c10::get_default_dtype_as_scalartype()));
}

at::Tensor vulkan_copy_from(const at::Tensor &source, const at::Tensor &destination,
                            bool non_blocking) {
    at::Tensor result = destination;
    pytorch_vulkan::copy_tensor(result, source, non_blocking);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &vulkan_empty);
    m.impl("empty_strided", &vulkan_empty_strided);
    m.impl("_copy_from", &vulkan_copy_from);
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);
