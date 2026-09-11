#include "vulkan_allocator.h"

#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_platform.h"

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

struct VulkanAllocation {
    std::shared_ptr<VulkanPlatform> platform;
    std::unique_ptr<VulkanBuffer> buffer;
};

std::string allocation_context(c10::Device device, size_t nbytes) {
    std::ostringstream message;
    message << "requested " << nbytes << " bytes for PrivateUse1 device index "
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

at::DataPtr VulkanAllocator::allocate(size_t nbytes) {
    const c10::Device device = allocation_device_override.value_or(
        pytorch_vulkan::current_device());
    const std::string context = allocation_context(device, nbytes);
    if (nbytes == 0) {
        throw std::invalid_argument("Vulkan allocation rejected: " + context);
    }

    constexpr size_t kMaximumAllocationSize = size_t{1} << 40;
    if (nbytes > kMaximumAllocationSize) {
        throw std::invalid_argument("Vulkan allocation rejected: " + context +
                                    " exceeds the supported maximum");
    }

    auto allocation = std::make_unique<VulkanAllocation>();
    try {
        check_device(device);
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

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &vulkan_empty);
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);
