#include "vulkan_allocator.h"

#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>
#include <memory>
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
    bool validated_label = false;
};

std::string allocation_context(c10::Device device, size_t nbytes) {
    std::ostringstream message;
    message << "requested " << nbytes << " bytes for "
            << device_type_name(device.type()) << " device index "
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
    auto *allocation = static_cast<VulkanAllocation *>(context);
    if (allocation->platform != nullptr) {
        try {
            auto &execution = allocation->platform->execution_context();
            if (execution.retain_until_completion([allocation] { delete allocation; }))
                return;
        } catch (...) {
            try {
                // If retention failed while work is pending, leaking is safer
                // than destroying a resource referenced by submitted commands.
                if (allocation->platform->execution_context().pending_count() != 0)
                    return;
            } catch (...) {
                return;
            }
        }
    }
    delete allocation;
}

VulkanAllocator allocator;
thread_local c10::optional<c10::Device> allocation_device_override;
thread_local bool allow_index_output_allocation = false;
thread_local bool allow_label_allocation = false;

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

std::size_t vulkan_storage_bytes(c10::ScalarType dtype) {
    if (dtype == at::kLong) {
        return sizeof(int64_t);
    }
    if (dtype == at::kDouble) {
        TORCH_CHECK(formatter_double_supported(),
                    "Vulkan formatter Double support requires the shaderFloat64 device "
                    "feature");
        return sizeof(double);
    }
    validate_vulkan_dtype(dtype, "storage");
    return dtype == at::kFloat ? sizeof(float) : sizeof(bool);
}

VulkanIndexOutputAllocationGuard::VulkanIndexOutputAllocationGuard() {
    TORCH_CHECK(!allow_index_output_allocation,
                "Vulkan index output allocation guard cannot be nested");
    allow_index_output_allocation = true;
}

VulkanIndexOutputAllocationGuard::~VulkanIndexOutputAllocationGuard() {
    allow_index_output_allocation = false;
}

VulkanLabelAllocationGuard::VulkanLabelAllocationGuard() {
    TORCH_CHECK(!allow_label_allocation,
                "Vulkan label allocation guard cannot be nested");
    allow_label_allocation = true;
}

VulkanLabelAllocationGuard::~VulkanLabelAllocationGuard() {
    allow_label_allocation = false;
}

void validate_allocation(const at::DataPtr &data, VkDeviceSize required_bytes,
                         const char *label) {
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "Vulkan add ", label, " has an invalid allocation payload");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->platform != nullptr && allocation->buffer != nullptr,
                "Vulkan add ", label, " allocation payload is incomplete");
    TORCH_CHECK(required_bytes <= allocation->buffer->size(), "Vulkan add ", label,
                " allocation is undersized (required ", required_bytes,
                " bytes, allocation is ", allocation->buffer->size(), " bytes)");
}

bool is_vulkan_allocation(const at::DataPtr &data) {
    return data.get_context() != nullptr && data.get_deleter() == &delete_allocation;
}

VulkanBuffer &allocation_buffer(const at::DataPtr &data) {
    ensure_process_local_vulkan();
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->buffer != nullptr,
                "Vulkan allocation payload has no buffer");
    return *allocation->buffer;
}

const VulkanPlatform &allocation_platform(const at::DataPtr &data) {
    ensure_process_local_vulkan();
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = data.cast_context<VulkanAllocation>(&delete_allocation);
    TORCH_CHECK(allocation->platform != nullptr,
                "Vulkan allocation payload has no platform");
    return *allocation->platform;
}

bool is_validated_label_allocation(const at::DataPtr &data) {
    if (!is_vulkan_allocation(data))
        return false;
    return data.cast_context<VulkanAllocation>(&delete_allocation)->validated_label;
}

void mark_validated_label_allocation(const at::DataPtr &data) {
    TORCH_CHECK(is_vulkan_allocation(data),
                "cannot mark a non-Vulkan allocation as a validated label");
    data.cast_context<VulkanAllocation>(&delete_allocation)->validated_label = true;
}

} // namespace pytorch_vulkan

at::DataPtr VulkanAllocator::allocate(size_t nbytes) {
    const c10::Device device =
        allocation_device_override.value_or(pytorch_vulkan::current_device());
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
        allocation->buffer =
            std::make_unique<VulkanBuffer>(*allocation->platform, nbytes);
        allocation->validated_label = allow_label_allocation;
    } catch (const VulkanUnavailable &error) {
        throw VulkanUnavailable("Vulkan allocation failed (" + context +
                                "): " + error.what());
    } catch (const std::exception &error) {
        throw std::runtime_error("Vulkan allocation failed (" + context +
                                 "): " + error.what());
    }
    VulkanAllocation *payload = allocation.release();
    return at::DataPtr(payload, payload, delete_allocation, device);
}

void VulkanAllocator::copy_data(void *, const void *, std::size_t) const {
    throw std::runtime_error("Vulkan allocator opaque-pointer copies are unsupported");
}

VulkanAllocator *vulkan_allocator_instance() { return &allocator; }

namespace {

at::Tensor vulkan_empty(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype,
                        c10::optional<at::Layout> layout,
                        c10::optional<c10::Device> device,
                        c10::optional<bool> pin_memory,
                        c10::optional<c10::MemoryFormat> memory_format) {
    TORCH_CHECK(!layout || *layout == at::Layout::Strided,
                "Vulkan allocator supports only strided tensors");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "Vulkan allocator does not support pinned memory");
    const c10::Device requested = device.value_or(pytorch_vulkan::current_device());
    const c10::Device target(requested.type(), requested.index() == c10::DeviceIndex(-1)
                                                   ? c10::DeviceIndex(0)
                                                   : requested.index());
    check_device(target);
    const auto allocation_dtype =
        dtype.value_or(c10::get_default_dtype_as_scalartype());
    TORCH_CHECK((allocation_dtype == at::kLong &&
                 (allow_index_output_allocation || allow_label_allocation)) ||
                    allocation_dtype != at::kLong,
                "Vulkan allocation dtype int64 is limited to indices and labels; "
                "supports only float32 and bool tensors, got Long");
    if (allocation_dtype != at::kLong)
        pytorch_vulkan::validate_vulkan_dtype(allocation_dtype, "allocation");
    AllocationDeviceOverrideGuard override(target);
    return at::detail::empty_generic_symint(
        size, vulkan_allocator_instance(),
        c10::DispatchKeySet(target.type() == c10::DeviceType::Vulkan
                                ? c10::DispatchKey::Vulkan
                                : c10::DispatchKey::PrivateUse1),
        allocation_dtype, memory_format);
}

at::Tensor vulkan_empty_strided(c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
                                c10::optional<at::ScalarType> dtype,
                                c10::optional<c10::Layout> layout,
                                c10::optional<c10::Device> device,
                                c10::optional<bool> pin_memory) {
    TORCH_CHECK(!layout || *layout == at::Layout::Strided,
                "Vulkan allocator supports only strided tensors");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "Vulkan allocator does not support pinned memory");
    const c10::Device requested = device.value_or(pytorch_vulkan::current_device());
    const c10::Device target(requested.type(), requested.index() == c10::DeviceIndex(-1)
                                                   ? c10::DeviceIndex(0)
                                                   : requested.index());
    check_device(target);
    const auto allocation_dtype =
        dtype.value_or(c10::get_default_dtype_as_scalartype());
    TORCH_CHECK((allocation_dtype == at::kLong &&
                 (allow_index_output_allocation || allow_label_allocation)) ||
                    allocation_dtype != at::kLong,
                "Vulkan allocation dtype int64 is limited to indices and labels; "
                "supports only float32 and bool tensors, got Long");
    if (allocation_dtype != at::kLong)
        pytorch_vulkan::validate_vulkan_dtype(allocation_dtype, "allocation");
    AllocationDeviceOverrideGuard override(target);
    return at::detail::empty_strided_symint_generic(
        size, stride, vulkan_allocator_instance(),
        c10::DispatchKeySet(target.type() == c10::DeviceType::Vulkan
                                ? c10::DispatchKey::Vulkan
                                : c10::DispatchKey::PrivateUse1),
        allocation_dtype);
}

at::Tensor vulkan_copy_from(const at::Tensor &source, const at::Tensor &destination,
                            bool non_blocking) {
    at::Tensor result = destination;
    if (destination.device().is_cpu() && !source.device().is_cpu() &&
        source.scalar_type() == at::kFloat &&
        destination.scalar_type() == source.scalar_type()) {
        TORCH_CHECK(!non_blocking,
                    "Vulkan formatter presentation does not support non_blocking=True");
        pytorch_vulkan::formatter_presentation_copy(result, source);
    } else {
        pytorch_vulkan::copy_tensor(result, source, non_blocking);
    }
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &vulkan_empty);
    m.impl("empty_strided", &vulkan_empty_strided);
    m.impl("_copy_from", &vulkan_copy_from);
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);
