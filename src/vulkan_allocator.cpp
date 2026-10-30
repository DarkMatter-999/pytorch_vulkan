#include "vulkan_allocator.h"

#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>
#include <algorithm>
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
    bool counted = false;
};

struct VulkanAllocationHandle {
    std::shared_ptr<VulkanAllocation> core;
};

void destroy_allocation(VulkanAllocation *allocation) noexcept {
    if (allocation->platform) {
        auto platform = allocation->platform;
        pytorch_vulkan::VulkanCleanupScope cleanup(platform->runtime_lifecycle());
        if (allocation->counted) platform->record_allocation_destroyed();
        delete allocation;
        return;
    }
    if (allocation->platform != nullptr && allocation->counted) {
        allocation->platform->record_allocation_destroyed();
        allocation->counted = false;
    }
    delete allocation;
}

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

void delete_allocation_core(VulkanAllocation *allocation) noexcept {
    if (pytorch_vulkan::inherited_fork_state()) return;
    if (allocation->platform != nullptr) {
        auto platform = allocation->platform;
        pytorch_vulkan::VulkanCleanupScope cleanup(platform->runtime_lifecycle());
        try {
            if (allocation->platform->device_lost())
                return;
            auto &execution = allocation->platform->execution_context();
            if (execution.retain_until_completion(
                    [allocation] { destroy_allocation(allocation); }))
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
    destroy_allocation(allocation);
}

void delete_allocation(void *context) noexcept {
    delete static_cast<VulkanAllocationHandle *>(context);
}

VulkanAllocation *allocation_core(const at::DataPtr &data) {
    auto *handle = data.cast_context<VulkanAllocationHandle>(&delete_allocation);
    TORCH_CHECK(handle->core != nullptr, "Vulkan allocation payload has no core");
    return handle->core.get();
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
    auto *allocation = allocation_core(data);
    TORCH_CHECK(allocation->platform != nullptr && allocation->buffer != nullptr,
                "Vulkan add ", label, " allocation payload is incomplete");
    TORCH_CHECK(required_bytes <= allocation->buffer->size(), "Vulkan add ", label,
                " allocation is undersized (required ", required_bytes,
                " bytes, allocation is ", allocation->buffer->size(), " bytes)");
}

bool is_vulkan_allocation(const at::DataPtr &data) {
    return data.get_context() != nullptr && data.get_deleter() == &delete_allocation;
}

std::shared_ptr<void> allocation_lease(const at::DataPtr &data) {
    TORCH_CHECK(is_vulkan_allocation(data), "DataPtr is not a Vulkan allocation");
    auto *handle = data.cast_context<VulkanAllocationHandle>(&delete_allocation);
    TORCH_CHECK(handle->core != nullptr, "Vulkan allocation payload has no core");
    return std::shared_ptr<void>(handle->core, handle->core.get());
}

VulkanBuffer &allocation_buffer(const at::DataPtr &data) {
    ensure_process_local_vulkan();
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = allocation_core(data);
    TORCH_CHECK(allocation->buffer != nullptr,
                "Vulkan allocation payload has no buffer");
    return *allocation->buffer;
}

const VulkanPlatform &allocation_platform(const at::DataPtr &data) {
    ensure_process_local_vulkan();
    TORCH_CHECK(data.get_context() != nullptr &&
                    data.get_deleter() == &delete_allocation,
                "DataPtr context is not a Vulkan allocation");
    auto *allocation = allocation_core(data);
    TORCH_CHECK(allocation->platform != nullptr,
                "Vulkan allocation payload has no platform");
    return *allocation->platform;
}
VulkanRuntimeLifecycle* allocation_runtime(const at::DataPtr &data) {
    if (!is_vulkan_allocation(data)) return nullptr;
    ensure_process_local_vulkan();
    auto* allocation = allocation_core(data);
    return allocation->platform ? &allocation->platform->runtime_lifecycle() : nullptr;
}

bool is_validated_label_allocation(const at::DataPtr &data) {
    if (!is_vulkan_allocation(data))
        return false;
    return allocation_core(data)->validated_label;
}

void mark_validated_label_allocation(const at::DataPtr &data) {
    TORCH_CHECK(is_vulkan_allocation(data),
                "cannot mark a non-Vulkan allocation as a validated label");
    allocation_core(data)->validated_label = true;
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

    try { check_device(device); }
    catch (const std::exception &error) {
        throw std::runtime_error("Vulkan allocation failed (" + context + "): " + error.what());
    }
    if (nbytes == 0) {
        auto allocation = std::make_unique<VulkanAllocation>();
        try {
            auto core = std::shared_ptr<VulkanAllocation>(allocation.release(), delete_allocation_core);
            auto *handle = new VulkanAllocationHandle{std::move(core)};
            return at::DataPtr(handle, handle, delete_allocation, device);
        } catch (const std::exception &error) {
            throw std::runtime_error("Vulkan allocation failed (" + context + "): " + error.what());
        }
    }
    std::shared_ptr<VulkanPlatform> platform;
    try { platform = pytorch_vulkan::platform(); }
    catch (const VulkanUnavailable &error) {
        throw VulkanUnavailable("Vulkan allocation failed (" + context + "): " + error.what());
    } catch (const std::exception &error) {
        throw std::runtime_error("Vulkan allocation failed (" + context + "): " + error.what());
    }
    return pytorch_vulkan::with_vulkan_operation(platform->runtime_lifecycle(), [&]() -> at::DataPtr {
    auto allocation = std::make_unique<VulkanAllocation>();
    try {
        allocation->platform = platform;
        allocation->platform->throw_if_device_lost();
        allocation->buffer =
            std::make_unique<VulkanBuffer>(*allocation->platform, nbytes);
        allocation->platform->record_allocation_created();
        allocation->counted = true;
        allocation->validated_label = allow_label_allocation;
    } catch (const VulkanUnavailable &error) {
        throw VulkanUnavailable("Vulkan allocation failed (" + context +
                                "): " + error.what());
    } catch (const std::exception &error) {
        throw std::runtime_error("Vulkan allocation failed (" + context +
                                 "): " + error.what());
    }
    auto core = std::shared_ptr<VulkanAllocation>(allocation.release(), delete_allocation_core);
    auto *handle = new VulkanAllocationHandle{std::move(core)};
    return at::DataPtr(handle, handle, delete_allocation, device);
    });
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
        // F32 non_blocking readback routes through copy_tensor (pinned destinations
        // enqueue; pageable ones complete before return). Blocking keeps the
        // formatter presentation path unchanged.
        if (non_blocking)
            pytorch_vulkan::copy_tensor(result, source, true);
        else
            pytorch_vulkan::formatter_presentation_copy(result, source);
    } else {
        pytorch_vulkan::copy_tensor(result, source, non_blocking);
    }
    return result;
}

at::Tensor vulkan_copy_from_and_resize(const at::Tensor &source,
                                       const at::Tensor &destination) {
    TORCH_CHECK(source.layout() == at::kStrided && destination.layout() == at::kStrided,
                "Vulkan _copy_from_and_resize supports only strided tensors");
    TORCH_CHECK(source.scalar_type() == at::kFloat &&
                    destination.scalar_type() == at::kFloat,
                "Vulkan _copy_from_and_resize currently supports only float32 tensors");

    at::Tensor result = destination;
    // Match the pinned MPS implementation: an empty source and exact self-copy
    // are no-ops; a zero-element destination is resized before the ordinary
    // copy route handles strides, offsets, and any broadcast validation.
    if (source.numel() == 0 || source.is_same(destination))
        return result;
    const auto& device_tensor = source.device().is_cpu() ? destination : source;
    return pytorch_vulkan::with_vulkan_operation(pytorch_vulkan::allocation_runtime(device_tensor.storage().data_ptr()), [&]() -> at::Tensor {
    if (destination.numel() == 0)
        result.resize_as_(source);
    result.copy_(source, false);
    return result;
    });
}

const at::Tensor &vulkan_resize_(
    const at::Tensor &self, c10::SymIntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(self.layout() == at::kStrided,
                "Vulkan resize_ supports only strided tensors");
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "Vulkan resize_ currently supports only float32 tensors");
    auto *impl = self.unsafeGetTensorImpl();
    std::vector<int64_t> concrete_size;
    concrete_size.reserve(size.size());
    for (const auto &dim : size) {
        const auto value = dim.expect_int();
        TORCH_CHECK(value >= 0, "Trying to create tensor with negative dimension ", value);
        concrete_size.push_back(value);
    }

    // Match ResizeCommon.h's pinned named route: same-shape, no-format is a
    // no-op; named shape changes and explicit formats reject before mutation.
    if (self.has_names())
        return at::native::resize_named_tensor_(self, concrete_size, memory_format);

    const auto old_sizes = impl->sym_sizes();
    if (old_sizes.equals(size) && !memory_format.has_value())
        return self;

    TORCH_CHECK(!memory_format || *memory_format != at::MemoryFormat::Preserve,
                "Unsupported memory format", *memory_format);
    if (memory_format == at::MemoryFormat::ChannelsLast)
        TORCH_CHECK(size.size() == 4,
                    "channels_last memory format requires a 4D tensor");
    if (memory_format == at::MemoryFormat::ChannelsLast3d)
        TORCH_CHECK(size.size() == 5,
                    "channels_last_3d memory format requires a 5D tensor");

    const auto storage_offset = impl->sym_storage_offset().expect_int();
    TORCH_CHECK(storage_offset >= 0,
                "Storage size calculation overflowed with sizes=", concrete_size);
    const auto target_bytes = at::detail::computeStorageNbytesContiguous(
        concrete_size, self.itemsize(), static_cast<size_t>(storage_offset));
    const auto old_storage_nbytes = impl->unsafe_storage()
        ? impl->unsafe_storage().sym_nbytes().maybe_as_int().value_or(-1)
        : 0;

    // The stock CPU implementation explicitly handles CPU and Meta, and asserts
    // for PrivateUse1. Keep its metadata/byte-count rules, but route byte growth
    // through the registered PrivateUse1 storage hook before touching metadata.
    const bool new_shape_is_empty = std::any_of(
        concrete_size.begin(), concrete_size.end(), [](int64_t dim) { return dim == 0; });
    // The hook owns copy/old-DataPtr retirement; this scope also owns metadata
    // mutation and the deterministic-fill tail.
    return pytorch_vulkan::with_vulkan_operation(
        pytorch_vulkan::allocation_runtime(self.storage().data_ptr()), [&]() -> const at::Tensor& {
    if (!new_shape_is_empty &&
        static_cast<uint64_t>(target_bytes) > self.storage().nbytes()) {
        at::GetPrivateUse1HooksInterface()->resizePrivateUse1Bytes(
            self.storage(), static_cast<size_t>(target_bytes));
    }

    impl->set_sizes_contiguous(concrete_size);
    if (memory_format.has_value())
        impl->empty_tensor_restride(*memory_format);

    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() &&
                     at::globalContext().deterministicFillUninitializedMemory() &&
                     old_storage_nbytes >= 0)) {
        const int64_t itemsize = static_cast<int64_t>(self.itemsize());
        const int64_t old_storage_numel = old_storage_nbytes / itemsize;
        const int64_t new_storage_numel = self.storage().nbytes() / itemsize;
        if (new_storage_numel > old_storage_numel) {
            auto tail = self.as_strided(
                {new_storage_numel - old_storage_numel}, {1}, old_storage_numel);
            at::native::fill_empty_deterministic_(tail);
        }
    }
    return self;
    });
}

void check_set_f32(const at::Tensor &self) {
    TORCH_CHECK(self.layout() == at::kStrided,
                "Vulkan set_ supports only strided tensors");
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "Vulkan set_ currently supports only float32 tensors");
}

at::Tensor &vulkan_set_storage_(at::Tensor &self, c10::Storage storage,
                                int64_t storage_offset,
                                at::IntArrayRef size,
                                at::IntArrayRef stride) {
    check_set_f32(self);
    TORCH_CHECK(stride.empty() || size.size() == stride.size(),
                "unequal size length (", size.size(), ") and stride length (",
                stride.size(), ")");
    TORCH_CHECK(storage, "Vulkan set_ requires a defined source storage");
    TORCH_CHECK(self.device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", self.device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
    for (const auto dim : size)
        TORCH_CHECK(dim >= 0, "Trying to create tensor with negative dimension ", dim);
    for (const auto dim_stride : stride)
        TORCH_CHECK(dim_stride >= 0, "set_: negative strides are not supported");

    // Match native set_storage_cpu_: empty stride means contiguous geometry.
    std::vector<int64_t> contiguous_strides;
    if (stride.empty()) {
        contiguous_strides.resize(size.size());
        if (!size.empty()) {
            contiguous_strides.back() = 1;
            for (int64_t i = static_cast<int64_t>(size.size()) - 2; i >= 0; --i) {
                const auto next = contiguous_strides[static_cast<size_t>(i + 1)];
                const auto dim = std::max<int64_t>(size[static_cast<size_t>(i + 1)], 1);
                TORCH_CHECK(next <= std::numeric_limits<int64_t>::max() / dim,
                            "set_: contiguous stride calculation overflow");
                contiguous_strides[static_cast<size_t>(i)] =
                    next * dim;
            }
        }
        stride = contiguous_strides;
    }

    const bool empty = std::any_of(size.begin(), size.end(), [](int64_t dim) { return dim == 0; });
    size_t required_bytes = 0;
    if (!empty) {
        required_bytes = at::detail::computeStorageNbytes(
            size, stride, self.itemsize(), static_cast<size_t>(storage_offset));
    }

    return pytorch_vulkan::with_vulkan_operation(
        pytorch_vulkan::allocation_runtime(self.storage().data_ptr()), [&]() -> at::Tensor& {
    // Native CPU grows the shared source storage when the requested view
    // exceeds it. Route that growth through the accepted PrivateUse1 hook.
    if (!empty && required_bytes > storage.nbytes()) {
        TORCH_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");
        at::GetPrivateUse1HooksInterface()->resizePrivateUse1Bytes(storage, required_bytes);
    }

    auto *impl = self.unsafeGetTensorImpl();
    // All validation, including capacity computation, precedes target mutation.
    at::native::checkSetStorage(self, storage, storage_offset, size, stride);
    impl->set_storage_offset(storage_offset);
    impl->set_sizes_and_strides(size, stride);
    return self;
    });
}

at::Tensor &vulkan_set_storage_symint_(at::Tensor &self, c10::Storage storage,
                                       c10::SymInt storage_offset,
                                       c10::SymIntArrayRef size,
                                       c10::SymIntArrayRef stride) {
    std::vector<int64_t> concrete_size;
    std::vector<int64_t> concrete_stride;
    concrete_size.reserve(size.size());
    concrete_stride.reserve(stride.size());
    for (const auto &dim : size)
        concrete_size.push_back(dim.expect_int());
    for (const auto &dim : stride)
        concrete_stride.push_back(dim.expect_int());
    return vulkan_set_storage_(self, std::move(storage), storage_offset.expect_int(),
                               concrete_size,
                               stride.empty() ? at::IntArrayRef() : at::IntArrayRef(concrete_stride));
}

at::Tensor &vulkan_set_storage_all_(at::Tensor &self, c10::Storage storage) {
    check_set_f32(self);
    TORCH_CHECK(storage.nbytes() / self.itemsize() <=
                    static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "Vulkan set_ storage is too large");
    const int64_t count = static_cast<int64_t>(storage.nbytes() / self.itemsize());
    const std::vector<int64_t> size{count};
    return vulkan_set_storage_(self, std::move(storage), 0, size, {});
}

at::Tensor &vulkan_set_tensor_(at::Tensor &self, const at::Tensor &source) {
    check_set_f32(self);
    if (self.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
        std::vector<int64_t> size(source.sizes().begin(), source.sizes().end());
        std::vector<int64_t> stride(source.strides().begin(), source.strides().end());
        return vulkan_set_storage_(self, source.storage(), source.storage_offset(),
                                   size, stride);
    }
    return self;
}

at::Tensor &vulkan_set_empty_(at::Tensor &self) {
    check_set_f32(self);
    return pytorch_vulkan::with_vulkan_operation(pytorch_vulkan::allocation_runtime(self.storage().data_ptr()), [&]() -> at::Tensor& {
    auto empty = at::empty({0}, self.options());
    auto *impl = self.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(empty.storage());
    impl->set_storage_offset(0);
    impl->set_sizes_contiguous({0});
    return self;
    });
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &vulkan_empty);
    m.impl("empty_strided", &vulkan_empty_strided);
    m.impl("_copy_from", &vulkan_copy_from);
    m.impl("_copy_from_and_resize", &vulkan_copy_from_and_resize);
    // PyTorch's autograd wrapper and CompositeExplicitAutograd callers remain
    // in use; PrivateUse1 needs its own leaf because the pinned native resize
    // implementation asserts for non-CPU/non-Meta storage. Byte growth itself
    // is delegated to PrivateUse1HooksInterface.
    m.impl("resize_", TORCH_FN(vulkan_resize_));
    m.impl("set_", TORCH_FN(vulkan_set_empty_));
    m.impl("set_.source_Tensor", TORCH_FN(vulkan_set_tensor_));
    m.impl("set_.source_Storage", TORCH_FN(vulkan_set_storage_all_));
    m.impl("set_.source_Storage_storage_offset", TORCH_FN(vulkan_set_storage_symint_));
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);
