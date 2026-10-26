#include "vulkan_pinned_allocator.h"

#include "vulkan_platform.h"

#include <c10/core/alignment.h>
#include <c10/core/Device.h>

#include <cstring>
#include <memory>
#include <stdexcept>

namespace {

struct VulkanPinnedHostOwner {
    std::shared_ptr<VulkanPlatform> platform;
    std::unique_ptr<VulkanBuffer> buffer;
    std::byte *mapped_base = nullptr;
    size_t logical_bytes = 0;
    VulkanPinnedHostTestOps ops{};
    bool counted = false;
};

struct VulkanPinnedHostHandle {
    std::shared_ptr<VulkanPinnedHostOwner> core;
};

void destroy_pinned_host_core(VulkanPinnedHostOwner *owner) noexcept {
    if (owner == nullptr || pytorch_vulkan::inherited_fork_state()) return;
    if (owner->platform != nullptr) {
        try {
            if (owner->ops.should_leak_for_testing != nullptr &&
                owner->ops.should_leak_for_testing(owner->ops.cookie, *owner->platform)) return;
            if (owner->platform->device_lost()) return;
        } catch (...) { return; }
    }
    if (owner->platform != nullptr && owner->counted)
        owner->platform->record_pinned_host_allocation_destroyed();
    delete owner;
}

void delete_pinned_host_owner(void *context) noexcept {
    delete static_cast<VulkanPinnedHostHandle *>(context);
}

VulkanPinnedHostOwner *pinned_host_owner(const at::DataPtr &data) noexcept {
    if (data.get_context() == nullptr || data.get_deleter() != &delete_pinned_host_owner)
        return nullptr;
    auto *handle = static_cast<VulkanPinnedHostHandle *>(data.get_context());
    return handle->core.get();
}

} // namespace

VulkanPinnedHostAllocator::VulkanPinnedHostAllocator(
    std::shared_ptr<VulkanPlatform> platform, VulkanPinnedHostTestOps ops)
    : platform_(std::move(platform)), ops_(ops) {}

at::DataPtr VulkanPinnedHostAllocator::allocate(size_t nbytes) {
    if (nbytes == 0)
        return at::DataPtr(nullptr, nullptr, &delete_pinned_host_owner,
                           c10::Device(c10::DeviceType::CPU));

    auto owner = std::make_unique<VulkanPinnedHostOwner>();
    owner->platform = platform_ ? platform_ : pytorch_vulkan::platform();
    owner->platform->throw_if_device_lost();
    owner->ops = ops_;
    auto buffer = std::make_unique<VulkanBuffer>(
        *owner->platform, nbytes,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &owner->ops.buffer);
    buffer->set_map_ops_for_testing(owner->ops.map);
    void *mapped = buffer->map_persistent();
    if (mapped == nullptr)
        throw std::runtime_error("Vulkan pinned map returned a null pointer");
    if (reinterpret_cast<std::uintptr_t>(mapped) % c10::gAlignment != 0)
        throw std::runtime_error("Vulkan pinned allocation base is not c10-aligned");
    owner->mapped_base = static_cast<std::byte *>(mapped);
    owner->logical_bytes = nbytes;
    owner->buffer = std::move(buffer);
    owner->platform->record_pinned_host_allocation_created();
    owner->counted = true;

    auto *data = owner->mapped_base;
    auto core = std::shared_ptr<VulkanPinnedHostOwner>(owner.release(), destroy_pinned_host_core);
    auto *context = new VulkanPinnedHostHandle{std::move(core)};
    return at::DataPtr(data, context, &delete_pinned_host_owner,
                       c10::Device(c10::DeviceType::CPU));
}

bool vulkan_pinned_host_range(const c10::Storage &storage, const void *data,
                              size_t bytes, VulkanPinnedHostRange *out) {
    if (out == nullptr) return false;
    const auto &ptr = storage.data_ptr();
    auto *owner = pinned_host_owner(ptr);
    if (owner == nullptr || owner->buffer == nullptr || data == nullptr) return false;
    const auto base = reinterpret_cast<std::uintptr_t>(owner->mapped_base);
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto storage_bytes = storage.nbytes();
    if (address < base || address - base > owner->logical_bytes ||
        address - base > storage_bytes) return false;
    const auto offset = address - base;
    if (bytes > owner->logical_bytes - offset || bytes > storage_bytes - offset) return false;
    auto *handle = static_cast<VulkanPinnedHostHandle *>(ptr.get_context());
    out->lease = std::shared_ptr<void>(handle->core, handle->core.get());
    out->buffer = owner->buffer.get();
    out->offset = static_cast<VkDeviceSize>(offset);
    out->bytes = static_cast<VkDeviceSize>(bytes);
    return true;
}

void VulkanPinnedHostAllocator::copy_data(void *dest, const void *src,
                                          std::size_t count) const {
    std::memcpy(dest, src, count);
}

VulkanPinnedHostAllocator *vulkan_pinned_host_allocator_instance() {
    static VulkanPinnedHostAllocator allocator;
    return &allocator;
}

bool is_vulkan_pinned_storage(const c10::Storage &storage) {
    return storage.data_ptr().get() != nullptr &&
           pinned_host_owner(storage.data_ptr()) != nullptr;
}

VkMemoryPropertyFlags vulkan_pinned_host_memory_properties_for_testing(
    const at::DataPtr &data) {
    auto *owner = pinned_host_owner(data);
    if (owner == nullptr)
        throw std::invalid_argument("DataPtr is not Vulkan pinned host storage");
    return owner->buffer->memory_properties();
}
