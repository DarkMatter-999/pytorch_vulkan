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
};

void delete_pinned_host_owner(void *context) noexcept {
    auto *owner = static_cast<VulkanPinnedHostOwner *>(context);
    if (owner == nullptr || pytorch_vulkan::inherited_fork_state()) return;
    if (owner->platform != nullptr) {
        try {
            if (owner->ops.should_leak_for_testing != nullptr) {
                if (owner->ops.should_leak_for_testing(owner->ops.cookie, *owner->platform))
                    return;
            } else {
                if (owner->platform->device_lost()) return;
                if (owner->platform->execution_context().pending_count() != 0) return;
            }
        } catch (...) {
            return;
        }
    }
    delete owner;
}

VulkanPinnedHostOwner *pinned_host_owner(const at::DataPtr &data) noexcept {
    if (data.get_context() == nullptr || data.get_deleter() != &delete_pinned_host_owner)
        return nullptr;
    return static_cast<VulkanPinnedHostOwner *>(data.get_context());
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

    auto *data = owner->mapped_base;
    auto *context = owner.release();
    return at::DataPtr(data, context, &delete_pinned_host_owner,
                       c10::Device(c10::DeviceType::CPU));
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
