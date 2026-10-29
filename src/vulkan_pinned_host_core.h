#pragma once

#include "vulkan_lifecycle.h"

#include <cstddef>
#include <memory>

namespace pytorch_vulkan::detail {
// Private dependency extraction: production and CPU fixtures share the actual
// retained member layout and release decision, without constructing a platform.
template<class Platform, class Buffer, class Ops>
struct PinnedHostCore {
    std::shared_ptr<Platform> platform;
    std::unique_ptr<Buffer> buffer;
    std::byte *mapped_base = nullptr;
    size_t logical_bytes = 0;
    Ops ops{};
    bool counted = false;
};

// The concrete custom deleter must reject inherited fork state before entry.
template<class Platform, class Buffer, class Ops>
void destroy_pinned_host_core(PinnedHostCore<Platform, Buffer, Ops> *owner) noexcept {
    if (owner == nullptr) return;
    if (owner->platform != nullptr) {
        auto platform = owner->platform;
        VulkanCleanupScope cleanup(platform->runtime_lifecycle());
        try {
            if (owner->ops.should_leak_for_testing != nullptr &&
                owner->ops.should_leak_for_testing(owner->ops.cookie, *owner->platform)) return;
            if (owner->platform->device_lost()) return;
        } catch (...) { return; }
        if (owner->counted) platform->record_pinned_host_allocation_destroyed();
        delete owner;
        return;
    }
    delete owner;
}
} // namespace pytorch_vulkan::detail
