#pragma once

#include "vulkan_execution.h"

enum class VulkanAsyncCopyKind { HostToDevice, DeviceToHost, DeviceToDevice };

struct VulkanAsyncCopyRequest {
    VulkanAsyncCopyKind kind = VulkanAsyncCopyKind::HostToDevice;
    VkBuffer source = VK_NULL_HANDLE;
    VkBuffer destination = VK_NULL_HANDLE;
    // Nonempty F32 byte regions; callers validate allocation bounds. The engine
    // rechecks handles, alignment, overflow and source/destination overlap.
    std::vector<VkBufferCopy> regions;
    VulkanTransferReservation reservation;
    std::vector<std::shared_ptr<void>> owners;
    std::function<void()> before_submit;
    std::function<void()> publish_success;
};

// Pure recorder: no platform/global state, ownership, submission or host access.
void record_async_copy(VkCommandBuffer cb, const VulkanAsyncCopyRequest &request);
