// Vulkan 1.3 specification, §7.9 (host write ordering): host writes that
// happen-before queue submission are made available to device access. The
// platform flushes noncoherent source ranges before submitting, so H2D needs no
// extra HOST_WRITE barrier. These shared device/host-read dependencies are still
// required for compute/transfer consumers and D2H host visibility.
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html
#include "vulkan_async_transfer.h"
#include "vulkan_synchronization.h"

void record_async_copy(VkCommandBuffer cb, const VulkanAsyncCopyRequest &request) {
    vulkan_record_memory_barrier(cb, vulkan_transfer_ready_spec());
    vkCmdCopyBuffer(cb, request.source, request.destination,
                    static_cast<uint32_t>(request.regions.size()), request.regions.data());
    vulkan_record_memory_barrier(cb, vulkan_compute_ready_spec());
    if (request.kind == VulkanAsyncCopyKind::DeviceToHost)
        vulkan_record_memory_barrier(cb, vulkan_host_read_spec());
}
