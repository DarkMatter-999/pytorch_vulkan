#pragma once

#include <vulkan/vulkan.h>

// Single source of truth for the same-queue memory dependencies between compute
// and transfer work this backend records (Vulkan 1.3 §7.9 availability and
// visibility). Every copy/fill/dispatch boundary references these helpers
// instead of duplicating stage/access literals at each site.

// Stage/access four-tuple recorded as one VkMemoryBarrier by the recording
// helpers below.
struct VulkanMemoryBarrierSpec {
    VkPipelineStageFlags src_stage = 0;
    VkAccessFlags src_access = 0;
    VkPipelineStageFlags dst_stage = 0;
    VkAccessFlags dst_access = 0;
};

// Stage mask used for the execution-context submission semaphore wait. Covers
// both device consumers of prior submissions on this queue (compute and
// transfer) and never includes HOST: host completion is ordered by the fence
// wait and the host-access exclusion, not by the submit wait mask.
VkPipelineStageFlags vulkan_execution_wait_stages();

// Producer -> transfer consumer. Includes both producer stages so preceding
// reads are also ordered against the new transfer write (WAR).
VulkanMemoryBarrierSpec vulkan_transfer_ready_spec();

// Producer -> compute consumer (storage shader read/write).
VulkanMemoryBarrierSpec vulkan_compute_ready_spec();

// Transfer producer -> host readback of mapped memory, recorded before the
// completion fence of the owning command buffer. Coherent memory still skips
// cache maintenance; the barrier is not a substitute for the fence wait.
VulkanMemoryBarrierSpec vulkan_host_read_spec();

// Records one spec as a single memory barrier on the given command buffer.
void vulkan_record_memory_barrier(VkCommandBuffer command_buffer,
                                  const VulkanMemoryBarrierSpec &spec);
