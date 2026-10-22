#include "vulkan_synchronization.h"

VkPipelineStageFlags vulkan_execution_wait_stages() {
    return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
}

VulkanMemoryBarrierSpec vulkan_transfer_ready_spec() {
    return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
}

VulkanMemoryBarrierSpec vulkan_compute_ready_spec() {
    return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
}

VulkanMemoryBarrierSpec vulkan_host_read_spec() {
    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT};
}

void vulkan_record_memory_barrier(VkCommandBuffer command_buffer,
                                  const VulkanMemoryBarrierSpec &spec) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, spec.src_access,
                            spec.dst_access};
    vkCmdPipelineBarrier(command_buffer, spec.src_stage, spec.dst_stage, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
}
