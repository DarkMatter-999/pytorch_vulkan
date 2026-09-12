#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <atomic>

class VulkanPlatform;

class VulkanCompute final {
  public:
    explicit VulkanCompute(const VulkanPlatform &platform);
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute &) = delete;
    VulkanCompute &operator=(const VulkanCompute &) = delete;

    void add(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes) const;
    std::size_t dispatch_count() const;

  private:
    const VulkanPlatform &platform_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDeviceSize max_storage_buffer_range_ = 0;
    uint32_t max_compute_workgroup_count_x_ = 0;
    mutable std::atomic<std::size_t> dispatch_count_{0};
};
