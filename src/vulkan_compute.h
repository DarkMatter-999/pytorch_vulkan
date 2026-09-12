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
    void tensor_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes,
                       uint32_t operation = 0) const;
    void tensor_scalar(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                       float scalar, uint32_t operation = 0) const;
    void scalar_tensor(float scalar, VkBuffer tensor, VkBuffer output,
                       VkDeviceSize bytes, uint32_t operation = 0) const;
    void scalar_tensor_alias(float scalar, VkBuffer tensor, VkBuffer output,
                             VkDeviceSize bytes, uint32_t operation = 0) const;
    void unary(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
               uint32_t operation) const;
    void unary_alias(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                     uint32_t operation) const;
    void tensor_tensor_alias(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                             VkDeviceSize bytes, uint32_t operation = 0) const;
    void tensor_scalar_alias(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                             float scalar, uint32_t operation = 0) const;
    std::size_t dispatch_count() const;

  private:
    void dispatch(uint32_t mode, VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                  VkDeviceSize bytes, float scalar, uint32_t operation,
                  bool exact_alias = false) const;
    const VulkanPlatform &platform_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layouts_[4]{};
    VkPipelineLayout pipeline_layouts_[4]{};
    VkShaderModule shader_modules_[4]{};
    VkPipeline pipelines_[4]{};
    VkDeviceSize max_storage_buffer_range_ = 0;
    uint32_t max_compute_workgroup_count_x_ = 0;
    mutable std::atomic<std::size_t> dispatch_count_{0};
};
