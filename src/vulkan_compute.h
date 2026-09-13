#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>

class VulkanPlatform;

class VulkanCompute final {
  public:
    explicit VulkanCompute(const VulkanPlatform &platform);
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute &) = delete;
    VulkanCompute &operator=(const VulkanCompute &) = delete;

    void add(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes) const;
    void tensor_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes,
                       uint32_t operation = 0, bool bool_dtype = false) const;
    void tensor_scalar(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                       float scalar, uint32_t operation = 0,
                       bool bool_dtype = false) const;
    void scalar_tensor(float scalar, VkBuffer tensor, VkBuffer output,
                       VkDeviceSize bytes, uint32_t operation = 0,
                       bool bool_dtype = false) const;
    void scalar_tensor_alias(float scalar, VkBuffer tensor, VkBuffer output,
                             VkDeviceSize bytes, uint32_t operation = 0,
                             bool bool_dtype = false) const;
    void unary(VkBuffer input, VkBuffer output, VkDeviceSize bytes, uint32_t operation,
               bool bool_dtype = false) const;
    void unary_alias(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                     uint32_t operation, bool bool_dtype = false) const;
    void unary_offset(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                      uint32_t operation, VkDeviceSize input_offset) const;
    void comparison_scalar(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                           float scalar) const;
    void comparison_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                           VkDeviceSize bytes, uint32_t operation = 0) const;
    void isfinite(VkBuffer input, VkBuffer output, VkDeviceSize bytes) const;
    void masked_select_count(VkBuffer input, VkBuffer mask, VkBuffer counter,
                             uint32_t element_count) const;
    void masked_select_compact(VkBuffer input, VkBuffer mask, VkBuffer output,
                               VkBuffer counter, uint32_t element_count,
                               uint32_t output_count) const;
    void tensor_tensor_alias(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                             VkDeviceSize bytes, uint32_t operation = 0,
                             bool bool_dtype = false) const;
    void tensor_scalar_alias(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                             float scalar, uint32_t operation = 0,
                             bool bool_dtype = false) const;
    void reduction(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                   uint32_t rank, const uint32_t *sizes, uint32_t reduce_mask,
                   uint32_t reduce_numel, uint32_t output_numel, bool mean) const;
    void argmax(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                uint32_t rank, const uint32_t *sizes, uint32_t dim,
                uint32_t reduce_size, uint32_t output_numel) const;
    void broadcast(VkBuffer input, VkBuffer output, uint32_t rank,
                   const uint32_t *input_sizes, const uint32_t *output_sizes,
                   uint32_t output_numel, float scale) const;
    void linear(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                uint32_t rows, uint32_t features, uint32_t outputs,
                bool transposed_weight = false, bool has_bias = true,
                uint32_t operation = 0) const;
    void convolution(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                     uint32_t operation = 0) const;
    void pooling(VkBuffer input, VkBuffer output, uint32_t batch, uint32_t channels,
                 uint32_t height, uint32_t width, uint32_t operation = 0) const;
    void f32_to_double(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                       VkDeviceSize output_bytes, uint32_t element_count) const;
    void formatter_double(VkBuffer input, VkBuffer rhs, VkBuffer output,
                          VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                          VkDeviceSize output_bytes, uint32_t element_count,
                          uint32_t operation, double scalar, uint32_t output_numel,
                          bool bool_output = false) const;
    std::size_t dispatch_count() const;

  private:
    void dispatch(uint32_t mode, VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                   VkDeviceSize bytes, float scalar, uint32_t operation,
                   bool exact_alias, bool bool_dtype, bool bool_output = false,
                   VkDeviceSize lhs_offset = 0, VkDeviceSize output_offset = 0) const;
    void dispatch_extra(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                        VkDeviceSize output_bytes, VkPipeline pipeline,
                        VkPipelineLayout pipeline_layout,
                        VkDescriptorSetLayout descriptor_layout, const void *params,
                        uint32_t params_size, uint32_t output_numel) const;
    void dispatch_formatter(VkBuffer input, VkBuffer rhs, VkBuffer output,
                            VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                            VkDeviceSize output_bytes, const void *params,
                            uint32_t params_size, uint32_t output_numel,
                            bool bool_output) const;
    void dispatch_model(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                        VkDeviceSize input_bytes, VkDeviceSize weight_bytes,
                        VkDeviceSize bias_bytes, VkDeviceSize output_bytes,
                        const void *params, uint32_t params_size, uint32_t output_numel,
                        VkPipeline pipeline = VK_NULL_HANDLE,
                        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE,
                        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE) const;
    void dispatch_masked(VkBuffer input, VkBuffer mask, VkBuffer output,
                         VkBuffer counter, uint32_t element_count,
                         VkDeviceSize output_bytes, VkPipeline pipeline,
                         VkPipelineLayout pipeline_layout,
                         VkDescriptorSetLayout descriptor_layout,
                         uint32_t descriptor_count) const;
    const VulkanPlatform &platform_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layouts_[8]{};
    VkPipelineLayout pipeline_layouts_[8]{};
    VkShaderModule shader_modules_[8]{};
    VkPipeline pipelines_[8]{};
    VkDescriptorSetLayout reduction_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout reduction_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule reduction_shader_ = VK_NULL_HANDLE;
    VkPipeline reduction_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout indexing_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout indexing_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule indexing_shader_ = VK_NULL_HANDLE;
    VkPipeline indexing_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout broadcast_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout broadcast_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule broadcast_shader_ = VK_NULL_HANDLE;
    VkPipeline broadcast_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout model_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule model_shader_ = VK_NULL_HANDLE;
    VkPipeline model_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout convolution_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout convolution_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule convolution_shader_ = VK_NULL_HANDLE;
    VkPipeline convolution_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pooling_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pooling_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule pooling_shader_ = VK_NULL_HANDLE;
    VkPipeline pooling_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout masked_count_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout masked_count_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule masked_count_shader_ = VK_NULL_HANDLE;
    VkPipeline masked_count_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout masked_compact_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout masked_compact_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule masked_compact_shader_ = VK_NULL_HANDLE;
    VkPipeline masked_compact_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout f32_to_double_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout f32_to_double_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule f32_to_double_shader_ = VK_NULL_HANDLE;
    VkPipeline f32_to_double_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout formatter_double_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout formatter_double_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule formatter_double_shader_ = VK_NULL_HANDLE;
    VkPipeline formatter_double_pipeline_ = VK_NULL_HANDLE;
    VkDeviceSize max_storage_buffer_range_ = 0;
    uint32_t max_compute_workgroup_count_x_ = 0;
    mutable std::atomic<std::size_t> dispatch_count_{0};
};
