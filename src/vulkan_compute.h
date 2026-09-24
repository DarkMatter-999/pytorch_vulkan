#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

class VulkanPlatform;
class VulkanBuffer;
class DescriptorArena;
struct DescriptorArenaSnapshot;
#include "vulkan_tensor_layout.h"

using pytorch_vulkan::VulkanTensorLayout;

class VulkanCompute final {
  public:
    explicit VulkanCompute(const VulkanPlatform &platform);
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute &) = delete;
    VulkanCompute &operator=(const VulkanCompute &) = delete;

    void add(VkBuffer lhs, const VulkanTensorLayout &lhs_layout, VkBuffer rhs,
             const VulkanTensorLayout &rhs_layout, VkBuffer output,
             const VulkanTensorLayout &output_layout) const;
    void tensor_tensor(VkBuffer lhs, const VulkanTensorLayout &lhs_layout, VkBuffer rhs,
                       const VulkanTensorLayout &rhs_layout, VkBuffer output,
                       const VulkanTensorLayout &output_layout, uint32_t operation = 0,
                       bool bool_dtype = false, float alpha = 1.0F) const;
    void tensor_scalar(VkBuffer tensor, const VulkanTensorLayout &tensor_layout,
                       VkBuffer output, const VulkanTensorLayout &output_layout,
                       float scalar, uint32_t operation = 0,
                       bool bool_dtype = false) const;
    void scalar_tensor(float scalar, VkBuffer tensor,
                       const VulkanTensorLayout &tensor_layout, VkBuffer output,
                       const VulkanTensorLayout &output_layout, uint32_t operation = 0,
                       bool bool_dtype = false) const;
    void scalar_tensor_alias(float scalar, VkBuffer tensor,
                             const VulkanTensorLayout &tensor_layout, VkBuffer output,
                             const VulkanTensorLayout &output_layout,
                             uint32_t operation = 0, bool bool_dtype = false) const;
    void unary(VkBuffer input, const VulkanTensorLayout &input_layout, VkBuffer output,
               const VulkanTensorLayout &output_layout, uint32_t operation,
               bool bool_dtype = false) const;
    void unary_alias(VkBuffer input, const VulkanTensorLayout &input_layout,
                     VkBuffer output, const VulkanTensorLayout &output_layout,
                     uint32_t operation, bool bool_dtype = false) const;
    void fill_alias(VkBuffer input, const VulkanTensorLayout &input_layout,
                    VkBuffer output, const VulkanTensorLayout &output_layout,
                    float scalar) const;
    void comparison_scalar(VkBuffer input, const VulkanTensorLayout &input_layout,
                           VkBuffer output, const VulkanTensorLayout &output_layout,
                           float scalar) const;
    void comparison_tensor(VkBuffer lhs, const VulkanTensorLayout &lhs_layout,
                           VkBuffer rhs, const VulkanTensorLayout &rhs_layout,
                           VkBuffer output, const VulkanTensorLayout &output_layout,
                           uint32_t operation = 0) const;
    void isfinite(VkBuffer input, const VulkanTensorLayout &input_layout,
                  VkBuffer output, const VulkanTensorLayout &output_layout) const;
    void masked_select_count(VkBuffer input, VkBuffer mask, VkBuffer counter,
                             uint32_t element_count) const;
    void masked_select_compact(VkBuffer input, VkBuffer mask, VkBuffer output,
                               VkBuffer counter, uint32_t element_count,
                               uint32_t output_count) const;
    void tensor_tensor_alias(VkBuffer lhs, const VulkanTensorLayout &lhs_layout,
                             VkBuffer rhs, const VulkanTensorLayout &rhs_layout,
                             VkBuffer output, const VulkanTensorLayout &output_layout,
                             uint32_t operation = 0, bool bool_dtype = false,
                             float alpha = 1.0F) const;
    void tensor_scalar_alias(VkBuffer tensor, const VulkanTensorLayout &tensor_layout,
                             VkBuffer output, const VulkanTensorLayout &output_layout,
                             float scalar, uint32_t operation = 0,
                             bool bool_dtype = false) const;
    void compound_tensor_tensor_alias(
        VkBuffer self, const VulkanTensorLayout &self_layout, VkBuffer tensor1,
        const VulkanTensorLayout &tensor1_layout, VkBuffer tensor2,
        const VulkanTensorLayout &tensor2_layout, VkBuffer output,
        const VulkanTensorLayout &output_layout, float value, uint32_t operation) const;
    void reduction(VkBuffer input, const VulkanTensorLayout &input_layout,
                   VkBuffer output, const VulkanTensorLayout &output_layout,
                   uint32_t reduce_mask, uint32_t reduce_numel, uint32_t output_numel,
                   uint32_t operation, uint32_t reduce_dim = 0) const;
    void reduction_backward(
        const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
        const VulkanBuffer *forward, const VulkanTensorLayout &forward_layout,
        const VulkanBuffer *grad_output, const VulkanTensorLayout &grad_output_layout,
        const VulkanBuffer *grad_input, const VulkanTensorLayout &grad_input_layout,
        uint32_t reduce_mask, uint32_t reduce_numel, uint32_t reduce_dim,
        uint32_t operation, bool keepdim) const;
    void mse_loss(const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
                  const VulkanBuffer *target, const VulkanTensorLayout &target_layout,
                  const VulkanBuffer *aux, const VulkanTensorLayout &aux_layout,
                  const VulkanBuffer *output, const VulkanTensorLayout &output_layout,
                  uint32_t element_count, uint32_t reduction, bool backward) const;
    void argmax(VkBuffer input, const VulkanTensorLayout &input_layout, VkBuffer output,
                const VulkanTensorLayout &output_layout, uint32_t dim,
                uint32_t reduce_size, uint32_t output_numel) const;
    void broadcast(VkBuffer input, const VulkanTensorLayout &input_layout,
                   VkBuffer output, const VulkanTensorLayout &output_layout,
                   uint32_t output_numel, float scale) const;
    void linear(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                const VulkanTensorLayout &input_layout,
                const VulkanTensorLayout &weight_layout,
                const VulkanTensorLayout &bias_layout,
                const VulkanTensorLayout &output_layout, uint32_t rows,
                uint32_t features, uint32_t outputs, bool transposed_weight = false,
                bool has_bias = true, uint32_t operation = 0) const;
    void gemm(VkBuffer a, const VulkanTensorLayout &a_layout, VkBuffer b,
              const VulkanTensorLayout &b_layout, VkBuffer c,
              const VulkanTensorLayout &c_layout, VkBuffer output,
              const VulkanTensorLayout &output_layout, VkBuffer bias,
              const VulkanTensorLayout &bias_layout, uint32_t m, uint32_t n, uint32_t k,
              float alpha = 1.0F, float beta = 0.0F, bool has_bias = false,
              uint32_t batch_count = 0, uint32_t batch_stride_a = 0,
              uint32_t batch_stride_b = 0, uint32_t batch_stride_c = 0,
               uint32_t batch_stride_d = 0) const;
    void rnn_sequence(VkBuffer input, VkBuffer weight, VkBuffer recurrent_weight,
                      VkBuffer bias, VkBuffer output, uint32_t batch,
                      uint32_t sequence, uint32_t input_dimension,
                      uint32_t hidden_dimension) const;
    void rnn_sequence_backward(
        VkBuffer input, VkBuffer weight, VkBuffer recurrent_weight, VkBuffer bias,
        VkBuffer output, VkBuffer gradient_output, VkBuffer gradient_input,
        VkBuffer partial_input_weight, VkBuffer partial_recurrent_weight,
        VkBuffer partial_bias, VkBuffer gradient_weight, VkBuffer gradient_recurrent_weight,
        VkBuffer gradient_bias, uint32_t batch, uint32_t sequence,
        uint32_t input_dimension, uint32_t hidden_dimension) const;
    void validate_rnn_sequence(uint32_t batch, uint32_t sequence,
                               uint32_t input_dimension,
                               uint32_t hidden_dimension) const;
    void validate_rnn_sequence_backward(uint32_t batch, uint32_t sequence,
                                        uint32_t input_dimension,
                                        uint32_t hidden_dimension) const;
    static void validate_rnn_sequence_limits(uint32_t batch, uint32_t sequence,
                                             uint32_t input_dimension,
                                             uint32_t hidden_dimension,
                                             VkDeviceSize max_storage_buffer_range,
                                             uint32_t max_compute_workgroup_count_x);
    static void validate_rnn_sequence_backward_limits(
        uint32_t batch, uint32_t sequence, uint32_t input_dimension,
        uint32_t hidden_dimension, VkDeviceSize max_storage_buffer_range,
        uint32_t max_compute_workgroup_count_x);
    void linear_relu_backward_input(VkBuffer, VkBuffer, VkBuffer, VkBuffer,
                                    const VulkanTensorLayout &,
                                    const VulkanTensorLayout &,
                                    const VulkanTensorLayout &,
                                    const VulkanTensorLayout &, uint32_t, uint32_t,
                                    uint32_t) const;
    void linear_relu_backward_weight(VkBuffer, VkBuffer, VkBuffer, VkBuffer,
                                     const VulkanTensorLayout &,
                                     const VulkanTensorLayout &,
                                     const VulkanTensorLayout &,
                                     const VulkanTensorLayout &, uint32_t, uint32_t,
                                     uint32_t) const;
    void linear_relu_backward_bias(VkBuffer, VkBuffer, VkBuffer,
                                   const VulkanTensorLayout &,
                                   const VulkanTensorLayout &,
                                   const VulkanTensorLayout &, uint32_t,
                                   uint32_t) const;
    // Records the shared multi-output backward invocation.  The shader and
    // operator integration are intentionally supplied by a later task.
    void linear_relu_backward(
        const VulkanBuffer *grad_output, const VulkanTensorLayout &grad_output_layout,
        const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
        const VulkanBuffer *weight, const VulkanTensorLayout &weight_layout,
        const VulkanBuffer *activation, const VulkanTensorLayout &activation_layout,
        const VulkanBuffer *d_input, const VulkanTensorLayout &d_input_layout,
        const VulkanBuffer *d_weight, const VulkanTensorLayout &d_weight_layout,
        const VulkanBuffer *d_bias, const VulkanTensorLayout &d_bias_layout,
        uint32_t rows, uint32_t features, uint32_t outputs) const;
    void convolution(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                     const VulkanTensorLayout &input_layout,
                     const VulkanTensorLayout &weight_layout,
                     const VulkanTensorLayout &bias_layout,
                     const VulkanTensorLayout &output_layout,
                     uint32_t operation = 0) const;
    void pooling(VkBuffer input, VkBuffer output,
                 const VulkanTensorLayout &input_layout,
                 const VulkanTensorLayout &output_layout, uint32_t batch,
                 uint32_t channels, uint32_t height, uint32_t width,
                 uint32_t operation = 0) const;
    void compute_multi_output(const VulkanBuffer *const *inputs,
                              const VulkanTensorLayout *const *input_layouts,
                              const VulkanBuffer *const *outputs,
                              const VulkanTensorLayout *const *output_layouts,
                              uint32_t input_count, uint32_t output_count,
                              uint32_t invocation_count, const void *params,
                              uint32_t params_size, bool classification) const;
    void f32_to_double(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                       VkDeviceSize output_bytes, uint32_t element_count) const;
    void formatter_double(VkBuffer input, VkBuffer rhs, VkBuffer output,
                          VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                          VkDeviceSize output_bytes, uint32_t element_count,
                          uint32_t operation, double scalar, uint32_t output_numel,
                          bool bool_output = false) const;
    std::size_t dispatch_count() const;
    void reset_dispatch_count() const;
    std::size_t submission_count() const;
    void reset_submission_count() const;
    std::size_t descriptor_pool_creation_count() const;
    std::size_t descriptor_set_allocation_count() const;
    std::size_t descriptor_set_reuse_count() const;
    std::size_t live_descriptor_pool_count() const;
    std::size_t live_descriptor_set_count() const;
    void invalidate_device_loss() const;
    DescriptorArenaSnapshot descriptor_arena_snapshot() const;
    std::size_t pipeline_count() const;
    std::size_t shader_module_count() const;
    void reset_descriptor_resource_counters() const;
    void begin_training_step() const;
    void end_training_step() const;
    void cancel_training_step() const;
    bool training_step_active() const;

  private:
    void dispatch(uint32_t mode, VkBuffer lhs, const VulkanTensorLayout *lhs_layout,
                  VkBuffer rhs, const VulkanTensorLayout *rhs_layout, VkBuffer output,
                  const VulkanTensorLayout &output_layout, float scalar,
                  uint32_t operation, bool exact_alias, bool bool_dtype,
                  bool bool_output = false, VkDeviceSize lhs_offset = 0,
                  VkDeviceSize output_offset = 0) const;
    void dispatch_compound(VkBuffer self, const VulkanTensorLayout &self_layout,
                           VkBuffer tensor1, const VulkanTensorLayout &tensor1_layout,
                           VkBuffer tensor2, const VulkanTensorLayout &tensor2_layout,
                           VkBuffer output, const VulkanTensorLayout &output_layout,
                           float value, uint32_t operation) const;
    void dispatch_extra(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                        VkDeviceSize output_bytes, VkPipeline pipeline,
                        VkPipelineLayout pipeline_layout,
                        VkDescriptorSetLayout descriptor_layout, const void *params,
                         uint32_t params_size, uint32_t output_numel,
                         const void *metadata = nullptr,
                         VkDeviceSize metadata_size = 0,
                         bool one_workgroup_per_output = false) const;
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
                         VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE,
                         const void *metadata = nullptr,
                         VkDeviceSize metadata_size = 0,
                         bool workgroup_per_output = false) const;
    void dispatch_multi_output(const VulkanBuffer *const *inputs,
                               const VulkanTensorLayout *const *input_layouts,
                               const VulkanBuffer *const *outputs,
                               const VulkanTensorLayout *const *output_layouts,
                               const void *params, uint32_t params_size,
                               const void *metadata, VkDeviceSize metadata_size,
                               VkPipeline pipeline, VkPipelineLayout pipeline_layout,
                               VkDescriptorSetLayout descriptor_layout,
                               uint32_t invocation_count, uint32_t input_count = 4,
                               uint32_t output_count = 3,
                               uint32_t descriptor_capacity = 0) const;
    void dispatch_masked(VkBuffer input, VkBuffer mask, VkBuffer output,
                         VkBuffer counter, uint32_t element_count,
                         VkDeviceSize output_bytes, VkPipeline pipeline,
                         VkPipelineLayout pipeline_layout,
                         VkDescriptorSetLayout descriptor_layout,
                         uint32_t descriptor_count) const;
    void record_dispatch(const char *scope = "operator") const;
    void finish_dispatch() const;
    void cancel_recording() const;
    VkDescriptorSet acquire_descriptor_set(VkDescriptorSetLayout descriptor_layout,
                                           uint32_t descriptor_count,
                                           uint32_t pool_capacity = 64) const;
    const VulkanPlatform &platform_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::unique_ptr<DescriptorArena> descriptor_arena_;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layouts_[10]{};
    VkPipelineLayout pipeline_layouts_[10]{};
    VkShaderModule shader_modules_[10]{};
    VkPipeline pipelines_[10]{};
    VkDescriptorSetLayout compound_descriptor_layouts_[2]{};
    VkPipelineLayout compound_pipeline_layouts_[2]{};
    VkShaderModule compound_shader_modules_[2]{};
    VkPipeline compound_pipelines_[2]{};
    VkDescriptorSetLayout reduction_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout reduction_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule reduction_shader_ = VK_NULL_HANDLE;
    VkPipeline reduction_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout reduction_backward_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout reduction_backward_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule reduction_backward_shader_ = VK_NULL_HANDLE;
    VkPipeline reduction_backward_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout loss_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout loss_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule loss_shader_ = VK_NULL_HANDLE;
    VkPipeline loss_pipeline_ = VK_NULL_HANDLE;
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
    // Kept separate from model_* to preserve the existing five-binding ABI.
    VkDescriptorSetLayout backward_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout backward_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule backward_shader_ = VK_NULL_HANDLE;
    VkPipeline backward_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout convolution_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout convolution_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule convolution_shader_ = VK_NULL_HANDLE;
    VkPipeline convolution_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pooling_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pooling_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule pooling_shader_ = VK_NULL_HANDLE;
    VkPipeline pooling_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout normalization_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout normalization_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule normalization_shader_ = VK_NULL_HANDLE;
    VkPipeline normalization_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout classification_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout classification_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule classification_shader_ = VK_NULL_HANDLE;
    VkPipeline classification_pipeline_ = VK_NULL_HANDLE;
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
    VkDescriptorSetLayout gemm_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout gemm_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule gemm_shader_ = VK_NULL_HANDLE;
    VkPipeline gemm_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout rnn_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout rnn_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule rnn_shader_ = VK_NULL_HANDLE;
    VkPipeline rnn_pipeline_ = VK_NULL_HANDLE;
    VkDeviceSize max_storage_buffer_range_ = 0;
    uint32_t max_push_constants_size_ = 0;
    uint32_t max_compute_workgroup_count_x_ = 0;
    uint32_t max_compute_workgroup_count_y_ = 0;
    uint32_t max_compute_workgroup_count_z_ = 0;
    uint32_t max_compute_shared_memory_size_ = 0;
    mutable std::atomic<std::size_t> dispatch_count_{0};
    mutable std::atomic<std::size_t> submission_count_{0};
    mutable std::size_t descriptor_pool_baseline_ = 0;
    mutable std::size_t descriptor_allocation_baseline_ = 0;
    mutable std::size_t descriptor_reuse_baseline_ = 0;
    mutable bool training_step_ = false;
};
