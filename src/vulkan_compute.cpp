#include "vulkan_compute.h"

#include "vulkan/shaders/generated/convolution_spv.h"
#include "vulkan/shaders/generated/masked_select_spv.h"
#include "vulkan/shaders/generated/model_spv.h"
#include "vulkan/shaders/generated/pointwise_spv.h"
#include "vulkan/shaders/generated/pooling_spv.h"
#include "vulkan/shaders/generated/reduction_indexing_spv.h"
#include "vulkan/shaders/generated/f32_to_double_spv.h"
#include "vulkan/shaders/generated/formatter_double_spv.h"
#include "vulkan_buffer.h"
#include "vulkan_platform.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

std::runtime_error contextual_error(const char *operation,
                                    const std::exception &error) {
    return std::runtime_error(std::string("Vulkan compute ") + operation + ": " +
                              error.what());
}

struct Params {
    float scalar;
    uint32_t element_count;
    uint32_t operation;
};

struct ReductionParams {
    uint32_t rank, output_numel, reduce_mask, reduce_numel;
    uint32_t sizes[8];
    uint32_t mean;
};

struct IndexingParams {
    uint32_t rank, output_numel, reduce_dim, reduce_size;
    uint32_t sizes[8];
};

struct BroadcastParams {
    uint32_t rank, output_numel;
    float scale;
    uint32_t input_sizes[8];
    uint32_t output_sizes[8];
};

struct LinearParams {
    uint32_t rows, features, outputs, transposed_weight, has_bias, operation;
};
struct ConvolutionParams {
    uint32_t batch, input_channels, input_height, input_width, output_channels,
        output_height, output_width, kernel_height, kernel_width, operation;
};
struct MaskedParams {
    uint32_t element_count;
};
struct PoolingParams {
    uint32_t batch, channels, height, width, operation;
};
struct F32ToDoubleParams { uint32_t element_count; };
struct FormatterDoubleParams {
    double scalar;
    uint32_t element_count;
    uint32_t operation;
};

constexpr uint32_t kAdd = 0;
constexpr uint32_t kWorkgroupSize = 256;

uint64_t checked_product(uint64_t lhs, uint64_t rhs, const char *name) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::invalid_argument(std::string("Vulkan pooling ") + name +
                                    " product overflow");
    return lhs * rhs;
}

VkDeviceSize checked_bytes(uint64_t elements, const char *name) {
    const uint64_t bytes = checked_product(elements, sizeof(float), name);
    if (bytes > std::numeric_limits<VkDeviceSize>::max())
        throw std::invalid_argument(std::string("Vulkan pooling ") + name +
                                    " byte range overflow");
    return static_cast<VkDeviceSize>(bytes);
}

} // namespace

VulkanCompute::VulkanCompute(const VulkanPlatform &platform)
    : platform_(platform), device_(platform.device()), queue_(platform.compute_queue()),
      command_pool_(platform.command_pool()) {
    try {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(platform.physical_device(), &properties);
        max_storage_buffer_range_ = properties.limits.maxStorageBufferRange;
        max_compute_workgroup_count_x_ = properties.limits.maxComputeWorkGroupCount[0];
        if (properties.limits.maxPushConstantsSize < sizeof(ReductionParams))
            throw std::runtime_error(
                "device maxPushConstantsSize is smaller than pointwise ABI");

        const VkDescriptorSetLayoutBinding lhs_binding = {
            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding rhs_binding = {
            1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding output_binding = {
            2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding tensor_tensor_bindings[] = {
            lhs_binding, rhs_binding, output_binding};
        const VkDescriptorSetLayoutBinding tensor_scalar_bindings[] = {lhs_binding,
                                                                       output_binding};
        const VkDescriptorSetLayoutBinding scalar_tensor_bindings[] = {lhs_binding,
                                                                       output_binding};
        const VkDescriptorSetLayoutBinding unary_bindings[] = {lhs_binding,
                                                               output_binding};

        const VkShaderModuleCreateInfo tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kTensorTensorCodeSize,
            vulkan_pointwise_shader::kTensorTensorCode};
        const VkShaderModuleCreateInfo tensor_scalar_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kTensorScalarCodeSize,
            vulkan_pointwise_shader::kTensorScalarCode};
        const VkShaderModuleCreateInfo scalar_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kScalarTensorCodeSize,
            vulkan_pointwise_shader::kScalarTensorCode};
        const VkShaderModuleCreateInfo unary_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kUnaryCodeSize,
            vulkan_pointwise_shader::kUnaryCode};
        const VkShaderModuleCreateInfo bool_tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolTensorTensorCodeSize,
            vulkan_pointwise_shader::kBoolTensorTensorCode};
        const VkShaderModuleCreateInfo bool_output_tensor_scalar_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputTensorScalarCodeSize,
            vulkan_pointwise_shader::kBoolOutputTensorScalarCode};
        const VkShaderModuleCreateInfo bool_output_unary_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputUnaryCodeSize,
            vulkan_pointwise_shader::kBoolOutputUnaryCode};
        const VkShaderModuleCreateInfo bool_output_tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputTensorTensorCodeSize,
            vulkan_pointwise_shader::kBoolOutputTensorTensorCode};

        const auto create_mode = [&](uint32_t mode,
                                     const VkDescriptorSetLayoutBinding *bindings,
                                     uint32_t binding_count,
                                     const VkShaderModuleCreateInfo &shader_info) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = binding_count;
            layout.pBindings = bindings;
            check_result(vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                                     &descriptor_set_layouts_[mode]),
                         "could not create pointwise descriptor-set layout");
            check_result(vkCreateShaderModule(device_, &shader_info, nullptr,
                                              &shader_modules_[mode]),
                         "could not create pointwise shader module");
        };
        create_mode(0, tensor_tensor_bindings, 3, tensor_tensor_shader);
        create_mode(1, tensor_scalar_bindings, 2, tensor_scalar_shader);
        create_mode(2, scalar_tensor_bindings, 2, scalar_tensor_shader);
        create_mode(3, unary_bindings, 2, unary_shader);
        if (platform.supports_bool_pointwise()) {
            create_mode(4, tensor_tensor_bindings, 3, bool_tensor_tensor_shader);
            create_mode(5, tensor_scalar_bindings, 2, bool_output_tensor_scalar_shader);
            create_mode(6, unary_bindings, 2, bool_output_unary_shader);
            create_mode(7, tensor_tensor_bindings, 3, bool_output_tensor_tensor_shader);
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
        const auto create_pipeline = [&](uint32_t mode) {
            VkPipelineLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &descriptor_set_layouts_[mode];
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &layout, nullptr,
                                                &pipeline_layouts_[mode]),
                         "could not create pointwise pipeline layout");
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader_modules_[mode];
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline.stage = stage;
            pipeline.layout = pipeline_layouts_[mode];
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline,
                                                  nullptr, &pipelines_[mode]),
                         "could not create pointwise compute pipeline");
        };
        create_pipeline(0);
        create_pipeline(1);
        create_pipeline(2);
        create_pipeline(3);
        if (platform.supports_bool_pointwise()) {
            create_pipeline(4);
            create_pipeline(5);
            create_pipeline(6);
            create_pipeline(7);
        }
        const VkDescriptorSetLayoutBinding reduction_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        const auto create_extra = [&](VkDescriptorSetLayout &descriptor_layout,
                                      VkShaderModule &module,
                                      VkPipelineLayout &pipeline_layout,
                                       VkPipeline &pipeline, const uint32_t *code,
                                       std::size_t code_size,
                                       uint32_t push_size = sizeof(ReductionParams)) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = 2;
            layout.pBindings = reduction_bindings;
            check_result(vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                                     &descriptor_layout),
                         "could not create reduction descriptor-set layout");
            VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            nullptr, 0, code_size, code};
            check_result(vkCreateShaderModule(device_, &shader, nullptr, &module),
                         "could not create reduction shader module");
             VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
            VkPipelineLayoutCreateInfo pipeline_layout_info{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor_layout;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                                &pipeline_layout),
                         "could not create reduction pipeline layout");
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                  &pipeline_info, nullptr, &pipeline),
                         "could not create reduction pipeline");
        };
        create_extra(reduction_descriptor_layout_, reduction_shader_,
                     reduction_pipeline_layout_, reduction_pipeline_,
                     vulkan_reduction_shader::kReductionCode,
                     vulkan_reduction_shader::kReductionCodeSize);
        create_extra(indexing_descriptor_layout_, indexing_shader_,
                     indexing_pipeline_layout_, indexing_pipeline_,
                     vulkan_reduction_shader::kIndexingCode,
                     vulkan_reduction_shader::kIndexingCodeSize);
        create_extra(broadcast_descriptor_layout_, broadcast_shader_,
                     broadcast_pipeline_layout_, broadcast_pipeline_,
                     vulkan_reduction_shader::kBroadcastCode,
                     vulkan_reduction_shader::kBroadcastCodeSize);
        create_extra(pooling_descriptor_layout_, pooling_shader_,
                     pooling_pipeline_layout_, pooling_pipeline_,
                     vulkan_pooling_shader::kCode, vulkan_pooling_shader::kCodeSize);
         if (platform.supports_formatter_double()) {
            create_extra(f32_to_double_descriptor_layout_, f32_to_double_shader_,
                         f32_to_double_pipeline_layout_, f32_to_double_pipeline_,
                         vulkan_f32_to_double_shader::kCode,
                         vulkan_f32_to_double_shader::kCodeSize,
                          sizeof(F32ToDoubleParams));
             const VkDescriptorSetLayoutBinding formatter_bindings[] = {
                 lhs_binding,
                 {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                 {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                 {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
             VkDescriptorSetLayoutCreateInfo formatter_layout{
                 VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
             formatter_layout.bindingCount = 4;
             formatter_layout.pBindings = formatter_bindings;
             check_result(vkCreateDescriptorSetLayout(device_, &formatter_layout, nullptr,
                                                       &formatter_double_descriptor_layout_),
                          "could not create formatter descriptor layout");
             VkShaderModuleCreateInfo formatter_shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                       nullptr, 0,
                                                       vulkan_formatter_double_shader::kCodeSize,
                                                       vulkan_formatter_double_shader::kCode};
             check_result(vkCreateShaderModule(device_, &formatter_shader, nullptr,
                                               &formatter_double_shader_),
                          "could not create formatter shader module");
             VkPushConstantRange formatter_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                sizeof(FormatterDoubleParams)};
             VkPipelineLayoutCreateInfo formatter_pipeline_layout{
                 VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
             formatter_pipeline_layout.setLayoutCount = 1;
             formatter_pipeline_layout.pSetLayouts = &formatter_double_descriptor_layout_;
             formatter_pipeline_layout.pushConstantRangeCount = 1;
             formatter_pipeline_layout.pPushConstantRanges = &formatter_push;
             check_result(vkCreatePipelineLayout(device_, &formatter_pipeline_layout, nullptr,
                                                  &formatter_double_pipeline_layout_),
                          "could not create formatter pipeline layout");
             VkPipelineShaderStageCreateInfo formatter_stage{
                 VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
             formatter_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
             formatter_stage.module = formatter_double_shader_;
             formatter_stage.pName = "main";
             VkComputePipelineCreateInfo formatter_pipeline{
                 VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
             formatter_pipeline.stage = formatter_stage;
             formatter_pipeline.layout = formatter_double_pipeline_layout_;
             check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                   &formatter_pipeline, nullptr,
                                                   &formatter_double_pipeline_),
                          "could not create formatter pipeline");
        }
        const VkDescriptorSetLayoutBinding model_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        VkDescriptorSetLayoutCreateInfo model_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        model_layout.bindingCount = 4;
        model_layout.pBindings = model_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &model_layout, nullptr,
                                                 &model_descriptor_layout_),
                     "could not create model descriptor-set layout");
        VkShaderModuleCreateInfo model_shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_model_shader::kLinearCodeSize, vulkan_model_shader::kLinearCode};
        check_result(
            vkCreateShaderModule(device_, &model_shader_info, nullptr, &model_shader_),
            "could not create model shader module");
        VkPushConstantRange model_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(LinearParams)};
        VkPipelineLayoutCreateInfo model_pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        model_pipeline_layout_info.setLayoutCount = 1;
        model_pipeline_layout_info.pSetLayouts = &model_descriptor_layout_;
        model_pipeline_layout_info.pushConstantRangeCount = 1;
        model_pipeline_layout_info.pPushConstantRanges = &model_push;
        check_result(vkCreatePipelineLayout(device_, &model_pipeline_layout_info,
                                            nullptr, &model_pipeline_layout_),
                     "could not create model pipeline layout");
        VkPipelineShaderStageCreateInfo model_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        model_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        model_stage.module = model_shader_;
        model_stage.pName = "main";
        VkComputePipelineCreateInfo model_pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        model_pipeline_info.stage = model_stage;
        model_pipeline_info.layout = model_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                              &model_pipeline_info, nullptr,
                                              &model_pipeline_),
                     "could not create model pipeline");
        VkDescriptorSetLayoutCreateInfo convolution_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        convolution_layout.bindingCount = 4;
        convolution_layout.pBindings = model_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &convolution_layout, nullptr,
                                                 &convolution_descriptor_layout_),
                     "could not create convolution descriptor-set layout");
        VkShaderModuleCreateInfo convolution_shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_convolution_shader::kCodeSize, vulkan_convolution_shader::kCode};
        check_result(vkCreateShaderModule(device_, &convolution_shader_info, nullptr,
                                          &convolution_shader_),
                     "could not create convolution shader module");
        VkPushConstantRange convolution_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                             sizeof(ConvolutionParams)};
        VkPipelineLayoutCreateInfo convolution_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        convolution_layout_info.setLayoutCount = 1;
        convolution_layout_info.pSetLayouts = &convolution_descriptor_layout_;
        convolution_layout_info.pushConstantRangeCount = 1;
        convolution_layout_info.pPushConstantRanges = &convolution_push;
        check_result(vkCreatePipelineLayout(device_, &convolution_layout_info, nullptr,
                                            &convolution_pipeline_layout_),
                     "could not create convolution pipeline layout");
        VkPipelineShaderStageCreateInfo convolution_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        convolution_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        convolution_stage.module = convolution_shader_;
        convolution_stage.pName = "main";
        VkComputePipelineCreateInfo convolution_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        convolution_info.stage = convolution_stage;
        convolution_info.layout = convolution_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                              &convolution_info, nullptr,
                                              &convolution_pipeline_),
                     "could not create convolution pipeline");
        const VkDescriptorSetLayoutBinding masked_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        const auto create_masked = [&](VkDescriptorSetLayout &descriptor,
                                       VkShaderModule &module,
                                       VkPipelineLayout &layout_handle,
                                       VkPipeline &pipeline, const uint32_t *code,
                                       std::size_t code_size, uint32_t binding_count) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = binding_count;
            layout.pBindings = masked_bindings;
            check_result(
                vkCreateDescriptorSetLayout(device_, &layout, nullptr, &descriptor),
                "could not create masked-select descriptor layout");
            VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            nullptr, 0, code_size, code};
            check_result(vkCreateShaderModule(device_, &shader, nullptr, &module),
                         "could not create masked-select shader module");
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(MaskedParams)};
            VkPipelineLayoutCreateInfo pipeline_layout_info{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                                &layout_handle),
                         "could not create masked-select pipeline layout");
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = layout_handle;
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info,
                                                  nullptr, &pipeline),
                         "could not create masked-select pipeline");
        };
        create_masked(masked_count_descriptor_layout_, masked_count_shader_,
                      masked_count_pipeline_layout_, masked_count_pipeline_,
                      vulkan_masked_select_shader::kCountCode,
                      vulkan_masked_select_shader::kCountCodeSize, 3);
        create_masked(masked_compact_descriptor_layout_, masked_compact_shader_,
                      masked_compact_pipeline_layout_, masked_compact_pipeline_,
                      vulkan_masked_select_shader::kCompactCode,
                      vulkan_masked_select_shader::kCompactCodeSize, 4);
    } catch (const std::exception &error) {
        for (auto pipeline : pipelines_) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
        }
        if (reduction_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
        if (indexing_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
        if (broadcast_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
        if (model_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, model_pipeline_, nullptr);
        if (convolution_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, convolution_pipeline_, nullptr);
        if (pooling_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, pooling_pipeline_, nullptr);
        if (masked_count_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, masked_count_pipeline_, nullptr);
        if (masked_compact_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, masked_compact_pipeline_, nullptr);
        for (auto module : shader_modules_) {
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, module, nullptr);
            }
        }
        if (reduction_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, reduction_shader_, nullptr);
        if (indexing_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, indexing_shader_, nullptr);
        if (broadcast_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
        if (model_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, model_shader_, nullptr);
        if (convolution_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, convolution_shader_, nullptr);
        if (pooling_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, pooling_shader_, nullptr);
        if (masked_count_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, masked_count_shader_, nullptr);
        if (masked_compact_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, masked_compact_shader_, nullptr);
        for (auto layout : pipeline_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
        }
        if (reduction_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
        if (indexing_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
        if (broadcast_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
        if (model_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
        if (convolution_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, convolution_pipeline_layout_, nullptr);
        if (pooling_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, pooling_pipeline_layout_, nullptr);
        if (masked_count_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, masked_count_pipeline_layout_, nullptr);
        if (masked_compact_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, masked_compact_pipeline_layout_, nullptr);
        for (auto layout : descriptor_set_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
        }
        if (reduction_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_,
                                         nullptr);
        if (indexing_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
        if (broadcast_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_,
                                         nullptr);
        if (model_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
        if (convolution_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, convolution_descriptor_layout_,
                                         nullptr);
        if (pooling_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, pooling_descriptor_layout_, nullptr);
        if (masked_count_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, masked_count_descriptor_layout_,
                                         nullptr);
        if (masked_compact_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, masked_compact_descriptor_layout_,
                                         nullptr);
        throw contextual_error("initialization failed", error);
    }
}

VulkanCompute::~VulkanCompute() {
    for (auto pipeline : pipelines_) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline, nullptr);
        }
    }
    if (reduction_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
    if (indexing_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
    if (broadcast_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
    if (model_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, model_pipeline_, nullptr);
    if (convolution_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, convolution_pipeline_, nullptr);
    if (pooling_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pooling_pipeline_, nullptr);
    if (masked_count_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, masked_count_pipeline_, nullptr);
    if (masked_compact_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, masked_compact_pipeline_, nullptr);
        if (f32_to_double_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, f32_to_double_pipeline_, nullptr);
        if (formatter_double_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, formatter_double_pipeline_, nullptr);
    for (auto module : shader_modules_) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module, nullptr);
        }
    }
    if (reduction_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, reduction_shader_, nullptr);
    if (indexing_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, indexing_shader_, nullptr);
    if (broadcast_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
    if (model_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, model_shader_, nullptr);
    if (convolution_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, convolution_shader_, nullptr);
    if (pooling_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, pooling_shader_, nullptr);
    if (masked_count_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, masked_count_shader_, nullptr);
    if (masked_compact_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, masked_compact_shader_, nullptr);
        if (f32_to_double_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, f32_to_double_shader_, nullptr);
        if (formatter_double_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, formatter_double_shader_, nullptr);
    for (auto layout : pipeline_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout, nullptr);
        }
    }
    if (reduction_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
    if (indexing_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
    if (broadcast_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
    if (model_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
    if (convolution_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, convolution_pipeline_layout_, nullptr);
    if (pooling_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, pooling_pipeline_layout_, nullptr);
    if (masked_count_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, masked_count_pipeline_layout_, nullptr);
    if (masked_compact_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, masked_compact_pipeline_layout_, nullptr);
        if (f32_to_double_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, f32_to_double_pipeline_layout_, nullptr);
        if (formatter_double_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, formatter_double_pipeline_layout_, nullptr);
    for (auto layout : descriptor_set_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
        }
    }
    if (reduction_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_, nullptr);
    if (indexing_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
    if (broadcast_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_, nullptr);
    if (model_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
    if (convolution_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, convolution_descriptor_layout_, nullptr);
    if (pooling_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, pooling_descriptor_layout_, nullptr);
    if (masked_count_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, masked_count_descriptor_layout_, nullptr);
    if (masked_compact_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, masked_compact_descriptor_layout_,
                                     nullptr);
        if (f32_to_double_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, f32_to_double_descriptor_layout_, nullptr);
        if (formatter_double_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, formatter_double_descriptor_layout_, nullptr);
}

void VulkanCompute::add(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                        VkDeviceSize bytes) const {
    tensor_tensor(lhs, rhs, output, bytes, kAdd);
}

void VulkanCompute::tensor_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation,
                                  bool bool_dtype) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation, false, bool_dtype);
}

void VulkanCompute::tensor_tensor_alias(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                        VkDeviceSize bytes, uint32_t operation,
                                        bool bool_dtype) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation, true, bool_dtype);
}

void VulkanCompute::tensor_scalar(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                                  float scalar, uint32_t operation,
                                  bool bool_dtype) const {
    dispatch(1, tensor, VK_NULL_HANDLE, output, bytes, scalar, operation, false,
             bool_dtype);
}

void VulkanCompute::tensor_scalar_alias(VkBuffer tensor, VkBuffer output,
                                        VkDeviceSize bytes, float scalar,
                                        uint32_t operation, bool bool_dtype) const {
    dispatch(1, tensor, VK_NULL_HANDLE, output, bytes, scalar, operation, true,
             bool_dtype);
}

void VulkanCompute::scalar_tensor(float scalar, VkBuffer tensor, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation,
                                  bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, tensor, output, bytes, scalar, operation, false,
             bool_dtype);
}

void VulkanCompute::scalar_tensor_alias(float scalar, VkBuffer tensor, VkBuffer output,
                                        VkDeviceSize bytes, uint32_t operation,
                                        bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, tensor, output, bytes, scalar, operation, true,
             bool_dtype);
}

void VulkanCompute::unary(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                          uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation, false,
             bool_dtype);
}

void VulkanCompute::unary_alias(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                                uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation, true,
             bool_dtype);
}

void VulkanCompute::unary_offset(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                                 uint32_t operation, VkDeviceSize input_offset) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation, false, false, false,
             input_offset, 0);
}

void VulkanCompute::comparison_scalar(VkBuffer input, VkBuffer output,
                                      VkDeviceSize bytes, float scalar) const {
    dispatch(1, input, VK_NULL_HANDLE, output, bytes, scalar, 8, false, false, true);
}

void VulkanCompute::comparison_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                      VkDeviceSize bytes, uint32_t operation) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation, false, false, true);
}

void VulkanCompute::isfinite(VkBuffer input, VkBuffer output,
                             VkDeviceSize bytes) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, 9, false, false, true);
}

void VulkanCompute::masked_select_count(VkBuffer input, VkBuffer mask, VkBuffer counter,
                                        uint32_t element_count) const {
    dispatch_masked(input, mask, VK_NULL_HANDLE, counter, element_count,
                    sizeof(uint32_t), masked_count_pipeline_,
                    masked_count_pipeline_layout_, masked_count_descriptor_layout_, 3);
}

void VulkanCompute::masked_select_compact(VkBuffer input, VkBuffer mask,
                                          VkBuffer output, VkBuffer counter,
                                          uint32_t element_count,
                                          uint32_t output_count) const {
    dispatch_masked(input, mask, output, counter, element_count,
                    static_cast<VkDeviceSize>(output_count) * sizeof(float),
                    masked_compact_pipeline_, masked_compact_pipeline_layout_,
                    masked_compact_descriptor_layout_, 4);
}

void VulkanCompute::reduction(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                              uint32_t rank, const uint32_t *sizes,
                              uint32_t reduce_mask, uint32_t reduce_numel,
                              uint32_t output_numel, bool mean) const {
    ReductionParams params{rank,         output_numel, reduce_mask,
                           reduce_numel, {},           mean ? 1U : 0U};
    for (uint32_t i = 0; i < rank; ++i)
        params.sizes[i] = sizes[i];
    dispatch_extra(input, output, input_bytes,
                   static_cast<VkDeviceSize>(output_numel) * sizeof(float),
                   reduction_pipeline_, reduction_pipeline_layout_,
                   reduction_descriptor_layout_, &params, sizeof(params), output_numel);
}

void VulkanCompute::argmax(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                           uint32_t rank, const uint32_t *sizes, uint32_t dim,
                           uint32_t reduce_size, uint32_t output_numel) const {
    IndexingParams params{rank, output_numel, dim, reduce_size, {}};
    for (uint32_t i = 0; i < rank; ++i)
        params.sizes[i] = sizes[i];
    dispatch_extra(input, output, input_bytes,
                   static_cast<VkDeviceSize>(output_numel) * sizeof(int64_t),
                   indexing_pipeline_, indexing_pipeline_layout_,
                   indexing_descriptor_layout_, &params, sizeof(params), output_numel);
}

void VulkanCompute::broadcast(VkBuffer input, VkBuffer output, uint32_t rank,
                              const uint32_t *input_sizes, const uint32_t *output_sizes,
                              uint32_t output_numel, float scale) const {
    if (rank > 8)
        throw std::invalid_argument("Vulkan compute broadcast supports ranks up to 8");
    BroadcastParams params{rank, output_numel, scale, {}, {}};
    for (uint32_t i = 0; i < rank; ++i) {
        params.input_sizes[i] = input_sizes[i];
        params.output_sizes[i] = output_sizes[i];
    }
    uint64_t input_numel = 1;
    for (uint32_t i = 0; i < rank; ++i)
        input_numel *= input_sizes[i];
    dispatch_extra(input, output,
                   static_cast<VkDeviceSize>(input_numel * sizeof(float)),
                   static_cast<VkDeviceSize>(output_numel * sizeof(float)),
                   broadcast_pipeline_, broadcast_pipeline_layout_,
                   broadcast_descriptor_layout_, &params, sizeof(params), output_numel);
}

void VulkanCompute::linear(VkBuffer input, VkBuffer weight, VkBuffer bias,
                           VkBuffer output, uint32_t rows, uint32_t features,
                           uint32_t outputs, bool transposed_weight, bool has_bias,
                           uint32_t operation) const {
    const uint64_t rows64 = rows;
    const uint64_t outputs64 = outputs;
    if (outputs64 != 0 && rows64 > std::numeric_limits<uint64_t>::max() / outputs64)
        throw std::invalid_argument(
            "Vulkan linear output count multiplication overflows uint64");
    const uint64_t output_numel64 = rows64 * outputs64;
    if (output_numel64 > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan linear output count overflow");
    LinearParams params{rows,     features, outputs, transposed_weight,
                        has_bias, operation};
    dispatch_model(input, weight, bias, output,
                   static_cast<VkDeviceSize>(rows) * features * sizeof(float),
                   static_cast<VkDeviceSize>(features) * outputs * sizeof(float),
                   static_cast<VkDeviceSize>(outputs) * sizeof(float),
                   static_cast<VkDeviceSize>(output_numel64) * sizeof(float), &params,
                   sizeof(params), static_cast<uint32_t>(output_numel64));
}

void VulkanCompute::convolution(VkBuffer input, VkBuffer weight, VkBuffer bias,
                                VkBuffer output, uint32_t operation) const {
    ConvolutionParams params{2, 1, 8, 8, 4, 8, 8, 3, 3, operation};
    const uint32_t output_numel = operation == 0   ? 512
                                  : operation == 1 ? 128
                                  : operation == 2 ? 36
                                                   : 4;
    const VkDeviceSize output_bytes =
        static_cast<VkDeviceSize>(output_numel) * sizeof(float);
    dispatch_model(input, weight, bias, output, 512 * sizeof(float),
                   (operation == 2 ? 512 : 36) * sizeof(float), 4 * sizeof(float),
                   output_bytes, &params, sizeof(params), output_numel,
                   convolution_pipeline_, convolution_pipeline_layout_,
                   convolution_descriptor_layout_);
}

void VulkanCompute::pooling(VkBuffer input, VkBuffer output, uint32_t batch,
                            uint32_t channels, uint32_t height, uint32_t width,
                            uint32_t operation) const {
    if (operation > 1)
        throw std::invalid_argument("Vulkan pooling operation is unsupported");
    const uint64_t spatial = checked_product(height, width, "spatial");
    const uint64_t batch_channels = checked_product(batch, channels, "batch-channel");
    const uint64_t input_numel = checked_product(batch_channels, spatial, "input");
    const uint64_t output_numel64 = operation == 0 ? batch_channels : input_numel;
    if (output_numel64 > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan pooling output count truncation");
    PoolingParams params{batch, channels, height, width, operation};
    const uint32_t output_numel = static_cast<uint32_t>(output_numel64);
    dispatch_extra(
        input, output,
        checked_bytes(operation == 0 ? input_numel : batch_channels, "input"),
        checked_bytes(output_numel64, "output"), pooling_pipeline_,
        pooling_pipeline_layout_, pooling_descriptor_layout_, &params, sizeof(params),
        output_numel);
}

void VulkanCompute::f32_to_double(VkBuffer input, VkBuffer output,
                                  VkDeviceSize input_bytes, VkDeviceSize output_bytes,
                                  uint32_t element_count) const {
    if (!platform_.supports_formatter_double())
        throw std::invalid_argument("Vulkan formatter Double support is unavailable");
    F32ToDoubleParams params{element_count};
    dispatch_extra(input, output, input_bytes, output_bytes, f32_to_double_pipeline_,
                   f32_to_double_pipeline_layout_, f32_to_double_descriptor_layout_,
                   &params, sizeof(params), element_count);
}

void VulkanCompute::formatter_double(VkBuffer input, VkBuffer rhs, VkBuffer output,
                                     VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                                     VkDeviceSize output_bytes, uint32_t element_count,
                                     uint32_t operation, double scalar,
                                     uint32_t output_numel, bool bool_output) const {
    if (!platform_.supports_formatter_double())
        throw std::invalid_argument("Vulkan formatter Double support is unavailable");
    FormatterDoubleParams params{scalar, element_count, operation};
    dispatch_formatter(input, rhs, output, input_bytes, rhs_bytes, output_bytes, &params,
                       sizeof(params), output_numel, bool_output);
}

void VulkanCompute::dispatch_model(VkBuffer input, VkBuffer weight, VkBuffer bias,
                                   VkBuffer output, VkDeviceSize input_bytes,
                                   VkDeviceSize weight_bytes, VkDeviceSize bias_bytes,
                                   VkDeviceSize output_bytes, const void *params,
                                   uint32_t params_size, uint32_t output_numel,
                                   VkPipeline pipeline,
                                   VkPipelineLayout pipeline_layout,
                                   VkDescriptorSetLayout descriptor_layout) const {
    if (pipeline == VK_NULL_HANDLE) {
        pipeline = model_pipeline_;
        pipeline_layout = model_pipeline_layout_;
        descriptor_layout = model_descriptor_layout_;
    }
    if (input == VK_NULL_HANDLE || weight == VK_NULL_HANDLE || bias == VK_NULL_HANDLE ||
        output == VK_NULL_HANDLE || !input_bytes || !weight_bytes || !bias_bytes ||
        !output_bytes || !output_numel || input_bytes > max_storage_buffer_range_ ||
        weight_bytes > max_storage_buffer_range_ ||
        bias_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        static_cast<uint64_t>(output_numel) >
            static_cast<uint64_t>(max_compute_workgroup_count_x_) * kWorkgroupSize ||
        (output_numel - 1) / kWorkgroupSize + 1 > max_compute_workgroup_count_x_)
        throw std::invalid_argument("Vulkan model compute has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    const auto cleanup = [&] {
        if (fence)
            vkDestroyFence(device_, fence, nullptr);
        if (cmd)
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool)
            vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create model descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate model descriptor set");
        VkDescriptorBufferInfo infos[] = {{input, 0, input_bytes},
                                          {weight, 0, weight_bytes},
                                          {bias, 0, bias_bytes},
                                          {output, 0, output_bytes}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
        VkCommandBufferAllocateInfo alloc{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = command_pool_;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &alloc, &cmd),
                     "could not allocate model command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin),
                     "could not begin model command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           params_size, params);
        vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        check_result(vkEndCommandBuffer(cmd), "could not end model command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create model fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence),
                     "could not submit model dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for model dispatch");
        cleanup();
    } catch (const std::exception &error) {
        if (submitted)
            vkQueueWaitIdle(queue_);
        cleanup();
        throw contextual_error("model dispatch failed", error);
    }
}

void VulkanCompute::dispatch_extra(VkBuffer input, VkBuffer output,
                                   VkDeviceSize input_bytes, VkDeviceSize output_bytes,
                                   VkPipeline pipeline,
                                   VkPipelineLayout pipeline_layout,
                                   VkDescriptorSetLayout descriptor_layout,
                                    const void *params, uint32_t params_size,
                                    uint32_t output_numel) const {
    const uint64_t max_elements = static_cast<uint64_t>(max_compute_workgroup_count_x_) *
                                  static_cast<uint64_t>(kWorkgroupSize);
    if (input == VK_NULL_HANDLE || output == VK_NULL_HANDLE || output_numel == 0 ||
        input_bytes == 0 || input_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        static_cast<uint64_t>(output_numel) > max_elements)
        throw std::invalid_argument("Vulkan compute reduction has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&] {
        if (fence != VK_NULL_HANDLE)
            vkDestroyFence(device_, fence, nullptr);
        if (cmd != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    bool submitted = false;
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create reduction descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate reduction descriptor set");
        VkDescriptorBufferInfo buffers[] = {{input, 0, input_bytes},
                                            {output, 0, output_bytes}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &cmd),
                     "could not allocate reduction command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin),
                     "could not begin reduction command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           params_size, params);
        vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        check_result(vkEndCommandBuffer(cmd), "could not end reduction command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create reduction fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence),
                     "could not submit reduction dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for reduction dispatch");
        cleanup();
    } catch (const std::exception &error) {
        if (submitted)
            vkQueueWaitIdle(queue_);
        cleanup();
        throw contextual_error("reduction dispatch failed", error);
    }
}

void VulkanCompute::dispatch_formatter(VkBuffer input, VkBuffer rhs, VkBuffer output,
                                       VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                                       VkDeviceSize output_bytes, const void *params,
                                       uint32_t params_size, uint32_t output_numel,
                                       bool bool_output) const {
    const uint64_t max_elements = static_cast<uint64_t>(max_compute_workgroup_count_x_) *
                                  static_cast<uint64_t>(kWorkgroupSize);
    if (input == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE || output == VK_NULL_HANDLE ||
        output_numel == 0 || input_bytes == 0 || rhs_bytes == 0 || output_bytes == 0 ||
        input_bytes > max_storage_buffer_range_ || rhs_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ || output_numel > max_elements)
        throw std::invalid_argument("Vulkan formatter Double compute has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&] {
        if (fence) vkDestroyFence(device_, fence, nullptr);
        if (cmd) vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool) vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    bool submitted = false;
    try {
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create formatter descriptor pool");
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &formatter_double_descriptor_layout_;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate formatter descriptor set");
        const VkDeviceSize bool_bytes = output_bytes;
        VkDescriptorBufferInfo buffers[] = {{input, 0, input_bytes}, {rhs, 0, rhs_bytes},
                                            {output, 0, output_bytes},
                                            {output, 0, bool_bytes}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &cmd),
                     "could not allocate formatter command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin), "could not begin formatter command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, formatter_double_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                formatter_double_pipeline_layout_, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, formatter_double_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, params_size, params);
        vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        check_result(vkEndCommandBuffer(cmd), "could not end formatter command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create formatter fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence), "could not submit formatter dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for formatter dispatch");
        cleanup();
    } catch (const std::exception &error) {
        if (submitted) vkQueueWaitIdle(queue_);
        cleanup();
        throw contextual_error("formatter dispatch failed", error);
    }
}

void VulkanCompute::dispatch_masked(VkBuffer input, VkBuffer mask, VkBuffer output,
                                    VkBuffer counter, uint32_t element_count,
                                    VkDeviceSize output_bytes, VkPipeline pipeline,
                                    VkPipelineLayout pipeline_layout,
                                     VkDescriptorSetLayout descriptor_layout,
                                     uint32_t descriptor_count) const {
    const uint64_t max_elements = static_cast<uint64_t>(max_compute_workgroup_count_x_) *
                                  static_cast<uint64_t>(kWorkgroupSize);
    const VkDeviceSize input_bytes =
        static_cast<VkDeviceSize>(element_count) * sizeof(float);
    const VkDeviceSize mask_bytes = element_count;
    if (input == VK_NULL_HANDLE || mask == VK_NULL_HANDLE ||
        counter == VK_NULL_HANDLE ||
        (descriptor_count == 4 && output == VK_NULL_HANDLE) || element_count == 0 ||
        input_bytes > max_storage_buffer_range_ ||
        mask_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ || output_bytes == 0 ||
         static_cast<uint64_t>(element_count) > max_elements)
        throw std::invalid_argument("Vulkan masked-select has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    const auto cleanup = [&] {
        if (fence)
            vkDestroyFence(device_, fence, nullptr);
        if (cmd)
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool)
            vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       descriptor_count};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create masked-select descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate masked-select descriptor set");
        VkDescriptorBufferInfo infos[] = {{input, 0, input_bytes},
                                          {mask, 0, mask_bytes},
                                          {output, 0, output_bytes},
                                          {counter, 0, sizeof(uint32_t)}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i == 3 ? 3 : i];
        }
        if (descriptor_count == 3)
            writes[2].pBufferInfo = &infos[3];
        vkUpdateDescriptorSets(device_, descriptor_count, writes, 0, nullptr);
        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &cmd),
                     "could not allocate masked-select command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin),
                     "could not begin masked-select command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        MaskedParams params{element_count};
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(params), &params);
        vkCmdDispatch(cmd, 1, 1, 1);
        check_result(vkEndCommandBuffer(cmd),
                     "could not end masked-select command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create masked-select fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence),
                     "could not submit masked-select dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for masked-select dispatch");
        cleanup();
    } catch (const std::exception &error) {
        if (submitted) {
            const VkResult recovery = vkQueueWaitIdle(queue_);
            if (recovery != VK_SUCCESS) {
                platform_.defer_compute_resources(pool, cmd, fence);
                pool = VK_NULL_HANDLE;
                cmd = VK_NULL_HANDLE;
                fence = VK_NULL_HANDLE;
                throw std::runtime_error(
                    std::string("Vulkan masked-select dispatch failed: ") +
                    error.what() + "; could not confirm cleanup completion");
            }
        }
        cleanup();
        throw contextual_error("masked-select dispatch failed", error);
    }
}

void VulkanCompute::dispatch(uint32_t mode, VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                             VkDeviceSize bytes, float scalar, uint32_t operation,
                             bool exact_alias, bool bool_dtype,
                             bool bool_output, VkDeviceSize lhs_offset,
                             VkDeviceSize output_offset) const {
    if (mode > 3 || output == VK_NULL_HANDLE ||
        (mode == 0 && (lhs == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE)) ||
        (mode != 0 && ((lhs == VK_NULL_HANDLE) == (rhs == VK_NULL_HANDLE)))) {
        throw std::invalid_argument("Vulkan compute pointwise requires valid buffers");
    }
    if ((bool_dtype || bool_output) && !platform_.supports_bool_pointwise()) {
        throw std::invalid_argument("Vulkan compute bool pointwise is unavailable");
    }
    if (exact_alias && ((mode == 0 && output != lhs && output != rhs) ||
                        (mode == 1 && output != lhs) || (mode == 2 && output != rhs) ||
                        (mode == 3 && output != lhs))) {
        throw std::invalid_argument(
            "Vulkan compute exact alias does not match an input buffer");
    }
    std::scoped_lock lock(platform_.queue_mutex());
    const std::size_t element_bytes = bool_dtype ? sizeof(bool) : sizeof(float);
    const VkDeviceSize elements = bytes / element_bytes;
    const VkDeviceSize output_bytes = bool_output ? elements * sizeof(bool) : bytes;
    if (!bytes || bytes % element_bytes || bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        elements > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(
            "Vulkan compute pointwise has an invalid byte range");
    const uint32_t groups =
        static_cast<uint32_t>((elements + kWorkgroupSize - 1) / kWorkgroupSize);
    if (groups > max_compute_workgroup_count_x_) {
        throw std::invalid_argument("Vulkan compute pointwise exceeds workgroup limit");
    }
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    const auto cleanup = [&] {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence, nullptr);
        }
        if (cmd != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, pool, nullptr);
        }
    };
    try {
        const uint32_t pipeline_mode = bool_output
                                           ? (mode == 0 ? 7U : (mode == 1 ? 5U : 6U))
                                           : mode + (bool_dtype ? 4 : 0);
        const uint32_t descriptor_count = mode == 0 ? 3 : 2;
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       descriptor_count};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create pointwise descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_set_layouts_[pipeline_mode];
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate pointwise descriptor set");
        VkDescriptorBufferInfo buffers[] = {
            {lhs, lhs_offset, bytes}, {rhs, 0, bytes}, {output, output_offset, output_bytes}};
        VkWriteDescriptorSet writes[3]{};
        const uint32_t bindings[] = {0, mode == 0 ? 1U : 2U, 2};
        const uint32_t write_count = mode == 0 ? 3 : 2;
        for (uint32_t i = 0; i < write_count; ++i) {
            const uint32_t source =
                mode == 0 ? i : (i == 0 ? ((mode == 1 || mode == 3) ? 0 : 1) : 2);
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = bindings[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[source];
        }
        vkUpdateDescriptorSets(device_, write_count, writes, 0, nullptr);

        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &cmd),
                     "could not allocate pointwise command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin),
                     "could not begin pointwise command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelines_[pipeline_mode]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layouts_[pipeline_mode], 0, 1, &set, 0,
                                nullptr);
        Params params{scalar, static_cast<uint32_t>(elements), operation};
        vkCmdPushConstants(cmd, pipeline_layouts_[pipeline_mode],
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
        vkCmdDispatch(cmd, groups, 1, 1);
        check_result(vkEndCommandBuffer(cmd), "could not end pointwise command buffer");

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create pointwise fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence),
                     "could not submit pointwise dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for pointwise dispatch");
        cleanup();
    } catch (const std::exception &error) {
        VkResult recovery = VK_SUCCESS;
        if (submitted) {
            recovery = vkQueueWaitIdle(queue_);
        }
        if (recovery != VK_SUCCESS) {
            platform_.defer_compute_resources(pool, cmd, fence);
            pool = VK_NULL_HANDLE;
            cmd = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            throw std::runtime_error(std::string("Vulkan compute dispatch failed: ") +
                                     error.what() +
                                     "; could not confirm cleanup completion");
        }
        cleanup();
        throw contextual_error("dispatch failed", error);
    }
}

std::size_t VulkanCompute::dispatch_count() const {
    return dispatch_count_.load(std::memory_order_relaxed);
}
