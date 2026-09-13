#include "vulkan_compute.h"

#include "vulkan/shaders/generated/pointwise_spv.h"
#include "vulkan/shaders/generated/reduction_indexing_spv.h"
#include "vulkan/shaders/generated/model_spv.h"
#include "vulkan_platform.h"

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

std::runtime_error contextual_error(const char *operation, const std::exception &error) {
    return std::runtime_error(std::string("Vulkan compute ") + operation + ": " + error.what());
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

struct LinearParams { uint32_t rows, features, outputs; };

constexpr uint32_t kAdd = 0;
constexpr uint32_t kWorkgroupSize = 256;

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
            throw std::runtime_error("device maxPushConstantsSize is smaller than pointwise ABI");

        const VkDescriptorSetLayoutBinding lhs_binding = {
            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const VkDescriptorSetLayoutBinding rhs_binding = {
            1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const VkDescriptorSetLayoutBinding output_binding = {
            2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const VkDescriptorSetLayoutBinding tensor_tensor_bindings[] = {
            lhs_binding, rhs_binding, output_binding};
        const VkDescriptorSetLayoutBinding tensor_scalar_bindings[] = {
            lhs_binding, output_binding};
        const VkDescriptorSetLayoutBinding scalar_tensor_bindings[] = {
            lhs_binding, output_binding};
        const VkDescriptorSetLayoutBinding unary_bindings[] = {lhs_binding, output_binding};

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
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
        const auto create_pipeline = [&](uint32_t mode) {
            VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &descriptor_set_layouts_[mode];
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layouts_[mode]),
                         "could not create pointwise pipeline layout");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader_modules_[mode];
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline.stage = stage;
            pipeline.layout = pipeline_layouts_[mode];
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr,
                                                   &pipelines_[mode]),
                         "could not create pointwise compute pipeline");
        };
        create_pipeline(0);
        create_pipeline(1);
        create_pipeline(2);
        create_pipeline(3);
        if (platform.supports_bool_pointwise()) {
            create_pipeline(4);
        }
        const VkDescriptorSetLayoutBinding reduction_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const auto create_extra = [&](VkDescriptorSetLayout &descriptor_layout,
                                      VkShaderModule &module, VkPipelineLayout &pipeline_layout,
                                      VkPipeline &pipeline, const uint32_t *code,
                                      std::size_t code_size) {
            VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = 2;
            layout.pBindings = reduction_bindings;
            check_result(vkCreateDescriptorSetLayout(device_, &layout, nullptr, &descriptor_layout),
                         "could not create reduction descriptor-set layout");
            VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                            code_size, code};
            check_result(vkCreateShaderModule(device_, &shader, nullptr, &module),
                         "could not create reduction shader module");
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     static_cast<uint32_t>(sizeof(ReductionParams))};
            VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor_layout;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout),
                         "could not create reduction pipeline layout");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                                   &pipeline), "could not create reduction pipeline");
        };
        create_extra(reduction_descriptor_layout_, reduction_shader_, reduction_pipeline_layout_,
                     reduction_pipeline_, vulkan_reduction_shader::kReductionCode,
                     vulkan_reduction_shader::kReductionCodeSize);
        create_extra(indexing_descriptor_layout_, indexing_shader_, indexing_pipeline_layout_,
                     indexing_pipeline_, vulkan_reduction_shader::kIndexingCode,
                     vulkan_reduction_shader::kIndexingCodeSize);
        create_extra(broadcast_descriptor_layout_, broadcast_shader_, broadcast_pipeline_layout_,
                     broadcast_pipeline_, vulkan_reduction_shader::kBroadcastCode,
                     vulkan_reduction_shader::kBroadcastCodeSize);
        const VkDescriptorSetLayoutBinding model_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo model_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        model_layout.bindingCount = 4; model_layout.pBindings = model_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &model_layout, nullptr, &model_descriptor_layout_),
                     "could not create model descriptor-set layout");
        VkShaderModuleCreateInfo model_shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                   vulkan_model_shader::kLinearCodeSize,
                                                   vulkan_model_shader::kLinearCode};
        check_result(vkCreateShaderModule(device_, &model_shader_info, nullptr, &model_shader_),
                     "could not create model shader module");
        VkPushConstantRange model_push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LinearParams)};
        VkPipelineLayoutCreateInfo model_pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        model_pipeline_layout_info.setLayoutCount = 1; model_pipeline_layout_info.pSetLayouts = &model_descriptor_layout_;
        model_pipeline_layout_info.pushConstantRangeCount = 1; model_pipeline_layout_info.pPushConstantRanges = &model_push;
        check_result(vkCreatePipelineLayout(device_, &model_pipeline_layout_info, nullptr, &model_pipeline_layout_),
                     "could not create model pipeline layout");
        VkPipelineShaderStageCreateInfo model_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        model_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; model_stage.module = model_shader_; model_stage.pName = "main";
        VkComputePipelineCreateInfo model_pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        model_pipeline_info.stage = model_stage; model_pipeline_info.layout = model_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &model_pipeline_info, nullptr, &model_pipeline_),
                     "could not create model pipeline");
    } catch (const std::exception &error) {
        for (auto pipeline : pipelines_) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
        }
        if (reduction_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
        if (indexing_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
        if (broadcast_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
        if (model_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, model_pipeline_, nullptr);
        for (auto module : shader_modules_) {
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, module, nullptr);
            }
        }
        if (reduction_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, reduction_shader_, nullptr);
        if (indexing_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, indexing_shader_, nullptr);
        if (broadcast_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
        if (model_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, model_shader_, nullptr);
        for (auto layout : pipeline_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
        }
        if (reduction_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
        if (indexing_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
        if (broadcast_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
        if (model_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
        for (auto layout : descriptor_set_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
        }
        if (reduction_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_, nullptr);
        if (indexing_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
        if (broadcast_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_, nullptr);
        if (model_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
        throw contextual_error("initialization failed", error);
    }
}

VulkanCompute::~VulkanCompute() {
    for (auto pipeline : pipelines_) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline, nullptr);
        }
    }
    if (reduction_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
    if (indexing_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
    if (broadcast_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
    if (model_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, model_pipeline_, nullptr);
    for (auto module : shader_modules_) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module, nullptr);
        }
    }
    if (reduction_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, reduction_shader_, nullptr);
    if (indexing_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, indexing_shader_, nullptr);
    if (broadcast_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
    if (model_shader_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, model_shader_, nullptr);
    for (auto layout : pipeline_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout, nullptr);
        }
    }
    if (reduction_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
    if (indexing_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
    if (broadcast_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
    if (model_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
    for (auto layout : descriptor_set_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
        }
    }
    if (reduction_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_, nullptr);
    if (indexing_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
    if (broadcast_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_, nullptr);
    if (model_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
}

void VulkanCompute::add(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes) const {
    tensor_tensor(lhs, rhs, output, bytes, kAdd);
}

void VulkanCompute::tensor_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation, bool bool_dtype) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation, false, bool_dtype);
}

void VulkanCompute::tensor_tensor_alias(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                        VkDeviceSize bytes, uint32_t operation, bool bool_dtype) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation, true, bool_dtype);
}

void VulkanCompute::tensor_scalar(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                                  float scalar, uint32_t operation, bool bool_dtype) const {
    dispatch(1, tensor, VK_NULL_HANDLE, output, bytes, scalar, operation, false, bool_dtype);
}

void VulkanCompute::tensor_scalar_alias(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                                        float scalar, uint32_t operation, bool bool_dtype) const {
    dispatch(1, tensor, VK_NULL_HANDLE, output, bytes, scalar, operation, true, bool_dtype);
}

void VulkanCompute::scalar_tensor(float scalar, VkBuffer tensor, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation, bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, tensor, output, bytes, scalar, operation, false, bool_dtype);
}

void VulkanCompute::scalar_tensor_alias(float scalar, VkBuffer tensor, VkBuffer output,
                                        VkDeviceSize bytes, uint32_t operation, bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, tensor, output, bytes, scalar, operation, true, bool_dtype);
}

void VulkanCompute::unary(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                          uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation, false, bool_dtype);
}

void VulkanCompute::unary_alias(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                                uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation, true, bool_dtype);
}

void VulkanCompute::reduction(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                              uint32_t rank,
                              const uint32_t *sizes, uint32_t reduce_mask,
                              uint32_t reduce_numel, uint32_t output_numel, bool mean) const {
    ReductionParams params{rank, output_numel, reduce_mask, reduce_numel, {}, mean ? 1U : 0U};
    for (uint32_t i = 0; i < rank; ++i) params.sizes[i] = sizes[i];
    dispatch_extra(input, output, input_bytes,
                   static_cast<VkDeviceSize>(output_numel) * sizeof(float), reduction_pipeline_,
                   reduction_pipeline_layout_, reduction_descriptor_layout_, &params, sizeof(params),
                   output_numel);
}

void VulkanCompute::argmax(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                           uint32_t rank,
                           const uint32_t *sizes, uint32_t dim,
                           uint32_t reduce_size, uint32_t output_numel) const {
    IndexingParams params{rank, output_numel, dim, reduce_size, {}};
    for (uint32_t i = 0; i < rank; ++i) params.sizes[i] = sizes[i];
    dispatch_extra(input, output, input_bytes,
                   static_cast<VkDeviceSize>(output_numel) * sizeof(int64_t), indexing_pipeline_,
                   indexing_pipeline_layout_, indexing_descriptor_layout_, &params, sizeof(params),
                   output_numel);
}

void VulkanCompute::broadcast(VkBuffer input, VkBuffer output, uint32_t rank,
                              const uint32_t *input_sizes, const uint32_t *output_sizes,
                              uint32_t output_numel, float scale) const {
    if (rank > 8) throw std::invalid_argument("Vulkan compute broadcast supports ranks up to 8");
    BroadcastParams params{rank, output_numel, scale, {}, {}};
    for (uint32_t i = 0; i < rank; ++i) {
        params.input_sizes[i] = input_sizes[i];
        params.output_sizes[i] = output_sizes[i];
    }
    uint64_t input_numel = 1;
    for (uint32_t i = 0; i < rank; ++i) input_numel *= input_sizes[i];
    dispatch_extra(input, output, static_cast<VkDeviceSize>(input_numel * sizeof(float)),
                   static_cast<VkDeviceSize>(output_numel * sizeof(float)), broadcast_pipeline_,
                   broadcast_pipeline_layout_, broadcast_descriptor_layout_, &params, sizeof(params),
                   output_numel);
}

void VulkanCompute::linear(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                           uint32_t rows, uint32_t features, uint32_t outputs) const {
    LinearParams params{rows, features, outputs};
    dispatch_model(input, weight, bias, output,
                   static_cast<VkDeviceSize>(rows) * features * sizeof(float),
                   static_cast<VkDeviceSize>(features) * outputs * sizeof(float),
                   static_cast<VkDeviceSize>(outputs) * sizeof(float),
                   static_cast<VkDeviceSize>(rows) * outputs * sizeof(float),
                   &params, sizeof(params), rows * outputs);
}

void VulkanCompute::dispatch_model(VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
                                   VkDeviceSize input_bytes, VkDeviceSize weight_bytes,
                                   VkDeviceSize bias_bytes, VkDeviceSize output_bytes,
                                   const void *params, uint32_t params_size, uint32_t output_numel) const {
    if (input == VK_NULL_HANDLE || weight == VK_NULL_HANDLE || bias == VK_NULL_HANDLE || output == VK_NULL_HANDLE ||
        !input_bytes || !weight_bytes || !bias_bytes || !output_bytes || !output_numel ||
        input_bytes > max_storage_buffer_range_ || weight_bytes > max_storage_buffer_range_ ||
        bias_bytes > max_storage_buffer_range_ || output_bytes > max_storage_buffer_range_ ||
        static_cast<uint64_t>(output_numel) >
            static_cast<uint64_t>(max_compute_workgroup_count_x_) * kWorkgroupSize ||
        (output_numel - 1) / kWorkgroupSize + 1 > max_compute_workgroup_count_x_)
        throw std::invalid_argument("Vulkan model compute has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE; bool submitted = false;
    const auto cleanup = [&] { if (fence) vkDestroyFence(device_, fence, nullptr); if (cmd) vkFreeCommandBuffers(device_, command_pool_, 1, &cmd); if (pool) vkDestroyDescriptorPool(device_, pool, nullptr); };
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool_info.maxSets = 1; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool), "could not create model descriptor pool");
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; set_info.descriptorPool = pool; set_info.descriptorSetCount = 1; set_info.pSetLayouts = &model_descriptor_layout_;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set), "could not allocate model descriptor set");
        VkDescriptorBufferInfo infos[] = {{input, 0, input_bytes}, {weight, 0, weight_bytes}, {bias, 0, bias_bytes}, {output, 0, output_bytes}};
        VkWriteDescriptorSet writes[4]{}; for (uint32_t i = 0; i < 4; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i]; }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; alloc.commandPool = command_pool_; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &alloc, &cmd), "could not allocate model command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; check_result(vkBeginCommandBuffer(cmd, &begin), "could not begin model command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, model_pipeline_); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, model_pipeline_layout_, 0, 1, &set, 0, nullptr); vkCmdPushConstants(cmd, model_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params); vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1); check_result(vkEndCommandBuffer(cmd), "could not end model command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check_result(vkCreateFence(device_, &fence_info, nullptr, &fence), "could not create model fence"); platform_.reserve_compute_resources(); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd; check_result(vkQueueSubmit(queue_, 1, &submit, fence), "could not submit model dispatch"); submitted = true; dispatch_count_.fetch_add(1, std::memory_order_relaxed); check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "could not wait for model dispatch"); cleanup();
    } catch (const std::exception &error) { if (submitted) vkQueueWaitIdle(queue_); cleanup(); throw contextual_error("model dispatch failed", error); }
}

void VulkanCompute::dispatch_extra(VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
                                   VkDeviceSize output_bytes, VkPipeline pipeline,
                                   VkPipelineLayout pipeline_layout,
                                   VkDescriptorSetLayout descriptor_layout, const void *params,
                                   uint32_t params_size, uint32_t output_numel) const {
    if (input == VK_NULL_HANDLE || output == VK_NULL_HANDLE || output_numel == 0 ||
        input_bytes == 0 || input_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        output_numel > max_compute_workgroup_count_x_ * kWorkgroupSize)
        throw std::invalid_argument("Vulkan compute reduction has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&] {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
        if (cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    bool submitted = false;
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool), "could not create reduction descriptor pool");
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool; set_info.descriptorSetCount = 1; set_info.pSetLayouts = &descriptor_layout;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set), "could not allocate reduction descriptor set");
        VkDescriptorBufferInfo buffers[] = {{input, 0, input_bytes}, {output, 0, output_bytes}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set;
            writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &cmd), "could not allocate reduction command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin), "could not begin reduction command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
        vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        check_result(vkEndCommandBuffer(cmd), "could not end reduction command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence), "could not create reduction fence");
        platform_.reserve_compute_resources();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
        check_result(vkQueueSubmit(queue_, 1, &submit, fence), "could not submit reduction dispatch");
        submitted = true; dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "could not wait for reduction dispatch");
        cleanup();
    } catch (const std::exception &error) {
        if (submitted) vkQueueWaitIdle(queue_);
        cleanup();
        throw contextual_error("reduction dispatch failed", error);
    }
}

void VulkanCompute::dispatch(uint32_t mode, VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                             VkDeviceSize bytes, float scalar, uint32_t operation,
                             bool exact_alias, bool bool_dtype) const {
    if (mode > 3 || output == VK_NULL_HANDLE ||
        (mode == 0 && (lhs == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE)) ||
        (mode != 0 && ((lhs == VK_NULL_HANDLE) == (rhs == VK_NULL_HANDLE)))) {
        throw std::invalid_argument("Vulkan compute pointwise requires valid buffers");
    }
    if (bool_dtype && !platform_.supports_bool_pointwise()) {
        throw std::invalid_argument("Vulkan compute bool pointwise is unavailable");
    }
    if (exact_alias &&
        ((mode == 0 && output != lhs && output != rhs) ||
         (mode == 1 && output != lhs) ||
         (mode == 2 && output != rhs) ||
         (mode == 3 && output != lhs))) {
        throw std::invalid_argument("Vulkan compute exact alias does not match an input buffer");
    }
    std::scoped_lock lock(platform_.queue_mutex());
    const std::size_t element_bytes = bool_dtype ? sizeof(bool) : sizeof(float);
    const VkDeviceSize elements = bytes / element_bytes;
    if (!bytes || bytes % element_bytes || bytes > max_storage_buffer_range_ ||
        elements > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan compute pointwise has an invalid byte range");
    const uint32_t groups = static_cast<uint32_t>(
        (elements + kWorkgroupSize - 1) / kWorkgroupSize);
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
        const uint32_t pipeline_mode = mode + (bool_dtype ? 4 : 0);
        const uint32_t descriptor_count = mode == 0 ? 3 : 2;
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
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
        VkDescriptorBufferInfo buffers[] = {{lhs, 0, bytes}, {rhs, 0, bytes},
                                            {output, 0, bytes}};
        VkWriteDescriptorSet writes[3]{};
        const uint32_t bindings[] = {0, mode == 0 ? 1U : 2U, 2};
        const uint32_t write_count = mode == 0 ? 3 : 2;
        for (uint32_t i = 0; i < write_count; ++i) {
            const uint32_t source = mode == 0 ? i : (i == 0 ? ((mode == 1 || mode == 3) ? 0 : 1) : 2);
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
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(cmd, &begin),
                     "could not begin pointwise command buffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[pipeline_mode]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layouts_[pipeline_mode], 0, 1, &set, 0, nullptr);
        Params params{scalar, static_cast<uint32_t>(elements), operation};
        vkCmdPushConstants(cmd, pipeline_layouts_[pipeline_mode], VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(params), &params);
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
                                     error.what() + "; could not confirm cleanup completion");
        }
        cleanup();
        throw contextual_error("dispatch failed", error);
    }
}

std::size_t VulkanCompute::dispatch_count() const {
    return dispatch_count_.load(std::memory_order_relaxed);
}
