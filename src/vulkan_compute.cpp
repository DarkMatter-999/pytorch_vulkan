#include "vulkan_compute.h"

#include "vulkan/shaders/generated/pointwise_spv.h"
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
        if (properties.limits.maxPushConstantsSize < sizeof(Params))
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
    } catch (const std::exception &error) {
        for (auto pipeline : pipelines_) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
        }
        for (auto module : shader_modules_) {
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, module, nullptr);
            }
        }
        for (auto layout : pipeline_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
        }
        for (auto layout : descriptor_set_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
        }
        throw contextual_error("initialization failed", error);
    }
}

VulkanCompute::~VulkanCompute() {
    for (auto pipeline : pipelines_) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline, nullptr);
        }
    }
    for (auto module : shader_modules_) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module, nullptr);
        }
    }
    for (auto layout : pipeline_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout, nullptr);
        }
    }
    for (auto layout : descriptor_set_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
        }
    }
}

void VulkanCompute::add(VkBuffer lhs, VkBuffer rhs, VkBuffer output, VkDeviceSize bytes) const {
    tensor_tensor(lhs, rhs, output, bytes, kAdd);
}

void VulkanCompute::tensor_tensor(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation) const {
    dispatch(0, lhs, rhs, output, bytes, 0.0F, operation);
}

void VulkanCompute::tensor_scalar(VkBuffer tensor, VkBuffer output, VkDeviceSize bytes,
                                  float scalar, uint32_t operation) const {
    dispatch(1, tensor, VK_NULL_HANDLE, output, bytes, scalar, operation);
}

void VulkanCompute::scalar_tensor(float scalar, VkBuffer tensor, VkBuffer output,
                                  VkDeviceSize bytes, uint32_t operation) const {
    dispatch(2, VK_NULL_HANDLE, tensor, output, bytes, scalar, operation);
}

void VulkanCompute::unary(VkBuffer input, VkBuffer output, VkDeviceSize bytes,
                          uint32_t operation) const {
    dispatch(3, input, VK_NULL_HANDLE, output, bytes, 0.0F, operation);
}

void VulkanCompute::dispatch(uint32_t mode, VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                             VkDeviceSize bytes, float scalar, uint32_t operation) const {
    if (mode > 3 || output == VK_NULL_HANDLE ||
        (mode == 0 && (lhs == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE)) ||
        (mode != 0 && ((lhs == VK_NULL_HANDLE) == (rhs == VK_NULL_HANDLE)))) {
        throw std::invalid_argument("Vulkan compute pointwise requires valid buffers");
    }
    std::scoped_lock lock(platform_.queue_mutex());
    const VkDeviceSize elements = bytes / sizeof(float);
    if (!bytes || bytes % sizeof(float) || bytes > max_storage_buffer_range_ ||
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
        set_info.pSetLayouts = &descriptor_set_layouts_[mode];
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[mode]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layouts_[mode], 0, 1, &set, 0, nullptr);
        Params params{scalar, static_cast<uint32_t>(elements), operation};
        vkCmdPushConstants(cmd, pipeline_layouts_[mode], VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
