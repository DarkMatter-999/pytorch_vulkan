#include "vulkan_compute.h"

#include "vulkan/shaders/generated/add_spv.h"
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
    return std::runtime_error(std::string("Vulkan compute ") + operation + ": " +
                              error.what());
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
        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr,
                                                  &descriptor_set_layout_),
                     "could not create add descriptor-set layout");

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = sizeof(uint32_t);
        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        check_result(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                            &pipeline_layout_),
                     "could not create add pipeline layout");

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = vulkan_add_shader::kCodeSize;
        shader_info.pCode = vulkan_add_shader::kCode;
        check_result(vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_),
                     "could not create add shader module");

        VkPipelineShaderStageCreateInfo stage_info{};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = shader_module_;
        stage_info.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                               nullptr, &pipeline_),
                     "could not create add compute pipeline");
    } catch (const std::exception &error) {
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (shader_module_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, shader_module_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_set_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        throw contextual_error("initialization failed", error);
    }
}

VulkanCompute::~VulkanCompute() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
    }
}

void VulkanCompute::add(VkBuffer lhs, VkBuffer rhs, VkBuffer output,
                        VkDeviceSize bytes) const {
    if (lhs == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE || output == VK_NULL_HANDLE) {
        throw std::invalid_argument("Vulkan compute add requires non-null buffers");
    }
    if (bytes == 0 || bytes % sizeof(float) != 0) {
        throw std::invalid_argument("Vulkan compute add bytes must be a non-zero float size");
    }
    const VkDeviceSize elements = bytes / sizeof(float);
    if (bytes > max_storage_buffer_range_) {
        throw std::invalid_argument(
            "Vulkan compute add byte range " + std::to_string(bytes) +
            " exceeds maxStorageBufferRange " +
            std::to_string(max_storage_buffer_range_));
    }
    if (elements > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Vulkan compute add element count exceeds uint32_t");
    }
    const VkDeviceSize groups = elements / 256 + (elements % 256 != 0 ? 1 : 0);
    if (groups > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Vulkan compute add workgroup count exceeds uint32_t");
    }
    if (groups > max_compute_workgroup_count_x_) {
        throw std::invalid_argument(
            "Vulkan compute add workgroup count " + std::to_string(groups) +
            " exceeds maxComputeWorkGroupCount[0] " +
            std::to_string(max_compute_workgroup_count_x_));
    }

    std::scoped_lock lock(platform_.queue_mutex());

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    const auto cleanup = [&]() {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
        if (command_buffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    try {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create add descriptor pool");
        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_set_layout_;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &descriptor_set),
                     "could not allocate add descriptor set");
        VkDescriptorBufferInfo buffers[] = {{lhs, 0, bytes}, {rhs, 0, bytes},
                                            {output, 0, bytes}};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check_result(vkAllocateCommandBuffers(device_, &allocation, &command_buffer),
                     "could not allocate add command buffer");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(command_buffer, &begin),
                     "could not begin add command buffer");
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr);
        const uint32_t count = static_cast<uint32_t>(elements);
        vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(count), &count);
        vkCmdDispatch(command_buffer, static_cast<uint32_t>(groups), 1, 1);
        check_result(vkEndCommandBuffer(command_buffer), "could not end add command buffer");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "could not create add fence");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        // Ensure the platform can take ownership if completion cannot be confirmed.
        platform_.reserve_compute_resources();
        check_result(vkQueueSubmit(queue_, 1, &submit, fence), "could not submit add dispatch");
        submitted = true;
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "could not wait for add dispatch");
        cleanup();
    } catch (const std::exception &error) {
        VkResult recovery_result = VK_SUCCESS;
        if (submitted) recovery_result = vkQueueWaitIdle(queue_);
        if (recovery_result != VK_SUCCESS) {
            platform_.defer_compute_resources(pool, command_buffer, fence);
            pool = VK_NULL_HANDLE;
            command_buffer = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            throw std::runtime_error(
                std::string("Vulkan compute dispatch failed: ") + error.what() +
                "; could not confirm cleanup completion with VkResult " +
                std::to_string(static_cast<int>(recovery_result)));
        }
        cleanup();
        throw contextual_error("dispatch failed", error);
    }
}

std::size_t VulkanCompute::dispatch_count() const {
    return dispatch_count_.load(std::memory_order_relaxed);
}
