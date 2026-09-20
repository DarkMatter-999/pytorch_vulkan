#include "vulkan/pipeline_cache.h"

#include "vulkan/shader_registry.h"
#include "vulkan/shaders/generated/gemm_spv.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_key_identity() {
    PipelineKey base{"gemm", 11, 22, {1, 2, 3}, 0x10};
    PipelineKey equivalent{"gemm", 11, 22, {1, 2, 3}, 0x10};
    PipelineKey shader_changed = base;
    shader_changed.shader_identity = 12;
    PipelineKey layout_changed = base;
    layout_changed.descriptor_layout_identity = 23;
    PipelineKey variant_changed = base;
    variant_changed.specialization_values[1] = 4;
    PipelineKey features_changed = base;
    features_changed.required_feature_bits = 0x20;

    expect(base == equivalent, "equivalent pipeline keys differ");
    expect(std::hash<PipelineKey>{}(base) == std::hash<PipelineKey>{}(equivalent),
           "equivalent pipeline keys have different hashes");
    expect(!(base == shader_changed) && !(base == layout_changed) &&
               !(base == variant_changed) && !(base == features_changed),
           "pipeline key identity fields were ignored");
}

void test_bound_and_invalidation(VulkanPlatform &platform) {
    VulkanExecutionContext &execution = platform.execution_context();
    VulkanPipelineCache cache(platform.device(), platform, execution, 1);

    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 1;
    descriptor_info.pBindings = &binding;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    expect(vkCreateDescriptorSetLayout(platform.device(), &descriptor_info, nullptr,
                                       &descriptor_layout) == VK_SUCCESS,
           "could not create cache test descriptor layout");

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    expect(vkCreatePipelineLayout(platform.device(), &layout_info, nullptr,
                                  &pipeline_layout) == VK_SUCCESS,
           "could not create cache test pipeline layout");

    const auto shader = platform.shader_registry().get_or_create(
        {"pipeline-cache-test", vulkan_shader_code_hash(
                                     vulkan_gemm_shader::kCode,
                                     vulkan_gemm_shader::kCodeSize / sizeof(uint32_t))},
        vulkan_gemm_shader::kCode,
        vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo create_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    create_info.stage = stage;
    create_info.layout = pipeline_layout;

    const PipelineKey first{"gemm", 1, 2, {3}, 4};
    const PipelineKey second{"pointwise", 5, 6, {7}, 8};
    const PipelineLayoutKey layout_key{1, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    const PipelineLayoutKey equivalent_layout_key{1, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    const PipelineLayoutKey changed_layout_key{1, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    expect(layout_key == equivalent_layout_key &&
               std::hash<PipelineLayoutKey>{}(layout_key) ==
                   std::hash<PipelineLayoutKey>{}(equivalent_layout_key),
           "equivalent pipeline layout keys differ");
    expect(!(layout_key == changed_layout_key),
           "pipeline layout push-constant identity was ignored");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo cached_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    cached_layout_info.setLayoutCount = 1;
    cached_layout_info.pSetLayouts = &descriptor_layout;
    cached_layout_info.pushConstantRangeCount = 1;
    cached_layout_info.pPushConstantRanges = &push;
    const auto cached_layout = cache.get_or_create_layout(layout_key, cached_layout_info);
    expect(cached_layout != VK_NULL_HANDLE, "cache failed to create pipeline layout");
    expect(cache.get_or_create_layout(layout_key, cached_layout_info) == cached_layout,
           "equivalent pipeline layout keys did not reuse a layout");
    expect(cache.snapshot().layout_count == 1, "pipeline layout cache count is incorrect");
    expect(cache.get_or_create(first, pipeline_layout, shader, create_info) != VK_NULL_HANDLE,
           "cache failed to create first pipeline");
    expect(cache.get_or_create(first, pipeline_layout, shader, create_info) != VK_NULL_HANDLE,
           "cache failed to reuse first pipeline");
    execution.begin();
    expect(cache.get_or_create(second, pipeline_layout, shader, create_info) != VK_NULL_HANDLE,
           "cache failed to create replacement pipeline");
    expect(cache.snapshot().pending_destructions == 1,
           "evicted pipeline destruction was not retained for completion");
    expect(cache.snapshot().pipeline_count == 2,
           "pipeline count omitted the pending evicted pipeline");
    execution.submit();
    const auto bounded = cache.snapshot();
    expect(bounded.entry_count == 1, "pipeline cache exceeded configured bound");
    expect(bounded.hits == 1 && bounded.misses == 2 && bounded.evictions == 1,
           "pipeline cache counters are incorrect");
    execution.retire_completed();
    execution.wait();
    expect(cache.snapshot().pending_destructions == 0,
           "evicted pipeline destruction did not run after completion");
    expect(cache.snapshot().pipeline_count == 1,
           "pipeline count did not include the resident cached pipeline");

    execution.begin();
    cache.destroy_all();
    expect(cache.snapshot().pending_destructions == 1 &&
               cache.snapshot().pending_layout_destructions == 1,
           "cache-owned resources were not retained during teardown");
    execution.submit();
    execution.retire_completed();
    execution.wait();
    expect(cache.snapshot().pipeline_count == 0 && cache.snapshot().layout_count == 0 &&
               cache.snapshot().pending_layout_destructions == 0,
           "cache-owned resources were not destroyed after completion");

    cache.invalidate_device_loss();
    expect(cache.snapshot().invalidated, "pipeline cache was not invalidated");
    bool rejected = false;
    try {
        cache.get_or_create({"rejected", 9, 10, {11}, 12}, pipeline_layout, shader,
                             create_info);
    } catch (const VulkanDeviceLost &) {
        rejected = true;
    }
    expect(rejected, "pipeline cache accepted creation after device loss");

    cache.destroy_all();
    vkDestroyPipelineLayout(platform.device(), pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(platform.device(), descriptor_layout, nullptr);
}

} // namespace

int main() {
    test_key_identity();
    if (!VulkanPlatform::is_available())
        return 77;
    try {
        VulkanPlatform platform;
        test_bound_and_invalidation(platform);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
