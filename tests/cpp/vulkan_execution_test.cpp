#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan_tensor_layout.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void expect_rejected(Function &&function, const char *message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    expect(rejected, message);
}

void test_invalid_states(VulkanExecutionContext &context) {
    expect(!context.recording(), "new context is recording");
    expect_rejected([&] { (void)context.command_buffer(); },
                    "inactive command_buffer was accepted");
    expect_rejected([&] { context.submit(); }, "inactive submit was accepted");
    expect_rejected([&] { context.defer_destruction([] {}); },
                    "inactive deferred callback was accepted");
    expect(!context.retain_until_completion([] {}),
           "inactive completion retention was accepted");
    context.wait();
    context.synchronize();
    context.begin();
    expect(context.recording(), "begin did not start recording");
    expect_rejected([&] { context.begin(); }, "nested begin was accepted");
    context.submit();
    context.begin();
    expect(context.recording(), "begin did not acquire reusable slot");
    context.cancel();
    expect(!context.recording(), "wait left context recording");
}

void test_deferred_callback(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    bool called = false;
    context.begin();
    context.defer_destruction([&] { called = true; });
    context.submit();
    context.synchronize();
    expect(called, "deferred callback was not retired by synchronize");
}

void test_submitted_callback_retirement(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.retain_until_completion([&] { ++callback_count; });
    context.submit();
    expect(callback_count == 0, "submitted callback ran before completion");
    expect(context.pending_count() == 1, "submitted callback was not retained");
    context.retire_completed();
    context.wait();
    expect(callback_count == 1, "submitted callback did not run exactly once");
    context.wait();
    expect(callback_count == 1, "submitted callback ran more than once");
}

void test_stale_slot_callback_retirement(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.submit();
    context.begin();
    context.cancel();
    context.retain_until_completion([&] { ++callback_count; });
    expect(callback_count == 0, "stale-slot callback ran before completion");
    context.wait();
    expect(callback_count == 1, "stale-slot callback did not run after completion");
    context.wait();
    expect(callback_count == 1, "stale-slot callback ran more than once");
}

void test_callback_waits_for_all_submissions(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.retain_until_completion([&] { ++callback_count; });
    context.submit();
    context.begin();
    context.submit();
    context.wait();
    expect(callback_count == 1,
           "callback did not execute exactly once after all submissions");
    context.wait();
    expect(callback_count == 1,
           "callback executed more than once after all submissions");
}

void test_abandoned_callback(VulkanPlatform &platform) {
    bool called = false;
    {
        VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                       platform.command_pool());
        context.begin();
        context.defer_destruction([&] { called = true; });
    }
    expect(called, "abandoned recording callback was discarded");
}

void test_cancelled_callback(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    bool called = false;
    context.begin();
    context.defer_destruction([&] { called = true; });
    context.cancel();
    expect(called, "cancelled recording callback was not invoked");
    expect(!context.recording(), "cancel left context recording");
    context.begin();
    context.cancel();
}

void test_fill(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    VulkanBuffer buffer(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 64, 0x12345678U);
    context.submit();
    context.wait();

    uint32_t values[16]{};
    buffer.read(values, sizeof(values));
    for (uint32_t value : values) {
        expect(value == 0x12345678U, "vkCmdFillBuffer produced the wrong pattern");
    }
}

void test_retained_descriptor_pool(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    expect(vkCreateDescriptorSetLayout(platform.device(), &layout_info, nullptr,
                                       &layout) == VK_SUCCESS,
           "could not create retained descriptor layout");
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    expect(vkCreatePipelineLayout(platform.device(), &pipeline_layout_info, nullptr,
                                  &pipeline_layout) == VK_SUCCESS,
           "could not create retained pipeline layout");
    VkDescriptorPool pool = VK_NULL_HANDLE;
    expect(vkCreateDescriptorPool(platform.device(), &pool_info, nullptr, &pool) ==
               VK_SUCCESS,
           "could not create retained descriptor pool");
    auto buffer = std::make_shared<VulkanBuffer>(
        platform, 64,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorSetAllocateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    expect(vkAllocateDescriptorSets(platform.device(), &set_info, &set) == VK_SUCCESS,
           "could not allocate retained descriptor set");
    VkDescriptorBufferInfo buffer_info{buffer->buffer(), 0, 64};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(platform.device(), 1, &write, 0, nullptr);
    context.begin();
    context.retain(pool);
    pool = VK_NULL_HANDLE;
    vkCmdBindDescriptorSets(context.command_buffer(), VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdFillBuffer(context.command_buffer(), buffer->buffer(), 0, 64, 0xdeadbeefU);
    context.defer_destruction([owner = std::move(buffer)] {});
    expect(buffer == nullptr, "recorded buffer owner was not released before submit");
    context.submit();
    expect(context.pending_count() == 1, "retained submission was not pending");
    context.wait();
    expect(context.pending_count() == 0, "retained submission was not retired");
    vkDestroyPipelineLayout(platform.device(), pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(platform.device(), layout, nullptr);

    VulkanBuffer ring_buffer(platform, 64,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), ring_buffer.buffer(), 0, 64, 0xcafebabeU);
    context.submit();
    expect(context.pending_count() == 1, "ring reuse left an incorrect pending count");
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), ring_buffer.buffer(), 0, 64, 0xfeedfaceU);
    context.submit();
    expect(context.pending_count() == 2,
           "ring did not retain both bounded submissions");
    context.begin();
    expect(context.pending_count() == 2,
           "ring reuse did not retire the reused submission before recording");
    context.cancel();
    expect(context.pending_count() == 1,
           "ring reuse lost the remaining submitted slot");
    context.wait();
    VkDescriptorPool replacement = VK_NULL_HANDLE;
    expect(vkCreateDescriptorPool(platform.device(), &pool_info, nullptr,
                                  &replacement) == VK_SUCCESS,
           "descriptor pool could not be reused after retirement");
    vkDestroyDescriptorPool(platform.device(), replacement, nullptr);
}

void test_platform_pending_compute_count(VulkanPlatform &platform) {
    VulkanBuffer lhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer rhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer output(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout layout{1,
                                                    {16},
                                                    {1},
                                                    0,
                                                    sizeof(float),
                                                    0,
                                                    16,
                                                    0,
                                                    64,
                                                    64,
                                                    pytorch_vulkan::VulkanOverlap::No};
    expect(platform.pending_compute_count() == 0,
           "platform reported pending compute work before training step");
    platform.compute().begin_training_step();
    platform.compute().add(lhs.buffer(), layout, rhs.buffer(), layout, output.buffer(),
                           layout);
    expect(platform.pending_compute_count() == 1,
           "platform did not report the active compute recording");
    platform.compute().end_training_step();
    expect(platform.pending_compute_count() == 0,
           "platform retained compute work after training step completion");
}

void test_descriptor_cache_rollover_and_cancellation(VulkanPlatform &platform) {
    VulkanBuffer lhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer rhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer output(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout pointwise_layout{
        1,
        {16},
        {1},
        0,
        sizeof(float),
        0,
        16,
        0,
        64,
        64,
        pytorch_vulkan::VulkanOverlap::No};

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 65; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 1,
           "generic descriptor cache did not roll over at pool capacity");

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 64; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    platform.compute().begin_training_step();
    for (int i = 0; i < 64; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 0,
           "cancelled recording did not reset generic descriptor cursors");

    VulkanBuffer a(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer b(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer c(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer gemm_output(platform, 4,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout matrix_layout{
        2,
        {1, 1},
        {1, 1},
        6,
        sizeof(float),
        0,
        1,
        0,
        4,
        4,
        pytorch_vulkan::VulkanOverlap::No};

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 4097; ++i) {
        platform.compute().gemm(a.buffer(), matrix_layout, b.buffer(), matrix_layout,
                                c.buffer(), matrix_layout, gemm_output.buffer(),
                                matrix_layout, VK_NULL_HANDLE, matrix_layout, 1, 1, 1,
                                1.0F, 0.0F, false);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 2,
           "GEMM descriptor cache did not roll over at pool capacity");
}

void test_execution_counters_and_timing(VulkanPlatform &platform) {
    platform.reset_execution_counters();
    platform.reset_timing();
    VulkanExecutionContext &context = platform.execution_context();
    context.begin();
    context.submit();
    expect(platform.compute_submitted_count() == 1,
           "successful submit was not counted independently");
    expect(platform.compute_completed_count() == 0,
           "submission was counted as completed before wait");
    context.wait();
    expect(platform.compute_completed_count() == 1,
           "successful wait was not counted as completion");
    const auto timing = platform.timing_snapshot();
    expect(timing.allocation > 0.0, "execution allocation timing was not recorded");
    expect(timing.recording > 0.0, "execution recording timing was not recorded");
    expect(timing.submit > 0.0, "execution submit timing was not recorded");
    expect(timing.host_fence_wait > 0.0,
           "execution host fence-wait timing was not recorded");
}

void test_bounded_platform_lifetimes() {
    for (int iteration = 0; iteration < 3; ++iteration) {
        VulkanPlatform platform;
        VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                       platform.command_pool());
        context.begin();
        context.submit();
        context.wait();
        expect(context.pending_count() == 0,
               "repeated platform lifetime retained submitted work");
    }
}

} // namespace

int main() {
    try {
        VulkanPlatform platform;
        VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                       platform.command_pool());
        test_invalid_states(context);
        test_deferred_callback(platform);
        test_submitted_callback_retirement(platform);
        test_stale_slot_callback_retirement(platform);
        test_callback_waits_for_all_submissions(platform);
        test_abandoned_callback(platform);
        test_cancelled_callback(platform);
        test_fill(platform);
        test_retained_descriptor_pool(platform);
        test_platform_pending_compute_count(platform);
        test_descriptor_cache_rollover_and_cancellation(platform);
        test_execution_counters_and_timing(platform);
        test_bounded_platform_lifetimes();
        std::cout << "Vulkan execution lifecycle tests passed\n";
        return 0;
    } catch (const VulkanUnavailable &) {
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
