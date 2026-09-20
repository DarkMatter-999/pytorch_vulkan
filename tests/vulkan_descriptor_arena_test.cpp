#include "vulkan/descriptor_arena.h"

#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan_compute.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_rollover_lifetime_and_quarantine(VulkanPlatform &platform) {
    VulkanExecutionContext &execution = platform.execution_context();
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    expect(vkCreateDescriptorSetLayout(platform.device(), &layout_info, nullptr, &layout) ==
               VK_SUCCESS,
           "could not create descriptor arena test layout");
    VkDescriptorSetLayoutBinding second_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo second_layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    second_layout_info.bindingCount = 1;
    second_layout_info.pBindings = &second_binding;
    VkDescriptorSetLayout second_layout = VK_NULL_HANDLE;
    expect(vkCreateDescriptorSetLayout(platform.device(), &second_layout_info, nullptr,
                                        &second_layout) == VK_SUCCESS,
           "could not create second descriptor arena test layout");

    try {
        DescriptorArena arena(platform.device(), execution);
        std::vector<VkDescriptorSet> sets;
        for (uint64_t id = 1; id <= 64; ++id)
            sets.push_back(arena.acquire(layout, id));
        auto full = arena.snapshot();
        expect(full.pool_count == 1 && full.live_sets == 64,
               "descriptor arena did not fill its bounded pool");

        execution.begin("descriptor-arena-test");
        VkDescriptorSet pending = arena.acquire(layout, 65);
        arena.release_after_completion(pending);
        auto pending_snapshot = arena.snapshot();
        expect(pending_snapshot.pending == 1 && pending_snapshot.live_sets == 65,
               "descriptor arena released a pending set too early");
        VkDescriptorSet not_reused = arena.acquire(layout, 65);
        expect(not_reused != pending, "descriptor arena reused a pending set");
        execution.submit();
        execution.wait();
        expect(arena.snapshot().pending == 0,
               "descriptor arena pending count did not clear after completion");

        VkDescriptorSet reused = arena.acquire(layout, 66);
        expect(reused == pending, "descriptor arena did not reuse a completed set");
        arena.release_after_completion(reused);
        VkDescriptorSet different_layout = arena.acquire(second_layout, 67);
        expect(different_layout != reused,
               "descriptor arena reused a set across incompatible layouts");
        auto rolled = arena.snapshot();
        expect(rolled.pool_count >= 2 && rolled.pool_count <= rolled.pool_limit &&
                   rolled.rollovers >= 1 && rolled.reuses >= 1,
               "descriptor arena rollover or reuse accounting is incorrect");

        VkDescriptorSetLayoutCreateInfo three_binding_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayoutBinding three_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        three_binding_info.bindingCount = 3;
        three_binding_info.pBindings = three_bindings;
        VkDescriptorSetLayout three_binding_layout = VK_NULL_HANDLE;
        expect(vkCreateDescriptorSetLayout(platform.device(), &three_binding_info, nullptr,
                                            &three_binding_layout) == VK_SUCCESS,
               "could not create descriptor-count test layout");
        VkDescriptorSet three_binding_set = arena.acquire(three_binding_layout, 68, 3);
        expect(three_binding_set != VK_NULL_HANDLE,
               "descriptor arena rejected a non-five descriptor layout");
        arena.release_after_completion(three_binding_set);
        const std::size_t created_before_loss = arena.snapshot().pool_creations;

        DescriptorArena bounded_arena(platform.device(), execution, 1);
        for (uint64_t id = 100; id < 164; ++id)
            bounded_arena.acquire(layout, id);
        bool rejected = false;
        try {
            bounded_arena.acquire(second_layout, 164);
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find("pool limit") != std::string::npos;
        }
        expect(rejected, "descriptor arena did not reject growth at its configured pool limit");

        arena.invalidate_device_loss();
        auto direct_quarantine = arena.snapshot();
        expect(direct_quarantine.quarantined > 0 && direct_quarantine.invalidated,
               "descriptor arena did not quarantine device-loss resources");
        vkDestroyDescriptorSetLayout(platform.device(), three_binding_layout, nullptr);
        platform.mark_device_lost(VK_ERROR_DEVICE_LOST);
        const auto platform_snapshot = platform.compute().descriptor_arena_snapshot();
        expect(platform_snapshot.invalidated,
               "platform-triggered device loss did not invalidate the compute arena");
        expect(direct_quarantine.live_sets == 0 && direct_quarantine.pending == 0,
               "descriptor arena reported live resources after device loss");
        expect(direct_quarantine.pool_creations == created_before_loss,
               "descriptor arena lost cumulative pool creation accounting on device loss");

    } catch (...) {
        vkDestroyDescriptorSetLayout(platform.device(), second_layout, nullptr);
        vkDestroyDescriptorSetLayout(platform.device(), layout, nullptr);
        throw;
    }
    vkDestroyDescriptorSetLayout(platform.device(), second_layout, nullptr);
    vkDestroyDescriptorSetLayout(platform.device(), layout, nullptr);
}

} // namespace

int main() {
    if (!VulkanPlatform::is_available())
        return 77;
    try {
        VulkanPlatform platform;
        test_rollover_lifetime_and_quarantine(platform);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
