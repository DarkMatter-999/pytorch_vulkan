#include "vulkan_platform.h"
#include "vulkan_execution.h"
#include "vulkan/shader_registry.h"
#include "vulkan/shaders/generated/gemm_spv.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_lookup_and_invalidation(VulkanPlatform &platform) {
    VulkanShaderRegistry &registry = platform.shader_registry();
    const uint32_t *code = vulkan_gemm_shader::kCode;
    const VulkanShaderRegistry::ShaderKey pointwise{"pointwise", 17U};
    const VulkanShaderRegistry::ShaderKey other{"other", 17U};
    const auto baseline = registry.snapshot();

    const auto first = registry.get_or_create(
        pointwise, code, vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
    const auto second = registry.get_or_create(
        pointwise, code, vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
    expect(first != VK_NULL_HANDLE, "registry returned a null shader module");
    expect(first == second, "equivalent shader keys did not reuse a module");

    const auto different = registry.get_or_create(
        other, code, vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
    expect(different != first, "different shader identities were merged");
    const auto snapshot = registry.snapshot();
    expect(snapshot.module_count == baseline.module_count + 2,
           "registry module count is incorrect");
    expect(snapshot.cache_misses == baseline.cache_misses + 2,
           "registry miss count is incorrect");
    expect(snapshot.cache_hits == baseline.cache_hits + 1,
           "registry hit count is incorrect");

    platform.mark_device_lost(VK_ERROR_DEVICE_LOST);
    const auto invalidated = registry.snapshot();
    expect(invalidated.invalidated, "registry was not invalidated on device loss");
    bool rejected = false;
    try {
        registry.get_or_create(
            {"new", 19U}, code, vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
    } catch (const VulkanDeviceLost &) {
        rejected = true;
    }
    expect(rejected, "registry accepted creation after device loss");
}

void test_deferred_module_destruction(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    VkShaderModule module = VK_NULL_HANDLE;
    const uint32_t *code = vulkan_gemm_shader::kCode;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = vulkan_gemm_shader::kCodeSize;
    info.pCode = code;
    expect(vkCreateShaderModule(platform.device(), &info, nullptr, &module) == VK_SUCCESS,
           "could not create deferred shader module");
    bool destroyed = false;
    context.begin();
    expect(context.retain_until_completion([&] {
               vkDestroyShaderModule(platform.device(), module, nullptr);
               module = VK_NULL_HANDLE;
               destroyed = true;
           }),
           "shader destruction was not retained for submitted work");
    context.submit();
    expect(!destroyed, "shader module was destroyed before completion");
    context.retire_completed();
    context.wait();
    expect(destroyed, "shader module was not destroyed after completion");
    expect(module == VK_NULL_HANDLE, "deferred shader module handle was not cleared");
}

} // namespace

int main() {
    if (!VulkanPlatform::is_available())
        return 77;
    try {
        VulkanPlatform platform;
        test_deferred_module_destruction(platform);
        test_lookup_and_invalidation(platform);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
