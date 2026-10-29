#include "vulkan/shader_registry.h"

#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <stdexcept>
#include <utility>
#include <vector>

VulkanShaderRegistry::VulkanShaderRegistry(VkDevice device, VulkanPlatform &platform,
                                           VulkanExecutionContext &execution)
    : device_(device), platform_(platform), execution_(execution) {}

VulkanShaderRegistry::~VulkanShaderRegistry() noexcept {
    if (native_destroyed_) return;
    std::vector<VkShaderModule> modules;
    bool invalidated = false;
    {
        std::scoped_lock lock(mutex_);
        for (const auto &entry : modules_)
            modules.push_back(entry.second);
        modules_.clear();
        invalidated = invalidated_;
    }
    if (invalidated || modules.empty())
        return;
    std::function<void()> destroy_modules = [device = device_,
                                             modules = std::move(modules)] {
        for (const auto module : modules) {
            if (module != VK_NULL_HANDLE)
                vkDestroyShaderModule(device, module, nullptr);
        }
    };
    if (!execution_.retain_until_completion(destroy_modules))
        destroy_modules();
}

void VulkanShaderRegistry::destroy_empty_native_table(VkDevice device,
    const pytorch_vulkan::VulkanQuiescence &empty, bool invalidated,
    NativeModules &modules, void *cookie, void (*destroy)(void *, VkDevice, VkShaderModule)) {
    if (!empty.healthy_empty() || invalidated || !destroy)
        throw std::logic_error("native shader cleanup requires empty healthy execution/services");
    for (auto &entry : modules) {
        if (entry.second != VK_NULL_HANDLE) destroy(cookie, device, entry.second);
        entry.second = VK_NULL_HANDLE;
    }
    modules.clear(); // Shader keys and VkShaderModule only, no arbitrary captures.
}

void VulkanShaderRegistry::destroy_empty_native(const pytorch_vulkan::VulkanQuiescence &empty) {
    std::lock_guard<std::mutex> lock(mutex_);
    destroy_empty_native_table(device_, empty, invalidated_, modules_, nullptr,
        [](void *, VkDevice device, VkShaderModule module) { vkDestroyShaderModule(device, module, nullptr); });
    native_destroyed_ = true;
}

std::size_t
VulkanShaderRegistry::KeyHash::operator()(const ShaderKey &key) const noexcept {
    return std::hash<ShaderKey>{}(key);
}

VkShaderModule VulkanShaderRegistry::get_or_create(const ShaderKey &key,
                                                   const uint32_t *code,
                                                   std::size_t word_count) {
    if (code == nullptr || word_count == 0)
        throw std::invalid_argument("Vulkan shader registry received empty code");
    if (platform_.device_lost() || execution_.invalidated())
        throw VulkanDeviceLost("Vulkan shader registry is invalidated");
    std::scoped_lock lock(mutex_);
    if (invalidated_ || platform_.device_lost())
        throw VulkanDeviceLost("Vulkan shader registry is invalidated");
    const auto existing = modules_.find(key);
    if (existing != modules_.end()) {
        ++cache_hits_;
        return existing->second;
    }
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = word_count * sizeof(uint32_t);
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult result = vkCreateShaderModule(device_, &info, nullptr, &module);
    if (result != VK_SUCCESS)
        throw std::runtime_error("could not create Vulkan shader module");
    modules_.emplace(key, module);
    ++cache_misses_;
    return module;
}

void VulkanShaderRegistry::invalidate_device_loss() {
    std::scoped_lock lock(mutex_);
    invalidated_ = true;
}

VulkanShaderRegistry::Snapshot VulkanShaderRegistry::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {modules_.size(), cache_hits_, cache_misses_, invalidated_};
}
