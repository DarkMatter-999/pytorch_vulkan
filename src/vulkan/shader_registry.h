#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

class VulkanExecutionContext;
class VulkanPlatform;
namespace pytorch_vulkan { struct VulkanQuiescence; }

struct ShaderKey {
    std::string name;
    uint64_t code_hash = 0;

    bool operator==(const ShaderKey &other) const {
        return name == other.name && code_hash == other.code_hash;
    }
};

constexpr uint64_t vulkan_shader_code_hash(const uint32_t *code,
                                           std::size_t word_count) {
    uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < word_count; ++index) {
        hash ^= static_cast<uint64_t>(code[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct ShaderRegistrySnapshot {
    std::size_t module_count = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    bool invalidated = false;
};

class VulkanShaderRegistry final {
  public:
    using ShaderKey = ::ShaderKey;
    using Snapshot = ShaderRegistrySnapshot;

    VulkanShaderRegistry(VkDevice device, VulkanPlatform &platform,
                         VulkanExecutionContext &execution);
    ~VulkanShaderRegistry() noexcept;

    VulkanShaderRegistry(const VulkanShaderRegistry &) = delete;
    VulkanShaderRegistry &operator=(const VulkanShaderRegistry &) = delete;

    VkShaderModule get_or_create(const ShaderKey &key, const uint32_t *code,
                                 std::size_t word_count);
    void invalidate_device_loss();
    Snapshot snapshot() const;
    // Caller owns queue serialization after admission/epilogues are closed.
    void destroy_empty_native(const pytorch_vulkan::VulkanQuiescence &);

  private:
    friend struct VulkanNativeServiceTestPeer;
    struct KeyHash {
        std::size_t operator()(const ShaderKey &key) const noexcept;
    };
    using NativeModules = std::unordered_map<ShaderKey, VkShaderModule, KeyHash>;
    static void destroy_empty_native_table(VkDevice,
        const pytorch_vulkan::VulkanQuiescence &, bool invalidated,
        NativeModules &, void *, void (*destroy)(void *, VkDevice, VkShaderModule));

    VkDevice device_ = VK_NULL_HANDLE;
    VulkanPlatform &platform_;
    VulkanExecutionContext &execution_;
    mutable std::mutex mutex_;
    NativeModules modules_;
    std::size_t cache_hits_ = 0;
    std::size_t cache_misses_ = 0;
    bool invalidated_ = false;
    bool native_destroyed_ = false;
};

namespace std {
template <> struct hash<ShaderKey> {
    std::size_t operator()(const ShaderKey &key) const noexcept {
        const std::size_t name_hash = std::hash<std::string>{}(key.name);
        const std::size_t code_hash = std::hash<uint64_t>{}(key.code_hash);
        return name_hash ^ (code_hash + static_cast<std::size_t>(0x9e3779b9U) +
                            (name_hash << 6) + (name_hash >> 2));
    }
};
} // namespace std
