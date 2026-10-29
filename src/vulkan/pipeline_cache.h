#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class VulkanExecutionContext;
class VulkanPlatform;
namespace pytorch_vulkan { struct VulkanQuiescence; }

struct PipelineKey {
    std::string shader_identity;
    uint64_t descriptor_layout_identity = 0;
    uint64_t shader_code_identity = 0;
    std::vector<uint32_t> specialization_values;
    uint64_t required_feature_bits = 0;

    bool operator==(const PipelineKey &other) const {
        return shader_identity == other.shader_identity &&
               descriptor_layout_identity == other.descriptor_layout_identity &&
               shader_code_identity == other.shader_code_identity &&
               specialization_values == other.specialization_values &&
               required_feature_bits == other.required_feature_bits;
    }
};

struct PipelineLayoutKey {
    uint64_t descriptor_layout_identity = 0;
    uint32_t stage_flags = 0;
    uint32_t push_constant_offset = 0;
    uint32_t push_constant_size = 0;

    bool operator==(const PipelineLayoutKey &other) const {
        return descriptor_layout_identity == other.descriptor_layout_identity &&
               stage_flags == other.stage_flags &&
               push_constant_offset == other.push_constant_offset &&
               push_constant_size == other.push_constant_size;
    }
};

struct PipelineCacheSnapshot {
    std::size_t entry_count = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
    std::size_t pipeline_count = 0;
    std::size_t layout_count = 0;
    std::size_t pending_destructions = 0;
    std::size_t pending_layout_destructions = 0;
    bool invalidated = false;
};

class VulkanPipelineCache final {
  public:
    VulkanPipelineCache(VkDevice device, VulkanPlatform &platform,
                        VulkanExecutionContext &execution, std::size_t max_entries);
    ~VulkanPipelineCache() noexcept;

    VulkanPipelineCache(const VulkanPipelineCache &) = delete;
    VulkanPipelineCache &operator=(const VulkanPipelineCache &) = delete;

    VkPipeline get_or_create(const PipelineKey &, VkPipelineLayout, VkShaderModule,
                             const VkComputePipelineCreateInfo &);
    VkPipelineLayout get_or_create_layout(const PipelineLayoutKey &,
                                          const VkPipelineLayoutCreateInfo &);
    void invalidate_device_loss();
    PipelineCacheSnapshot snapshot() const;
    void destroy_all() noexcept;
    // Caller owns queue serialization after admission/epilogues are closed.
    void destroy_empty_native(const pytorch_vulkan::VulkanQuiescence &);

  private:
    friend struct VulkanNativeServiceTestPeer;
    struct KeyHash {
        std::size_t operator()(const PipelineKey &) const noexcept;
    };
    struct LayoutKeyHash {
        std::size_t operator()(const PipelineLayoutKey &) const noexcept;
    };
    struct Entry {
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::list<PipelineKey>::iterator lru;
    };
    struct LayoutEntry {
        VkPipelineLayout layout = VK_NULL_HANDLE;
    };
    using NativePipelines = std::unordered_map<PipelineKey, Entry, KeyHash>;
    using NativeLayouts = std::unordered_map<PipelineLayoutKey, LayoutEntry, LayoutKeyHash>;
    using NativeLru = std::list<PipelineKey>;
    static void destroy_empty_native_tables(VkDevice,
        const pytorch_vulkan::VulkanQuiescence &, bool invalidated,
        std::size_t pending, std::size_t pending_layouts,
        NativePipelines &, NativeLayouts &, NativeLru &, void *,
        void (*destroy_pipeline)(void *, VkDevice, VkPipeline),
        void (*destroy_layout)(void *, VkDevice, VkPipelineLayout));

    void destroy_pipeline(VkPipeline) noexcept;
    void destroy_layout(VkPipelineLayout) noexcept;
    VkPipeline evict_one_locked();

    VkDevice device_ = VK_NULL_HANDLE;
    VulkanPlatform &platform_;
    VulkanExecutionContext &execution_;
    std::size_t max_entries_ = 0;
    mutable std::mutex mutex_;
    NativePipelines entries_;
    NativeLayouts layouts_;
    NativeLru lru_;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
    std::size_t evictions_ = 0;
    std::shared_ptr<std::atomic<std::size_t>> pending_destructions_ =
        std::make_shared<std::atomic<std::size_t>>(0);
    std::shared_ptr<std::atomic<std::size_t>> pending_layout_destructions_ =
        std::make_shared<std::atomic<std::size_t>>(0);
    bool invalidated_ = false;
    bool native_destroyed_ = false;
};

namespace std {
template <> struct hash<PipelineKey> {
    std::size_t operator()(const PipelineKey &key) const noexcept;
};
template <> struct hash<PipelineLayoutKey> {
    std::size_t operator()(const PipelineLayoutKey &key) const noexcept;
};
} // namespace std
