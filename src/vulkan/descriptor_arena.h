#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace pytorch_vulkan { struct VulkanQuiescence; }

class VulkanExecutionContext;

struct DescriptorArenaSnapshot {
    std::size_t pool_count = 0;
    std::size_t pool_limit = 0;
    std::size_t pool_creations = 0;
    std::size_t live_sets = 0;
    std::size_t allocations = 0;
    std::size_t reuses = 0;
    std::size_t rollovers = 0;
    std::size_t pending = 0;
    std::size_t quarantined = 0;
    bool invalidated = false;
};

class DescriptorArena final {
  public:
    DescriptorArena(VkDevice device, VulkanExecutionContext &execution,
                    std::size_t pool_limit = 64);
    ~DescriptorArena();

    DescriptorArena(const DescriptorArena &) = delete;
    DescriptorArena &operator=(const DescriptorArena &) = delete;

    VkDescriptorSet acquire(VkDescriptorSetLayout layout, uint64_t submission_id,
                            uint32_t descriptor_count = 5);
    void release_after_completion(VkDescriptorSet set);
    void invalidate_device_loss();
    DescriptorArenaSnapshot snapshot() const;
    // Caller owns queue serialization after admission/epilogues are closed.
    void destroy_empty_native(const pytorch_vulkan::VulkanQuiescence &);

  private:
    friend class VulkanCompute;
    friend struct VulkanNativeServiceTestPeer;
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        uint32_t descriptor_count = 5;
        std::size_t capacity = 64;
        std::vector<VkDescriptorSet> sets;
        std::vector<VkDescriptorSet> free_sets;
        std::size_t pending = 0;
    };
    static void destroy_empty_native_pools(VkDevice,
        const pytorch_vulkan::VulkanQuiescence &, bool invalidated,
        std::size_t pending, std::size_t quarantined, std::vector<Pool> &,
        void *, void (*destroy)(void *, VkDevice, VkDescriptorPool));
    struct State;
    VkDescriptorSet acquire(VkDescriptorSetLayout layout, uint64_t submission_id,
                            uint32_t descriptor_count, std::size_t pool_capacity);
    VkDevice device_ = VK_NULL_HANDLE;
    VulkanExecutionContext &execution_;
    std::shared_ptr<State> state_;
    bool native_destroyed_ = false;
};
