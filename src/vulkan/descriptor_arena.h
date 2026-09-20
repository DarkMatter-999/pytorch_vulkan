#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>

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

  private:
    friend class VulkanCompute;
    struct State;
    VkDescriptorSet acquire(VkDescriptorSetLayout layout, uint64_t submission_id,
                            uint32_t descriptor_count, std::size_t pool_capacity);
    VkDevice device_ = VK_NULL_HANDLE;
    VulkanExecutionContext &execution_;
    std::shared_ptr<State> state_;
};
