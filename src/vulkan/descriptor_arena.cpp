#include "vulkan/descriptor_arena.h"

#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kPoolCapacity = 64;
constexpr std::size_t kMaxPoolCount = 64;
constexpr uint32_t kDescriptorBindingsPerSet = 5;

struct Pool {
    VkDescriptorPool handle = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    uint32_t descriptor_count = kDescriptorBindingsPerSet;
    std::size_t capacity = kPoolCapacity;
    std::vector<VkDescriptorSet> sets;
    std::vector<VkDescriptorSet> free_sets;
    std::size_t pending = 0;
};
} // namespace

struct DescriptorArena::State {
    mutable std::mutex mutex;
    std::vector<Pool> pools;
    std::size_t allocations = 0;
    std::size_t pool_creations = 0;
    std::size_t reuses = 0;
    std::size_t rollovers = 0;
    std::size_t pending = 0;
    std::size_t quarantined = 0;
    bool invalidated = false;
    bool alive = true;
    std::size_t pool_limit = kMaxPoolCount;
};

DescriptorArena::DescriptorArena(VkDevice device, VulkanExecutionContext &execution,
                                 std::size_t pool_limit)
    : device_(device), execution_(execution), state_(std::make_shared<State>()) {
    if (device_ == VK_NULL_HANDLE)
        throw std::invalid_argument("descriptor arena requires a Vulkan device");
    if (pool_limit == 0 || pool_limit > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("descriptor arena has invalid pool limit");
    state_->pool_limit = pool_limit;
}

DescriptorArena::~DescriptorArena() {
    std::vector<VkDescriptorPool> pools;
    {
        std::scoped_lock lock(state_->mutex);
        state_->alive = false;
        if (state_->invalidated) {
            state_->pools.clear();
            return;
        }
        pools.reserve(state_->pools.size());
        for (auto &pool : state_->pools) {
            if (pool.handle != VK_NULL_HANDLE)
                pools.push_back(pool.handle);
            pool.handle = VK_NULL_HANDLE;
        }
        state_->pools.clear();
    }
    const auto destroy = [device = device_, pools = std::move(pools)] {
        for (const auto pool : pools)
            vkDestroyDescriptorPool(device, pool, nullptr);
    };
    try {
        if (!execution_.retain_until_completion(destroy))
            destroy();
    } catch (...) {
        // Completion is not knowable here; leak safely rather than destroy in-flight pools.
    }
}

VkDescriptorSet DescriptorArena::acquire(VkDescriptorSetLayout layout,
                                         uint64_t submission_id,
                                         uint32_t descriptor_count) {
    return acquire(layout, submission_id, descriptor_count, kPoolCapacity);
}

VkDescriptorSet DescriptorArena::acquire(VkDescriptorSetLayout layout,
                                         uint64_t /*submission_id*/,
                                         uint32_t descriptor_count,
                                         std::size_t pool_capacity) {
    if (layout == VK_NULL_HANDLE)
        throw std::invalid_argument("descriptor arena requires a descriptor layout");
    if (descriptor_count == 0)
        throw std::invalid_argument("descriptor arena requires descriptors");
    if (pool_capacity == 0 || pool_capacity > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("descriptor arena has invalid pool capacity");
    std::scoped_lock lock(state_->mutex);
    if (state_->invalidated)
        throw VulkanDeviceLost("descriptor arena is invalid after device loss");

    for (auto &pool : state_->pools) {
        if (pool.layout != layout || pool.descriptor_count != descriptor_count ||
            pool.capacity != pool_capacity)
            continue;
        if (!pool.free_sets.empty()) {
            const VkDescriptorSet set = pool.free_sets.back();
            pool.free_sets.pop_back();
            ++state_->reuses;
            return set;
        }
        if (pool.sets.size() >= pool.capacity)
            continue;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        info.descriptorPool = pool.handle;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;
        const VkResult result = vkAllocateDescriptorSets(device_, &info, &set);
        if (result != VK_SUCCESS)
            throw std::runtime_error("could not allocate descriptor arena set");
        pool.sets.push_back(set);
        ++state_->allocations;
        return set;
    }

    Pool pool;
    pool.layout = layout;
    pool.descriptor_count = descriptor_count;
    pool.capacity = pool_capacity;
    const uint64_t pool_descriptor_count =
        static_cast<uint64_t>(pool.descriptor_count) * pool_capacity;
    if (pool_descriptor_count > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("descriptor arena descriptor count overflows");
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    static_cast<uint32_t>(pool_descriptor_count)};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = static_cast<uint32_t>(pool_capacity);
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &size;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool.handle) != VK_SUCCESS)
        throw std::runtime_error("could not create descriptor arena pool");
    try {
        if (state_->pools.size() >= state_->pool_limit)
            throw std::runtime_error("descriptor arena pool limit reached");
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        info.descriptorPool = pool.handle;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(device_, &info, &set) != VK_SUCCESS)
            throw std::runtime_error("could not allocate descriptor arena set");
        pool.sets.push_back(set);
        state_->pools.push_back(std::move(pool));
        ++state_->allocations;
        ++state_->pool_creations;
        if (state_->pools.size() > 1)
            ++state_->rollovers;
        return set;
    } catch (...) {
        vkDestroyDescriptorPool(device_, pool.handle, nullptr);
        throw;
    }
}

void DescriptorArena::release_after_completion(VkDescriptorSet set) {
    if (set == VK_NULL_HANDLE)
        return;
    std::size_t pool_index = std::numeric_limits<std::size_t>::max();
    {
        std::scoped_lock lock(state_->mutex);
        if (state_->invalidated)
            return;
        for (std::size_t i = 0; i < state_->pools.size(); ++i) {
            auto &pool = state_->pools[i];
            if (std::find(pool.sets.begin(), pool.sets.end(), set) == pool.sets.end())
                continue;
            pool_index = i;
            ++pool.pending;
            ++state_->pending;
            break;
        }
    }
    if (pool_index == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("descriptor arena release received an unknown set");

    const auto state = state_;
    const auto release = [state, pool_index, set] {
        std::scoped_lock lock(state->mutex);
        if (!state->alive || state->invalidated || pool_index >= state->pools.size())
            return;
        auto &pool = state->pools[pool_index];
        if (pool.pending != 0)
            --pool.pending;
        if (state->pending != 0)
            --state->pending;
        pool.free_sets.push_back(set);
    };
    if (!execution_.retain_until_completion(release))
        release();
}

void DescriptorArena::invalidate_device_loss() {
    std::scoped_lock lock(state_->mutex);
    if (state_->invalidated)
        return;
    state_->invalidated = true;
    state_->quarantined = state_->pools.size();
    state_->pending = 0;
    for (auto &pool : state_->pools)
        pool.pending = 0;
}

DescriptorArenaSnapshot DescriptorArena::snapshot() const {
    std::scoped_lock lock(state_->mutex);
    DescriptorArenaSnapshot result;
    result.pool_limit = state_->pool_limit;
    result.pool_count = state_->invalidated ? 0 : state_->pools.size();
    result.pool_creations = state_->pool_creations;
    result.allocations = state_->allocations;
    result.reuses = state_->reuses;
    result.rollovers = state_->rollovers;
    result.pending = state_->pending;
    result.quarantined = state_->quarantined;
    result.invalidated = state_->invalidated;
    if (!state_->invalidated)
        for (const auto &pool : state_->pools)
            result.live_sets += pool.sets.size();
    return result;
}
