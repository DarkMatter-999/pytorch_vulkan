#include "vulkan/pipeline_cache.h"

#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void combine(std::size_t &seed, std::size_t value) {
    seed ^= value + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6) + (seed >> 2);
}
} // namespace

std::size_t std::hash<PipelineKey>::operator()(const PipelineKey &key) const noexcept {
    std::size_t result = std::hash<std::string>{}(key.shader_identity);
    combine(result, std::hash<uint64_t>{}(key.descriptor_layout_identity));
    combine(result, std::hash<uint64_t>{}(key.shader_code_identity));
    for (const uint32_t value : key.specialization_values)
        combine(result, std::hash<uint32_t>{}(value));
    combine(result, std::hash<uint64_t>{}(key.required_feature_bits));
    return result;
}

std::size_t std::hash<PipelineLayoutKey>::operator()(
    const PipelineLayoutKey &key) const noexcept {
    std::size_t result = std::hash<uint64_t>{}(key.descriptor_layout_identity);
    combine(result, std::hash<uint32_t>{}(key.stage_flags));
    combine(result, std::hash<uint32_t>{}(key.push_constant_offset));
    combine(result, std::hash<uint32_t>{}(key.push_constant_size));
    return result;
}

std::size_t VulkanPipelineCache::KeyHash::operator()(const PipelineKey &key) const noexcept {
    return std::hash<PipelineKey>{}(key);
}

std::size_t VulkanPipelineCache::LayoutKeyHash::operator()(
    const PipelineLayoutKey &key) const noexcept {
    return std::hash<PipelineLayoutKey>{}(key);
}

VulkanPipelineCache::VulkanPipelineCache(VkDevice device, VulkanPlatform &platform,
                                         VulkanExecutionContext &execution,
                                         std::size_t max_entries)
    : device_(device), platform_(platform), execution_(execution), max_entries_(max_entries) {
    if (max_entries == 0)
        throw std::invalid_argument("Vulkan pipeline cache maximum must be positive");
}

VulkanPipelineCache::~VulkanPipelineCache() noexcept { destroy_all(); }

void VulkanPipelineCache::destroy_pipeline(VkPipeline pipeline) noexcept {
    if (pipeline == VK_NULL_HANDLE)
        return;
    const auto pending = pending_destructions_;
    pending->fetch_add(1, std::memory_order_relaxed);
    std::function<void()> destroy = [device = device_, pipeline, pending] {
        vkDestroyPipeline(device, pipeline, nullptr);
        pending->fetch_sub(1, std::memory_order_relaxed);
    };
    try {
        if (!execution_.retain_until_completion(destroy))
            destroy();
    } catch (...) {
        // Device-loss teardown must not issue destruction against an invalid device.
        pending->fetch_sub(1, std::memory_order_relaxed);
    }
}

void VulkanPipelineCache::destroy_layout(VkPipelineLayout layout) noexcept {
    if (layout == VK_NULL_HANDLE)
        return;
    const auto pending = pending_layout_destructions_;
    pending->fetch_add(1, std::memory_order_relaxed);
    std::function<void()> destroy = [device = device_, layout, pending] {
        vkDestroyPipelineLayout(device, layout, nullptr);
        pending->fetch_sub(1, std::memory_order_relaxed);
    };
    try {
        if (!execution_.retain_until_completion(destroy))
            destroy();
    } catch (...) {
        pending->fetch_sub(1, std::memory_order_relaxed);
    }
}

VkPipeline VulkanPipelineCache::evict_one_locked() {
    const PipelineKey key = lru_.front();
    lru_.pop_front();
    const auto found = entries_.find(key);
    if (found == entries_.end())
        return VK_NULL_HANDLE;
    const VkPipeline pipeline = found->second.pipeline;
    entries_.erase(found);
    ++evictions_;
    return pipeline;
}

VkPipeline VulkanPipelineCache::get_or_create(
    const PipelineKey &key, VkPipelineLayout pipeline_layout, VkShaderModule shader,
    const VkComputePipelineCreateInfo &create_info) {
    if (platform_.device_lost() || execution_.invalidated())
        throw VulkanDeviceLost("Vulkan pipeline cache is invalidated");
    VkPipeline evicted = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    {
        std::scoped_lock lock(mutex_);
        if (invalidated_ || platform_.device_lost())
            throw VulkanDeviceLost("Vulkan pipeline cache is invalidated");
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            ++hits_;
            lru_.splice(lru_.end(), lru_, found->second.lru);
            return found->second.pipeline;
        }
        ++misses_;
        VkComputePipelineCreateInfo info = create_info;
        info.layout = pipeline_layout;
        info.stage.module = shader;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) !=
            VK_SUCCESS)
            throw std::runtime_error("could not create cached compute pipeline");
        lru_.push_back(key);
        entries_.emplace(key, Entry{pipeline, std::prev(lru_.end())});
        if (entries_.size() > max_entries_)
            evicted = evict_one_locked();
    }
    if (evicted != VK_NULL_HANDLE)
        destroy_pipeline(evicted);
    return pipeline;
}

VkPipelineLayout VulkanPipelineCache::get_or_create_layout(
    const PipelineLayoutKey &key, const VkPipelineLayoutCreateInfo &create_info) {
    if (platform_.device_lost() || execution_.invalidated())
        throw VulkanDeviceLost("Vulkan pipeline cache is invalidated");
    std::scoped_lock lock(mutex_);
    if (invalidated_ || platform_.device_lost())
        throw VulkanDeviceLost("Vulkan pipeline cache is invalidated");
    const auto found = layouts_.find(key);
    if (found != layouts_.end())
        return found->second.layout;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device_, &create_info, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("could not create cached pipeline layout");
    layouts_.emplace(key, LayoutEntry{layout});
    return layout;
}

void VulkanPipelineCache::invalidate_device_loss() {
    std::scoped_lock lock(mutex_);
    invalidated_ = true;
}

PipelineCacheSnapshot VulkanPipelineCache::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {entries_.size(), hits_, misses_, evictions_,
            entries_.size() + pending_destructions_->load(std::memory_order_relaxed),
            layouts_.size(), pending_destructions_->load(std::memory_order_relaxed),
            pending_layout_destructions_->load(std::memory_order_relaxed), invalidated_};
}

void VulkanPipelineCache::destroy_all() noexcept {
    std::vector<VkPipeline> pipelines;
    std::vector<VkPipelineLayout> layouts;
    bool invalidated = false;
    {
        std::scoped_lock lock(mutex_);
        for (const auto &entry : entries_)
            pipelines.push_back(entry.second.pipeline);
        for (const auto &entry : layouts_)
            layouts.push_back(entry.second.layout);
        entries_.clear();
        layouts_.clear();
        lru_.clear();
        invalidated = invalidated_;
    }
    if (invalidated || platform_.device_lost())
        return;
    for (const VkPipeline pipeline : pipelines)
        destroy_pipeline(pipeline);
    for (const VkPipelineLayout layout : layouts)
        destroy_layout(layout);
}
