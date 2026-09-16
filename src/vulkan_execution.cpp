#include "vulkan_execution.h"

#include "vulkan_platform.h"

#include <array>
#include <exception>
#include <stdexcept>
#include <string>
#include <chrono>

namespace {
struct QuarantinedExecution {
    bool occupied = false;
    std::array<VkCommandBuffer, 2> command_buffers{};
    std::array<VkFence, 2> fences{};
    std::array<VkSemaphore, 2> signal_semaphores{};
    std::vector<std::function<void()>> callbacks;
};
constexpr std::size_t kQuarantineCapacity = 16;
std::mutex quarantine_mutex;
std::array<QuarantinedExecution, kQuarantineCapacity> quarantined;

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
}
} // namespace

VulkanExecutionContext::VulkanExecutionContext(VkDevice device, VkQueue queue,
                                               VkCommandPool command_pool)
    : device_(device), queue_(queue), command_pool_(command_pool) {
    if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
        command_pool_ == VK_NULL_HANDLE)
        throw std::invalid_argument("Vulkan execution context requires valid handles");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = command_pool_;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = kRingSize;
    VkCommandBuffer buffers[kRingSize]{};
    try {
        check_result(vkAllocateCommandBuffers(device_, &allocation, buffers),
                     "Could not allocate Vulkan execution command buffers");
        for (std::size_t i = 0; i < kRingSize; ++i)
            ring_[i].command_buffer = buffers[i];
        for (std::size_t i = 0; i < kRingSize; ++i) {
            VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check_result(vkCreateFence(device_, &fence_info, nullptr, &ring_[i].fence),
                         "Could not create Vulkan execution fence");
            VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            check_result(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                           &ring_[i].signal_semaphore),
                         "Could not create Vulkan execution semaphore");
        }
    } catch (...) {
        for (auto &record : ring_) {
            if (record.fence != VK_NULL_HANDLE)
                vkDestroyFence(device_, record.fence, nullptr);
            if (record.signal_semaphore != VK_NULL_HANDLE)
                vkDestroySemaphore(device_, record.signal_semaphore, nullptr);
        }
        VkCommandBuffer allocated[kRingSize]{};
        std::size_t count = 0;
        for (const auto &record : ring_) {
            if (record.command_buffer != VK_NULL_HANDLE)
                allocated[count++] = record.command_buffer;
        }
        if (count)
            vkFreeCommandBuffers(device_, command_pool_, count, allocated);
        throw;
    }
}

VulkanExecutionContext::VulkanExecutionContext(VkDevice device, VkQueue queue,
                                               VkCommandPool command_pool,
                                               VulkanPlatform *platform)
    : VulkanExecutionContext(device, queue, command_pool) {
    platform_ = platform;
}

VulkanExecutionContext::~VulkanExecutionContext() noexcept {
    std::scoped_lock lock(mutex_);
    if (device_ == VK_NULL_HANDLE)
        return;
    if (vkQueueWaitIdle(queue_) != VK_SUCCESS) {
        std::scoped_lock quarantine_lock(quarantine_mutex);
        for (auto &slot : quarantined) {
            if (slot.occupied)
                continue;
            slot.occupied = true;
            for (std::size_t i = 0; i < kRingSize; ++i) {
                slot.command_buffers[i] = ring_[i].command_buffer;
                slot.fences[i] = ring_[i].fence;
                slot.signal_semaphores[i] = ring_[i].signal_semaphore;
            }
            slot.callbacks = std::move(deferred_callbacks_);
            for (const auto &record : ring_)
                for (const auto &callback : record.callbacks)
                    slot.callbacks.push_back(callback);
            for (auto &record : ring_) {
                record.command_buffer = VK_NULL_HANDLE;
                record.fence = VK_NULL_HANDLE;
                record.signal_semaphore = VK_NULL_HANDLE;
            }
            latest_signal_semaphore_ = VK_NULL_HANDLE;
            return;
        }
        // Completion is unknown and no allocation-free owner remains: leak safely.
        return;
    }
    for (auto &record : ring_) {
        for (auto &callback : record.callbacks) {
            try { callback(); } catch (...) {}
        }
        record.callbacks.clear();
        if (record.fence != VK_NULL_HANDLE)
            vkDestroyFence(device_, record.fence, nullptr);
        if (record.signal_semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(device_, record.signal_semaphore, nullptr);
        if (record.command_buffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device_, command_pool_, 1, &record.command_buffer);
    }
    for (auto &callback : deferred_callbacks_) {
        try { callback(); } catch (...) {}
    }
}

void VulkanExecutionContext::begin() {
    std::scoped_lock lock(mutex_);
    if (recording_)
        throw std::logic_error("Vulkan execution context is already recording");
    active_slot_ = (active_slot_ + 1) % kRingSize;
    if (ring_[active_slot_].submitted)
        wait_and_retire(ring_[active_slot_]);
    if (latest_signal_semaphore_ == ring_[active_slot_].signal_semaphore)
        recreate_signal_semaphore(ring_[active_slot_]);
    const auto allocation_start = std::chrono::steady_clock::now();
    reset_reusable_resources(ring_[active_slot_]);
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_result(vkBeginCommandBuffer(ring_[active_slot_].command_buffer, &info),
                 "Could not begin Vulkan execution command buffer");
    if (platform_)
        platform_->record_timing(
            VulkanTimingCategory::Allocation,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          allocation_start)
                .count());
    recording_ = true;
}

VkCommandBuffer VulkanExecutionContext::command_buffer() const {
    std::scoped_lock lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution command buffer is not recording");
    return ring_[active_slot_].command_buffer;
}

void VulkanExecutionContext::submit() {
    std::scoped_lock lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    try {
        const auto recording_start = std::chrono::steady_clock::now();
        check_result(vkEndCommandBuffer(ring_[active_slot_].command_buffer),
                     "Could not end Vulkan execution command buffer");
        if (platform_)
            platform_->record_timing(
                VulkanTimingCategory::Recording,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              recording_start)
                    .count());
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        const VkSemaphore wait_semaphore = latest_signal_semaphore_;
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (wait_semaphore != VK_NULL_HANDLE) {
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &wait_semaphore;
            submit.pWaitDstStageMask = &wait_stage;
        }
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &ring_[active_slot_].command_buffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &ring_[active_slot_].signal_semaphore;
        const auto submit_start = std::chrono::steady_clock::now();
        check_result(vkQueueSubmit(queue_, 1, &submit, ring_[active_slot_].fence),
                     "Could not submit Vulkan execution command buffer");
        if (platform_) {
            platform_->record_compute_submitted();
            platform_->record_timing(
                VulkanTimingCategory::SubmitWait,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              submit_start)
                    .count());
        }
        ring_[active_slot_].callbacks = std::move(deferred_callbacks_);
        ring_[active_slot_].submitted = true;
        latest_signal_semaphore_ = ring_[active_slot_].signal_semaphore;
        recording_ = false;
    } catch (...) {
        abandon_recording(std::current_exception());
    }
}

void VulkanExecutionContext::wait() {
    std::scoped_lock lock(mutex_);
    if (recording_)
        throw std::logic_error("Cannot wait while Vulkan execution is recording");
    bool found = false;
    for (auto &record : ring_) {
        if (record.submitted) {
            found = true;
            wait_and_retire(record);
        }
    }
    (void)found;
}

void VulkanExecutionContext::synchronize() { wait(); }

void VulkanExecutionContext::retire_completed() {
    std::scoped_lock lock(mutex_);
    if (recording_)
        throw std::logic_error("Cannot retire while Vulkan execution is recording");
    for (auto &record : ring_) {
        if (record.submitted && vkGetFenceStatus(device_, record.fence) == VK_SUCCESS) {
            if (platform_)
                platform_->record_compute_completed();
            retire(record);
        }
    }
}

void VulkanExecutionContext::cancel() {
    std::scoped_lock lock(mutex_);
    if (!recording_)
        return;
    recording_ = false;
    for (auto &callback : deferred_callbacks_) {
        try { callback(); } catch (...) {}
    }
    deferred_callbacks_.clear();
    try { reset_reusable_resources(ring_[active_slot_]); } catch (...) {}
}

bool VulkanExecutionContext::recording() const {
    std::scoped_lock lock(mutex_);
    return recording_;
}

void VulkanExecutionContext::defer_destruction(std::function<void()> callback) {
    if (!callback)
        throw std::invalid_argument("Vulkan execution callback is empty");
    std::scoped_lock lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    deferred_callbacks_.push_back(std::move(callback));
}

void VulkanExecutionContext::retain(VkDescriptorPool descriptor_pool) {
    if (descriptor_pool == VK_NULL_HANDLE)
        return;
    std::scoped_lock lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    deferred_callbacks_.push_back([device = device_, descriptor_pool] {
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    });
}

std::size_t VulkanExecutionContext::pending_count() const {
    std::scoped_lock lock(mutex_);
    std::size_t count = 0;
    for (const auto &record : ring_)
        count += record.submitted ? 1 : 0;
    if (recording_)
        ++count;
    return count;
}

void VulkanExecutionContext::retire(InFlightRecord &record) {
    std::exception_ptr error;
    for (auto &callback : record.callbacks) {
        try { callback(); } catch (...) { if (!error) error = std::current_exception(); }
    }
    record.callbacks.clear();
    record.submitted = false;
    if (error)
        std::rethrow_exception(error);
}

void VulkanExecutionContext::recreate_signal_semaphore(InFlightRecord &record) {
    if (record.signal_semaphore == VK_NULL_HANDLE)
        return;
    vkDestroySemaphore(device_, record.signal_semaphore, nullptr);
    record.signal_semaphore = VK_NULL_HANDLE;
    latest_signal_semaphore_ = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check_result(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                   &record.signal_semaphore),
                 "Could not recreate Vulkan execution semaphore");
}

void VulkanExecutionContext::reset_reusable_resources(InFlightRecord &record) {
    check_result(vkResetCommandBuffer(record.command_buffer, 0),
                 "Could not reset Vulkan execution command buffer");
    check_result(vkResetFences(device_, 1, &record.fence),
                 "Could not reset Vulkan execution fence");
}

void VulkanExecutionContext::wait_and_retire(InFlightRecord &record) {
    const auto wait_start = std::chrono::steady_clock::now();
    const VkResult result = vkWaitForFences(device_, 1, &record.fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        check_result(vkQueueWaitIdle(queue_),
                     "Could not confirm Vulkan execution completion");
        check_result(result, "Could not wait for Vulkan execution fence");
    }
    const double wait_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start)
            .count();
    if (platform_) {
        platform_->record_compute_wait();
        platform_->record_compute_completed();
        platform_->record_timing(VulkanTimingCategory::SubmitWait, wait_seconds);
        platform_->record_timing(VulkanTimingCategory::Compute, wait_seconds);
    }
    retire(record);
}

[[noreturn]] void VulkanExecutionContext::abandon_recording(std::exception_ptr original) {
    recording_ = false;
    for (auto &callback : deferred_callbacks_) {
        try { callback(); } catch (...) {}
    }
    deferred_callbacks_.clear();
    try { reset_reusable_resources(ring_[active_slot_]); } catch (...) {}
    std::rethrow_exception(original);
}
