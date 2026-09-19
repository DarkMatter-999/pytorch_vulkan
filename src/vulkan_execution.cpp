#include "vulkan_execution.h"

#include "vulkan_platform.h"

#include <array>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>

namespace {
struct QuarantinedExecution {
    bool occupied = false;
    std::array<VkCommandBuffer, 2> command_buffers{};
    std::array<VkFence, 2> fences{};
    std::array<VkSemaphore, 2> signal_semaphores{};
    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
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

[[noreturn]] void throw_device_lost(VkResult result) {
    throw VulkanDeviceLost("Vulkan device lost; execution state invalidated (VkResult " +
                           std::to_string(static_cast<int>(result)) + ")");
}

std::exception_ptr
execute_callbacks(std::vector<std::function<void()>> &callbacks) noexcept {
    std::exception_ptr error;
    for (auto &callback : callbacks) {
        try {
            callback();
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
    }
    callbacks.clear();
    return error;
}
} // namespace

VulkanExecutionContext::VulkanExecutionContext(VkDevice device, VkQueue queue,
                                               VkCommandPool command_pool)
    : device_(device), queue_(queue), command_pool_(command_pool) {
    if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
        command_pool_ == VK_NULL_HANDLE)
        throw std::invalid_argument("Vulkan execution context requires valid handles");
    VkCommandBufferAllocateInfo allocation{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
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
            VkSemaphoreCreateInfo semaphore_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
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
    initialize_timestamp_queries();
}

VulkanExecutionContext::~VulkanExecutionContext() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
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
            slot.timestamp_query_pool = timestamp_query_pool_;
            slot.callbacks = std::move(deferred_callbacks_);
            for (auto &record : ring_) {
                for (auto &callback : record.callbacks)
                    slot.callbacks.push_back(std::move(callback));
                record.callbacks.clear();
            }
            for (auto &callback : completion_callbacks_)
                slot.callbacks.push_back(std::move(callback));
            completion_callbacks_.clear();
            for (auto &record : ring_) {
                record.command_buffer = VK_NULL_HANDLE;
                record.fence = VK_NULL_HANDLE;
                record.signal_semaphore = VK_NULL_HANDLE;
            }
            latest_signal_semaphore_ = VK_NULL_HANDLE;
            timestamp_query_pool_ = VK_NULL_HANDLE;
            return;
        }
        // Completion is unknown and no allocation-free owner remains: leak safely.
        return;
    }
    std::vector<std::function<void()>> callbacks;
    for (auto &record : ring_) {
        for (auto &callback : record.callbacks)
            callbacks.push_back(std::move(callback));
        record.callbacks.clear();
    }
    for (auto &callback : deferred_callbacks_)
        callbacks.push_back(std::move(callback));
    deferred_callbacks_.clear();
    for (auto &callback : completion_callbacks_)
        callbacks.push_back(std::move(callback));
    completion_callbacks_.clear();
    lock.unlock();
    execute_callbacks(callbacks);
    lock.lock();
    for (auto &record : ring_) {
        if (record.fence != VK_NULL_HANDLE)
            vkDestroyFence(device_, record.fence, nullptr);
        if (record.signal_semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(device_, record.signal_semaphore, nullptr);
        if (record.command_buffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device_, command_pool_, 1, &record.command_buffer);
    }
    if (timestamp_query_pool_ != VK_NULL_HANDLE)
        vkDestroyQueryPool(device_, timestamp_query_pool_, nullptr);
}

void VulkanExecutionContext::begin(const char *scope) {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (recording_)
        throw std::logic_error("Vulkan execution context is already recording");
    active_slot_ = (active_slot_ + 1) % kRingSize;
    if (ring_[active_slot_].submitted)
        wait_and_retire(ring_[active_slot_], lock);
    if (latest_signal_semaphore_ == ring_[active_slot_].signal_semaphore)
        recreate_signal_semaphore(ring_[active_slot_]);
    const auto allocation_start = std::chrono::steady_clock::now();
    reset_reusable_resources(ring_[active_slot_]);
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_result(vkBeginCommandBuffer(ring_[active_slot_].command_buffer, &info),
                 "Could not begin Vulkan execution command buffer");
    ring_[active_slot_].timestamp_scope = scope == nullptr ? "operator" : scope;
    ring_[active_slot_].timestamp_recorded = false;
    if (timestamp_queries_supported_) {
        const uint32_t query_base = static_cast<uint32_t>(active_slot_ * 2);
        vkCmdResetQueryPool(ring_[active_slot_].command_buffer, timestamp_query_pool_,
                            query_base, 2);
        vkCmdWriteTimestamp(ring_[active_slot_].command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool_,
                            query_base);
        ring_[active_slot_].timestamp_begin = query_base;
        ring_[active_slot_].timestamp_end = query_base + 1;
        ring_[active_slot_].timestamp_recorded = true;
    }
    if (platform_)
        platform_->record_timing(
            VulkanTimingCategory::Allocation,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          allocation_start)
                .count());
    recording_ = true;
}

VkCommandBuffer VulkanExecutionContext::command_buffer() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution command buffer is not recording");
    return ring_[active_slot_].command_buffer;
}

void VulkanExecutionContext::submit() {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    try {
        const auto recording_start = std::chrono::steady_clock::now();
        if (ring_[active_slot_].timestamp_recorded)
            vkCmdWriteTimestamp(ring_[active_slot_].command_buffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                timestamp_query_pool_,
                                ring_[active_slot_].timestamp_end);
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
        const VkResult submit_result =
            vkQueueSubmit(queue_, 1, &submit, ring_[active_slot_].fence);
        if (submit_result == VK_ERROR_DEVICE_LOST) {
            invalidate(submit_result, lock);
            throw_device_lost(submit_result);
        }
        check_result(submit_result, "Could not submit Vulkan execution command buffer");
        if (platform_) {
            platform_->record_compute_submitted();
            platform_->record_timing(
                VulkanTimingCategory::Submit,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              submit_start)
                    .count());
        }
        ring_[active_slot_].callbacks = std::move(deferred_callbacks_);
        ring_[active_slot_].submission_id = ++next_submission_id_;
        ring_[active_slot_].submitted = true;
        latest_signal_semaphore_ = ring_[active_slot_].signal_semaphore;
        recording_ = false;
    } catch (...) {
        if (invalidated_)
            throw;
        abandon_recording(std::current_exception(), lock);
    }
}

void VulkanExecutionContext::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (recording_)
        throw std::logic_error("Cannot wait while Vulkan execution is recording");
    bool found = false;
    for (auto &record : ring_) {
        if (record.submitted) {
            found = true;
            wait_and_retire(record, lock);
        }
    }
    (void)found;
}

void VulkanExecutionContext::synchronize() { wait(); }

void VulkanExecutionContext::retire_completed() {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (recording_)
        throw std::logic_error("Cannot retire while Vulkan execution is recording");
    for (auto &record : ring_) {
        const VkResult status = record.submitted
                                    ? vkGetFenceStatus(device_, record.fence)
                                    : VK_NOT_READY;
        if (status == VK_ERROR_DEVICE_LOST) {
            invalidate(status, lock);
            throw_device_lost(status);
        }
        if (record.submitted && status == VK_SUCCESS) {
            if (platform_)
                platform_->record_compute_completed();
            resolve_timestamp(record);
            retire(record, lock);
        }
    }
}

void VulkanExecutionContext::cancel() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (invalidated_) {
        recording_ = false;
        return;
    }
    if (!recording_)
        return;
    recording_ = false;
    std::vector<std::function<void()>> callbacks;
    callbacks.swap(deferred_callbacks_);
    lock.unlock();
    execute_callbacks(callbacks);
    lock.lock();
    try {
        reset_reusable_resources(ring_[active_slot_]);
    } catch (...) {
    }
}

bool VulkanExecutionContext::recording() const {
    std::scoped_lock lock(mutex_);
    return recording_;
}

bool VulkanExecutionContext::invalidated() const {
    std::scoped_lock lock(mutex_);
    return invalidated_;
}

void VulkanExecutionContext::defer_destruction(std::function<void()> callback) {
    if (!callback)
        throw std::invalid_argument("Vulkan execution callback is empty");
    std::scoped_lock lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    deferred_callbacks_.push_back(std::move(callback));
}

bool VulkanExecutionContext::retain_until_completion(std::function<void()> callback) {
    if (!callback)
        throw std::invalid_argument("Vulkan execution callback is empty");
    std::scoped_lock lock(mutex_);
    if (recording_) {
        deferred_callbacks_.push_back(std::move(callback));
        return true;
    }
    bool submitted = false;
    for (const auto &record : ring_) {
        if (record.submitted) {
            submitted = true;
            break;
        }
    }
    if (submitted) {
        completion_callbacks_.push_back(std::move(callback));
        return true;
    }
    return false;
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

bool VulkanExecutionContext::timestamp_queries_supported() const {
    std::scoped_lock lock(mutex_);
    return timestamp_queries_supported_;
}

std::string VulkanExecutionContext::timestamp_query_support_reason() const {
    std::scoped_lock lock(mutex_);
    return timestamp_query_support_reason_;
}

std::vector<VulkanTimestampSample> VulkanExecutionContext::timestamp_samples() const {
    std::scoped_lock lock(mutex_);
    return timestamp_samples_;
}

void VulkanExecutionContext::reset_timestamp_samples() {
    std::scoped_lock lock(mutex_);
    timestamp_samples_.clear();
}

std::size_t VulkanExecutionContext::timestamp_query_capacity() const {
    std::scoped_lock lock(mutex_);
    return timestamp_queries_supported_ ? kRingSize * 2 : 0;
}

std::size_t VulkanExecutionContext::timestamp_query_in_use() const {
    std::scoped_lock lock(mutex_);
    if (!timestamp_queries_supported_)
        return 0;
    std::size_t in_use = 0;
    for (const auto &record : ring_) {
        if (record.timestamp_recorded && (record.submitted ||
                                          (&record == &ring_[active_slot_] && recording_)))
            in_use += 2;
    }
    return in_use;
}

bool VulkanExecutionContext::timestamp_query_quarantined() const {
    std::scoped_lock lock(mutex_);
    return timestamp_query_quarantined_;
}

void VulkanExecutionContext::retire(InFlightRecord &record,
                                    std::unique_lock<std::mutex> &lock) {
    std::vector<std::function<void()>> callbacks;
    callbacks.swap(record.callbacks);
    record.submitted = false;
    bool submitted = false;
    for (const auto &pending : ring_)
        submitted |= pending.submitted;
    if (submitted) {
        for (auto &callback : callbacks)
            completion_callbacks_.push_back(std::move(callback));
        callbacks.clear();
        return;
    }
    for (auto &callback : completion_callbacks_)
        callbacks.push_back(std::move(callback));
    completion_callbacks_.clear();
    lock.unlock();
    std::exception_ptr error = execute_callbacks(callbacks);
    lock.lock();
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
    check_result(
        vkCreateSemaphore(device_, &semaphore_info, nullptr, &record.signal_semaphore),
        "Could not recreate Vulkan execution semaphore");
}

void VulkanExecutionContext::reset_reusable_resources(InFlightRecord &record) {
    check_result(vkResetCommandBuffer(record.command_buffer, 0),
                 "Could not reset Vulkan execution command buffer");
    check_result(vkResetFences(device_, 1, &record.fence),
                 "Could not reset Vulkan execution fence");
}

void VulkanExecutionContext::initialize_timestamp_queries() {
    if (platform_ == nullptr)
        return;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(platform_->physical_device(), &properties);
    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(platform_->physical_device(), &queue_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(platform_->physical_device(), &queue_count,
                                             queues.data());
    const uint32_t queue_family = platform_->device_info().compute_queue_family;
    timestamp_period_ = properties.limits.timestampPeriod;
    if (!properties.limits.timestampComputeAndGraphics || timestamp_period_ <= 0.0F ||
        queue_family >= queues.size() || queues[queue_family].timestampValidBits == 0) {
        timestamp_query_support_reason_ =
            "device does not support compute timestamp queries";
        return;
    }
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = static_cast<uint32_t>(kRingSize * 2);
    if (vkCreateQueryPool(device_, &info, nullptr, &timestamp_query_pool_) != VK_SUCCESS) {
        timestamp_query_support_reason_ = "timestamp query pool creation failed";
        return;
    }
    timestamp_queries_supported_ = true;
    timestamp_query_support_reason_ = "supported";
}

void VulkanExecutionContext::resolve_timestamp(InFlightRecord &record) {
    if (!record.timestamp_recorded || !timestamp_queries_supported_)
        return;
    uint64_t values[4]{};
    const VkResult result = vkGetQueryPoolResults(
        device_, timestamp_query_pool_, record.timestamp_begin, 2, sizeof(values), values,
        sizeof(uint64_t) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY)
        return;
    VulkanTimestampSample sample;
    sample.available = result == VK_SUCCESS && values[1] != 0 && values[3] != 0;
    sample.submission_id = record.submission_id;
    sample.scope = record.timestamp_scope;
    if (sample.available && values[2] >= values[0])
        sample.gpu_time_ns = static_cast<uint64_t>(
            static_cast<double>(values[2] - values[0]) * timestamp_period_);
    timestamp_samples_.push_back(std::move(sample));
}

void VulkanExecutionContext::wait_and_retire(InFlightRecord &record,
                                              std::unique_lock<std::mutex> &lock) {
    const auto wait_start = std::chrono::steady_clock::now();
    const VkResult result =
        vkWaitForFences(device_, 1, &record.fence, VK_TRUE, UINT64_MAX);
    if (result == VK_ERROR_DEVICE_LOST) {
        invalidate(result, lock);
        throw_device_lost(result);
    }
    if (result != VK_SUCCESS) {
        const VkResult recovery_result = vkQueueWaitIdle(queue_);
        if (recovery_result == VK_ERROR_DEVICE_LOST) {
            invalidate(recovery_result, lock);
            throw_device_lost(recovery_result);
        }
        check_result(recovery_result, "Could not confirm Vulkan execution completion");
        check_result(result, "Could not wait for Vulkan execution fence");
    }
    const double wait_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start)
            .count();
    if (platform_) {
        platform_->record_compute_wait();
        platform_->record_compute_completed();
        platform_->record_timing(VulkanTimingCategory::HostFenceWait, wait_seconds);
    }
    resolve_timestamp(record);
    retire(record, lock);
}

[[noreturn]] void
VulkanExecutionContext::abandon_recording(std::exception_ptr original,
                                          std::unique_lock<std::mutex> &lock) {
    recording_ = false;
    std::vector<std::function<void()>> callbacks;
    callbacks.swap(deferred_callbacks_);
    lock.unlock();
    execute_callbacks(callbacks);
    lock.lock();
    if (!invalidated_) {
        try {
            reset_reusable_resources(ring_[active_slot_]);
        } catch (...) {
        }
    }
    std::rethrow_exception(original);
}

void VulkanExecutionContext::throw_if_invalidated(
    std::unique_lock<std::mutex> &lock) {
    if (invalidated_) {
        throw_device_lost(invalidation_result_);
    }
    if (platform_ && platform_->device_lost()) {
        invalidate(VK_ERROR_DEVICE_LOST, lock);
        throw_device_lost(VK_ERROR_DEVICE_LOST);
    }
}

void VulkanExecutionContext::invalidate(VkResult result,
                                        std::unique_lock<std::mutex> &lock) {
    if (invalidated_)
        return;
    invalidated_ = true;
    invalidation_result_ = result;
    recording_ = false;
    if (platform_)
        platform_->mark_device_lost(result);

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
        slot.timestamp_query_pool = timestamp_query_pool_;
        for (auto &record : ring_) {
            for (auto &callback : record.callbacks)
                slot.callbacks.push_back(std::move(callback));
            record.callbacks.clear();
            record.command_buffer = VK_NULL_HANDLE;
            record.fence = VK_NULL_HANDLE;
            record.signal_semaphore = VK_NULL_HANDLE;
            record.submitted = false;
        }
        for (auto &callback : completion_callbacks_)
            slot.callbacks.push_back(std::move(callback));
        completion_callbacks_.clear();
        latest_signal_semaphore_ = VK_NULL_HANDLE;
        timestamp_query_pool_ = VK_NULL_HANDLE;
        timestamp_query_quarantined_ = true;
        device_ = VK_NULL_HANDLE;
        queue_ = VK_NULL_HANDLE;
        command_pool_ = VK_NULL_HANDLE;
        (void)lock;
        return;
    }
    for (auto &record : ring_)
        record.submitted = false;
    // The quarantine table is bounded. If it is full, leak the handles rather
    // than letting the destructor attempt to use an invalid device.
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    command_pool_ = VK_NULL_HANDLE;
}
