#include "vulkan_execution.h"

#include "vulkan_platform.h"
#include "vulkan_synchronization.h"

#include <array>
#include <chrono>
#include <cxxabi.h>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <unistd.h>

struct VulkanTransferReservationState {
    const int process = ::getpid();
    std::mutex mutex;
    std::condition_variable cv;
    std::size_t occupied = 0;
    uint64_t generation = 0;
    bool closing = false;
    bool lifecycle_closed = false;
    bool poisoned = false;
};

struct VulkanTransferSlotLease {
    explicit VulkanTransferSlotLease(std::shared_ptr<VulkanTransferReservationState> state)
        : state(std::move(state)) {}
    ~VulkanTransferSlotLease() {
        if (!state || state->process != ::getpid()) return;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->occupied) --state->occupied;
        ++state->generation;
        state->cv.notify_all();
    }
    std::shared_ptr<VulkanTransferReservationState> state;
};

namespace {
// Identity outlives callback invocation and last capture destruction, including
// the foreign pump's final member accesses. Intrusive TLS has no allocation or
// execution/lifecycle lock and supports reentrant ready-only observation.
struct PumpIdentity {
    pytorch_vulkan::VulkanCallbackObservationScope observation;
    const VulkanExecutionContext* context;
    PumpIdentity* previous;
    static thread_local PumpIdentity* current;
    explicit PumpIdentity(const VulkanExecutionContext* c) : context(c), previous(current) { current = this; }
    ~PumpIdentity() { current = previous; }
    static bool contains(const VulkanExecutionContext* c) noexcept {
        for (auto* p = current; p; p = p->previous) if (p->context == c) return true;
        return false;
    }
};
thread_local PumpIdentity* PumpIdentity::current = nullptr;
constexpr uint32_t kQueriesPerSlot = 10;

VkResult real_allocate(void *, VkDevice d, const VkCommandBufferAllocateInfo *i,
                       VkCommandBuffer *b) { return vkAllocateCommandBuffers(d, i, b); }
void real_free(void *, VkDevice d, VkCommandPool p, uint32_t n,
               const VkCommandBuffer *b) { vkFreeCommandBuffers(d, p, n, b); }
VkResult real_create_fence(void *, VkDevice d, const VkFenceCreateInfo *i,
                           const VkAllocationCallbacks *a, VkFence *f) {
    return vkCreateFence(d, i, a, f);
}
void real_destroy_fence(void *, VkDevice d, VkFence f, const VkAllocationCallbacks *a) {
    vkDestroyFence(d, f, a);
}
VkResult real_create_semaphore(void *, VkDevice d, const VkSemaphoreCreateInfo *i,
                               const VkAllocationCallbacks *a, VkSemaphore *s) {
    return vkCreateSemaphore(d, i, a, s);
}
void real_destroy_semaphore(void *, VkDevice d, VkSemaphore s,
                            const VkAllocationCallbacks *a) {
    vkDestroySemaphore(d, s, a);
}
VkResult real_begin(void *, VkCommandBuffer b, const VkCommandBufferBeginInfo *i) {
    return vkBeginCommandBuffer(b, i);
}
VkResult real_end(void *, VkCommandBuffer b) { return vkEndCommandBuffer(b); }
VkResult real_reset_command(void *, VkCommandBuffer b, VkCommandBufferResetFlags f) {
    return vkResetCommandBuffer(b, f);
}
VkResult real_reset_fences(void *, VkDevice d, uint32_t n, const VkFence *f) {
    return vkResetFences(d, n, f);
}
VkResult real_submit(void *, VkQueue q, uint32_t n, const VkSubmitInfo *i, VkFence f) {
    return vkQueueSubmit(q, n, i, f);
}
VkResult real_fence_status(void *, VkDevice d, VkFence f) {
    return vkGetFenceStatus(d, f);
}
VkResult real_wait_fences(void *, VkDevice d, uint32_t n, const VkFence *f,
                          VkBool32 all, uint64_t timeout) {
    return vkWaitForFences(d, n, f, all, timeout);
}
VkResult real_queue_idle(void *, VkQueue q) { return vkQueueWaitIdle(q); }

VulkanExecutionOps real_execution_ops() {
    return {nullptr, real_allocate, real_free, real_create_fence, real_destroy_fence,
            real_create_semaphore, real_destroy_semaphore, real_begin, real_end,
            real_reset_command, real_reset_fences, real_submit, real_fence_status,
            real_wait_fences, real_queue_idle};
}

constexpr std::size_t kQuarantineCapacity = 16;
std::mutex quarantine_mutex;
// Intentionally never destroyed: uncertain device ownership outlives process static teardown.
std::array<VulkanExecutionQuarantine *, kQuarantineCapacity> quarantined{};

void retain_production_quarantine(void *, VulkanExecutionQuarantine *payload) noexcept {
    std::lock_guard<std::mutex> lock(quarantine_mutex);
    for (auto &slot : quarantined) {
        if (slot == nullptr) {
            slot = payload;
            return;
        }
    }
    // Full bounded diagnostics table: leak the ENTIRE payload, including capture owners.
}
VulkanExecutionQuarantineSink production_quarantine{nullptr, retain_production_quarantine};

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
}

[[noreturn]] void throw_device_lost(VkResult result) {
    throw VulkanDeviceLost(
        "Vulkan device lost; execution state invalidated (VkResult " +
        std::to_string(static_cast<int>(result)) + ")");
}

std::exception_ptr
execute_callbacks(VulkanExecutionCallbacks &callbacks) {
    std::exception_ptr error;
    for (auto &callback : callbacks) {
        try {
            callback();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            if (!error)
                error = std::current_exception();
        }
    }
    callbacks.clear();
    return error;
}
} // namespace

VulkanCallbackDrainOnExit::~VulkanCallbackDrainOnExit() noexcept {
    context_.drain_ready_cleanup();
}

VulkanExecutionContext::VulkanExecutionContext(VkDevice device, VkQueue queue,
                                               VkCommandPool command_pool)
    : device_(device), queue_(queue), command_pool_(command_pool),
      ops_(real_execution_ops()), quarantine_sink_(&production_quarantine),
      reservation_state_(std::make_shared<VulkanTransferReservationState>()) {
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
        check_result(ops_.allocate_command_buffers(ops_.cookie, device_, &allocation, buffers),
                     "Could not allocate Vulkan execution command buffers");
        for (std::size_t i = 0; i < kRingSize; ++i)
            ring_[i].command_buffer = buffers[i];
        for (std::size_t i = 0; i < kRingSize; ++i) {
            VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check_result(ops_.create_fence(ops_.cookie, device_, &fence_info, nullptr, &ring_[i].fence),
                         "Could not create Vulkan execution fence");
            VkSemaphoreCreateInfo semaphore_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            check_result(ops_.create_semaphore(ops_.cookie, device_, &semaphore_info, nullptr,
                                           &ring_[i].signal_semaphore),
                         "Could not create Vulkan execution semaphore");
        }
    } catch (...) {
        for (auto &record : ring_) {
            if (record.fence != VK_NULL_HANDLE)
                ops_.destroy_fence(ops_.cookie, device_, record.fence, nullptr);
            if (record.signal_semaphore != VK_NULL_HANDLE)
                ops_.destroy_semaphore(ops_.cookie, device_, record.signal_semaphore, nullptr);
        }
        VkCommandBuffer allocated[kRingSize]{};
        std::size_t count = 0;
        for (const auto &record : ring_) {
            if (record.command_buffer != VK_NULL_HANDLE)
                allocated[count++] = record.command_buffer;
        }
        if (count)
            ops_.free_command_buffers(ops_.cookie, device_, command_pool_, count, allocated);
        throw;
    }
}

VulkanExecutionContext::VulkanExecutionContext(
    VkDevice device, VkQueue queue, VkCommandPool command_pool,
    const VulkanExecutionOps &ops, VulkanExecutionQuarantineSink &quarantine,
    std::mutex &shared_queue_mutex)
    : device_(device), queue_(queue), command_pool_(command_pool), ops_(ops),
      quarantine_sink_(&quarantine), queue_mutex_(&shared_queue_mutex),
      reservation_state_(std::make_shared<VulkanTransferReservationState>()) {
    if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
        command_pool_ == VK_NULL_HANDLE || !ops_.allocate_command_buffers ||
        !ops_.free_command_buffers || !ops_.create_fence || !ops_.destroy_fence ||
        !ops_.create_semaphore || !ops_.destroy_semaphore || !ops_.begin_command_buffer ||
        !ops_.end_command_buffer || !ops_.reset_command_buffer || !ops_.reset_fences ||
        !ops_.queue_submit || !ops_.get_fence_status || !ops_.wait_for_fences ||
        !ops_.queue_wait_idle || !quarantine.retain)
        throw std::invalid_argument("Invalid Vulkan execution test ops or handles");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = command_pool_;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = kRingSize;
    VkCommandBuffer buffers[kRingSize]{};
    check_result(ops_.allocate_command_buffers(ops_.cookie, device_, &allocation, buffers),
                 "Could not allocate Vulkan execution command buffers");
    for (std::size_t i = 0; i < kRingSize; ++i)
        ring_[i].command_buffer = buffers[i];
    try {
        for (std::size_t i = 0; i < kRingSize; ++i) {
            VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check_result(ops_.create_fence(ops_.cookie, device_, &fence_info, nullptr,
                                           &ring_[i].fence),
                         "Could not create Vulkan execution fence");
            VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            check_result(ops_.create_semaphore(ops_.cookie, device_, &semaphore_info,
                                               nullptr, &ring_[i].signal_semaphore),
                         "Could not create Vulkan execution semaphore");
        }
    } catch (...) {
        for (auto &record : ring_) {
            if (record.fence) ops_.destroy_fence(ops_.cookie, device_, record.fence, nullptr);
            if (record.signal_semaphore)
                ops_.destroy_semaphore(ops_.cookie, device_, record.signal_semaphore, nullptr);
        }
        ops_.free_command_buffers(ops_.cookie, device_, command_pool_, kRingSize, buffers);
        throw;
    }
}

VulkanExecutionContext::VulkanExecutionContext(VkDevice device, VkQueue queue,
                                               VkCommandPool command_pool,
                                               VulkanPlatform *platform)
    : VulkanExecutionContext(device, queue, command_pool) {
    platform_ = platform;
    if (platform_ != nullptr) {
        queue_mutex_ = &platform_->queue_mutex();
        ops_ = platform_->counting_execution_ops(ops_);
    }
    initialize_timestamp_queries();
}

VulkanExecutionContext::~VulkanExecutionContext() noexcept {
    // Empty-native terminal teardown already destroyed all handles and never
    // enters the cancellation/pump semantics of dedicated-context destruction.
    if (device_ == VK_NULL_HANDLE) return;
    try {
        if (!teardown_with_services({})) return;
    } catch (const __cxxabiv1::__forced_unwind&) {
        pytorch_vulkan::vulkan_terminal_fault("forced unwind through noexcept execution teardown");
    } catch (...) {
        // Never release possibly referenced owners on an uncertain teardown path.
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        if (!invalidated_) invalidate(VK_ERROR_UNKNOWN, lock);
        return;
    }
    std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    destroy_native_handles_locked();
}

void VulkanExecutionContext::destroy_native_handles_locked() {
    if (device_ == VK_NULL_HANDLE) return;
    for (auto &record : ring_) {
        if (record.fence != VK_NULL_HANDLE)
            ops_.destroy_fence(ops_.cookie, device_, record.fence, nullptr);
        if (record.signal_semaphore != VK_NULL_HANDLE)
            ops_.destroy_semaphore(ops_.cookie, device_, record.signal_semaphore, nullptr);
        if (record.command_buffer != VK_NULL_HANDLE)
            ops_.free_command_buffers(ops_.cookie, device_, command_pool_, 1,
                                       &record.command_buffer);
        record.fence = VK_NULL_HANDLE;
        record.signal_semaphore = VK_NULL_HANDLE;
        record.command_buffer = VK_NULL_HANDLE;
    }
    if (timestamp_query_pool_ != VK_NULL_HANDLE)
        vkDestroyQueryPool(device_, timestamp_query_pool_, nullptr);
    timestamp_query_pool_ = VK_NULL_HANDLE;
    latest_signal_semaphore_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

void VulkanExecutionContext::begin(const char *scope) {
    VulkanCallbackDrainOnExit drain(*this);
    std::unique_lock<std::mutex> queue_lock(*queue_mutex_);
    begin_locked(scope);
    queue_lock.unlock();
    drain_ready_callbacks(false);
}

VulkanTransferReservation VulkanExecutionContext::reserve_transfer(
    const std::function<void()> &flush_locked) {
    if (reservation_state_->process != ::getpid())
        throw std::logic_error("Vulkan reservations inherited across fork");
    if (callback_active_on_this_thread())
        throw std::logic_error("completion pump cannot reserve a transfer slot");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
            throw std::logic_error("completion callback cannot reserve a transfer slot");
    }
    for (;;) {
        {
            std::unique_lock<std::mutex> slots(reservation_state_->mutex);
            if (reservation_state_->closing || reservation_state_->lifecycle_closed || reservation_state_->poisoned)
                throw std::logic_error("Vulkan transfer reservations are unavailable");
            if (reservation_state_->occupied < 2) {
                if (ops_.fail_reservation_allocation &&
                    ops_.fail_reservation_allocation(ops_.cookie))
                    throw std::bad_alloc();
                auto lease = std::make_shared<VulkanTransferSlotLease>(reservation_state_);
                ++reservation_state_->occupied;
                return VulkanTransferReservation(std::move(lease));
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
                throw std::logic_error("completion callback cannot reserve a transfer slot");
            if (invalidated_ || closing_)
                throw std::logic_error("Vulkan transfer reservations are unavailable");
        }
        // Retire only the oldest submitted transfer; this boundary owns queue serialization.
        // No slot/context lock is held while taking the queue lock or pumping.
        // Count the actual capacity-pressure boundary, including oldest-fence
        // waits (not merely the later CV wait). No execution/reservation lock.
        if (ops_.reservation_wait_begin) ops_.reservation_wait_begin(ops_.cookie);
        wait_oldest_transfer(flush_locked);
        std::unique_lock<std::mutex> slots(reservation_state_->mutex);
        const uint64_t generation = reservation_state_->generation;
        const auto pending = [&] {
            return reservation_state_->occupied >= 2 && !reservation_state_->closing &&
                !reservation_state_->lifecycle_closed && !reservation_state_->poisoned &&
                reservation_state_->generation == generation;
        };
        // The reservation lock is the FULL lock set here. Re-evaluate capacity,
        // lifecycle and generation at the outer loop after restoring the GIL.
        pytorch_vulkan::vulkan_host_wait(slots, pending, [&] {
            reservation_state_->cv.wait(slots, [&] { return !pending(); });
        });
    }
}

void VulkanExecutionContext::notify_reservation_change() noexcept {
    std::lock_guard<std::mutex> lock(reservation_state_->mutex);
    ++reservation_state_->generation;
    reservation_state_->cv.notify_all();
}

void VulkanExecutionContext::wait_oldest_transfer(
    const std::function<void()> &flush_locked) {
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        try {
            // An open recording can hold slots that no fence will ever free; submit it first.
            bool flush = false;
            {
                std::lock_guard<std::mutex> state(mutex_);
                if (recording_)
                    for (const auto &transfer : transfer_records_)
                        flush |= transfer->state == TransferRecord::State::Recording;
            }
            if (flush) {
                if (flush_locked) flush_locked();
                else submit_locked();
            }
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            error = std::current_exception();
        }
        std::unique_lock<std::mutex> lock(mutex_);
        try {
            throw_if_invalidated(lock);
            std::shared_ptr<TransferRecord> oldest;
            for (const auto &transfer : transfer_records_) {
                if (transfer->state != TransferRecord::State::Pending &&
                    transfer->state != TransferRecord::State::Ready &&
                    transfer->state != TransferRecord::State::Failed &&
                    transfer->state != TransferRecord::State::Cancelled) continue;
                if (!oldest || transfer->submission_id < oldest->submission_id ||
                    (transfer->submission_id == oldest->submission_id &&
                     transfer->operation_id < oldest->operation_id))
                    oldest = transfer;
            }
            if (oldest && oldest->state == TransferRecord::State::Pending &&
                !callbacks_active_) {
                for (auto &record : ring_) {
                    if (record.submitted && record.submission_id == oldest->submission_id) {
                        wait_and_retire(record, lock);
                        break;
                    }
                }
            }
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            error = std::current_exception();
        }
    }
    try { drain_ready_callbacks(false); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
}

VulkanCompletionId VulkanExecutionContext::defer_transfer(
    VulkanTransferReservation &&reservation, std::vector<std::shared_ptr<void>> &&owners,
    std::function<void()> &&publish_success) {
    if (!reservation.lease_)
        throw std::invalid_argument("Vulkan transfer requires a live reservation");
    // Everything that can throw is done before the caller's objects are moved from.
    auto transfer = std::make_shared<TransferRecord>();
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (closing_ || !recording_)
        throw std::logic_error("Vulkan transfer attachment requires active recording");
    if (reservation.lease_->state != reservation_state_)
        throw std::invalid_argument("Vulkan transfer reservation belongs to another context");
    if (ops_.fail_transfer_attachment && ops_.fail_transfer_attachment(ops_.cookie))
        throw std::bad_alloc();
    transfer_records_.push_back(transfer);
    // Nothrow commit: moves of vectors/functions/shared_ptrs.
    transfer->operation_id = ++next_operation_id_;
    transfer->state = TransferRecord::State::Recording;
    transfer->owners = std::move(owners);
    transfer->publish_success = std::move(publish_success);
    transfer->lease = std::move(reservation.lease_);
    const auto operation_id = transfer->operation_id;
    lock.unlock();
    notify_reservation_change();
    return operation_id;
}

VulkanCompletionId VulkanExecutionContext::enqueue_transfer_locked(
    VulkanTransferReservation &&reservation, std::vector<std::shared_ptr<void>> &&owners,
    std::function<void()> &&publish_success,
    const std::function<void(VkCommandBuffer)> &record, bool submit_now, const char *scope) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
            throw std::logic_error("completion callback cannot enqueue a transfer");
    }
    if (!recording()) begin_locked(scope);
    // Keep attachment outside the catch: its strong guarantee leaves all
    // caller-owned captures intact on failure, to be destroyed unlocked.
    const auto id = defer_transfer(std::move(reservation), std::move(owners),
                                   std::move(publish_success));
    try {
        record(command_buffer());
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        const auto original = std::current_exception();
        cancel_locked(); // existing Cancelled state + persistent cancel error
        std::rethrow_exception(original);
    }
    if (submit_now) submit_locked(); // existing Failed/recovery/quarantine policy
    return id;
}

void VulkanExecutionContext::begin_locked(const char *scope) {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (closing_)
        throw std::logic_error("Vulkan execution context is closing");
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
    check_result(ops_.begin_command_buffer(ops_.cookie,
                                           ring_[active_slot_].command_buffer, &info),
                 "Could not begin Vulkan execution command buffer");
    ring_[active_slot_].timestamp_scope = scope == nullptr ? "operator" : scope;
    ring_[active_slot_].timestamp_recorded = false;
    if (timestamp_queries_supported_) {
        const uint32_t query_base = static_cast<uint32_t>(active_slot_ * kQueriesPerSlot);
        vkCmdResetQueryPool(ring_[active_slot_].command_buffer, timestamp_query_pool_,
                            query_base, kQueriesPerSlot);
        vkCmdWriteTimestamp(ring_[active_slot_].command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool_,
                            query_base);
        ring_[active_slot_].timestamp_begin = query_base;
        ring_[active_slot_].timestamp_end = query_base + 1;
        ring_[active_slot_].timestamp_recorded = true;
        ring_[active_slot_].timestamp_regions.clear();
        next_timestamp_region_ = 0;
        timestamp_region_active_ = false;
    }
    if (platform_)
        platform_->record_timing(
            VulkanTimingCategory::Allocation,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          allocation_start)
                .count());
    recording_ = true;
}

void VulkanExecutionContext::begin_timestamp_scope(const char *scope) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!recording_ || timestamp_region_active_)
        throw std::logic_error("Invalid Vulkan timestamp scope begin");
    if (!timestamp_queries_supported_) {
        timestamp_region_active_ = true;
        return;
    }
    if (next_timestamp_region_ >= 4)
        throw std::logic_error("Vulkan timestamp region capacity exceeded");
    const uint32_t query_base = static_cast<uint32_t>(active_slot_ * kQueriesPerSlot) +
                                2 + next_timestamp_region_ * 2;
    auto &record = ring_[active_slot_];
    record.timestamp_regions.push_back({query_base, query_base + 1,
                                        scope == nullptr ? "region" : scope});
    vkCmdWriteTimestamp(record.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        timestamp_query_pool_, query_base);
    timestamp_region_active_ = true;
}

bool VulkanExecutionContext::timestamp_region_capacity_exhausted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return timestamp_queries_supported_ && next_timestamp_region_ >= 4;
}

void VulkanExecutionContext::end_timestamp_scope() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!recording_ || !timestamp_region_active_)
        throw std::logic_error("Invalid Vulkan timestamp scope end");
    if (timestamp_queries_supported_) {
        auto &region = ring_[active_slot_].timestamp_regions.back();
        vkCmdWriteTimestamp(ring_[active_slot_].command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool_, region.end);
        ++next_timestamp_region_;
    }
    timestamp_region_active_ = false;
}

VkCommandBuffer VulkanExecutionContext::command_buffer() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!recording_)
        throw std::logic_error("Vulkan execution command buffer is not recording");
    return ring_[active_slot_].command_buffer;
}

void VulkanExecutionContext::submit() {
    VulkanCallbackDrainOnExit drain(*this);
    std::unique_lock<std::mutex> queue_lock(*queue_mutex_);
    submit_locked();
    queue_lock.unlock();
    drain_ready_callbacks(false);
}

void VulkanExecutionContext::submit_locked() {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (closing_)
        throw std::logic_error("Vulkan execution context is closing");
    if (!recording_)
        throw std::logic_error("Vulkan execution context is not recording");
    bool submit_attempted = false;
    try {
        const auto recording_start = std::chrono::steady_clock::now();
        if (ring_[active_slot_].timestamp_recorded)
            vkCmdWriteTimestamp(ring_[active_slot_].command_buffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                timestamp_query_pool_,
                                ring_[active_slot_].timestamp_end);
        check_result(ops_.end_command_buffer(ops_.cookie,
                                             ring_[active_slot_].command_buffer),
                     "Could not end Vulkan execution command buffer");
        if (platform_)
            platform_->record_timing(
                VulkanTimingCategory::Recording,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              recording_start)
                    .count());
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        const VkSemaphore wait_semaphore = latest_signal_semaphore_;
        const VkPipelineStageFlags wait_stage = vulkan_execution_wait_stages();
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
        submit_attempted = true;
        const VkResult submit_result =
            ops_.queue_submit(ops_.cookie, queue_, 1, &submit,
                              ring_[active_slot_].fence);
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
        ring_[active_slot_].callbacks.splice(ring_[active_slot_].callbacks.end(),
                                             deferred_callbacks_);
        ring_[active_slot_].submission_id = ++next_submission_id_;
        for (auto &transfer : transfer_records_)
            if (transfer->state == TransferRecord::State::Recording) {
                transfer->submission_id = ring_[active_slot_].submission_id;
                transfer->state = TransferRecord::State::Pending;
            }
        notify_reservation_change();
        ring_[active_slot_].submitted = true;
        latest_signal_semaphore_ = ring_[active_slot_].signal_semaphore;
        recording_ = false;
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        if (invalidated_)
            throw;
        const auto original = std::current_exception();
        bool has_recording_transfer = false;
        for (const auto &transfer : transfer_records_)
            has_recording_transfer |= transfer->state == TransferRecord::State::Recording;
        if (submit_attempted && has_recording_transfer) {
            const VkResult recovery = ops_.queue_wait_idle(ops_.cookie, queue_);
            if (recovery != VK_SUCCESS) {
                invalidate(recovery, lock);
                std::rethrow_exception(original);
            }
        }
        if (has_recording_transfer) {
            if (!callback_error_) callback_error_ = original;
            for (auto &transfer : transfer_records_)
                if (transfer->state == TransferRecord::State::Recording)
                    transfer->state = TransferRecord::State::Failed;
            notify_reservation_change();
        }
        abandon_recording(original, lock);
    }
}

void VulkanExecutionContext::wait() {
    synchronize_with_flush({});
}

void VulkanExecutionContext::wait_device_locked() {
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

bool VulkanExecutionContext::query_with_flush(const std::function<void()> &flush_locked) {
    if (callback_active_on_this_thread()) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (callbacks_active_) return false;
    }
    VulkanCallbackDrainOnExit drain(*this);
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        try {
            if (!closing_ && flush_locked) flush_locked();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) { error = std::current_exception(); }
        try {
            if (!recording()) poll_device_locked();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    try { drain_ready_callbacks(false); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        if (!error) error = std::current_exception();
    }
    if (error) std::rethrow_exception(error);
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (callback_error_) std::rethrow_exception(callback_error_);
    if (recording_ || callbacks_active_ || !ready_callbacks_.empty() ||
        !transfer_records_.empty()) return false;
    for (const auto &record : ring_)
        if (record.submitted) return false;
    return true;
}

void VulkanExecutionContext::synchronize_with_flush(const std::function<void()> &flush_locked) {
    if (callback_active_on_this_thread())
        throw std::logic_error("completion pump cannot wait for itself");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
            throw std::logic_error("completion callback cannot wait for itself");
    }
    VulkanCallbackDrainOnExit drain(*this);
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        try {
            if (!closing_ && flush_locked) flush_locked();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) { error = std::current_exception(); }
        try { wait_device_locked(); }
        catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    try { drain_ready_callbacks(true); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        if (!error) error = std::current_exception();
    }
    if (error) std::rethrow_exception(error);
}

pytorch_vulkan::VulkanQuiescence VulkanExecutionContext::snapshot_locked() const {
    pytorch_vulkan::VulkanQuiescence s;
    s.recording = recording_;
    for (const auto &record : ring_) {
        s.submitted_slots += record.submitted ? 1 : 0;
        s.ring_callbacks += record.callbacks.size();
    }
    s.deferred_callbacks = deferred_callbacks_.size();
    s.completion_callbacks = completion_callbacks_.size();
    s.ready_callbacks = ready_callbacks_.size();
    s.pump_active = callbacks_active_ || callback_epilogues_ != 0;
    s.pump_owner = callback_owner_;
    s.transfer_records = transfer_records_.size();
    s.reservations = reservation_state_->occupied;
    s.reservation_generation = reservation_state_->generation;
    s.reservation_closing = reservation_state_->closing || reservation_state_->lifecycle_closed;
    s.reservation_poisoned = reservation_state_->poisoned;
    s.invalidated = invalidated_;
    s.sticky_error = static_cast<bool>(callback_error_);
    return s;
}

pytorch_vulkan::VulkanQuiescence VulkanExecutionContext::lifecycle_snapshot() const {
    if (reservation_state_->process != ::getpid())
        throw std::logic_error("Vulkan execution inherited across fork");
    std::lock_guard<std::mutex> queue(*queue_mutex_);
    std::lock_guard<std::mutex> execution(mutex_);
    std::lock_guard<std::mutex> reservations(reservation_state_->mutex);
    return snapshot_locked(); // Value only; all three locks release before caller consults lifecycle.
}

void VulkanExecutionContext::set_lifecycle_reservations_closed(bool closed) {
    if (reservation_state_->process != ::getpid()) return;
    std::lock_guard<std::mutex> slots(reservation_state_->mutex);
    // Independent of destructor closing and poison: reopening never revives either.
    reservation_state_->lifecycle_closed = closed;
    ++reservation_state_->generation;
    reservation_state_->cv.notify_all();
}

bool VulkanExecutionContext::callback_active_on_this_thread() const noexcept {
    if (reservation_state_->process != ::getpid()) return false;
    return PumpIdentity::contains(this);
}

void VulkanExecutionContext::wait_callback_epilogue(std::unique_lock<std::mutex> &lock,
    pytorch_vulkan::VulkanHostTime deadline, const pytorch_vulkan::VulkanHostClock *clock) {
    if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
        throw std::logic_error("completion callback cannot wait for itself");
    if (callbacks_active_ && ops_.callback_wait_begin) ops_.callback_wait_begin(ops_.cookie);
    while (callbacks_active_) {
        // Called only by unlocked drain routes: no caller queue/reservation lock.
        pytorch_vulkan::vulkan_host_wait(lock, [&] { return callbacks_active_; }, [&] {
            if (!clock) callback_cv_.wait(lock);
            else {
                if (clock->now(clock->cookie) >= deadline)
                    throw pytorch_vulkan::VulkanHostCloseTimeout("Vulkan foreign callback epilogue deadline");
                clock->wait_until(clock->cookie, callback_cv_, lock, deadline);
            }
        });
    }
}

void VulkanExecutionContext::wait_foreign_epilogues_until(
    pytorch_vulkan::VulkanHostTime deadline, const pytorch_vulkan::VulkanHostClock &clock) {
    if (reservation_state_->process != ::getpid()) return;
    if (callback_active_on_this_thread())
        throw std::logic_error("completion pump cannot wait for itself");
    std::unique_lock<std::mutex> lock(mutex_);
    if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
        throw std::logic_error("completion callback cannot wait for itself");
    if ((callbacks_active_ || callback_epilogues_) && ops_.callback_wait_begin)
        ops_.callback_wait_begin(ops_.cookie);
    // Include the pump's final error read and local owner destruction, not just
    // invocation/capture destruction. No context member access follows its
    // counted epilogue's final mutex release.
    while (callbacks_active_ || callback_epilogues_) {
        if (clock.now(clock.cookie) >= deadline)
            throw pytorch_vulkan::VulkanHostCloseTimeout("Vulkan foreign pump epilogue deadline");
        pytorch_vulkan::vulkan_host_wait(lock,
            [&] { return callbacks_active_ || callback_epilogues_; },
            [&] { clock.wait_until(clock.cookie, callback_cv_, lock, deadline); });
    }
}

void VulkanExecutionContext::synchronize_with_flush_until(
    const std::function<void()> &flush_locked, pytorch_vulkan::VulkanHostTime deadline,
    const pytorch_vulkan::VulkanHostClock &clock) {
    if (reservation_state_->process != ::getpid())
        throw std::logic_error("Vulkan execution inherited across fork");
    if (callback_active_on_this_thread())
        throw std::logic_error("completion callback cannot wait for itself");
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        try { if (!closing_ && flush_locked) flush_locked(); }
        catch (const __cxxabiv1::__forced_unwind &) { throw; }
        catch (...) { error = std::current_exception(); }
        try { wait_device_locked(); }
        catch (const __cxxabiv1::__forced_unwind &) { throw; }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    try { drain_ready_callbacks(true, deadline, &clock); }
    catch (const __cxxabiv1::__forced_unwind &) { throw; }
    catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
}

void VulkanExecutionContext::teardown_empty_native_services(
    const std::function<void()> &destroy_services) {
    if (reservation_state_->process != ::getpid()) return;
    std::lock_guard<std::mutex> queue(*queue_mutex_);
    {
        std::lock_guard<std::mutex> execution(mutex_);
        std::lock_guard<std::mutex> reservations(reservation_state_->mutex);
        if (!snapshot_locked().healthy_empty())
            throw std::logic_error("native-only teardown requires healthy empty execution");
        closing_ = true;
        reservation_state_->closing = true;
        ++reservation_state_->generation;
        reservation_state_->cv.notify_all();
    }
    // Only audited native services, under queue serialization. Their retain calls
    // see no recording/submissions and destroy native payloads inline, never pump.
    if (destroy_services) destroy_services();
    std::lock_guard<std::mutex> execution(mutex_);
    std::lock_guard<std::mutex> reservations(reservation_state_->mutex);
    if (!snapshot_locked().healthy_empty())
        throw std::logic_error("native service cleanup changed execution quiescence");
    destroy_native_handles_locked();
}

bool VulkanExecutionContext::teardown_with_services(const std::function<void()> &destroy_services) {
    {
        std::lock_guard<std::mutex> slots(reservation_state_->mutex);
        reservation_state_->closing = true;
        ++reservation_state_->generation;
        reservation_state_->cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        closing_ = true;
        if (invalidated_) return false;
        const VkResult result = ops_.queue_wait_idle(ops_.cookie, queue_);
        if (result != VK_SUCCESS) {
            invalidate(result, lock);
            return false;
        }
        recording_ = false;
        for (auto &transfer : transfer_records_)
            if (transfer->state == TransferRecord::State::Recording)
                transfer->state = TransferRecord::State::Cancelled;
        notify_reservation_change();
        for (auto &record : ring_) {
            if (record.submitted) {
                resolve_timestamp(record);
                // Queue idle proves release safety only, never successful completion.
                retire(record, lock, false);
            }
        }
        ready_callbacks_.splice(ready_callbacks_.end(), deferred_callbacks_);
    }
    try { drain_ready_callbacks(true); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        // Safe cleanup continues; the callback exception is already persistent.
    }
    std::lock_guard<std::mutex> queue_lock(*queue_mutex_);
    if (invalidated()) return false;
    if (destroy_services) destroy_services();
    return true;
}

void VulkanExecutionContext::retire_completed() {
    VulkanCallbackDrainOnExit drain(*this);
    std::unique_lock<std::mutex> queue_lock(*queue_mutex_);
    poll_device_locked();
    queue_lock.unlock();
    drain_ready_callbacks(false);
}

void VulkanExecutionContext::poll_device_locked() {
    std::unique_lock<std::mutex> lock(mutex_);
    throw_if_invalidated(lock);
    if (recording_)
        throw std::logic_error("Cannot retire while Vulkan execution is recording");
    poll_completed(lock);
}

bool VulkanExecutionContext::query_complete() {
    return query_with_flush({});
}

void VulkanExecutionContext::poll_completed(std::unique_lock<std::mutex> &lock) {
    for (auto &record : ring_) {
        if (!record.submitted)
            continue;
        const VkResult status = ops_.get_fence_status(ops_.cookie, device_, record.fence);
        if (status == VK_ERROR_DEVICE_LOST) {
            invalidate(status, lock);
            throw_device_lost(status);
        }
        if (status == VK_SUCCESS) {
            if (platform_)
                platform_->record_compute_completed();
            resolve_timestamp(record);
            retire(record, lock);
            throw_if_invalidated(lock);
        } else if (status != VK_NOT_READY) {
            const VkResult recovery = ops_.queue_wait_idle(ops_.cookie, queue_);
            if (recovery != VK_SUCCESS) {
                invalidate(recovery, lock);
                check_result(recovery, "Could not confirm Vulkan execution completion");
            }
            if (!callback_error_) {
                try {
                    check_result(status, "Could not query Vulkan execution fence");
                } catch (...) {
                    callback_error_ = std::current_exception();
                }
            }
            for (auto &pending : ring_) {
                if (!pending.submitted) continue;
                resolve_timestamp(pending);
                retire(pending, lock, false);
            }
            std::rethrow_exception(callback_error_);
        }
    }
}

void VulkanExecutionContext::cancel() {
    std::unique_lock<std::mutex> queue_lock(*queue_mutex_);
    cancel_locked();
    queue_lock.unlock();
    drain_ready_callbacks(false);
}

void VulkanExecutionContext::cancel_locked() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (invalidated_) {
        recording_ = false;
        return;
    }
    if (!recording_)
        return;
    recording_ = false;
    bool has_recording_transfer = false;
    for (const auto &transfer : transfer_records_)
        has_recording_transfer |= transfer->state == TransferRecord::State::Recording;
    if (has_recording_transfer) {
        if (!callback_error_)
            callback_error_ = std::make_exception_ptr(
                std::runtime_error("Vulkan transfer recording was cancelled"));
        for (auto &transfer : transfer_records_)
            if (transfer->state == TransferRecord::State::Recording)
                transfer->state = TransferRecord::State::Cancelled;
        notify_reservation_change();
    }
    // R2 can retain owners also referenced by an earlier submitted R1. Cancelling
    // R2 proves only its own commands were discarded, not that R1 is complete.
    bool submitted = false;
    for (const auto &pending : ring_) submitted |= pending.submitted;
    if (submitted)
        completion_callbacks_.splice(completion_callbacks_.end(), deferred_callbacks_);
    else
        ready_callbacks_.splice(ready_callbacks_.end(), deferred_callbacks_);
    if (!ready_callbacks_.empty())
        callback_cv_.notify_all();
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
    count += ready_callbacks_.size();
    count += callbacks_active_ ? 1 : 0;
    count += transfer_records_.size();
    return count;
}

std::size_t VulkanExecutionContext::pending_transfer_count() const {
    std::scoped_lock lock(mutex_);
    return transfer_records_.size();
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
    return timestamp_queries_supported_ ? kRingSize * kQueriesPerSlot : 0;
}

std::size_t VulkanExecutionContext::timestamp_query_in_use() const {
    std::scoped_lock lock(mutex_);
    if (!timestamp_queries_supported_)
        return 0;
    std::size_t in_use = 0;
    for (const auto &record : ring_) {
        if (record.timestamp_recorded &&
            (record.submitted || (&record == &ring_[active_slot_] && recording_)))
            in_use += 2;
    }
    return in_use;
}

bool VulkanExecutionContext::timestamp_query_quarantined() const {
    std::scoped_lock lock(mutex_);
    return timestamp_query_quarantined_;
}

void VulkanExecutionContext::retire(InFlightRecord &record,
                                    std::unique_lock<std::mutex> &lock,
                                    bool allow_success_publication) {
    record.submitted = false;
    for (auto &transfer : transfer_records_)
        if (transfer->state == TransferRecord::State::Pending &&
            transfer->submission_id == record.submission_id)
            transfer->state = allow_success_publication ? TransferRecord::State::Ready
                                                        : TransferRecord::State::Failed;
    notify_reservation_change();
    bool submitted = false;
    for (const auto &pending : ring_)
        submitted |= pending.submitted;
    if (submitted) {
        completion_callbacks_.splice(completion_callbacks_.end(), record.callbacks);
        return;
    }
    ready_callbacks_.splice(ready_callbacks_.end(), record.callbacks);
    ready_callbacks_.splice(ready_callbacks_.end(), completion_callbacks_);
    (void)lock;
}

void VulkanExecutionContext::recreate_signal_semaphore(InFlightRecord &record) {
    if (record.signal_semaphore == VK_NULL_HANDLE)
        return;
    ops_.destroy_semaphore(ops_.cookie, device_, record.signal_semaphore, nullptr);
    record.signal_semaphore = VK_NULL_HANDLE;
    latest_signal_semaphore_ = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check_result(
        ops_.create_semaphore(ops_.cookie, device_, &semaphore_info, nullptr,
                              &record.signal_semaphore),
        "Could not recreate Vulkan execution semaphore");
}

void VulkanExecutionContext::reset_reusable_resources(InFlightRecord &record) {
    check_result(ops_.reset_command_buffer(ops_.cookie, record.command_buffer, 0),
                 "Could not reset Vulkan execution command buffer");
    check_result(ops_.reset_fences(ops_.cookie, device_, 1, &record.fence),
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
    info.queryCount = static_cast<uint32_t>(kRingSize * kQueriesPerSlot);
    if (vkCreateQueryPool(device_, &info, nullptr, &timestamp_query_pool_) !=
        VK_SUCCESS) {
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
        device_, timestamp_query_pool_, record.timestamp_begin, 2, sizeof(values),
        values, sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
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
    for (const auto &region : record.timestamp_regions) {
        uint64_t region_values[4]{};
        const VkResult region_result = vkGetQueryPoolResults(
            device_, timestamp_query_pool_, region.begin, 2, sizeof(region_values),
            region_values, sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        VulkanTimestampSample region_sample;
        region_sample.submission_id = record.submission_id;
        region_sample.scope = region.scope;
        region_sample.available = region_result == VK_SUCCESS &&
                                  region_values[1] != 0 && region_values[3] != 0;
        if (region_sample.available && region_values[2] >= region_values[0])
            region_sample.gpu_time_ns = static_cast<uint64_t>(
                static_cast<double>(region_values[2] - region_values[0]) *
                timestamp_period_);
        timestamp_samples_.push_back(std::move(region_sample));
    }
}

void VulkanExecutionContext::wait_and_retire(InFlightRecord &record,
                                             std::unique_lock<std::mutex> &lock) {
    const auto wait_start = std::chrono::steady_clock::now();
    const VkResult result =
        ops_.wait_for_fences(ops_.cookie, device_, 1, &record.fence, VK_TRUE,
                             UINT64_MAX);
    if (result == VK_ERROR_DEVICE_LOST) {
        invalidate(result, lock);
        throw_device_lost(result);
    }
    if (result != VK_SUCCESS) {
        const VkResult recovery_result = ops_.queue_wait_idle(ops_.cookie, queue_);
        if (recovery_result == VK_ERROR_DEVICE_LOST) {
            invalidate(recovery_result, lock);
            throw_device_lost(recovery_result);
        }
        if (recovery_result != VK_SUCCESS) {
            invalidate(recovery_result, lock);
            check_result(recovery_result, "Could not confirm Vulkan execution completion");
        }
        for (auto &pending : ring_) {
            if (!pending.submitted) continue;
            resolve_timestamp(pending);
            retire(pending, lock, false);
        }
        if (!callback_error_) {
            try {
                check_result(result, "Could not wait for Vulkan execution fence");
            } catch (...) {
                callback_error_ = std::current_exception();
            }
        }
        std::rethrow_exception(callback_error_);
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
    ready_callbacks_.splice(ready_callbacks_.end(), deferred_callbacks_);
    if (!invalidated_) {
        try {
            reset_reusable_resources(ring_[active_slot_]);
        } catch (...) {
        }
    }
    std::rethrow_exception(original);
}

void VulkanExecutionContext::drain_ready_callbacks(bool wait_for_foreign_pump,
    pytorch_vulkan::VulkanHostTime deadline, const pytorch_vulkan::VulkanHostClock *clock) {
    PumpIdentity identity(this);
    struct PumpEpilogue {
        std::mutex &mutex;
        std::condition_variable &cv;
        std::size_t &count;
        ~PumpEpilogue() {
            std::lock_guard<std::mutex> lock(mutex);
            --count;
            cv.notify_all();
        }
    };
    { std::lock_guard<std::mutex> lock(mutex_); ++callback_epilogues_; }
    PumpEpilogue epilogue{mutex_, callback_cv_, callback_epilogues_};
    std::exception_ptr error;
    try { drain_ready_transfers(wait_for_foreign_pump, deadline, clock); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        error = std::current_exception();
    }
    VulkanExecutionCallbacks callbacks;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (callbacks_active_) {
            if (callback_owner_ == std::this_thread::get_id()) {
                if (wait_for_foreign_pump)
                    throw std::logic_error("completion callback cannot wait for itself");
                return;
            }
            if (!wait_for_foreign_pump)
                return;
            wait_callback_epilogue(lock, deadline, clock);
        }
        if (!ready_callbacks_.empty()) {
            callbacks.swap(ready_callbacks_);
            callbacks_active_ = true;
            callback_owner_ = std::this_thread::get_id();
        }
    }
    if (!callbacks.empty()) {
        const std::exception_ptr error = execute_callbacks(callbacks);
        std::unique_lock<std::mutex> lock(mutex_);
        if (error && !callback_error_)
            callback_error_ = error;
        callbacks_active_ = false;
        callback_owner_ = {};
        callback_cv_.notify_all();
        notify_reservation_change();
    }
    {
        std::scoped_lock lock(mutex_);
        if (!error) error = callback_error_;
    }
    if (error)
        std::rethrow_exception(error);
}

void VulkanExecutionContext::drain_ready_transfers(bool wait_for_foreign_pump,
    pytorch_vulkan::VulkanHostTime deadline, const pytorch_vulkan::VulkanHostClock *clock) {
    for (;;) {
        std::shared_ptr<TransferRecord> transfer;
        bool publish = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (callbacks_active_) {
                bool ready_transfer = false;
                for (const auto &candidate : transfer_records_)
                    ready_transfer |= candidate->state == TransferRecord::State::Ready ||
                                      candidate->state == TransferRecord::State::Failed ||
                                      candidate->state == TransferRecord::State::Cancelled;
                if (!ready_transfer) return;
                if (callback_owner_ == std::this_thread::get_id()) {
                    if (wait_for_foreign_pump)
                        throw std::logic_error("completion callback cannot wait for itself");
                    return;
                }
                if (!wait_for_foreign_pump) return;
                wait_callback_epilogue(lock, deadline, clock);
            }
            std::shared_ptr<TransferRecord> earliest_submitted;
            for (const auto &candidate : transfer_records_) {
                if (candidate->submission_id == 0 ||
                    (candidate->state != TransferRecord::State::Ready &&
                     candidate->state != TransferRecord::State::Pending)) continue;
                if (!earliest_submitted ||
                    candidate->submission_id < earliest_submitted->submission_id ||
                    (candidate->submission_id == earliest_submitted->submission_id &&
                     candidate->operation_id < earliest_submitted->operation_id))
                    earliest_submitted = candidate;
            }
            if (earliest_submitted && earliest_submitted->state == TransferRecord::State::Ready)
                transfer = earliest_submitted;
            if (!transfer) {
                for (const auto &candidate : transfer_records_)
                    if ((candidate->state == TransferRecord::State::Failed ||
                         candidate->state == TransferRecord::State::Cancelled) &&
                        (!transfer || candidate->operation_id < transfer->operation_id))
                        transfer = candidate;
            }
            if (!transfer) break;
            if (transfer->state == TransferRecord::State::Ready) {
                transfer->state = TransferRecord::State::Publishing;
                publish = true;
            }
            callbacks_active_ = true;
            callback_owner_ = std::this_thread::get_id();
        }

        std::exception_ptr error;
        if (publish && transfer->publish_success) {
            try { transfer->publish_success(); }
            catch (const __cxxabiv1::__forced_unwind&) { throw; }
            catch (...) { error = std::current_exception(); }
        }
        bool quarantined = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            quarantined = transfer->state == TransferRecord::State::Quarantined;
            if (!quarantined) {
                if (error && !callback_error_) callback_error_ = error;
                if (publish)
                    transfer->state = error ? TransferRecord::State::Failed
                                            : TransferRecord::State::Published;
            }
        }
        if (!quarantined) {
            // Capture release is part of visible pump activity and happens unlocked.
            transfer->owners.clear();
            transfer->publish_success = {};
            transfer->lease.reset();
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!quarantined) transfer_records_.remove(transfer);
            callbacks_active_ = false;
            callback_owner_ = {};
            callback_cv_.notify_all();
            notify_reservation_change();
        }
    }
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error = callback_error_;
    }
    if (error) std::rethrow_exception(error);
}

void VulkanExecutionContext::drain_ready_cleanup() noexcept {
    try {
        drain_ready_callbacks(false);
    } catch (const __cxxabiv1::__forced_unwind&) {
        // A noexcept legacy ready-only destructor cannot propagate a forced
        // unwind. Never swallow it (GNU aborts); retain state and fail closed.
        pytorch_vulkan::vulkan_terminal_fault("forced unwind through noexcept ready cleanup");
    } catch (...) {
        // drain_ready_callbacks persists the first callback exception before throwing.
    }
}

std::unique_lock<std::mutex> VulkanExecutionContext::acquire_host_access(
    const std::function<void()> &flush_locked,
    const std::function<void()> &wait_idle_locked) {
    if (callback_active_on_this_thread())
        throw std::logic_error("completion pump cannot acquire host access");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (callbacks_active_ && callback_owner_ == std::this_thread::get_id())
            throw std::logic_error(
                "completion callback cannot acquire host access while publishing");
    }

    for (;;) {
        std::unique_lock<std::mutex> queue_lock(*queue_mutex_);
        try {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                throw_if_invalidated(lock);
                if (closing_)
                    throw std::logic_error("Vulkan execution context is closing");
            }
            if (flush_locked) flush_locked();
            if (wait_idle_locked) wait_idle_locked();
            // Poll, retire, reset and any error recovery all remain serialized
            // with every other use of the shared Vulkan queue/command pool.
            poll_device_locked();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            const std::exception_ptr original = std::current_exception();
            queue_lock.unlock();
            try { drain_ready_callbacks(true); }
            catch (const __cxxabiv1::__forced_unwind&) { throw; }
            catch (...) {}
            std::rethrow_exception(original);
        }

        queue_lock.unlock();
        // This waits for a foreign publisher and its final capture release.
        // Reentrant self-wait is rejected by drain_ready_callbacks(true).
        drain_ready_callbacks(true);

        queue_lock.lock();
        bool repeat = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            throw_if_invalidated(lock);
            if (closing_)
                throw std::logic_error("Vulkan execution context is closing");
            repeat = recording_ || callbacks_active_ || !ready_callbacks_.empty() ||
                     !deferred_callbacks_.empty() || !completion_callbacks_.empty();
            for (const auto &record : ring_)
                repeat = repeat || record.submitted;
            for (const auto &transfer : transfer_records_)
                repeat = repeat || transfer->state == TransferRecord::State::Ready ||
                         transfer->state == TransferRecord::State::Failed ||
                         transfer->state == TransferRecord::State::Cancelled ||
                         transfer->state == TransferRecord::State::Publishing;
        }
        if (!repeat)
            return queue_lock;
        queue_lock.unlock();
    }
}

void VulkanExecutionContext::throw_if_invalidated(std::unique_lock<std::mutex> &lock) {
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
    {
        std::lock_guard<std::mutex> slots(reservation_state_->mutex);
        reservation_state_->closing = true;
        reservation_state_->poisoned = true;
        ++reservation_state_->generation;
        reservation_state_->cv.notify_all();
    }
    if (platform_)
        platform_->mark_device_lost(result);

    auto &payload = *quarantine_payload_;
    for (std::size_t i = 0; i < kRingSize; ++i) {
        auto &record = ring_[i];
        payload.command_buffers[i] = record.command_buffer;
        payload.fences[i] = record.fence;
        payload.signal_semaphores[i] = record.signal_semaphore;
        payload.record_callbacks[i].splice(payload.record_callbacks[i].end(), record.callbacks);
        record.command_buffer = VK_NULL_HANDLE;
        record.fence = VK_NULL_HANDLE;
        record.signal_semaphore = VK_NULL_HANDLE;
        record.submitted = false;
    }
    payload.timestamp_query_pool = timestamp_query_pool_;
    payload.deferred_callbacks.splice(payload.deferred_callbacks.end(), deferred_callbacks_);
    payload.completion_callbacks.splice(payload.completion_callbacks.end(), completion_callbacks_);
    payload.ready_callbacks.splice(payload.ready_callbacks.end(), ready_callbacks_);
    for (auto it = transfer_records_.begin(); it != transfer_records_.end();) {
        auto current = it++;
        const auto state = (*current)->state;
        if (state != TransferRecord::State::Published &&
            state != TransferRecord::State::Failed &&
            state != TransferRecord::State::Cancelled) {
            (*current)->state = TransferRecord::State::Quarantined;
            payload.transfer_records.splice(payload.transfer_records.end(),
                                             transfer_records_, current);
        }
    }
    latest_signal_semaphore_ = VK_NULL_HANDLE;
    timestamp_query_pool_ = VK_NULL_HANDLE;
    timestamp_query_quarantined_ = true;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    command_pool_ = VK_NULL_HANDLE;
    quarantine_sink_->retain(quarantine_sink_->cookie, quarantine_payload_.release());
    callback_cv_.notify_all();
    (void)lock;
}
