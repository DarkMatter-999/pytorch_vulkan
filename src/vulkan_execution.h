#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>
#include <utility>
#include <vector>

class VulkanPlatform;

struct VulkanExecutionOps {
    void *cookie = nullptr;
    VkResult (*allocate_command_buffers)(void *, VkDevice,
                                         const VkCommandBufferAllocateInfo *,
                                         VkCommandBuffer *) = nullptr;
    void (*free_command_buffers)(void *, VkDevice, VkCommandPool, uint32_t,
                                 const VkCommandBuffer *) = nullptr;
    VkResult (*create_fence)(void *, VkDevice, const VkFenceCreateInfo *,
                             const VkAllocationCallbacks *, VkFence *) = nullptr;
    void (*destroy_fence)(void *, VkDevice, VkFence,
                          const VkAllocationCallbacks *) = nullptr;
    VkResult (*create_semaphore)(void *, VkDevice, const VkSemaphoreCreateInfo *,
                                 const VkAllocationCallbacks *, VkSemaphore *) = nullptr;
    void (*destroy_semaphore)(void *, VkDevice, VkSemaphore,
                              const VkAllocationCallbacks *) = nullptr;
    VkResult (*begin_command_buffer)(void *, VkCommandBuffer,
                                     const VkCommandBufferBeginInfo *) = nullptr;
    VkResult (*end_command_buffer)(void *, VkCommandBuffer) = nullptr;
    VkResult (*reset_command_buffer)(void *, VkCommandBuffer, VkCommandBufferResetFlags) = nullptr;
    VkResult (*reset_fences)(void *, VkDevice, uint32_t, const VkFence *) = nullptr;
    VkResult (*queue_submit)(void *, VkQueue, uint32_t, const VkSubmitInfo *, VkFence) = nullptr;
    VkResult (*get_fence_status)(void *, VkDevice, VkFence) = nullptr;
    VkResult (*wait_for_fences)(void *, VkDevice, uint32_t, const VkFence *, VkBool32,
                                uint64_t) = nullptr;
    VkResult (*queue_wait_idle)(void *, VkQueue) = nullptr;
    // Optional observation of the actual callback-wait boundary; never calls back into execution.
    void (*callback_wait_begin)(void *) noexcept = nullptr;
    // Optional fixture observation immediately before capacity CV backpressure.
    void (*reservation_wait_begin)(void *) noexcept = nullptr;
    // Optional per-instance allocation-failure injection; production leaves null.
    bool (*fail_reservation_allocation)(void *) noexcept = nullptr;
    // Optional per-instance failure of the record-list node allocation in defer_transfer.
    bool (*fail_transfer_attachment)(void *) noexcept = nullptr;
};

// Nodes are allocated on callback admission, never on retirement/cancel/quarantine handoff.
using VulkanExecutionCallbacks = std::list<std::function<void()>>;
using VulkanCompletionId = uint64_t;
struct VulkanTransferSlotLease;
struct VulkanTransferReservationState;

struct VulkanTransferRecord {
    enum class State { Reserved, Recording, Pending, Ready, Publishing,
                       Published, Cancelled, Failed, Quarantined };
    VulkanCompletionId operation_id = 0;
    uint64_t submission_id = 0;
    State state = State::Reserved;
    std::vector<std::shared_ptr<void>> owners;
    std::function<void()> publish_success;
    std::shared_ptr<VulkanTransferSlotLease> lease;
};

class VulkanTransferReservation final {
  public:
    VulkanTransferReservation() = default;
    ~VulkanTransferReservation() = default;
    VulkanTransferReservation(VulkanTransferReservation &&) noexcept = default;
    VulkanTransferReservation &operator=(VulkanTransferReservation &&) noexcept = default;
    VulkanTransferReservation(const VulkanTransferReservation &) = delete;
    VulkanTransferReservation &operator=(const VulkanTransferReservation &) = delete;
    explicit operator bool() const noexcept { return static_cast<bool>(lease_); }
  private:
    friend class VulkanExecutionContext;
    explicit VulkanTransferReservation(std::shared_ptr<VulkanTransferSlotLease> lease)
        : lease_(std::move(lease)) {}
    std::shared_ptr<VulkanTransferSlotLease> lease_;
};

struct VulkanExecutionQuarantine {
    std::array<VkCommandBuffer, 2> command_buffers{};
    std::array<VkFence, 2> fences{};
    std::array<VkSemaphore, 2> signal_semaphores{};
    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    std::array<VulkanExecutionCallbacks, 2> record_callbacks;
    VulkanExecutionCallbacks deferred_callbacks;
    VulkanExecutionCallbacks completion_callbacks;
    VulkanExecutionCallbacks ready_callbacks;
    std::list<std::shared_ptr<VulkanTransferRecord>> transfer_records;
};

struct VulkanExecutionQuarantineSink {
    void *cookie = nullptr;
    // Consumes the preallocated payload. Must retain/leak it, not destroy its captures here.
    void (*retain)(void *, VulkanExecutionQuarantine *) noexcept = nullptr;
};

struct VulkanTimestampSample {
    uint64_t gpu_time_ns = 0;
    bool available = false;
    uint64_t submission_id = 0;
    std::string scope;
};

class VulkanExecutionContext final {
  public:
    VulkanExecutionContext(VkDevice device, VkQueue queue, VkCommandPool command_pool);
    ~VulkanExecutionContext() noexcept;

    VulkanExecutionContext(const VulkanExecutionContext &) = delete;
    VulkanExecutionContext &operator=(const VulkanExecutionContext &) = delete;

    void begin(const char *scope = "operator");
    // Call with no queue/transfer/execution lock held. Pressure may flush an open
    // recording that holds transfer slots (flush_locked, or submit_locked when empty;
    // platform callers pass the compute flush), synchronize and wait.
    VulkanTransferReservation reserve_transfer(const std::function<void()> &flush_locked = {});
    // Caller holds the shared queue lock; attach while recording before owner-referencing
    // commands. Strong guarantee: reservation, owners and action are moved from only on
    // success, so a throw leaves them with the caller to destroy outside the queue lock.
    VulkanCompletionId defer_transfer(VulkanTransferReservation &&reservation,
                                      std::vector<std::shared_ptr<void>> &&owners,
                                      std::function<void()> &&publish_success = {});
    // Caller holds queue mutex; declare VulkanCallbackDrainOnExit before that
    // guard. Attach strongly BEFORE recording owner-referencing commands. On
    // failure, cancel/submit persist their existing error and pump owners only
    // after the outer queue guard is released. Cache maintenance belongs to the
    // platform recorder/publication, not this execution-layer transaction.
    VulkanCompletionId enqueue_transfer_locked(
        VulkanTransferReservation &&reservation,
        std::vector<std::shared_ptr<void>> &&owners,
        std::function<void()> &&publish_success,
        const std::function<void(VkCommandBuffer)> &record,
        bool submit_now, const char *scope = "transfer");
    void begin_timestamp_scope(const char *scope);
    void end_timestamp_scope();
    VkCommandBuffer command_buffer() const;
    void submit();
    void wait();
    void synchronize();
    void retire_completed();
    bool query_complete();
    void cancel();
    bool recording() const;
    bool invalidated() const;
    void defer_destruction(std::function<void()> callback);
    // Retains a callback through all currently submitted work; false means idle.
    bool retain_until_completion(std::function<void()> callback);
    void retain(VkDescriptorPool descriptor_pool);
    std::size_t pending_count() const;
    // Host records only (including publication/quarantine), not compute/ring work.
    std::size_t pending_transfer_count() const;
    bool timestamp_queries_supported() const;
    bool timestamp_region_capacity_exhausted() const;
    std::string timestamp_query_support_reason() const;
    std::vector<VulkanTimestampSample> timestamp_samples() const;
    void reset_timestamp_samples();
    std::size_t timestamp_query_capacity() const;
    std::size_t timestamp_query_in_use() const;
    bool timestamp_query_quarantined() const;

    explicit VulkanExecutionContext(VkDevice device, VkQueue queue,
                                    VkCommandPool command_pool,
                                    VulkanPlatform *platform);
    VulkanExecutionContext(VkDevice device, VkQueue queue, VkCommandPool command_pool,
                           const VulkanExecutionOps &ops,
                           VulkanExecutionQuarantineSink &quarantine,
                           std::mutex &shared_queue_mutex);

    // Device/lifecycle phase; caller must own the shared queue mutex. These
    // enqueue cleanup only and never execute or destroy callback captures.
    void begin_locked(const char *scope = "operator");
    void submit_locked();
    void cancel_locked();
    void wait_device_locked();
    void poll_device_locked();
    // Ready-only cleanup: no device polling, safe even while another batch records.
    void drain_ready_cleanup() noexcept;
    // Shared host-map boundary used by the platform and fake fixtures. The
    // callbacks run under the shared queue lock. Returns owning that same lock
    // only after a queue-held completion cutoff, unlocked blocking publication
    // drain, and an atomic queue→execution final check with no intervening work.
    std::unique_lock<std::mutex> acquire_host_access(
        const std::function<void()> &flush_locked,
        const std::function<void()> &wait_idle_locked);
    // Shared production orchestration. flush_locked runs within the single queue phase.
    bool query_with_flush(const std::function<void()> &flush_locked);
    void synchronize_with_flush(const std::function<void()> &flush_locked);
    bool teardown_with_services(const std::function<void()> &destroy_services);

  private:
    struct InFlightRecord {
        struct TimestampRegion {
            uint32_t begin = 0;
            uint32_t end = 0;
            std::string scope;
        };
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore signal_semaphore = VK_NULL_HANDLE;
        bool submitted = false;
        uint64_t submission_id = 0;
        uint32_t timestamp_begin = 0;
        uint32_t timestamp_end = 0;
        bool timestamp_recorded = false;
        std::string timestamp_scope;
        std::vector<TimestampRegion> timestamp_regions;
        VulkanExecutionCallbacks callbacks;
    };

    using TransferRecord = VulkanTransferRecord;

    void retire(InFlightRecord &record, std::unique_lock<std::mutex> &lock,
                bool allow_success_publication = true);
    void recreate_signal_semaphore(InFlightRecord &record);
    void reset_reusable_resources(InFlightRecord &record);
    void wait_and_retire(InFlightRecord &record, std::unique_lock<std::mutex> &lock);
    void poll_completed(std::unique_lock<std::mutex> &lock);
    void drain_ready_callbacks(bool wait);
    void drain_ready_transfers(bool wait);
    void wait_oldest_transfer(const std::function<void()> &flush_locked);
    void notify_reservation_change() noexcept;
    [[noreturn]] void abandon_recording(std::exception_ptr original,
                                        std::unique_lock<std::mutex> &lock);
    void throw_if_invalidated(std::unique_lock<std::mutex> &lock);
    void invalidate(VkResult result, std::unique_lock<std::mutex> &lock);
    void initialize_timestamp_queries();
    void resolve_timestamp(InFlightRecord &record);

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VulkanPlatform *platform_ = nullptr;
    VulkanExecutionOps ops_{};
    VulkanExecutionQuarantineSink *quarantine_sink_ = nullptr;
    std::unique_ptr<VulkanExecutionQuarantine> quarantine_payload_ =
        std::make_unique<VulkanExecutionQuarantine>();
    mutable std::mutex owned_queue_mutex_;
    std::mutex *queue_mutex_ = &owned_queue_mutex_;
    static constexpr std::size_t kRingSize = 2;
    std::array<InFlightRecord, kRingSize> ring_{};
    VkSemaphore latest_signal_semaphore_ = VK_NULL_HANDLE;
    std::size_t active_slot_ = 0;
    bool recording_ = false;
    bool closing_ = false;
    bool invalidated_ = false;
    VkResult invalidation_result_ = VK_SUCCESS;
    VulkanExecutionCallbacks deferred_callbacks_;
    VulkanExecutionCallbacks completion_callbacks_;
    VulkanExecutionCallbacks ready_callbacks_;
    std::condition_variable callback_cv_;
    bool callbacks_active_ = false;
    std::thread::id callback_owner_{};
    std::exception_ptr callback_error_;
    std::list<std::shared_ptr<TransferRecord>> transfer_records_;
    std::shared_ptr<VulkanTransferReservationState> reservation_state_;
    VulkanCompletionId next_operation_id_ = 0;
    VkQueryPool timestamp_query_pool_ = VK_NULL_HANDLE;
    bool timestamp_queries_supported_ = false;
    bool timestamp_query_quarantined_ = false;
    float timestamp_period_ = 0.0F;
    std::string timestamp_query_support_reason_ = "timestamp queries are unavailable";
    uint64_t next_submission_id_ = 0;
    std::vector<VulkanTimestampSample> timestamp_samples_;
    uint32_t next_timestamp_region_ = 0;
    bool timestamp_region_active_ = false;
    mutable std::mutex mutex_;
};

// Outer operation scope: declare before its queue/transfer guards.
class VulkanCallbackDrainOnExit final {
  public:
    explicit VulkanCallbackDrainOnExit(VulkanExecutionContext &context) : context_(context) {}
    ~VulkanCallbackDrainOnExit() noexcept;
  private:
    VulkanExecutionContext &context_;
};
