#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

class VulkanPlatform;

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
    VkCommandBuffer command_buffer() const;
    void submit();
    void wait();
    void synchronize();
    void retire_completed();
    void cancel();
    bool recording() const;
    bool invalidated() const;
    void defer_destruction(std::function<void()> callback);
    // Retains a callback through all currently submitted work; false means idle.
    bool retain_until_completion(std::function<void()> callback);
    void retain(VkDescriptorPool descriptor_pool);
    std::size_t pending_count() const;
    bool timestamp_queries_supported() const;
    std::string timestamp_query_support_reason() const;
    std::vector<VulkanTimestampSample> timestamp_samples() const;
    void reset_timestamp_samples();
    std::size_t timestamp_query_capacity() const;
    std::size_t timestamp_query_in_use() const;
    bool timestamp_query_quarantined() const;

    explicit VulkanExecutionContext(VkDevice device, VkQueue queue,
                                    VkCommandPool command_pool,
                                    VulkanPlatform *platform);

  private:
    struct InFlightRecord {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore signal_semaphore = VK_NULL_HANDLE;
        bool submitted = false;
        uint64_t submission_id = 0;
        uint32_t timestamp_begin = 0;
        uint32_t timestamp_end = 0;
        bool timestamp_recorded = false;
        std::string timestamp_scope;
        std::vector<std::function<void()>> callbacks;
    };

    void retire(InFlightRecord &record, std::unique_lock<std::mutex> &lock);
    void recreate_signal_semaphore(InFlightRecord &record);
    void reset_reusable_resources(InFlightRecord &record);
    void wait_and_retire(InFlightRecord &record, std::unique_lock<std::mutex> &lock);
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
    static constexpr std::size_t kRingSize = 2;
    std::array<InFlightRecord, kRingSize> ring_{};
    VkSemaphore latest_signal_semaphore_ = VK_NULL_HANDLE;
    std::size_t active_slot_ = 0;
    bool recording_ = false;
    bool invalidated_ = false;
    VkResult invalidation_result_ = VK_SUCCESS;
    std::vector<std::function<void()>> deferred_callbacks_;
    std::vector<std::function<void()>> completion_callbacks_;
    VkQueryPool timestamp_query_pool_ = VK_NULL_HANDLE;
    bool timestamp_queries_supported_ = false;
    bool timestamp_query_quarantined_ = false;
    float timestamp_period_ = 0.0F;
    std::string timestamp_query_support_reason_ = "timestamp queries are unavailable";
    uint64_t next_submission_id_ = 0;
    std::vector<VulkanTimestampSample> timestamp_samples_;
    mutable std::mutex mutex_;
};
