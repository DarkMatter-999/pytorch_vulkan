#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <exception>
#include <functional>
#include <mutex>
#include <vector>

class VulkanPlatform;

class VulkanExecutionContext final {
  public:
    VulkanExecutionContext(VkDevice device, VkQueue queue, VkCommandPool command_pool);
    ~VulkanExecutionContext() noexcept;

    VulkanExecutionContext(const VulkanExecutionContext &) = delete;
    VulkanExecutionContext &operator=(const VulkanExecutionContext &) = delete;

    void begin();
    VkCommandBuffer command_buffer() const;
    void submit();
    void wait();
    void synchronize();
    void retire_completed();
    void cancel();
    bool recording() const;
    void defer_destruction(std::function<void()> callback);
    void retain(VkDescriptorPool descriptor_pool);
    std::size_t pending_count() const;

    explicit VulkanExecutionContext(VkDevice device, VkQueue queue,
                                    VkCommandPool command_pool,
                                    VulkanPlatform *platform);

  private:
    struct InFlightRecord {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore signal_semaphore = VK_NULL_HANDLE;
        bool submitted = false;
        std::vector<std::function<void()>> callbacks;
    };

    void retire(InFlightRecord &record);
    void recreate_signal_semaphore(InFlightRecord &record);
    void reset_reusable_resources(InFlightRecord &record);
    void wait_and_retire(InFlightRecord &record);
    [[noreturn]] void abandon_recording(std::exception_ptr original);

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VulkanPlatform *platform_ = nullptr;
    static constexpr std::size_t kRingSize = 2;
    std::array<InFlightRecord, kRingSize> ring_{};
    VkSemaphore latest_signal_semaphore_ = VK_NULL_HANDLE;
    std::size_t active_slot_ = 0;
    bool recording_ = false;
    std::vector<std::function<void()>> deferred_callbacks_;
    mutable std::mutex mutex_;
};
