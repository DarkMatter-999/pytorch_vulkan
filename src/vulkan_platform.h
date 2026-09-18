#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

class VulkanUnavailable : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class VulkanCompute;
class VulkanExecutionContext;
class VulkanBuffer;

struct VulkanDeviceInfo {
    std::string name;
    uint32_t compute_queue_family = 0;
    bool required_capabilities = false;
    bool storage_dispatch_limits = false;
    bool host_visible_memory = false;
    bool shader_capabilities = false;
};

struct VulkanExecutionCounterSnapshot {
    std::size_t dispatches = 0;
    std::size_t vulkan_copies = 0;
    std::size_t explicit_transfers = 0;
    std::size_t fallbacks = 0;
};

struct VulkanTimingSnapshot {
    double allocation = 0.0;
    double recording = 0.0;
    double submit_wait = 0.0;
    double compute = 0.0;
    double total = 0.0;
};

struct VulkanPendingTransferResources {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

enum class VulkanTimingCategory { Allocation, Recording, SubmitWait, Compute };

class VulkanPlatform {
  public:
    explicit VulkanPlatform(bool enable_validation = false);
    ~VulkanPlatform() noexcept;

    static bool is_available() noexcept;

    VulkanPlatform(const VulkanPlatform &) = delete;
    VulkanPlatform &operator=(const VulkanPlatform &) = delete;

    const VulkanDeviceInfo &device_info() const;
    uint32_t api_version() const;
    VkPhysicalDevice physical_device() const;
    VkDevice device() const;
    VkQueue compute_queue() const;
    VkCommandPool command_pool() const;
    VulkanCompute &compute() const;
    // Serializes every use of the shared compute queue and command pool.
    std::mutex &queue_mutex() const;
    // Serializes synchronous transfers that may reuse the shared staging buffer.
    std::mutex &transfer_mutex() const;
    void wait_for_transfer() const;
    // Exposes lifecycle invariants to native tests.
    std::size_t pending_transfer_count() const;
    std::size_t pending_compute_count() const;
    std::size_t compute_dispatch_count() const;
    std::size_t compute_submitted_count() const;
    std::size_t compute_completed_count() const;
    std::size_t compute_wait_count() const;
    void record_compute_submitted() const;
    void record_compute_completed() const;
    void record_compute_wait() const;
    VulkanExecutionCounterSnapshot execution_counter_snapshot() const;
    void reset_execution_counters() const;
    std::size_t explicit_transfer_count() const;
    void record_explicit_transfer() const;
    // Records an implicit CPU fallback, rejecting it when strict mode is on.
    void record_fallback() const;
    void set_strict_mode(bool enabled) const;
    bool strict_mode() const;
    std::size_t vulkan_copy_count() const;
    void record_vulkan_copy() const;
    // Counts recorded vkCmdCopyBuffer commands, distinct from logical transfers.
    std::size_t copy_command_count() const;
    void record_copy_command() const;
    void copy_buffer_sync(VkBuffer source, VkBuffer destination, VkDeviceSize size,
                          VkDeviceSize source_offset = 0,
                          VkDeviceSize destination_offset = 0) const;
    void fill_buffer_sync(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                          uint32_t data) const;
    void copy_buffer(VkBuffer source, VkBuffer destination, VkDeviceSize size) const;
    VulkanBuffer &staging_buffer(VkDeviceSize size) const;
    VulkanTimingSnapshot timing_snapshot() const;
    void reset_timing() const;
    void record_timing(VulkanTimingCategory category, double seconds) const;
    bool validation_enabled() const;
    bool supports_bool_pointwise() const;
    bool supports_formatter_double() const;
    VulkanExecutionContext &execution_context() const;

  private:
    using PendingTransferResources = VulkanPendingTransferResources;

    void cleanup() noexcept;
    friend class VulkanCompute;

    VkInstance instance_ = VK_NULL_HANDLE;
    mutable std::unique_ptr<VulkanCompute> compute_;
    mutable std::unique_ptr<VulkanExecutionContext> execution_;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    bool bool_pointwise_supported_ = false;
    bool formatter_double_supported_ = false;
    VulkanDeviceInfo device_info_;
    uint32_t api_version_ = VK_API_VERSION_1_1;
    mutable std::vector<PendingTransferResources> pending_transfer_resources_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex transfer_mutex_;
    mutable std::atomic<std::size_t> explicit_transfer_count_{0};
    mutable std::atomic<std::size_t> vulkan_copy_count_{0};
    mutable std::atomic<std::size_t> fallback_count_{0};
    mutable std::atomic<bool> strict_mode_{false};
    mutable std::atomic<std::size_t> copy_command_count_{0};
    mutable std::atomic<std::size_t> compute_submitted_count_{0};
    mutable std::atomic<std::size_t> compute_completed_count_{0};
    mutable std::atomic<std::size_t> compute_wait_count_{0};
    mutable std::unique_ptr<VulkanBuffer> staging_buffer_;
    mutable VulkanTimingSnapshot timing_;
};

namespace pytorch_vulkan {

namespace testing {
bool candidate_selection_falls_back() noexcept;
}

// A forked child cannot safely reuse Vulkan objects created by its parent.
bool inherited_fork_state() noexcept;
void register_fork_state_handler();
void ensure_process_local_vulkan();

std::shared_ptr<VulkanPlatform> platform();

} // namespace pytorch_vulkan
