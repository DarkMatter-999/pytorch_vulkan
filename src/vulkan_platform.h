#pragma once

#include <vulkan/vulkan.h>

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

struct VulkanDeviceInfo {
    std::string name;
    uint32_t compute_queue_family = 0;
};

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
    void wait_for_transfer() const;
    // Exposes the lifecycle invariant to native transfer tests only.
    std::size_t pending_transfer_count() const;
    std::size_t compute_dispatch_count() const;
    void copy_buffer_sync(VkBuffer source, VkBuffer destination,
                          VkDeviceSize size) const;
    void copy_buffer(VkBuffer source, VkBuffer destination, VkDeviceSize size) const;
    bool validation_enabled() const;

  private:
    struct PendingTransferResources {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    struct PendingComputeResources {
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    void cleanup() noexcept;
    void defer_compute_resources(VkDescriptorPool descriptor_pool,
                                 VkCommandBuffer command_buffer,
                                 VkFence fence) const noexcept;
    void reserve_compute_resources() const;
    friend class VulkanCompute;

    VkInstance instance_ = VK_NULL_HANDLE;
    mutable std::unique_ptr<VulkanCompute> compute_;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    VulkanDeviceInfo device_info_;
    uint32_t api_version_ = VK_API_VERSION_1_1;
    mutable std::vector<PendingTransferResources> pending_transfer_resources_;
    mutable std::vector<PendingComputeResources> pending_compute_resources_;
    mutable std::mutex queue_mutex_;
};

namespace pytorch_vulkan {

std::shared_ptr<VulkanPlatform> platform();

} // namespace pytorch_vulkan
