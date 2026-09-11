#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

class VulkanUnavailable : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct VulkanDeviceInfo {
    std::string name;
    uint32_t compute_queue_family = 0;
};

class VulkanPlatform {
  public:
    explicit VulkanPlatform(bool enable_validation = false);
    ~VulkanPlatform();

    static bool is_available() noexcept;

    VulkanPlatform(const VulkanPlatform &) = delete;
    VulkanPlatform &operator=(const VulkanPlatform &) = delete;

    const VulkanDeviceInfo &device_info() const;
    uint32_t api_version() const;
    VkPhysicalDevice physical_device() const;
    VkDevice device() const;
    VkQueue compute_queue() const;
    VkCommandPool command_pool() const;
    void copy_buffer(VkBuffer source, VkBuffer destination, VkDeviceSize size) const;
    bool validation_enabled() const;

  private:
    void cleanup();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    VulkanDeviceInfo device_info_;
    uint32_t api_version_ = VK_API_VERSION_1_1;
};

namespace pytorch_vulkan {

std::shared_ptr<VulkanPlatform> platform();

} // namespace pytorch_vulkan
