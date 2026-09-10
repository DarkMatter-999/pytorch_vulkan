#include "vulkan_platform.h"

#include <stdexcept>
#include <vector>

namespace {

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(operation);
    }
}

} // namespace

VulkanPlatform::VulkanPlatform() {
    uint32_t loader_version = VK_API_VERSION_1_0;
    const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerate_instance_version == nullptr ||
        enumerate_instance_version(&loader_version) != VK_SUCCESS ||
        VK_VERSION_MAJOR(loader_version) != 1 || VK_VERSION_MINOR(loader_version) < 1) {
        throw std::runtime_error("Vulkan 1.1 loader support is required");
    }

    VkApplicationInfo application_info{};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "pytorch_vulkan";
    application_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.pEngineName = "pytorch_vulkan";
    application_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.apiVersion = api_version_;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;
    check_result(vkCreateInstance(&instance_info, nullptr, &instance_),
                 "Could not create Vulkan instance");

    uint32_t device_count = 0;
    check_result(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
                 "Could not enumerate Vulkan physical devices");
    if (device_count == 0) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        throw std::runtime_error("No Vulkan physical device is available");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    check_result(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
                 "Could not enumerate Vulkan physical devices");

    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                                 queue_families.data());

        for (uint32_t family = 0; family < queue_family_count; ++family) {
            if ((queue_families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                device_info_.name = properties.deviceName;
                device_info_.compute_queue_family = family;
                float queue_priority = 1.0F;
                VkDeviceQueueCreateInfo queue_info{};
                queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queue_info.queueFamilyIndex = family;
                queue_info.queueCount = 1;
                queue_info.pQueuePriorities = &queue_priority;

                VkDeviceCreateInfo device_info{};
                device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                device_info.queueCreateInfoCount = 1;
                device_info.pQueueCreateInfos = &queue_info;
                check_result(vkCreateDevice(device, &device_info, nullptr, &device_),
                             "Could not create Vulkan logical device");
                physical_device_ = device;
                vkGetDeviceQueue(device_, family, 0, &compute_queue_);
                VkCommandPoolCreateInfo command_pool_info{};
                command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                command_pool_info.flags =
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                command_pool_info.queueFamilyIndex = family;
                if (vkCreateCommandPool(device_, &command_pool_info, nullptr,
                                        &command_pool_) != VK_SUCCESS) {
                    vkDestroyDevice(device_, nullptr);
                    device_ = VK_NULL_HANDLE;
                    vkDestroyInstance(instance_, nullptr);
                    instance_ = VK_NULL_HANDLE;
                    throw std::runtime_error("Could not create Vulkan command pool");
                }
                return;
            }
        }
    }

    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    throw std::runtime_error("No Vulkan physical device has a compute queue");
}

VulkanPlatform::~VulkanPlatform() {
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

const VulkanDeviceInfo &VulkanPlatform::device_info() const { return device_info_; }

uint32_t VulkanPlatform::api_version() const { return api_version_; }

VkPhysicalDevice VulkanPlatform::physical_device() const { return physical_device_; }

VkDevice VulkanPlatform::device() const { return device_; }

VkQueue VulkanPlatform::compute_queue() const { return compute_queue_; }

VkCommandPool VulkanPlatform::command_pool() const { return command_pool_; }

void VulkanPlatform::copy_buffer(VkBuffer source, VkBuffer destination,
                                 VkDeviceSize size) const {
    if (source == VK_NULL_HANDLE || destination == VK_NULL_HANDLE || size == 0) {
        throw std::invalid_argument("Invalid Vulkan buffer copy arguments");
    }

    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_result(vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer),
                 "Could not allocate Vulkan command buffer");

    try {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(command_buffer, &begin_info),
                     "Could not begin Vulkan command buffer");

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, source, destination, 1, &copy_region);
        check_result(vkEndCommandBuffer(command_buffer),
                     "Could not end Vulkan command buffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        check_result(vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE),
                     "Could not submit Vulkan command buffer");
        check_result(vkQueueWaitIdle(compute_queue_),
                     "Could not synchronize Vulkan compute queue");
    } catch (...) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw;
    }

    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}
