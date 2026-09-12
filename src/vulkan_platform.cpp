#include "vulkan_platform.h"

#include "vulkan_compute.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";

// If device-idle cannot be established, the Vulkan handles must remain owned
// by a live object.  These records intentionally retain their handles for the
// remainder of the process rather than destroying resources whose completion
// state is unknown.
struct QuarantinedVulkanResources {
    std::unique_ptr<VulkanCompute> compute;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

    // Completion is unknown, so retain the compute object's Vulkan handles
    // and parent device instead of invoking destruction during teardown. The
    // pending command/fence/pool handle values remain live under this device;
    // their vector metadata does not own or release Vulkan handles.
    // The slot intentionally never destroys compute: completion is unknown.
    ~QuarantinedVulkanResources() { compute.release(); }
};

// Quarantine bookkeeping must not allocate while cleanup is noexcept.  A
// bounded process-lifetime table is sufficient for platform instances; if it
// is exhausted, cleanup deliberately leaks the handles rather than risking a
// destruction of work whose completion is unknown.
constexpr std::size_t kQuarantineCapacity = 16;
std::mutex quarantine_mutex;
std::array<QuarantinedVulkanResources, kQuarantineCapacity> quarantined_resources;
std::size_t quarantined_resource_count = 0;

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) {
    if (callback_data != nullptr && callback_data->pMessage != nullptr) {
        std::cerr << "Vulkan validation: " << callback_data->pMessage << "\n";
    }
    return VK_FALSE;
}

bool has_validation_layer() {
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const VkLayerProperties &layer : layers) {
        if (std::string(layer.layerName) == kValidationLayer) {
            return true;
        }
    }
    return false;
}

bool has_debug_utils_extension() {
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    for (const VkExtensionProperties &extension : extensions) {
        if (std::string(extension.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) {
            return true;
        }
    }
    return false;
}

} // namespace

VulkanPlatform::VulkanPlatform(bool enable_validation)
    : validation_enabled_(enable_validation) {
    try {
        uint32_t loader_version = VK_API_VERSION_1_0;
        const auto enumerate_instance_version =
            reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
        if (enumerate_instance_version == nullptr ||
            enumerate_instance_version(&loader_version) != VK_SUCCESS ||
            VK_VERSION_MAJOR(loader_version) != 1 ||
            VK_VERSION_MINOR(loader_version) < 1) {
            throw VulkanUnavailable("Vulkan 1.1 loader support is required");
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
        const char *validation_layer = kValidationLayer;
        const char *debug_utils_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        if (validation_enabled_) {
            if (!has_validation_layer()) {
                throw VulkanUnavailable("Vulkan validation layer is unavailable");
            }
            if (!has_debug_utils_extension()) {
                throw VulkanUnavailable("Vulkan debug utils extension is unavailable");
            }
            instance_info.enabledLayerCount = 1;
            instance_info.ppEnabledLayerNames = &validation_layer;
            instance_info.enabledExtensionCount = 1;
            instance_info.ppEnabledExtensionNames = &debug_utils_extension;
        }
        if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
            throw VulkanUnavailable("Could not create Vulkan instance");
        }

        if (validation_enabled_) {
            VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
            messenger_info.sType =
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            messenger_info.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            messenger_info.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            messenger_info.pfnUserCallback = debug_callback;
            const auto create_messenger =
                reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
            if (create_messenger == nullptr ||
                create_messenger(instance_, &messenger_info, nullptr,
                                 &debug_messenger_) != VK_SUCCESS) {
                throw std::runtime_error("Could not create Vulkan debug messenger");
            }
        }

        uint32_t device_count = 0;
        if (vkEnumeratePhysicalDevices(instance_, &device_count, nullptr) !=
            VK_SUCCESS) {
            throw VulkanUnavailable("Could not enumerate Vulkan physical devices");
        }
        if (device_count == 0) {
            throw VulkanUnavailable("No Vulkan physical device is available");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        if (vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()) !=
            VK_SUCCESS) {
            throw VulkanUnavailable("Could not enumerate Vulkan physical devices");
        }

        for (VkPhysicalDevice device : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device, &properties);

            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                                     nullptr);
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
                    check_result(
                        vkCreateDevice(device, &device_info, nullptr, &device_),
                        "Could not create Vulkan logical device");
                    physical_device_ = device;
                    vkGetDeviceQueue(device_, family, 0, &compute_queue_);
                    VkCommandPoolCreateInfo command_pool_info{};
                    command_pool_info.sType =
                        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                    command_pool_info.flags =
                        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                    command_pool_info.queueFamilyIndex = family;
                    if (vkCreateCommandPool(device_, &command_pool_info, nullptr,
                                            &command_pool_) != VK_SUCCESS) {
                        throw std::runtime_error(
                            "Could not create Vulkan command pool");
                    }
                    compute_ = std::make_unique<VulkanCompute>(*this);
                    return;
                }
            }
        }

        throw VulkanUnavailable("No Vulkan physical device has a compute queue");
    } catch (...) {
        cleanup();
        throw;
    }
}

VulkanPlatform::~VulkanPlatform() noexcept { cleanup(); }

void VulkanPlatform::cleanup() noexcept {
    std::scoped_lock lock(queue_mutex_);
    if (device_ != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle(device_) != VK_SUCCESS) {
            std::scoped_lock quarantine_lock(quarantine_mutex);
            if (quarantined_resource_count < kQuarantineCapacity) {
                QuarantinedVulkanResources &resources =
                    quarantined_resources[quarantined_resource_count++];
                resources.compute = std::move(compute_);
                resources.instance = instance_;
                resources.device = device_;
                resources.command_pool = command_pool_;
                resources.debug_messenger = debug_messenger_;
                pending_compute_resources_.clear();
                pending_transfer_resources_.clear();
            } else {
                // No allocation-free owner remains.  Leak every handle and
                // the compute object rather than destroying unknown work.
                compute_.release();
                pending_compute_resources_.clear();
                pending_transfer_resources_.clear();
            }
            instance_ = VK_NULL_HANDLE;
            device_ = VK_NULL_HANDLE;
            command_pool_ = VK_NULL_HANDLE;
            debug_messenger_ = VK_NULL_HANDLE;
            compute_queue_ = VK_NULL_HANDLE;
            physical_device_ = VK_NULL_HANDLE;
            return;
        }
    }
    // Compute owns device objects; wait for all queue work before destroying them.
    compute_.reset();
    for (const PendingComputeResources &resources : pending_compute_resources_) {
        if (resources.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, resources.fence, nullptr);
        }
        if (resources.command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, command_pool_, 1, &resources.command_buffer);
        }
        if (resources.descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, resources.descriptor_pool, nullptr);
        }
    }
    pending_compute_resources_.clear();
    for (const PendingTransferResources &resources : pending_transfer_resources_) {
        if (resources.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, resources.fence, nullptr);
        }
        if (resources.command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, command_pool_, 1, &resources.command_buffer);
        }
    }
    pending_transfer_resources_.clear();
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (debug_messenger_ != VK_NULL_HANDLE) {
        const auto destroy_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_messenger != nullptr) {
            destroy_messenger(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VulkanPlatform::is_available() noexcept {
    try {
        const VulkanPlatform platform;
        return platform.device() != VK_NULL_HANDLE;
    } catch (...) {
        return false;
    }
}

const VulkanDeviceInfo &VulkanPlatform::device_info() const { return device_info_; }

uint32_t VulkanPlatform::api_version() const { return api_version_; }

VkPhysicalDevice VulkanPlatform::physical_device() const { return physical_device_; }

VkDevice VulkanPlatform::device() const { return device_; }

VkQueue VulkanPlatform::compute_queue() const { return compute_queue_; }

VkCommandPool VulkanPlatform::command_pool() const { return command_pool_; }

VulkanCompute &VulkanPlatform::compute() const { return *compute_; }

std::mutex &VulkanPlatform::queue_mutex() const { return queue_mutex_; }

void VulkanPlatform::defer_compute_resources(VkDescriptorPool descriptor_pool,
                                              VkCommandBuffer command_buffer,
                                              VkFence fence) const noexcept {
    if (pending_compute_resources_.size() < pending_compute_resources_.capacity()) {
        pending_compute_resources_.push_back({descriptor_pool, command_buffer, fence});
        return;
    }
    // This is a defensive fail-safe for allocator/container anomalies.  The
    // submitted resources are intentionally leaked, never destroyed while
    // completion is unknown.
}

void VulkanPlatform::reserve_compute_resources() const {
    pending_compute_resources_.reserve(pending_compute_resources_.size() + 1);
}

std::size_t VulkanPlatform::pending_transfer_count() const {
    return pending_transfer_resources_.size();
}

std::size_t VulkanPlatform::compute_dispatch_count() const {
    return compute_ == nullptr ? 0 : compute_->dispatch_count();
}

bool VulkanPlatform::validation_enabled() const { return validation_enabled_; }

void VulkanPlatform::copy_buffer_sync(VkBuffer source, VkBuffer destination,
                                       VkDeviceSize size) const {
    if (source == VK_NULL_HANDLE || destination == VK_NULL_HANDLE || size == 0) {
        throw std::invalid_argument("Invalid Vulkan buffer copy arguments");
    }

    std::scoped_lock lock(queue_mutex_);

    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_result(vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer),
                 "Could not allocate Vulkan command buffer");

    VkFence fence = VK_NULL_HANDLE;
    bool submission_may_be_pending = false;
    const auto release_resources = [&]() {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
            command_buffer = VK_NULL_HANDLE;
        }
    };
    const auto defer_resources = [&]() {
        if (pending_transfer_resources_.size() < pending_transfer_resources_.capacity()) {
            pending_transfer_resources_.push_back({command_buffer, fence});
        }
        // If bookkeeping capacity is unexpectedly unavailable, intentionally
        // leak submitted resources rather than destroy them with unknown
        // completion.
        command_buffer = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    };
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
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "Could not create Vulkan transfer fence");
        // Ensure deferred ownership cannot allocate after submission starts.
        pending_transfer_resources_.reserve(pending_transfer_resources_.size() + 1);
        submission_may_be_pending = true;
        check_result(vkQueueSubmit(compute_queue_, 1, &submit_info, fence),
                     "Could not submit Vulkan command buffer");
        const VkResult wait_result =
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        if (wait_result != VK_SUCCESS) {
            const VkResult recovery_result = vkQueueWaitIdle(compute_queue_);
            if (recovery_result == VK_SUCCESS) {
                release_resources();
            } else {
                defer_resources();
                std::ostringstream message;
                message << "Could not wait for Vulkan transfer fence failed with VkResult "
                        << static_cast<int>(wait_result)
                        << "; could not confirm Vulkan transfer completion failed with "
                           "VkResult "
                        << static_cast<int>(recovery_result);
                throw std::runtime_error(message.str());
            }
            check_result(wait_result, "Could not wait for Vulkan transfer fence");
        }
    } catch (...) {
        if (command_buffer == VK_NULL_HANDLE && fence == VK_NULL_HANDLE) {
            throw;
        }
        if (submission_may_be_pending) {
            const VkResult recovery_result = vkQueueWaitIdle(compute_queue_);
            if (recovery_result == VK_SUCCESS) {
                release_resources();
            } else {
                defer_resources();
                std::ostringstream message;
                try {
                    throw;
                } catch (const std::exception &error) {
                    message << error.what();
                } catch (...) {
                    message << "Vulkan transfer failed with a non-standard exception";
                }
                message << "; could not confirm Vulkan transfer completion failed with "
                           "VkResult "
                        << static_cast<int>(recovery_result);
                throw std::runtime_error(message.str());
            }
        } else {
            release_resources();
        }
        throw;
    }

    release_resources();
}

void VulkanPlatform::wait_for_transfer() const {
    std::scoped_lock lock(queue_mutex_);
    const VkResult result = vkQueueWaitIdle(compute_queue_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Could not wait for Vulkan transfer queue with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

void VulkanPlatform::copy_buffer(VkBuffer source, VkBuffer destination,
                                  VkDeviceSize size) const {
    copy_buffer_sync(source, destination, size);
}
