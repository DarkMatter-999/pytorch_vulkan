#include "vulkan_platform.h"

#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#endif
#include <array>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr uint32_t kMinimumPushConstantsSize = 36;
constexpr uint32_t kMinimumWorkgroupSize = 256;
constexpr VkDeviceSize kProbeStorageBufferSize = 4096;

struct DeviceSuitability {
    uint32_t compute_queue_family = 0;
    bool formatter_double_supported = false;
    bool bool_pointwise_supported = false;
    bool required_capabilities = false;
    bool storage_dispatch_limits = false;
    bool host_visible_memory = false;
    bool shader_capabilities = false;
    bool uses_core_12_features = false;
    bool has_8bit_storage_extension = false;
    bool has_float16_int8_extension = false;
    std::string reason;
};

template <typename Candidate, typename Suitable, typename Initialize>
int select_candidate(const std::vector<Candidate> &candidates, Suitable suitable,
                     Initialize initialize) {
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (suitable(candidates[index]) && initialize(candidates[index]))
            return static_cast<int>(index);
    }
    return -1;
}

bool has_device_extension(VkPhysicalDevice device, const char *name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                             extensions.data()) != VK_SUCCESS) {
        return false;
    }
    for (const auto &extension : extensions) {
        if (std::string(extension.extensionName) == name)
            return true;
    }
    return false;
}

bool has_storage_buffer_memory(VkPhysicalDevice physical_device, VkDevice device) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = kProbeStorageBufferSize;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
    bool suitable = false;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1U << i)) != 0 &&
            (memory.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            suitable = true;
            break;
        }
    }
    vkDestroyBuffer(device, buffer, nullptr);
    return suitable;
}

bool has_required_workgroup_limits(const VkPhysicalDeviceLimits &limits) {
    return limits.maxComputeWorkGroupCount[0] > 0 &&
           limits.maxComputeWorkGroupCount[1] > 0 &&
           limits.maxComputeWorkGroupCount[2] > 0 &&
           limits.maxComputeWorkGroupSize[0] >= kMinimumWorkgroupSize &&
           limits.maxComputeWorkGroupSize[1] >= 1 &&
           limits.maxComputeWorkGroupSize[2] >= 1 &&
           limits.maxComputeWorkGroupInvocations >= kMinimumWorkgroupSize;
}

DeviceSuitability assess_device(VkPhysicalDevice device,
                                uint32_t negotiated_api_version) {
    DeviceSuitability result;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
    for (uint32_t family = 0; family < queue_count; ++family) {
        if (queues[family].queueCount > 0 &&
            (queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            result.compute_queue_family = family;
            result.required_capabilities = true;
            break;
        }
    }

    // Storage-buffer-storage-class is core in Vulkan 1.1, but remains an
    // explicit requirement for a 1.0 physical device.
    const bool api_1_1 = VK_VERSION_MINOR(negotiated_api_version) >= 1 &&
                         VK_VERSION_MINOR(properties.apiVersion) >= 1;
    const bool api_1_2 = VK_VERSION_MINOR(negotiated_api_version) >= 2 &&
                         VK_VERSION_MINOR(properties.apiVersion) >= 2;
    const bool storage_extension =
        api_1_2 || has_device_extension(device, VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    result.has_8bit_storage_extension =
        has_device_extension(device, VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    result.has_float16_int8_extension =
        has_device_extension(device, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    result.storage_dispatch_limits =
        storage_extension &&
        properties.limits.maxStorageBufferRange >= kProbeStorageBufferSize &&
        has_required_workgroup_limits(properties.limits) &&
        properties.limits.maxPushConstantsSize >= kMinimumPushConstantsSize;

    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((memory.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            result.host_visible_memory = true;
            break;
        }
    }

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (api_1_2) {
        VkPhysicalDeviceVulkan12Features core12{};
        core12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features.pNext = &core12;
        vkGetPhysicalDeviceFeatures2(device, &features);
        result.formatter_double_supported = features.features.shaderFloat64;
        result.bool_pointwise_supported =
            core12.shaderInt8 && core12.storageBuffer8BitAccess;
        result.shader_capabilities =
            core12.storageBuffer8BitAccess && core12.uniformAndStorageBuffer8BitAccess;
        result.uses_core_12_features = true;
    } else {
        VkPhysicalDevice8BitStorageFeatures storage8{};
        storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
        float16_int8.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        features.pNext = &storage8;
        storage8.pNext = &float16_int8;
        vkGetPhysicalDeviceFeatures2(device, &features);
        result.formatter_double_supported = features.features.shaderFloat64;
        result.bool_pointwise_supported = result.has_float16_int8_extension &&
                                          float16_int8.shaderInt8 &&
                                          storage8.storageBuffer8BitAccess;
        result.shader_capabilities = storage8.storageBuffer8BitAccess &&
                                     storage8.uniformAndStorageBuffer8BitAccess;
    }

    if (!result.required_capabilities)
        result.reason = "no compute queue";
    else if (!result.storage_dispatch_limits)
        result.reason = "required extensions or limits are unavailable";
    else if (!result.host_visible_memory)
        result.reason = "no host-visible memory is available";
    else if (!result.shader_capabilities)
        result.reason = "required shader capabilities are unavailable";
    return result;
}

// If device-idle cannot be established, the Vulkan handles must remain owned
// by a live object.  These records intentionally retain their handles for the
// remainder of the process rather than destroying resources whose completion
// state is unknown.
struct QuarantinedVulkanResources {
    std::unique_ptr<VulkanCompute> compute;
    std::unique_ptr<VulkanExecutionContext> execution;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    std::unique_ptr<VulkanBuffer> staging_buffer;
    std::vector<VulkanPendingTransferResources> pending_transfer_resources;

    // Completion is unknown, so retain the compute object's Vulkan handles
    // and parent device instead of invoking destruction during teardown. The
    // pending command/fence/pool handle values remain live under this device;
    // their vector metadata does not own or release Vulkan handles.
    // The slot intentionally never destroys compute: completion is unknown.
    ~QuarantinedVulkanResources() {
        execution.release();
        compute.release();
        staging_buffer.release();
    }
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

namespace pytorch_vulkan {

namespace testing {
bool candidate_selection_falls_back() noexcept {
    struct Candidate {
        bool suitable;
        bool initializes;
        const char *reason;
    };
    const std::vector<Candidate> candidates{
        {false, false, "missing compute queue"},
        {true, false, "logical device creation failed"},
        {true, true, ""}};
    int attempts = 0;
    std::vector<std::string> reasons;
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 1;
    limits.maxComputeWorkGroupCount[1] = 1;
    limits.maxComputeWorkGroupCount[2] = 1;
    limits.maxComputeWorkGroupSize[0] = kMinimumWorkgroupSize;
    limits.maxComputeWorkGroupSize[1] = 1;
    limits.maxComputeWorkGroupSize[2] = 1;
    limits.maxComputeWorkGroupInvocations = kMinimumWorkgroupSize;
    const bool dimensions_pass = has_required_workgroup_limits(limits);
    limits.maxComputeWorkGroupSize[1] = 0;
    const bool y_rejected = !has_required_workgroup_limits(limits);
    limits.maxComputeWorkGroupSize[1] = 1;
    limits.maxComputeWorkGroupSize[2] = 0;
    const bool z_rejected = !has_required_workgroup_limits(limits);
    return dimensions_pass && y_rejected && z_rejected &&
           select_candidate(
               candidates,
               [&reasons](const Candidate &candidate) {
                   if (!candidate.suitable)
                       reasons.emplace_back(candidate.reason);
                   return candidate.suitable;
               },
               [&attempts, &reasons](const Candidate &candidate) {
                   ++attempts;
                   if (!candidate.initializes)
                       reasons.emplace_back(candidate.reason);
                   return candidate.initializes;
               }) == 2 &&
           attempts == 2 && reasons.size() == 2 &&
           reasons[0] == "missing compute queue" &&
           reasons[1] == "logical device creation failed";
}

} // namespace testing

namespace {

std::atomic<bool> forked_child{false};

void mark_forked_child() noexcept {
    forked_child.store(true, std::memory_order_relaxed);
}

} // namespace

bool inherited_fork_state() noexcept {
    return forked_child.load(std::memory_order_relaxed);
}

void register_fork_state_handler() {
#if defined(__unix__) || defined(__APPLE__)
    static std::once_flag registration;
    std::call_once(registration, [] {
        if (pthread_atfork(nullptr, nullptr, &mark_forked_child) != 0) {
            throw std::runtime_error("Could not register Vulkan fork-state handler");
        }
    });
#endif
}

void ensure_process_local_vulkan() {
    if (inherited_fork_state()) {
        throw std::runtime_error(
            "Vulkan cannot be used after fork because the child inherited parent "
            "Vulkan "
            "state; use multiprocessing spawn or materialize tensors on CPU");
    }
}

} // namespace pytorch_vulkan

VulkanPlatform::VulkanPlatform(bool enable_validation)
    : validation_enabled_(enable_validation) {
    try {
        pytorch_vulkan::register_fork_state_handler();
        pytorch_vulkan::ensure_process_local_vulkan();
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
        api_version_ = VK_MAKE_VERSION(
            1, std::min(VK_VERSION_MINOR(loader_version), uint32_t{2}), 0);

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

        std::string rejected_devices;
        const auto cleanup_candidate = [&]() noexcept {
            if (device_ != VK_NULL_HANDLE && vkDeviceWaitIdle(device_) != VK_SUCCESS) {
                std::scoped_lock quarantine_lock(quarantine_mutex);
                if (quarantined_resource_count < kQuarantineCapacity) {
                    auto &resources =
                        quarantined_resources[quarantined_resource_count++];
                    resources.compute = std::move(compute_);
                    resources.execution = std::move(execution_);
                    resources.instance = instance_;
                    resources.device = device_;
                    resources.command_pool = command_pool_;
                    resources.debug_messenger = debug_messenger_;
                    resources.staging_buffer = std::move(staging_buffer_);
                } else {
                    compute_.release();
                    execution_.release();
                    staging_buffer_.release();
                }
                instance_ = VK_NULL_HANDLE;
                command_pool_ = VK_NULL_HANDLE;
                device_ = VK_NULL_HANDLE;
                debug_messenger_ = VK_NULL_HANDLE;
            } else {
                compute_.reset();
                execution_.reset();
                if (command_pool_ != VK_NULL_HANDLE)
                    vkDestroyCommandPool(device_, command_pool_, nullptr);
                if (device_ != VK_NULL_HANDLE)
                    vkDestroyDevice(device_, nullptr);
                command_pool_ = VK_NULL_HANDLE;
                device_ = VK_NULL_HANDLE;
            }
            physical_device_ = VK_NULL_HANDLE;
            compute_queue_ = VK_NULL_HANDLE;
        };
        DeviceSuitability suitability;
        VkPhysicalDeviceProperties properties{};
        const int selected_candidate = select_candidate(
            devices,
            [&](VkPhysicalDevice device) {
                vkGetPhysicalDeviceProperties(device, &properties);
                suitability = assess_device(device, api_version_);
                if (!suitability.reason.empty()) {
                    if (!rejected_devices.empty())
                        rejected_devices += "; ";
                    rejected_devices +=
                        std::string(properties.deviceName) + ": " + suitability.reason;
                }
                return suitability.reason.empty();
            },
            [&](VkPhysicalDevice device) {
                try {
                    const uint32_t family = suitability.compute_queue_family;
                    device_info_.name = properties.deviceName;
                    device_info_.compute_queue_family = family;
                    device_info_.required_capabilities =
                        suitability.required_capabilities;
                    device_info_.storage_dispatch_limits =
                        suitability.storage_dispatch_limits;
                    device_info_.host_visible_memory = suitability.host_visible_memory;
                    device_info_.shader_capabilities = suitability.shader_capabilities;
                    float queue_priority = 1.0F;
                    VkDeviceQueueCreateInfo queue_info{};
                    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                    queue_info.queueFamilyIndex = family;
                    queue_info.queueCount = 1;
                    queue_info.pQueuePriorities = &queue_priority;
                    VkPhysicalDeviceFeatures2 enabled_features2{};
                    enabled_features2.sType =
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                    enabled_features2.features.shaderFloat64 =
                        suitability.formatter_double_supported;
                    VkDeviceCreateInfo device_info{};
                    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                    device_info.queueCreateInfoCount = 1;
                    device_info.pQueueCreateInfos = &queue_info;
                    VkPhysicalDeviceVulkan12Features core12{};
                    VkPhysicalDevice8BitStorageFeatures enabled_8bit{};
                    VkPhysicalDeviceShaderFloat16Int8Features enabled_int8{};
                    const char *device_extensions[] = {
                        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
                        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME};
                    if (suitability.uses_core_12_features) {
                        core12.sType =
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
                        core12.storageBuffer8BitAccess = VK_TRUE;
                        core12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
                        core12.shaderInt8 = suitability.bool_pointwise_supported;
                        enabled_features2.pNext = &core12;
                    } else {
                        enabled_8bit.sType =
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
                        enabled_8bit.storageBuffer8BitAccess = VK_TRUE;
                        enabled_8bit.uniformAndStorageBuffer8BitAccess = VK_TRUE;
                        enabled_int8.sType =
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
                        enabled_int8.shaderInt8 = suitability.bool_pointwise_supported;
                        enabled_8bit.pNext = suitability.bool_pointwise_supported
                                                 ? &enabled_int8
                                                 : nullptr;
                        enabled_features2.pNext = &enabled_8bit;
                        device_info.enabledExtensionCount =
                            suitability.bool_pointwise_supported ? 2 : 1;
                        device_info.ppEnabledExtensionNames = device_extensions;
                    }
                    device_info.pEnabledFeatures = nullptr;
                    device_info.pNext = &enabled_features2;
                    if (suitability.uses_core_12_features &&
                        suitability.has_8bit_storage_extension) {
                        device_info.enabledExtensionCount = 1;
                        device_info.ppEnabledExtensionNames = device_extensions;
                    }
                    check_result(
                        vkCreateDevice(device, &device_info, nullptr, &device_),
                        "Could not create Vulkan logical device");
                    if (!has_storage_buffer_memory(device, device_))
                        throw VulkanUnavailable(
                            "storage buffer has no host-visible memory type");
                    physical_device_ = device;
                    bool_pointwise_supported_ = suitability.bool_pointwise_supported;
                    formatter_double_supported_ =
                        suitability.formatter_double_supported;
                    vkGetDeviceQueue(device_, family, 0, &compute_queue_);
                    VkCommandPoolCreateInfo command_pool_info{};
                    command_pool_info.sType =
                        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                    command_pool_info.flags =
                        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                    command_pool_info.queueFamilyIndex = family;
                    check_result(vkCreateCommandPool(device_, &command_pool_info,
                                                     nullptr, &command_pool_),
                                 "Could not create Vulkan command pool");
                    execution_ = std::make_unique<VulkanExecutionContext>(
                        device_, compute_queue_, command_pool_, this);
                    compute_ = std::make_unique<VulkanCompute>(*this);
                    return true;
                } catch (const std::exception &error) {
                    if (!rejected_devices.empty())
                        rejected_devices += "; ";
                    rejected_devices +=
                        std::string(properties.deviceName) + ": " + error.what();
                    cleanup_candidate();
                    if (instance_ == VK_NULL_HANDLE)
                        throw VulkanUnavailable(
                            "Could not safely clean up Vulkan candidate");
                    return false;
                }
            });

        if (selected_candidate < 0)
            throw VulkanUnavailable("No suitable Vulkan physical device: " +
                                    rejected_devices);
    } catch (...) {
        cleanup();
        throw;
    }
}

VulkanPlatform::~VulkanPlatform() noexcept { cleanup(); }

void VulkanPlatform::cleanup() noexcept {
    if (pytorch_vulkan::inherited_fork_state()) {
        // Do not invoke Vulkan teardown on handles inherited across fork.
        compute_.release();
        execution_.release();
        return;
    }
    std::scoped_lock lock(queue_mutex_);
    if (device_ != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle(device_) != VK_SUCCESS) {
            std::scoped_lock quarantine_lock(quarantine_mutex);
            if (quarantined_resource_count < kQuarantineCapacity) {
                QuarantinedVulkanResources &resources =
                    quarantined_resources[quarantined_resource_count++];
                resources.compute = std::move(compute_);
                resources.execution = std::move(execution_);
                resources.instance = instance_;
                resources.device = device_;
                resources.command_pool = command_pool_;
                resources.debug_messenger = debug_messenger_;
                resources.staging_buffer = std::move(staging_buffer_);
                resources.pending_transfer_resources =
                    std::move(pending_transfer_resources_);
            } else {
                // No allocation-free owner remains.  Leak every handle and
                // the compute object rather than destroying unknown work.
                compute_.release();
                execution_.release();
                staging_buffer_.release();
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
    execution_.reset();
    compute_.reset();
    staging_buffer_.reset();
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
        pytorch_vulkan::register_fork_state_handler();
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

VulkanExecutionContext &VulkanPlatform::execution_context() const {
    return *execution_;
}

VulkanCompute &VulkanPlatform::compute() const { return *compute_; }

std::mutex &VulkanPlatform::queue_mutex() const { return queue_mutex_; }

std::mutex &VulkanPlatform::transfer_mutex() const { return transfer_mutex_; }

std::size_t VulkanPlatform::pending_transfer_count() const {
    return pending_transfer_resources_.size();
}

std::size_t VulkanPlatform::pending_compute_count() const {
    std::scoped_lock lock(queue_mutex_);
    return execution_ == nullptr ? 0 : execution_->pending_count();
}

std::size_t VulkanPlatform::compute_dispatch_count() const {
    std::scoped_lock lock(queue_mutex_);
    return compute_ == nullptr ? 0 : compute_->dispatch_count();
}

void VulkanPlatform::reset_execution_counters() const {
    std::scoped_lock lock(queue_mutex_);
    explicit_transfer_count_.store(0, std::memory_order_relaxed);
    vulkan_copy_count_.store(0, std::memory_order_relaxed);
    fallback_count_.store(0, std::memory_order_relaxed);
    copy_command_count_.store(0, std::memory_order_relaxed);
    compute_submitted_count_.store(0, std::memory_order_relaxed);
    compute_completed_count_.store(0, std::memory_order_relaxed);
    compute_wait_count_.store(0, std::memory_order_relaxed);
    if (compute_ != nullptr) {
        compute_->reset_dispatch_count();
        compute_->reset_submission_count();
    }
}

std::size_t VulkanPlatform::compute_submitted_count() const {
    return compute_submitted_count_.load(std::memory_order_relaxed);
}

std::size_t VulkanPlatform::compute_completed_count() const {
    return compute_completed_count_.load(std::memory_order_relaxed);
}

std::size_t VulkanPlatform::compute_wait_count() const {
    return compute_wait_count_.load(std::memory_order_relaxed);
}

void VulkanPlatform::record_compute_submitted() const {
    compute_submitted_count_.fetch_add(1, std::memory_order_relaxed);
}

void VulkanPlatform::record_compute_completed() const {
    compute_completed_count_.fetch_add(1, std::memory_order_relaxed);
}

void VulkanPlatform::record_compute_wait() const {
    compute_wait_count_.fetch_add(1, std::memory_order_relaxed);
}

VulkanExecutionCounterSnapshot VulkanPlatform::execution_counter_snapshot() const {
    std::scoped_lock lock(queue_mutex_);
    return {compute_ == nullptr ? 0 : compute_->dispatch_count(),
            vulkan_copy_count_.load(std::memory_order_relaxed),
            explicit_transfer_count_.load(std::memory_order_relaxed),
            fallback_count_.load(std::memory_order_relaxed)};
}

std::size_t VulkanPlatform::explicit_transfer_count() const {
    std::scoped_lock lock(queue_mutex_);
    return explicit_transfer_count_.load(std::memory_order_relaxed);
}

void VulkanPlatform::record_explicit_transfer() const {
    std::scoped_lock lock(queue_mutex_);
    explicit_transfer_count_.fetch_add(1, std::memory_order_relaxed);
}

void VulkanPlatform::record_fallback() const {
    fallback_count_.fetch_add(1, std::memory_order_relaxed);
    if (strict_mode_.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "Vulkan strict fallback mode rejects implicit CPU fallback");
    }
}

void VulkanPlatform::set_strict_mode(bool enabled) const {
    strict_mode_.store(enabled, std::memory_order_relaxed);
}

bool VulkanPlatform::strict_mode() const {
    return strict_mode_.load(std::memory_order_relaxed);
}

std::size_t VulkanPlatform::vulkan_copy_count() const {
    std::scoped_lock lock(queue_mutex_);
    return vulkan_copy_count_.load(std::memory_order_relaxed);
}

void VulkanPlatform::record_vulkan_copy() const {
    std::scoped_lock lock(queue_mutex_);
    vulkan_copy_count_.fetch_add(1, std::memory_order_relaxed);
}

std::size_t VulkanPlatform::copy_command_count() const {
    std::scoped_lock lock(queue_mutex_);
    return copy_command_count_.load(std::memory_order_relaxed);
}

void VulkanPlatform::record_copy_command() const {
    copy_command_count_.fetch_add(1, std::memory_order_relaxed);
}

bool VulkanPlatform::validation_enabled() const { return validation_enabled_; }

bool VulkanPlatform::supports_bool_pointwise() const {
    return bool_pointwise_supported_;
}

bool VulkanPlatform::supports_formatter_double() const {
    return formatter_double_supported_;
}

void VulkanPlatform::copy_buffer_sync(VkBuffer source, VkBuffer destination,
                                      VkDeviceSize size, VkDeviceSize source_offset,
                                      VkDeviceSize destination_offset) const {
    if (source == VK_NULL_HANDLE || destination == VK_NULL_HANDLE || size == 0) {
        throw std::invalid_argument("Invalid Vulkan buffer copy arguments");
    }

    std::scoped_lock lock(queue_mutex_);

    const auto total_start = std::chrono::steady_clock::now();
    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    const auto allocation_start = std::chrono::steady_clock::now();
    check_result(vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer),
                 "Could not allocate Vulkan command buffer");
    timing_.allocation += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - allocation_start)
                              .count();

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
        if (pending_transfer_resources_.size() <
            pending_transfer_resources_.capacity()) {
            pending_transfer_resources_.push_back({command_buffer, fence});
        }
        // If bookkeeping capacity is unexpectedly unavailable, intentionally
        // leak submitted resources rather than destroy them with unknown
        // completion.
        command_buffer = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    };
    try {
        const auto recording_start = std::chrono::steady_clock::now();
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(command_buffer, &begin_info),
                     "Could not begin Vulkan command buffer");

        VkBufferCopy copy_region{};
        copy_region.srcOffset = source_offset;
        copy_region.dstOffset = destination_offset;
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, source, destination, 1, &copy_region);
        copy_command_count_.fetch_add(1, std::memory_order_relaxed);
        check_result(vkEndCommandBuffer(command_buffer),
                     "Could not end Vulkan command buffer");
        timing_.recording += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - recording_start)
                                 .count();

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
        const auto submit_wait_start = std::chrono::steady_clock::now();
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
                message
                    << "Could not wait for Vulkan transfer fence failed with VkResult "
                    << static_cast<int>(wait_result)
                    << "; could not confirm Vulkan transfer completion failed with "
                       "VkResult "
                    << static_cast<int>(recovery_result);
                throw std::runtime_error(message.str());
            }
            check_result(wait_result, "Could not wait for Vulkan transfer fence");
        }
        timing_.submit_wait += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - submit_wait_start)
                                   .count();
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
    timing_.total +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start)
            .count();
}

void VulkanPlatform::fill_buffer_sync(VkBuffer buffer, VkDeviceSize offset,
                                      VkDeviceSize size, uint32_t data) const {
    if (buffer == VK_NULL_HANDLE || size == 0 || (offset % 4) != 0 || (size % 4) != 0) {
        throw std::invalid_argument("Invalid Vulkan buffer fill arguments");
    }
    std::scoped_lock lock(queue_mutex_);
    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_result(vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer),
                 "Could not allocate Vulkan fill command buffer");
    VkFence fence = VK_NULL_HANDLE;
    try {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_result(vkBeginCommandBuffer(command_buffer, &begin_info),
                     "Could not begin Vulkan fill command buffer");
        vkCmdFillBuffer(command_buffer, buffer, offset, size, data);
        check_result(vkEndCommandBuffer(command_buffer),
                     "Could not end Vulkan fill command buffer");
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check_result(vkCreateFence(device_, &fence_info, nullptr, &fence),
                     "Could not create Vulkan fill fence");
        check_result(vkQueueSubmit(compute_queue_, 1, &submit_info, fence),
                     "Could not submit Vulkan fill command buffer");
        check_result(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                     "Could not wait for Vulkan fill fence");
    } catch (...) {
        vkQueueWaitIdle(compute_queue_);
        if (fence != VK_NULL_HANDLE)
            vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw;
    }
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

void VulkanPlatform::wait_for_transfer() const {
    std::scoped_lock lock(queue_mutex_);
    const VkResult result = vkQueueWaitIdle(compute_queue_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Could not wait for Vulkan transfer queue with VkResult " +
            std::to_string(static_cast<int>(result)));
    }
}

void VulkanPlatform::copy_buffer(VkBuffer source, VkBuffer destination,
                                 VkDeviceSize size) const {
    copy_buffer_sync(source, destination, size);
}

VulkanBuffer &VulkanPlatform::staging_buffer(VkDeviceSize size) const {
    if (size == 0)
        throw std::invalid_argument(
            "Vulkan staging buffer size must be greater than zero");
    std::scoped_lock lock(queue_mutex_);
    const auto start = std::chrono::steady_clock::now();
    if (staging_buffer_ == nullptr || staging_buffer_->size() < size) {
        staging_buffer_ = std::make_unique<VulkanBuffer>(
            *this, size,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    const auto end = std::chrono::steady_clock::now();
    timing_.allocation += std::chrono::duration<double>(end - start).count();
    return *staging_buffer_;
}

VulkanTimingSnapshot VulkanPlatform::timing_snapshot() const {
    std::scoped_lock lock(queue_mutex_);
    return timing_;
}

void VulkanPlatform::reset_timing() const {
    std::scoped_lock lock(queue_mutex_);
    timing_ = {};
}

void VulkanPlatform::record_timing(VulkanTimingCategory category,
                                   double seconds) const {
    timing_.total += seconds;
    switch (category) {
    case VulkanTimingCategory::Allocation:
        timing_.allocation += seconds;
        break;
    case VulkanTimingCategory::Recording:
        timing_.recording += seconds;
        break;
    case VulkanTimingCategory::SubmitWait:
        timing_.submit_wait += seconds;
        break;
    case VulkanTimingCategory::Compute:
        timing_.compute += seconds;
        break;
    }
}
