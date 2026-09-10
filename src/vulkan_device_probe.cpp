#include <vulkan/vulkan.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

const char *result_name(VkResult result) {
    return result == VK_SUCCESS ? "success" : "failure";
}

} // namespace

int main() {
    uint32_t loader_version = VK_API_VERSION_1_0;
    const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerate_instance_version == nullptr ||
        enumerate_instance_version(&loader_version) != VK_SUCCESS ||
        VK_VERSION_MAJOR(loader_version) != 1 || VK_VERSION_MINOR(loader_version) < 1) {
        std::cerr << "Vulkan 1.1 loader support is required\n";
        return EXIT_FAILURE;
    }

    VkApplicationInfo application_info{};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "pytorch_vulkan device probe";
    application_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.pEngineName = "pytorch_vulkan";
    application_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult instance_result =
        vkCreateInstance(&instance_info, nullptr, &instance);
    if (instance_result != VK_SUCCESS) {
        std::cerr << "Could not create Vulkan instance: "
                  << result_name(instance_result) << "\n";
        return EXIT_FAILURE;
    }

    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        std::cerr << "No Vulkan physical device is available\n";
        vkDestroyInstance(instance, nullptr);
        return EXIT_FAILURE;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        std::cerr << "Could not enumerate Vulkan physical devices\n";
        vkDestroyInstance(instance, nullptr);
        return EXIT_FAILURE;
    }

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
                std::cout << "Vulkan device: " << properties.deviceName << "\n";
                std::cout << "Compute queue family: " << family << "\n";
                vkDestroyInstance(instance, nullptr);
                return EXIT_SUCCESS;
            }
        }
    }

    std::cerr << "No Vulkan physical device has a compute queue\n";
    vkDestroyInstance(instance, nullptr);
    return EXIT_FAILURE;
}
