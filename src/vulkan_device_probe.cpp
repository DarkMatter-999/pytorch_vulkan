#include "vulkan_platform.h"

#include <cstdlib>
#include <iostream>

int main() {
    try {
        const VulkanPlatform platform;
        const VulkanDeviceInfo &device = platform.device_info();
        std::cout << "Vulkan API: " << VK_VERSION_MAJOR(platform.api_version()) << "."
                  << VK_VERSION_MINOR(platform.api_version()) << "\n";
        std::cout << "Vulkan device: " << device.name << "\n";
        std::cout << "Compute queue family: " << device.compute_queue_family << "\n";
        std::cout << "Logical device: "
                  << (platform.device() != VK_NULL_HANDLE ? "ready" : "missing")
                  << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
