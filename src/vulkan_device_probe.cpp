#include "vulkan_buffer.h"

#include <array>
#include <cstdlib>
#include <iostream>

int main() {
    try {
        const VulkanPlatform platform;
        const VulkanDeviceInfo &device = platform.device_info();
        const VulkanBuffer buffer(platform, 4096);
        const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
        std::array<float, 4> output{};
        VulkanBuffer staging(platform, sizeof(input),
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        staging.write(input.data(), sizeof(input));
        staging.read(output.data(), sizeof(output));
        if (output != input) {
            throw std::runtime_error("Vulkan host transfer round trip failed");
        }
        std::cout << "Vulkan API: " << VK_VERSION_MAJOR(platform.api_version()) << "."
                  << VK_VERSION_MINOR(platform.api_version()) << "\n";
        std::cout << "Vulkan device: " << device.name << "\n";
        std::cout << "Compute queue family: " << device.compute_queue_family << "\n";
        std::cout << "Logical device: "
                  << (platform.device() != VK_NULL_HANDLE ? "ready" : "missing")
                  << "\n";
        std::cout << "Buffer: "
                  << (buffer.buffer() != VK_NULL_HANDLE ? "ready" : "missing") << "\n";
        std::cout << "Transfer: passed\n";
        std::cout << "Command pool: "
                  << (platform.command_pool() != VK_NULL_HANDLE ? "ready" : "missing")
                  << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
