#include "vulkan_buffer.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    try {
        const bool enable_validation =
            argc == 2 && std::string(argv[1]) == "--validation";
        const VulkanPlatform platform(enable_validation);
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
        VulkanBuffer download(platform, sizeof(output),
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        platform.copy_buffer(staging.buffer(), buffer.buffer(), sizeof(input));
        platform.copy_buffer(buffer.buffer(), download.buffer(), sizeof(output));
        output.fill(0.0F);
        download.read(output.data(), sizeof(output));
        if (output != input) {
            throw std::runtime_error("Vulkan device transfer round trip failed");
        }
        for (int iteration = 0; iteration < 32; ++iteration) {
            const VulkanBuffer temporary(platform, 4096);
            if (temporary.buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan buffer lifecycle failed");
            }
        }
        std::cout << "Vulkan API: " << VK_VERSION_MAJOR(platform.api_version()) << "."
                  << VK_VERSION_MINOR(platform.api_version()) << "\n";
        std::cout << "Availability: " << (VulkanPlatform::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "Validation: "
                  << (platform.validation_enabled() ? "enabled" : "disabled") << "\n";
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
        std::cout << "Device transfer: passed\n";
        std::cout << "Lifecycle: passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
