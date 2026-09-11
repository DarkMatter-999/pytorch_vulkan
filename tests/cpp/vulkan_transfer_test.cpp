#include "vulkan_buffer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_host_visible_buffer_copy() {
    const VulkanPlatform platform;
    constexpr VkDeviceSize kSize = 4096;
    VulkanBuffer source(platform, kSize,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer destination(platform, kSize,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::vector<std::uint8_t> input(static_cast<size_t>(kSize));
    std::vector<std::uint8_t> output(input.size(), 0);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>(index * 17U + 3U);
    }
    source.write(input.data(), kSize);
    platform.copy_buffer_sync(source.buffer(), destination.buffer(), kSize);
    destination.read(output.data(), kSize);
    expect(output == input, "synchronous Vulkan buffer copy produced incorrect data");
}

} // namespace

int main() {
    try {
        test_host_visible_buffer_copy();
        std::cout << "Vulkan synchronous transfer test passed\n";
        return 0;
    } catch (const VulkanUnavailable &error) {
        std::cerr << error.what() << '\n';
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
