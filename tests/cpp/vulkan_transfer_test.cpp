#include "vulkan_buffer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
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

void test_host_visible_bool_byte_ranges() {
    const VulkanPlatform platform;
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (const size_t size : {1U, 3U, 4U, 5U, 64U, 512U, 4096U}) {
        for (const size_t offset : {1U, 2U, 3U, 4U}) {
            std::vector<std::uint8_t> original(size + offset + 8, 1);
            std::vector<std::uint8_t> input(size);
            for (size_t index = 0; index < size; ++index)
                input[index] = static_cast<std::uint8_t>(index % 3 == 0);
            VulkanBuffer buffer(platform, original.size(), host);
            buffer.write(original.data(), original.size());
            platform.reset_execution_counters();
            // These are the same checked map-once leaves as the tensor Bool path.
            buffer.write(input.data(), size, offset);
            std::vector<std::uint8_t> readback(size);
            buffer.read(readback.data(), size, offset);
            expect(readback == input, "host-visible Bool subrange readback differed");
            expect(platform.transfer_submission_count() == 0 &&
                       platform.transfer_wait_count() == 0,
                   "host-visible Bool access unexpectedly submitted a transfer");
            std::vector<std::uint8_t> backing(original.size());
            buffer.read(backing.data(), backing.size());
            std::copy(input.begin(), input.end(), original.begin() + offset);
            expect(backing == original, "host-visible Bool write clobbered neighbors");
        }
    }
}

} // namespace

int main() {
    try {
        test_host_visible_buffer_copy();
        test_host_visible_bool_byte_ranges();
        std::cout << "Vulkan synchronous transfer test passed\n";
        return 0;
    } catch (const VulkanUnavailable &error) {
        expect(std::string(error.what()).find("Vulkan") != std::string::npos,
               "unavailable Vulkan transfer reported an unrelated error");
        std::cerr << error.what() << '\n';
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
