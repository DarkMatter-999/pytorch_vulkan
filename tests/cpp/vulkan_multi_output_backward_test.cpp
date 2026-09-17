#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"
#include "vulkan_tensor_layout.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using pytorch_vulkan::VulkanOverlap;
using pytorch_vulkan::VulkanTensorLayout;

void expect(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void expect_rejected(Function &&function, const char *message) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

VulkanTensorLayout layout(uint32_t rows, uint32_t columns, VkDeviceSize bytes) {
    return {2, {static_cast<int64_t>(rows), static_cast<int64_t>(columns)},
            {static_cast<int64_t>(columns), 1}, 0, sizeof(float), 0,
            static_cast<int64_t>(rows) * columns, 0, bytes, 64, VulkanOverlap::No};
}

void test_output_overlap_rejected(VulkanPlatform &platform) {
    VulkanBuffer grad_output(platform, 64), input(platform, 64), weight(platform, 64),
                 activation(platform, 64), d_input_and_weight(platform, 64),
                 d_bias(platform, 64);
    auto go = layout(1, 2, 8);
    auto x = layout(1, 2, 8);
    auto w = layout(2, 2, 16);
    auto a = layout(1, 2, 8);
    auto dx = layout(1, 2, 8);
    auto dw = layout(2, 2, 16);
    VulkanTensorLayout db{1, {2}, {1}, 0, sizeof(float), 0, 2, 0, 8, 64,
                          VulkanOverlap::No};
    dw.byte_offset = 4;
    bool rejected_overlap = false;
    std::string overlap_error;
    try {
            platform.compute().linear_relu_backward(
                &grad_output, go, &input, x, &weight, w, &activation, a,
                &d_input_and_weight, dx, &d_input_and_weight, dw, &d_bias, db, 1, 2, 2);
    } catch (const std::exception &error) {
        overlap_error = error.what();
        rejected_overlap = std::string(error.what()).find("outputs overlap") !=
                           std::string::npos;
    }
    if (!rejected_overlap)
        throw std::runtime_error("overlap fixture did not reach output-overlap validation: " +
                                 overlap_error);
}

void test_invalid_range_rejected(VulkanPlatform &platform) {
    VulkanBuffer grad_output(platform, 64), input(platform, 64), weight(platform, 64),
                 activation(platform, 64), d_input(platform, 64), d_weight(platform, 64),
                 d_bias(platform, 64);
    auto go = layout(1, 2, 8);
    auto x = layout(1, 2, 8);
    auto w = layout(2, 2, 16);
    auto a = layout(1, 2, 8);
    auto dx = layout(1, 2, 8);
    auto dw = layout(2, 2, 16);
    VulkanTensorLayout db{1, {2}, {1}, 0, sizeof(float), 0, 2, 0, 8, 64,
                          VulkanOverlap::No};
    dx.byte_range = 68;
    expect_rejected(
        [&] {
            platform.compute().linear_relu_backward(
                &grad_output, go, &input, x, &weight, w, &activation, a, &d_input, dx,
                &d_weight, dw, &d_bias, db, 1, 2, 2);
        },
        "invalid multi-output allocation range was accepted");
}
} // namespace

int main() {
    try {
        VulkanPlatform platform;
        test_output_overlap_rejected(platform);
        test_invalid_range_rejected(platform);
        std::cout << "Vulkan multi-output backward validation tests passed\n";
        return 0;
    } catch (const VulkanUnavailable &) {
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
