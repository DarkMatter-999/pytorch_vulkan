#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <c10/core/Allocator.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_one_allocation(c10::Allocator &allocator) {
    auto data = allocator.allocate(4096);
    expect(data.device() == kDevice, "allocation has the wrong device");
    expect(data.get() != nullptr, "allocation data token is null");
    expect(data.get() == data.get_context(), "data token is not the payload");
}

void test_independent_release_order(c10::Allocator &allocator) {
    auto first = allocator.allocate(1024);
    auto second = allocator.allocate(2048);
    auto third = allocator.allocate(4096);
    expect(first.get() != second.get(), "allocations are not independent");
    expect(second.get() != third.get(), "allocations are not independent");
    second.clear();
    first.clear();
    third.clear();
}

void test_repeated_lifetimes(c10::Allocator &allocator) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        auto data = allocator.allocate(256);
        expect(data.get() != nullptr, "repeated allocation returned null");
    }
}

void test_zero_byte_allocation(c10::Allocator &allocator) {
    auto data = allocator.allocate(0);
    expect(data.device() == kDevice, "zero-byte allocation has the wrong device");
    expect(data.get() != nullptr, "zero-byte allocation payload is null");
    expect(data.get() == data.get_context(), "zero-byte allocation context mismatch");
}

void test_oversized_allocation_rejected(c10::Allocator &allocator) {
    const size_t size = std::numeric_limits<size_t>::max();
    bool rejected = false;
    std::string message;
    try {
        (void)allocator.allocate(size);
    } catch (const std::exception &error) {
        message = error.what();
        rejected =
            message.find("requested 18446744073709551615 bytes") != std::string::npos &&
            message.find("PrivateUse1 device index 0") != std::string::npos;
    }
    if (!rejected) {
        throw std::runtime_error(
            "oversized allocation was not rejected with context: " + message);
    }
}

void test_copy_rejected(c10::Allocator &allocator) {
    bool rejected = false;
    try {
        allocator.copy_data(nullptr, nullptr, 1);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    expect(rejected, "opaque-pointer copy was accepted");
}

void test_explicit_shutdown_preserves_live_allocation(c10::Allocator &allocator) {
    auto data = allocator.allocate(4096);
    const VulkanPlatform &owned_platform = pytorch_vulkan::allocation_platform(data);
    pytorch_vulkan::shutdown_platform();
    expect(owned_platform.device() != VK_NULL_HANDLE,
           "explicit platform shutdown invalidated a live allocation");
    data.clear();
}

void test_release_after_submission(c10::Allocator &allocator) {
    auto &platform = *pytorch_vulkan::platform();
    auto &context = platform.execution_context();
    auto data = allocator.allocate(4096);
    VulkanBuffer &buffer = pytorch_vulkan::allocation_buffer(data);
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 4096, 0x12345678U);
    context.submit();
    data.clear();
    expect(context.pending_count() == 1,
           "allocation release unexpectedly retired submitted work");
    context.wait();
    expect(context.pending_count() == 0,
           "submitted allocation was not retired after synchronization");
}

void test_bounded_reuse_after_submission(c10::Allocator &allocator) {
    auto &platform = *pytorch_vulkan::platform();
    auto &context = platform.execution_context();
    for (int iteration = 0; iteration < 8; ++iteration) {
        auto small = allocator.allocate(256);
        VulkanBuffer &small_buffer = pytorch_vulkan::allocation_buffer(small);
        context.begin();
        vkCmdFillBuffer(context.command_buffer(), small_buffer.buffer(), 0, 256,
                        static_cast<uint32_t>(iteration));
        context.submit();
        small.clear();
        auto large = allocator.allocate(16384);
        VulkanBuffer &large_buffer = pytorch_vulkan::allocation_buffer(large);
        context.begin();
        vkCmdFillBuffer(context.command_buffer(), large_buffer.buffer(), 0, 16384,
                        static_cast<uint32_t>(iteration + 1));
        context.submit();
        large.clear();
        expect(context.pending_count() == 2,
               "submitted reuse sequence did not fill both ring slots");
        context.begin();
        context.cancel();
        auto final_small = allocator.allocate(256);
        VulkanBuffer &final_small_buffer =
            pytorch_vulkan::allocation_buffer(final_small);
        context.begin();
        vkCmdFillBuffer(context.command_buffer(), final_small_buffer.buffer(), 0, 256,
                        static_cast<uint32_t>(iteration + 2));
        context.submit();
        final_small.clear();
        context.wait();
        expect(context.pending_count() == 0,
               "submitted reuse sequence retained a completed slot");
    }
}

void test_bounded_allocator_transition_stress(c10::Allocator &allocator) {
    for (int iteration = 0; iteration < 16; ++iteration) {
        auto small = allocator.allocate(128 + iteration);
        auto large = allocator.allocate(8192 + iteration * 17);
        expect(small.get() != large.get(), "allocator transition aliased allocations");
        large.clear();
        small.clear();
    }
}

} // namespace

int main() {
    (void)vulkan_allocator_instance();
    auto *allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    try {
        expect(allocator != nullptr, "PrivateUse1 allocator is not registered");

        bool setup_available = true;
        try {
            (void)pytorch_vulkan::platform();
        } catch (const VulkanUnavailable &) {
            setup_available = false;
        }
        if (!setup_available) {
            expect(pytorch_vulkan::VulkanDeviceGuard().deviceCount() == 0,
                   "unavailable Vulkan device count was not zero");
            bool rejected = false;
            try {
                (void)allocator->allocate(4096);
            } catch (const VulkanUnavailable &) {
                rejected = true;
            } catch (const std::exception &) {
                throw std::runtime_error(
                    "unavailable Vulkan allocation failed with the wrong error type");
            }
            expect(rejected, "unavailable Vulkan allocation was accepted");
            return 77;
        }

        test_one_allocation(*allocator);
        test_independent_release_order(*allocator);
        test_repeated_lifetimes(*allocator);
        test_zero_byte_allocation(*allocator);
        test_oversized_allocation_rejected(*allocator);
        test_copy_rejected(*allocator);
        test_release_after_submission(*allocator);
        test_bounded_reuse_after_submission(*allocator);
        test_bounded_allocator_transition_stress(*allocator);
        test_explicit_shutdown_preserves_live_allocation(*allocator);
        std::cout << "Vulkan allocator lifetime tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
