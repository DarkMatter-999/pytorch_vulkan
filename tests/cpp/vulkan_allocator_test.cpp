#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_device_guard.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <ATen/ATen.h>
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

void test_allocation_lease_outlives_dataptr_and_rebind(c10::Allocator &allocator) {
    auto &platform = *pytorch_vulkan::platform();
    platform.synchronize_execution();
    const auto before = platform.live_resource_snapshot().allocations;
    auto tensor = at::empty({8}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto lease = pytorch_vulkan::allocation_lease(tensor.storage().data_ptr());
    auto view = at::empty({3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    view.set_(tensor.storage(), 1, {3}, {1});
    auto same_lease = pytorch_vulkan::allocation_lease(view.storage().data_ptr());
    expect(platform.live_resource_snapshot().allocations == before + 1, "lease core counted once");
    auto foreign = at::from_blob(tensor.data_ptr(), {8}, at::TensorOptions().dtype(at::kFloat));
    bool rejected = false;
    try { (void)pytorch_vulkan::allocation_lease(foreign.storage().data_ptr()); }
    catch (const c10::Error &) { rejected = true; }
    expect(rejected, "foreign payload lease was accepted");
    auto empty = allocator.allocate(0);
    auto zero_lease = pytorch_vulkan::allocation_lease(empty);
    expect(static_cast<bool>(zero_lease), "zero-byte lease was not safe to hold");
    tensor.resize_({32});
    tensor = at::Tensor();
    view = at::Tensor();
    platform.synchronize_execution();
    const auto during_rebind = platform.live_resource_snapshot().allocations;
    if (during_rebind != before + 1)
        throw std::runtime_error("old core did not survive rebind while lease held: " +
            std::to_string(during_rebind) + " vs " + std::to_string(before + 1));
    lease.reset(); same_lease.reset();
    platform.synchronize_execution();
    expect(platform.live_resource_snapshot().allocations == before,
           "last lease did not destroy old core exactly once");
}

void test_last_allocation_lease_defers_pending_destruction(c10::Allocator &allocator) {
    auto &platform = *pytorch_vulkan::platform();
    platform.synchronize_execution();
    const auto before = platform.live_resource_snapshot().allocations;
    auto data = allocator.allocate(4096);
    auto lease = pytorch_vulkan::allocation_lease(data);
    auto &buffer = pytorch_vulkan::allocation_buffer(data);
    auto &context = platform.execution_context();
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 4096, 0x12345678U);
    context.submit();
    data.clear();
    lease.reset();
    expect(platform.live_resource_snapshot().allocations == before + 1,
           "last lease destroyed a core still used by submitted work");
    context.wait();
    expect(platform.live_resource_snapshot().allocations == before,
           "pending core was not destroyed after synchronization");
}

void test_resize_retains_old_allocation_during_training_recording() {
    auto &platform = *pytorch_vulkan::platform();
    platform.synchronize_execution();
    auto tensor = at::empty({6}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto storage = tensor.storage();
    auto *storage_impl = storage.unsafeGetStorageImpl();
    const auto baseline = platform.live_resource_snapshot().allocations;
    platform.compute().begin_training_step();
    try {
        auto &old_buffer = pytorch_vulkan::allocation_buffer(storage.data_ptr());
        vkCmdFillBuffer(platform.execution_context().command_buffer(),
                        old_buffer.buffer(), 0, 6 * sizeof(float), 0x3f800000U);
        tensor.resize_({12});
        expect(tensor.storage().unsafeGetStorageImpl() == storage_impl &&
                   storage.unsafeGetStorageImpl() == storage_impl,
               "resize replaced StorageImpl instead of preserving aliases");
        expect(platform.live_resource_snapshot().allocations == baseline + 1,
               "old resize allocation was released before recorded commands completed");
        platform.compute().end_training_step();
    } catch (...) {
        platform.compute().cancel_training_step();
        throw;
    }

    expect(platform.live_resource_snapshot().allocations == baseline,
           "old resize allocation was not retired after training completion");
    auto &new_buffer = pytorch_vulkan::allocation_buffer(storage.data_ptr());
    VulkanBuffer host(platform,
                      6 * sizeof(float),
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    platform.copy_buffer_sync(new_buffer.buffer(), host.buffer(), 6 * sizeof(float));
    float result[6]{};
    host.read(result, sizeof(result));
    for (float value : result)
        expect(value == 1.0f,
               "resize prefix copy did not consume the queued old-allocation write");
}

void test_set_releases_old_allocation_only_after_recorded_consumer(bool rebind) {
    auto &platform = *pytorch_vulkan::platform();
    platform.synchronize_execution();
    const auto baseline = platform.live_resource_snapshot().allocations;
    auto tensor = at::empty({6}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = c10::GetAllocator(c10::DeviceType::PrivateUse1)->allocate(6 * sizeof(float));
    at::Tensor replacement;
    if (rebind)
        replacement = at::empty({2}, at::TensorOptions().dtype(at::kFloat).device(kDevice));

    auto &old_buffer = pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr());
    const VkBuffer old_handle = old_buffer.buffer();
    auto &output_buffer = pytorch_vulkan::allocation_buffer(output);
    const VkBuffer output_handle = output_buffer.buffer();
    auto &context = platform.execution_context();
    platform.compute().begin_training_step();
    try {
        vkCmdFillBuffer(context.command_buffer(), old_handle, 0, 6 * sizeof(float),
                        0x3f800000U);
        VkBufferCopy copy{};
        copy.size = 6 * sizeof(float);
        vkCmdCopyBuffer(context.command_buffer(), old_handle, output_handle, 1, &copy);

        // No Tensor, Storage, or owning allocation alias to the old buffer is
        // retained here. Only recorded GPU work may keep it alive.
        if (rebind)
            tensor.set_(replacement);
        else
            tensor.set_();
        const auto expected_pending_allocations = baseline + (rebind ? 3 : 2);
        const auto observed_pending_allocations =
            platform.live_resource_snapshot().allocations;
        const std::string pending_message =
            std::string(rebind
                            ? "set_(source) did not retain its command-owned old allocation: "
                            : "set_() did not retain its command-owned old allocation: ") +
            std::to_string(observed_pending_allocations) + " != " +
            std::to_string(expected_pending_allocations);
        expect(observed_pending_allocations == expected_pending_allocations,
               pending_message.c_str());
        platform.compute().end_training_step();
    } catch (...) {
        platform.compute().cancel_training_step();
        throw;
    }

    expect(platform.live_resource_snapshot().allocations == baseline + (rebind ? 2 : 1),
           rebind ? "set_(source) old allocation was not reclaimed after completion"
                  : "set_() old allocation was not reclaimed after completion");
    auto &completed_output = pytorch_vulkan::allocation_buffer(output);
    VulkanBuffer host(platform, 6 * sizeof(float),
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    platform.copy_buffer_sync(completed_output.buffer(), host.buffer(), 6 * sizeof(float));
    float result[6]{};
    host.read(result, sizeof(result));
    for (float value : result)
        expect(value == 1.0f,
               rebind ? "rebind consumer did not read the recorded old-allocation values"
                      : "reset consumer did not read the recorded old-allocation values");
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
        test_allocation_lease_outlives_dataptr_and_rebind(*allocator);
        test_last_allocation_lease_defers_pending_destruction(*allocator);
        test_resize_retains_old_allocation_during_training_recording();
        test_set_releases_old_allocation_only_after_recorded_consumer(false);
        test_set_releases_old_allocation_only_after_recorded_consumer(true);
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
