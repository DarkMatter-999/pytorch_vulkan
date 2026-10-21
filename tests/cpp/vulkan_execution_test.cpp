#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan/descriptor_arena.h"
#include "vulkan_tensor_layout.h"

#include <cstdint>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <new>
#include <deque>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct FakeExecution {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<VkFence, bool> fences;
    std::unordered_map<VkFence, VkResult> fence_overrides;
    std::vector<VkFence> submitted_fences;
    std::deque<VkResult> statuses;
    VkResult end_result = VK_SUCCESS;
    VkResult submit_result = VK_SUCCESS;
    VkResult idle_result = VK_SUCCESS;
    VkResult wait_result = VK_SUCCESS;
    std::function<void()> status_hook;
    std::function<void()> wait_hook;
    bool block_wait = false;
    bool release_wait = false;
    unsigned status_calls = 0;
    unsigned wait_calls = 0;
    unsigned submit_calls = 0;
    unsigned idle_calls = 0;
    unsigned quarantine_calls = 0;
    std::size_t quarantine_capacity = 4;
    unsigned pump_waits = 0;
    unsigned reservation_waits = 0;
    std::atomic<unsigned> fail_reservation_allocations{0};
    std::atomic<unsigned> fail_attachment_allocations{0};
    bool worker_failed = false;
    unsigned freed_buffers = 0;
    unsigned destroyed_fences = 0;
    unsigned destroyed_semaphores = 0;
    uint64_t next_handle = 10;
    std::array<std::unique_ptr<VulkanExecutionQuarantine>, 4> quarantines;
    std::array<std::unique_ptr<VulkanExecutionQuarantine>, 4> overflow_quarantines;

    void signal_all() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto &fence : fences) fence.second = true;
        release_wait = true;
        cv.notify_all();
    }
};

template <typename Handle> Handle fake_handle(uint64_t value) {
    return reinterpret_cast<Handle>(static_cast<uintptr_t>(value));
}

VkResult fake_allocate(void *cookie, VkDevice, const VkCommandBufferAllocateInfo *info,
                       VkCommandBuffer *buffers) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    for (uint32_t i = 0; i < info->commandBufferCount; ++i)
        buffers[i] = fake_handle<VkCommandBuffer>(++fake.next_handle);
    return VK_SUCCESS;
}
void fake_free(void *cookie, VkDevice, VkCommandPool, uint32_t count, const VkCommandBuffer *) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    fake.freed_buffers += count;
}
VkResult fake_create_fence(void *cookie, VkDevice, const VkFenceCreateInfo *,
                           const VkAllocationCallbacks *, VkFence *fence) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    *fence = fake_handle<VkFence>(++fake.next_handle);
    fake.fences[*fence] = false;
    return VK_SUCCESS;
}
void fake_destroy_fence(void *cookie, VkDevice, VkFence, const VkAllocationCallbacks *) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.destroyed_fences;
}
VkResult fake_create_semaphore(void *cookie, VkDevice, const VkSemaphoreCreateInfo *,
                              const VkAllocationCallbacks *, VkSemaphore *semaphore) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    *semaphore = fake_handle<VkSemaphore>(++fake.next_handle);
    return VK_SUCCESS;
}
void fake_destroy_semaphore(void *cookie, VkDevice, VkSemaphore, const VkAllocationCallbacks *) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.destroyed_semaphores;
}
VkResult fake_begin(void *, VkCommandBuffer, const VkCommandBufferBeginInfo *) {
    return VK_SUCCESS;
}
VkResult fake_end(void *cookie, VkCommandBuffer) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    return fake.end_result;
}
VkResult fake_reset_command(void *, VkCommandBuffer, VkCommandBufferResetFlags) {
    return VK_SUCCESS;
}
VkResult fake_reset_fences(void *cookie, VkDevice, uint32_t count, const VkFence *fences) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    for (uint32_t i = 0; i < count; ++i) fake.fences[fences[i]] = false;
    return VK_SUCCESS;
}
VkResult fake_submit(void *cookie, VkQueue, uint32_t, const VkSubmitInfo *, VkFence fence) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.submit_calls;
    fake.submitted_fences.push_back(fence);
    return fake.submit_result;
}
VkResult fake_fence_status(void *cookie, VkDevice, VkFence fence) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::function<void()> hook;
    VkResult result;
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        ++fake.status_calls;
        hook = fake.status_hook;
        result = fake.fences.at(fence) ? VK_SUCCESS : VK_NOT_READY;
        if (const auto override = fake.fence_overrides.find(fence);
            override != fake.fence_overrides.end()) result = override->second;
        if (!fake.statuses.empty()) {
            result = fake.statuses.front();
            fake.statuses.pop_front();
        }
    }
    if (hook) hook();
    return result;
}
VkResult fake_wait_fences(void *cookie, VkDevice, uint32_t count, const VkFence *fences, VkBool32,
                          uint64_t) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::function<void()> hook;
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        ++fake.wait_calls;
        hook = fake.wait_hook;
    }
    if (hook) hook();
    std::unique_lock<std::mutex> lock(fake.mutex);
    if (fake.block_wait) fake.cv.wait(lock, [&] { return fake.release_wait; });
    if (fake.wait_result == VK_SUCCESS)
        for (uint32_t i = 0; i < count; ++i) fake.fences[fences[i]] = true;
    return fake.wait_result;
}
VkResult fake_queue_idle(void *cookie, VkQueue) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.idle_calls;
    if (fake.idle_result == VK_SUCCESS)
        for (auto &fence : fake.fences) fence.second = true;
    return fake.idle_result;
}
void fake_quarantine(void *cookie, VulkanExecutionQuarantine *payload) noexcept {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.quarantine_calls;
    for (std::size_t i = 0; i < fake.quarantine_capacity; ++i) {
        if (!fake.quarantines[i]) {
            fake.quarantines[i].reset(payload);
            return;
        }
    }
    for (auto &slot : fake.overflow_quarantines) {
        if (!slot) {
            slot.reset(payload); // fixture leak-table models bounded sink saturation
            return;
        }
    }
    std::terminate();
}

void fake_callback_wait_begin(void *cookie) noexcept {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.pump_waits;
    fake.cv.notify_all();
}

void fake_reservation_wait_begin(void *cookie) noexcept {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.reservation_waits;
    fake.cv.notify_all();
}

bool fake_fail_reservation_allocation(void *cookie) noexcept {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    unsigned remaining = fake.fail_reservation_allocations.load();
    while (remaining && !fake.fail_reservation_allocations.compare_exchange_weak(
                             remaining, remaining - 1)) {}
    return remaining != 0;
}

bool fake_fail_attachment_allocation(void *cookie) noexcept {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    unsigned remaining = fake.fail_attachment_allocations.load();
    while (remaining && !fake.fail_attachment_allocations.compare_exchange_weak(
                             remaining, remaining - 1)) {}
    return remaining != 0;
}

VulkanExecutionOps fake_ops(FakeExecution &fake) {
    return {&fake, fake_allocate, fake_free, fake_create_fence, fake_destroy_fence,
            fake_create_semaphore, fake_destroy_semaphore, fake_begin, fake_end,
            fake_reset_command, fake_reset_fences, fake_submit, fake_fence_status,
            fake_wait_fences, fake_queue_idle, fake_callback_wait_begin,
            fake_reservation_wait_begin, fake_fail_reservation_allocation,
            fake_fail_attachment_allocation};
}

VulkanExecutionQuarantineSink fake_sink(FakeExecution &fake) {
    return {&fake, fake_quarantine};
}

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void expect_rejected(Function &&function, const char *message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    expect(rejected, message);
}

void test_invalid_states(VulkanExecutionContext &context) {
    expect(!context.recording(), "new context is recording");
    expect_rejected([&] { (void)context.command_buffer(); },
                    "inactive command_buffer was accepted");
    expect_rejected([&] { context.submit(); }, "inactive submit was accepted");
    expect_rejected([&] { context.defer_destruction([] {}); },
                    "inactive deferred callback was accepted");
    expect(!context.retain_until_completion([] {}),
           "inactive completion retention was accepted");
    context.wait();
    context.synchronize();
    context.begin();
    expect(context.recording(), "begin did not start recording");
    expect_rejected([&] { context.begin(); }, "nested begin was accepted");
    context.submit();
    context.begin();
    expect(context.recording(), "begin did not acquire reusable slot");
    context.cancel();
    expect(!context.recording(), "wait left context recording");
}

void test_noncoherent_mapped_range_calculation() {
    const auto middle = VulkanBuffer::mapped_range(97, 13, 1024, 64);
    expect(middle.offset == 64 && middle.size == 64,
           "noncoherent range did not cover and atom-align the requested bytes");
    const auto tail = VulkanBuffer::mapped_range(1000, 24, 1024, 64);
    expect(tail.offset == 960 && tail.size == VK_WHOLE_SIZE,
           "allocation-tail range did not use legal whole-size mapping");
    const auto exact = VulkanBuffer::mapped_range(64, 64, 1024, 64);
    expect(exact.offset == 64 && exact.size == 64,
           "already aligned range was changed");
    expect_rejected([&] { (void)VulkanBuffer::mapped_range(1020, 8, 1024, 64); },
                    "out-of-allocation mapped range was accepted");
    expect_rejected([&] { (void)VulkanBuffer::mapped_range(0, 4, 1024, 0); },
                    "zero atom size was accepted");
}

void test_host_buffer_offset_mapping(VulkanPlatform &platform) {
    VulkanBuffer buffer(platform, 128,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const uint32_t expected[] = {0x10203040U, 0x55667788U};
    uint32_t actual[] = {0U, 0U};
    buffer.write(expected, sizeof(expected), 12);
    buffer.read(actual, sizeof(actual), 12);
    expect(actual[0] == expected[0] && actual[1] == expected[1],
           "offset host mapping did not address requested buffer bytes");
}

void test_deferred_callback(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    bool called = false;
    context.begin();
    context.defer_destruction([&] { called = true; });
    context.submit();
    context.synchronize();
    expect(called, "deferred callback was not retired by synchronize");
}

void test_submitted_callback_retirement(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.retain_until_completion([&] { ++callback_count; });
    context.submit();
    expect(callback_count == 0, "submitted callback ran before completion");
    expect(context.pending_count() == 1, "submitted callback was not retained");
    context.retire_completed();
    context.wait();
    expect(callback_count == 1, "submitted callback did not run exactly once");
    context.wait();
    expect(callback_count == 1, "submitted callback ran more than once");
}

void test_callback_reentrancy_does_not_report_completion(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    bool query_reported_complete = false;
    bool synchronize_rejected = false;
    context.begin();
    context.defer_destruction([&] {
        query_reported_complete = context.query_complete();
        try {
            context.synchronize();
        } catch (const std::logic_error &) {
            synchronize_rejected = true;
        }
    });
    context.submit();
    context.synchronize();
    expect(!query_reported_complete,
           "callback reentrant query reported its own cleanup complete");
    expect(synchronize_rejected,
           "callback reentrant synchronize did not reject self-wait");
}

void test_controlled_execution_ops_retirement() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink,
                                   queue_mutex);
    int cleanup_calls = 0;
    bool queue_was_unlocked = false;
    bool reentrant_query_complete = true;
    bool reentrant_sync_rejected = false;
    context.begin();
    context.defer_destruction([&] {
        ++cleanup_calls;
        std::thread observer([&] {
            queue_was_unlocked = queue_mutex.try_lock();
            if (queue_was_unlocked) queue_mutex.unlock();
        });
        observer.join();
        reentrant_query_complete = context.query_complete();
        try {
            context.synchronize();
        } catch (const std::logic_error &) {
            reentrant_sync_rejected = true;
        }
    });
    context.submit();
    expect(!context.query_complete(), "fake incomplete fence retired cleanup");
    expect(cleanup_calls == 0, "fake incomplete fence released callback captures");
    fake.signal_all();
    expect(context.query_complete(), "signaled fake fence did not retire cleanup");
    expect(cleanup_calls == 1, "fake completion did not run cleanup exactly once");
    expect(queue_was_unlocked, "cleanup callback ran while queue mutex was held");
    expect(!reentrant_query_complete, "reentrant fake query reported completion");
    expect(reentrant_sync_rejected, "reentrant fake synchronize did not reject self-wait");
}

void test_controlled_cleanup_errors_are_persistent_and_nonblocking_to_cleanup() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(11), fake_handle<VkQueue>(12),
                                   fake_handle<VkCommandPool>(13), ops, sink,
                                   queue_mutex);
    int remaining_cleanup_calls = 0;
    context.begin();
    context.defer_destruction([] { throw std::runtime_error("controlled cleanup error"); });
    context.defer_destruction([&] { ++remaining_cleanup_calls; });
    context.submit();
    fake.signal_all();
    expect_rejected([&] { (void)context.query_complete(); },
                    "throwing cleanup did not surface through query");
    expect(remaining_cleanup_calls == 1,
           "throwing cleanup prevented later safe cleanup");
    expect_rejected([&] { (void)context.query_complete(); },
                    "cleanup error was not persistent on repeated query");
    expect(remaining_cleanup_calls == 1,
           "repeated query invoked a completed cleanup more than once");
}

void test_fake_transfer_record_success_only_publication() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(21), fake_handle<VkQueue>(22),
                                   fake_handle<VkCommandPool>(23), ops, sink, queue_mutex);
    std::atomic<int> owner_released{0};
    auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
        delete static_cast<int *>(value);
        ++owner_released;
    });
    std::atomic<int> publications{0};
    std::atomic<int> released_during_publication{-1};
    auto reservation = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(reservation), {owner}, [&] {
            released_during_publication = owner_released.load();
            ++publications;
        });
    }
    owner.reset();
    context.submit();
    expect(!context.query_complete(), "unready transfer fence reported completion");
    expect(publications == 0 && owner_released == 0,
           "pending transfer published or released its exact owner");
    fake.signal_all();
    expect(context.query_complete(), "successful transfer fence remained pending");
    expect(publications == 1 && owner_released == 1,
           "confirmed transfer did not publish once then release its owner");
    expect(released_during_publication == 0,
           "transfer owner was released before publication completed");
}

void test_fake_transfer_cancel_and_failed_submit_never_publish() {
    for (const char *failure : {"cancel", "end", "submit"}) {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(31), fake_handle<VkQueue>(32),
                                       fake_handle<VkCommandPool>(33), ops, sink, queue_mutex);
        std::atomic<int> owner_released{0};
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value);
            ++owner_released;
        });
        std::atomic<int> publications{0};
        auto reservation = context.reserve_transfer();
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++publications; });
        }
        owner.reset();
        bool synchronous_error = false;
        try {
            if (std::string(failure) == "cancel") {
                context.cancel();
            } else {
                if (std::string(failure) == "end") fake.end_result = VK_ERROR_UNKNOWN;
                else fake.submit_result = VK_ERROR_UNKNOWN;
                context.submit();
            }
        } catch (const std::exception &) {
            synchronous_error = true;
        }
        expect(synchronous_error, "transfer failure did not remain observable");
        expect(publications == 0, "cancelled or failed transfer published success");
        expect(owner_released == 1, "confirmed-safe failed transfer did not release owner");
        expect_rejected([&] { (void)context.query_complete(); },
                        "transfer contract error was not sticky");
        expect(publications == 0, "repeated query reran failed transfer publication");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::atomic<int> released{0}, published{0};
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(35), fake_handle<VkQueue>(36),
                                           fake_handle<VkCommandPool>(37), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
                delete static_cast<int *>(value); ++released;
            });
            auto reservation = context.reserve_transfer();
            context.begin();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
            }
            owner.reset();
            fake.submit_result = VK_ERROR_UNKNOWN;
            fake.idle_result = VK_ERROR_UNKNOWN;
            expect_rejected([&] { context.submit(); },
                            "failed queue-idle recovery did not reject submitted transfer");
            expect(published == 0 && released == 0 && fake.quarantine_calls == 1,
                   "uncertain submit failure released or published transfer owner");
        }
        expect(released == 0, "uncertain submit owner escaped quarantine at context teardown");
        fake.quarantines[0].reset();
        expect(released == 1, "submit quarantine did not retain owner until fake completion");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::atomic<int> released{0}, published{0};
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(38), fake_handle<VkQueue>(39),
                                           fake_handle<VkCommandPool>(40), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
                delete static_cast<int *>(value); ++released;
            });
            auto reservation = context.reserve_transfer();
            context.begin();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
            }
            owner.reset();
        }
        expect(published == 0 && released == 1,
               "unsubmitted teardown published success or leaked safely unused owner");
    }
}

void test_fake_transfer_fence_error_recovery_and_quarantine() {
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(41), fake_handle<VkQueue>(42),
                                       fake_handle<VkCommandPool>(43), ops, sink, queue_mutex);
        std::atomic<int> released{0}, published{0};
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        auto reservation = context.reserve_transfer();
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        }
        owner.reset();
        context.submit();
        fake.statuses = {VK_ERROR_UNKNOWN};
        fake.idle_result = VK_SUCCESS;
        expect_rejected([&] { (void)context.query_complete(); },
                        "fence query error was not reported after idle recovery");
        expect(published == 0 && released == 1,
               "idle recovery incorrectly published or leaked safe transfer owners");
        expect_rejected([&] { (void)context.query_complete(); },
                        "fence query error was not persistent");
        expect(published == 0, "fence error recovery invoked success action");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::atomic<int> released{0}, published{0};
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(51), fake_handle<VkQueue>(52),
                                           fake_handle<VkCommandPool>(53), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
                delete static_cast<int *>(value); ++released;
            });
            auto reservation = context.reserve_transfer();
            context.begin();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
            }
            owner.reset();
            context.submit();
            fake.statuses = {VK_ERROR_UNKNOWN};
            fake.idle_result = VK_ERROR_UNKNOWN;
            expect_rejected([&] { (void)context.query_complete(); },
                            "failed idle recovery did not reject uncertain completion");
            expect(published == 0 && released == 0 && fake.quarantine_calls == 1,
                   "uncertain transfer was published, released, or not quarantined");
        }
        expect(released == 0, "context destruction released uncertain transfer owner");
        fake.quarantines[0].reset(); // fixture declares fake device work complete
        expect(released == 1, "fake quarantine reclaim did not release transfer owner");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(55), fake_handle<VkQueue>(56),
                                       fake_handle<VkCommandPool>(57), ops, sink, queue_mutex);
        std::atomic<int> released{0}, published{0};
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        auto reservation = context.reserve_transfer();
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        }
        owner.reset();
        context.submit();
        fake.fence_overrides[fake.submitted_fences[0]] = VK_ERROR_DEVICE_LOST;
        expect_rejected([&] { (void)context.query_complete(); },
                        "device-lost fence query did not invalidate transfer context");
        expect(published == 0 && released == 0 && fake.quarantine_calls == 1,
               "device-lost transfer owner was published or released");
        fake.quarantines[0].reset(); // fixture declares fake work complete
        expect(released == 1, "device-lost quarantine did not retain the transfer owner");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(58), fake_handle<VkQueue>(59),
                                       fake_handle<VkCommandPool>(60), ops, sink, queue_mutex);
        std::atomic<int> released{0}, published{0};
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        auto reservation = context.reserve_transfer();
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        }
        owner.reset();
        context.submit();
        fake.wait_result = VK_ERROR_UNKNOWN;
        fake.idle_result = VK_SUCCESS;
        std::exception_ptr wait_error;
        try { context.synchronize(); } catch (...) { wait_error = std::current_exception(); }
        expect(static_cast<bool>(wait_error),
               "fence wait error was not reported after successful idle recovery");
        expect(published == 0 && released == 1,
               "wait idle recovery published or retained safe transfer owner");
        std::exception_ptr repeated_wait_error;
        try { context.synchronize(); }
        catch (...) { repeated_wait_error = std::current_exception(); }
        expect(repeated_wait_error == wait_error, "fence wait error identity was not persistent");
        expect(published == 0, "wait error recovery invoked success action");
    }
    {
        FakeExecution fake;
        fake.quarantine_capacity = 0;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::atomic<int> released{0}, published{0};
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(64), fake_handle<VkQueue>(65),
                                           fake_handle<VkCommandPool>(66), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
                delete static_cast<int *>(value); ++released;
            });
            auto reservation = context.reserve_transfer();
            context.begin();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
            }
            owner.reset();
            context.submit();
            fake.fence_overrides[fake.submitted_fences[0]] = VK_ERROR_DEVICE_LOST;
            expect_rejected([&] { (void)context.query_complete(); },
                            "saturated quarantine context did not fail closed");
            expect(published == 0 && released == 0 && fake.overflow_quarantines[0],
                   "saturated fake sink did not retain whole transfer payload");
        }
        fake.overflow_quarantines[0].reset();
        expect(released == 1, "saturated quarantine retained owner until fixture completion");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::atomic<int> released{0}, published{0};
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(67), fake_handle<VkQueue>(68),
                                           fake_handle<VkCommandPool>(69), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
                delete static_cast<int *>(value); ++released;
            });
            auto reservation = context.reserve_transfer();
            context.begin();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
            }
            owner.reset();
            context.submit();
            fake.wait_result = VK_ERROR_UNKNOWN;
            fake.idle_result = VK_ERROR_UNKNOWN;
            expect_rejected([&] { context.synchronize(); },
                            "failed wait recovery did not reject uncertain transfer");
            expect(published == 0 && released == 0 && fake.quarantine_calls == 1,
                   "uncertain wait failure released or published transfer owner");
        }
        expect(released == 0, "wait recovery quarantine lost owner on context destruction");
        fake.quarantines[0].reset();
        expect(released == 1, "wait recovery quarantine did not retain owner to completion");
    }
}

void test_fake_recording_query_and_single_protected_poll() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    context.begin();
    expect(!context.query_complete(), "recording query must be false, not an exception");
    expect(context.recording() && fake.status_calls == 0 && fake.wait_calls == 0,
           "recording query mutated or waited for recording");
    context.submit();
    fake.statuses = {VK_NOT_READY, VK_SUCCESS};
    expect(!context.query_complete(), "incomplete poll reported completion");
    expect(fake.status_calls == 1, "query polled a second time outside its device phase");
    expect(context.query_complete(), "later successful poll did not complete");
    expect(fake.status_calls == 2 && fake.wait_calls == 0, "query performed a blocking wait");
    context.begin();
    context.cancel();
    expect(context.query_complete(), "cancelled recording remained pending");
}

void test_fake_ready_cleanup_at_destruction() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    int calls = 0;
    {
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        context.begin();
        context.defer_destruction([&] { ++calls; });
        context.submit();
        fake.signal_all();
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        context.poll_device_locked();
        expect(calls == 0, "locked retirement invoked cleanup");
    }
    expect(calls == 1, "destructor dropped already-ready cleanup");
}

void test_fake_failed_submit_drains_abandoned_cleanup() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    int calls = 0;
    context.begin();
    context.defer_destruction([&] {
        ++calls;
        expect(!context.query_complete(), "abandon cleanup did not remain visible");
        throw std::runtime_error("secondary cleanup error");
    });
    fake.end_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    std::string error;
    try { context.submit(); } catch (const std::exception &e) { error = e.what(); }
    expect(error.find("end Vulkan execution command buffer") != std::string::npos,
           "submit lost its original recording error");
    expect(calls == 1, "failed public submit did not drain safe abandoned cleanup");
    expect_rejected([&] { context.synchronize(); }, "cleanup error was not saved for synchronize");
}

void test_fake_compute_exit_drains_without_device_poll() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    int calls = 0;
    {
        VulkanCallbackDrainOnExit drain(context);
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        context.begin_locked();
        context.defer_destruction([&] { ++calls; });
        context.cancel_locked();
        context.begin_locked();
    }
    expect(calls == 1, "compute exit failed to drain ready A while B was recording");
    expect(context.recording(), "ready-only drain mutated B's recording");
    expect(fake.status_calls == 0 && fake.wait_calls == 0, "exit drain polled/waited device");
    context.cancel();
    context.begin();
    context.submit();
    fake.statuses = {VK_ERROR_OUT_OF_HOST_MEMORY};
    { VulkanCallbackDrainOnExit drain(context); }
    expect(fake.status_calls == 0, "exit guard swallowed an injected device query error");
    expect_rejected([&] { context.query_complete(); }, "explicit query lost its device error");
}

bool other_thread_can_lock(std::mutex &mutex) {
    bool acquired = false;
    std::thread observer([&] {
        acquired = mutex.try_lock();
        if (acquired) mutex.unlock();
    });
    observer.join();
    return acquired;
}

struct ControlledGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
    bool failed = false;

    void pause() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return entered || failed; });
        expect(entered && !failed, "worker failed before controlled pause");
    }
    void open() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

class JoinedFakeWorkers final {
  public:
    JoinedFakeWorkers(ControlledGate &gate, FakeExecution &fake) : gate_(gate), fake_(fake) {
        threads_.reserve(3);
    }
    ~JoinedFakeWorkers() {
        gate_.open();
        fake_.signal_all();
        for (auto &thread : threads_) if (thread.joinable()) thread.join();
    }
    template <typename Function> void spawn(Function &&function) {
        threads_.emplace_back([this, fn = std::forward<Function>(function)]() mutable {
            try { fn(); } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    if (!error_) error_ = std::current_exception();
                }
                {
                    std::lock_guard<std::mutex> lock(gate_.mutex);
                    gate_.failed = true;
                    gate_.cv.notify_all();
                }
                {
                    std::lock_guard<std::mutex> lock(fake_.mutex);
                    fake_.worker_failed = true;
                    fake_.cv.notify_all();
                }
            }
        });
    }
    void join() {
        for (auto &thread : threads_) if (thread.joinable()) thread.join();
        if (error_) std::rethrow_exception(error_);
    }
  private:
    ControlledGate &gate_;
    FakeExecution &fake_;
    std::mutex error_mutex_;
    std::exception_ptr error_;
    std::vector<std::thread> threads_;
};

void test_fake_transfer_reservation_capacity_and_shutdown_lifetime() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(61), fake_handle<VkQueue>(62),
                                   fake_handle<VkCommandPool>(63), ops, sink, queue_mutex);
    fake.fail_reservation_allocations = 1;
    bool allocation_failed = false;
    try { (void)context.reserve_transfer(); }
    catch (const std::bad_alloc &) { allocation_failed = true; }
    expect(allocation_failed, "injected reservation allocation failure was not surfaced");
    auto first = context.reserve_transfer();
    auto second = context.reserve_transfer();
    ControlledGate gate;
    JoinedFakeWorkers workers(gate, fake);
    std::atomic<bool> third_returned{false};
    workers.spawn([&] {
        auto third = context.reserve_transfer();
        third_returned = true;
    });
    {
        std::unique_lock<std::mutex> lock(fake.mutex);
        fake.cv.wait(lock, [&] { return fake.reservation_waits != 0 || fake.worker_failed; });
        expect(!fake.worker_failed, "third reservation worker failed before capacity wait");
    }
    expect(!third_returned, "third reservation passed while both slots were occupied");
    expect(other_thread_can_lock(queue_mutex),
           "reservation capacity wait retained the shared queue lock");
    first = VulkanTransferReservation{};
    workers.join();
    expect(third_returned, "released reservation did not admit blocked waiter");
    (void)second;

    VulkanTransferReservation outliving_token;
    {
        VulkanExecutionContext short_lived(fake_handle<VkDevice>(71), fake_handle<VkQueue>(72),
                                           fake_handle<VkCommandPool>(73), ops, sink, queue_mutex);
        outliving_token = short_lived.reserve_transfer();
    }
    outliving_token = VulkanTransferReservation{};
}

void test_fake_transfer_capacity_waits_for_fence_and_capture_release() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(75), fake_handle<VkQueue>(76),
                                   fake_handle<VkCommandPool>(77), ops, sink, queue_mutex);
    std::atomic<int> published{0}, released{0};
    fake.block_wait = true;
    fake.wait_hook = [&] {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.cv.notify_all();
    };
    auto submit_one = [&](int id) {
        auto reservation = context.reserve_transfer();
        auto owner = std::shared_ptr<void>(new int(id), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        }
        owner.reset();
        context.submit();
    };
    submit_one(1);
    submit_one(2);
    ControlledGate gate;
    JoinedFakeWorkers workers(gate, fake);
    std::atomic<bool> third_returned{false};
    workers.spawn([&] { auto third = context.reserve_transfer(); third_returned = true; });
    {
        std::unique_lock<std::mutex> lock(fake.mutex);
        fake.cv.wait(lock, [&] { return fake.wait_calls != 0 || fake.worker_failed; });
        expect(!fake.worker_failed, "capacity worker failed before oldest-fence wait");
    }
    expect(!third_returned && published == 0 && released == 0,
           "capacity pressure bypassed pending fence/publication lifetime");
    fake.signal_all();
    workers.join();
    expect(third_returned && published == 1 && released == 1,
           "capacity did not advance after oldest fence and owner release");
    context.synchronize();
    expect(published == 2 && released == 2,
           "later transfer did not remain bounded and retire after oldest backpressure");
}

void test_fake_transfer_publication_submission_and_operation_order() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(81), fake_handle<VkQueue>(82),
                                   fake_handle<VkCommandPool>(83), ops, sink, queue_mutex);
    std::vector<int> order;
    auto first = context.reserve_transfer();
    auto second = context.reserve_transfer();
    auto attach = [&](VulkanTransferReservation &&reservation, int value) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(reservation), {}, [&] { order.push_back(value); });
    };
    context.begin();
    attach(std::move(first), 1);
    context.submit();
    context.begin();
    attach(std::move(second), 2);
    context.submit();
    expect(fake.submitted_fences.size() == 2, "fake did not observe both submissions");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.fence_overrides[fake.submitted_fences[0]] = VK_NOT_READY;
        fake.fence_overrides[fake.submitted_fences[1]] = VK_SUCCESS;
    }
    expect(!context.query_complete(), "earlier unready submission was ignored");
    expect(order.empty(), "later ring slot published ahead of earlier submission");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.fence_overrides[fake.submitted_fences[0]] = VK_SUCCESS;
    }
    expect(context.query_complete(), "all successful transfer submissions did not complete");
    expect(order == std::vector<int>({1, 2}),
           "success callbacks did not follow submission then operation ID order");
    auto third = context.reserve_transfer();
    auto fourth = context.reserve_transfer();
    context.begin();
    attach(std::move(third), 3);
    attach(std::move(fourth), 4);
    context.submit();
    fake.signal_all();
    expect(context.query_complete(), "same-recording transfer callbacks did not complete");
    expect(order == std::vector<int>({1, 2, 3, 4}),
           "multiple transfer callbacks in one recording lost operation order");
}

void test_fake_transfer_capacity_flushes_open_recording() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(91), fake_handle<VkQueue>(92),
                                   fake_handle<VkCommandPool>(93), ops, sink, queue_mutex);
    std::atomic<int> published{0}, released{0};
    fake.block_wait = true;
    fake.wait_hook = [&] {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.cv.notify_all();
    };
    context.begin();
    for (int id = 0; id < 2; ++id) {
        auto reservation = context.reserve_transfer();
        auto owner = std::shared_ptr<void>(new int(id), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
    }
    ControlledGate gate;
    JoinedFakeWorkers workers(gate, fake);
    std::atomic<bool> third_returned{false};
    workers.spawn([&] {
        auto third = context.reserve_transfer([&] { context.submit_locked(); });
        third_returned = true;
    });
    {
        std::unique_lock<std::mutex> lock(fake.mutex);
        fake.cv.wait(lock, [&] { return fake.wait_calls != 0 || fake.worker_failed; });
        expect(!fake.worker_failed, "capacity worker failed before flushed fence wait");
    }
    expect(fake.submit_calls == 1 && !third_returned && published == 0,
           "capacity pressure did not flush the open recording before waiting");
    fake.signal_all();
    workers.join();
    expect(third_returned && published == 2 && released == 2,
           "flushed recording did not publish and free capacity");
}

void test_fake_teardown_after_queue_idle_never_publishes() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    std::atomic<int> published{0}, released{0};
    {
        VulkanExecutionContext context(fake_handle<VkDevice>(95), fake_handle<VkQueue>(96),
                                       fake_handle<VkCommandPool>(97), ops, sink, queue_mutex);
        auto reservation = context.reserve_transfer();
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        }
        owner.reset();
        context.submit();
        expect(context.teardown_with_services({}),
               "fake teardown unexpectedly left the context unsafe");
        expect(published == 0 && released == 1,
               "queue-idle teardown published success or leaked its owner");
    }
    expect(published == 0, "context destruction published success after queue idle");
}

void test_fake_transfer_attachment_allocation_failure_is_atomic() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(101), fake_handle<VkQueue>(102),
                                   fake_handle<VkCommandPool>(103), ops, sink, queue_mutex);
    std::atomic<int> published{0}, released{0}, legacy{0}, action_destroyed{0};
    auto make_owner = [&](int id) {
        return std::shared_ptr<void>(new int(id), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
    };
    // Existing pending work plus legacy cleanup must be untouched by a failed attachment.
    auto first = context.reserve_transfer();
    context.begin();
    {
        std::vector<std::shared_ptr<void>> owners{make_owner(1)};
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(first), std::move(owners), [&] { ++published; });
    }
    expect(context.retain_until_completion([&] { ++legacy; }), "legacy cleanup was not retained");
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.submit_locked();
    }
    auto second = context.reserve_transfer();
    context.begin();
    const auto pending_before = context.pending_count();
    std::vector<std::shared_ptr<void>> owners{make_owner(2)};
    auto sentinel = std::shared_ptr<int>(new int(0), [&](int *value) {
        delete value; ++action_destroyed;
    });
    std::function<void()> action = [&, sentinel] { ++published; };
    sentinel.reset();
    fake.fail_attachment_allocations = 1;
    bool bad_alloc = false;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        try { context.defer_transfer(std::move(second), std::move(owners), std::move(action)); }
        catch (const std::bad_alloc &) { bad_alloc = true; }
    }
    expect(bad_alloc, "forced attachment allocation failure did not reach the caller");
    expect(second && owners.size() == 1 && static_cast<bool>(action),
           "failed attachment consumed the caller's reservation or captures");
    expect(released == 0 && action_destroyed == 0,
           "failed attachment destroyed captures under the caller's queue lock");
    expect(context.pending_count() == pending_before,
           "failed attachment left a partially attached record");
    expect(published == 0 && legacy == 0, "failed attachment disturbed pending work");
    // The same live reservation can still attach, and the record is not duplicated.
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(second), std::move(owners), std::move(action));
    }
    expect(!second && owners.empty(), "successful retry did not consume the reservation");
    context.submit();
    fake.signal_all();
    context.synchronize();
    expect(published == 2 && legacy == 1 && released == 2 && action_destroyed == 1,
           "attachment retry or earlier pending work did not complete exactly once");

    // Abandoning a reservation after a failed attachment frees capacity.
    auto third = context.reserve_transfer();
    context.begin();
    fake.fail_attachment_allocations = 1;
    {
        std::vector<std::shared_ptr<void>> abandoned_owners{make_owner(3)};
        std::lock_guard<std::mutex> lock(queue_mutex);
        expect_rejected([&] { context.defer_transfer(std::move(third), std::move(abandoned_owners)); },
                        "second forced attachment failure was accepted");
    }
    third = VulkanTransferReservation();
    context.cancel();
    expect(released == 3, "abandoned owner was not released after the failed attachment");
    auto a = context.reserve_transfer();
    auto b = context.reserve_transfer();
    expect(a && b, "failed attachment leaked reservation capacity");
}

void test_fake_transfer_publisher_and_capture_visibility() {
    for (const bool pause_capture : {false, true}) {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(91), fake_handle<VkQueue>(92),
                                       fake_handle<VkCommandPool>(93), ops, sink, queue_mutex);
        ControlledGate gate;
        JoinedFakeWorkers workers(gate, fake);
        std::atomic<int> published{0}, released{0};
        std::atomic<bool> locks_free{false};
        std::atomic<bool> reentrant_query_incomplete{false}, self_sync_rejected{false};
        auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
            ++released;
            locks_free = other_thread_can_lock(queue_mutex);
            if (pause_capture) gate.pause();
            delete static_cast<int *>(value);
        });
        auto reservation = context.reserve_transfer();
        context.begin();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            context.defer_transfer(std::move(reservation), {owner}, [&] {
                ++published;
                if (!pause_capture) gate.pause();
                reentrant_query_incomplete = !context.query_complete();
                try { context.synchronize(); }
                catch (const std::logic_error &) { self_sync_rejected = true; }
            });
        }
        owner.reset();
        context.submit();
        fake.signal_all();
        workers.spawn([&] { (void)context.query_complete(); });
        gate.wait_entered();
        std::promise<bool> queried;
        auto query_result = queried.get_future();
        workers.spawn([&, promise = std::move(queried)]() mutable {
            promise.set_value(context.query_complete());
        });
        expect(!query_result.get(), "foreign query completed during transfer pump");
        std::atomic<bool> sync_done{false};
        workers.spawn([&] { context.synchronize(); sync_done = true; });
        {
            std::unique_lock<std::mutex> lock(fake.mutex);
            fake.cv.wait(lock, [&] { return fake.pump_waits != 0 || fake.worker_failed; });
            expect(!fake.worker_failed, "foreign synchronize failed before pump wait");
        }
        expect(!context.query_complete() && !sync_done,
               "transfer query/synchronize ignored active action or capture release");
        gate.open();
        workers.join();
        expect(sync_done && published == 1 && released == 1,
               "transfer action/capture did not finish exactly once");
        expect(locks_free, "transfer capture release inherited queue lock");
        expect(reentrant_query_incomplete && self_sync_rejected,
               "transfer action reentrancy did not observe active pump");
        expect(context.query_complete(), "transfer remained pending after capture release");
    }
}

void test_fake_transfer_invalidation_quarantines_active_publisher() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(111), fake_handle<VkQueue>(112),
                                   fake_handle<VkCommandPool>(113), ops, sink, queue_mutex);
    ControlledGate gate;
    JoinedFakeWorkers workers(gate, fake);
    std::atomic<int> published{0}, released{0};
    auto make_owner = [&] {
        return std::shared_ptr<void>(new int(1), [&](void *value) {
            delete static_cast<int *>(value); ++released;
        });
    };
    auto first_owner = make_owner();
    auto second_owner = make_owner();
    auto first = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(first), {first_owner}, [&] {
            ++published;
            gate.pause();
        });
    }
    context.submit();
    auto second = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(second), {second_owner}, [&] { ++published; });
    }
    context.submit();
    first_owner.reset();
    second_owner.reset();
    fake.signal_all();
    workers.spawn([&] {
        try { (void)context.query_complete(); }
        catch (const std::exception &) { /* expected after concurrent invalidation */ }
    });
    gate.wait_entered();
    fake.idle_result = VK_ERROR_DEVICE_LOST;
    expect(!context.teardown_with_services({}), "device-lost teardown claimed success");
    expect(fake.quarantine_calls == 1 && released == 0 && published == 1,
           "invalidation did not quarantine publishing/ready transfer records");
    gate.open();
    workers.join();
    expect(released == 0 && published == 1,
           "publisher overwrote quarantine or released owners after invalidation");
    fake.quarantines[0].reset(); // fixture declares fake device work complete
    expect(released == 2, "quarantined publisher records did not retain both owners");
}

void test_fake_transfer_throw_is_sticky_and_does_not_skip_safe_work() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(101), fake_handle<VkQueue>(102),
                                   fake_handle<VkCommandPool>(103), ops, sink, queue_mutex);
    std::atomic<int> throwing_calls{0}, later_calls{0}, cleanup_calls{0}, released{0};
    auto first_owner = std::shared_ptr<void>(new int(1), [&](void *value) {
        delete static_cast<int *>(value); ++released;
    });
    auto second_owner = std::shared_ptr<void>(new int(2), [&](void *value) {
        delete static_cast<int *>(value); ++released;
    });
    auto first = context.reserve_transfer();
    auto second = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(first), {first_owner}, [&] {
            ++throwing_calls; throw std::runtime_error("transfer publish failure");
        });
        context.defer_transfer(std::move(second), {second_owner}, [&] { ++later_calls; });
        context.defer_destruction([&] { ++cleanup_calls; });
    }
    first_owner.reset();
    second_owner.reset();
    context.submit();
    fake.signal_all();
    std::exception_ptr original;
    try { (void)context.query_complete(); }
    catch (...) { original = std::current_exception(); }
    expect(static_cast<bool>(original), "throwing success action did not surface");
    expect(throwing_calls == 1 && later_calls == 1 && cleanup_calls == 1 && released == 2,
           "publish failure skipped later safe action/legacy cleanup or owner release");
    std::exception_ptr repeated;
    try { (void)context.query_complete(); }
    catch (...) { repeated = std::current_exception(); }
    expect(repeated == original, "transfer publication exception was not persistent");
    expect(throwing_calls == 1 && later_calls == 1,
           "repeated query invoked a completed success action again");
    std::exception_ptr sync_error;
    try { context.synchronize(); }
    catch (...) { sync_error = std::current_exception(); }
    expect(sync_error == original, "synchronize did not preserve the first transfer exception");
    expect(throwing_calls == 1 && later_calls == 1 && cleanup_calls == 1,
           "repeated synchronize reran already-completed transfer or legacy callbacks");
}

void test_fake_paused_callback_and_last_capture_visibility() {
    for (const bool pause_capture : {false, true}) {
        std::atomic<int> cleanup{0};
        std::atomic<int> destroyed_owner{0};
        std::atomic<bool> locks_free{false};
        std::atomic<bool> reentrant_incomplete{false};
        std::atomic<bool> sync_done{false};
        ControlledGate gate;
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::mutex transfer_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        JoinedFakeWorkers workers(gate, fake);
        const auto pause = [&] {
            locks_free = other_thread_can_lock(queue_mutex) && other_thread_can_lock(transfer_mutex);
            reentrant_incomplete = !context.query_complete();
            gate.pause();
        };
        auto owner = std::shared_ptr<int>(new int(1), [&](int *p) {
            ++destroyed_owner;
            if (pause_capture) pause();
            delete p;
        });
        context.begin();
        context.defer_destruction([&, owner] {
            ++cleanup;
            if (!pause_capture) pause();
        });
        owner.reset();
        context.submit();
        fake.signal_all();
        workers.spawn([&] { (void)context.query_complete(); });
        gate.wait_entered();
        std::promise<bool> queried;
        auto query_result = queried.get_future();
        workers.spawn([&, promise = std::move(queried)]() mutable {
            promise.set_value(context.query_with_flush([] { throw std::runtime_error("foreign flush"); }));
        });
        expect(!query_result.get(), "foreign query completed while callback/capture was paused");
        workers.spawn([&] {
            context.synchronize_with_flush([] {});
            sync_done = true;
        });
        {
            std::unique_lock<std::mutex> lock(fake.mutex);
            fake.cv.wait(lock, [&] { return fake.pump_waits != 0 || fake.worker_failed; });
            expect(!fake.worker_failed, "synchronize worker failed before pump wait");
        }
        expect(!context.query_complete() && !sync_done,
               "synchronize returned before callback/capture release");
        expect(locks_free && reentrant_incomplete, "callback/capture release retained a backend lock");
        gate.open();
        workers.join();
        expect(sync_done && cleanup == 1 && destroyed_owner == 1,
               "paused cleanup did not finish exactly once");
        expect(context.query_complete(), "completion remained pending after last capture release");
    }
}

void test_fake_boundary_excludes_producer_until_device_cutoff() {
    for (const bool synchronize : {false, true}) {
        std::atomic<bool> admitted{false};
        ControlledGate gate;
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        JoinedFakeWorkers workers(gate, fake);
        std::promise<void> attempted;
        auto attempt = attempted.get_future();
        fake.status_hook = [&] { expect(!admitted, "producer entered between flush and poll"); };
        fake.wait_hook = [&] { expect(!admitted, "producer entered between flush and wait"); };
        context.begin();
        const auto flush_locked = [&] {
            context.submit_locked();
            fake.signal_all();
            workers.spawn([&, promise = std::move(attempted)]() mutable {
                promise.set_value();
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.begin_locked();
                admitted = true;
            });
            attempt.get();
        };
        if (synchronize) context.synchronize_with_flush(flush_locked);
        else (void)context.query_with_flush(flush_locked);
        workers.join();
        expect(admitted && context.recording(), "producer was not admitted after device cutoff");
        expect(!context.query_complete(), "new producer recording was silently completed");
        context.cancel();
    }
}

void test_fake_persistent_error_does_not_skip_next_safe_cleanup() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    int first = 0;
    int second = 0;
    context.begin();
    context.defer_destruction([&] { ++first; throw std::runtime_error("first cleanup failure"); });
    context.submit();
    fake.signal_all();
    const auto expect_first_error = [&](const std::function<void()> &operation) {
        std::string message;
        try { operation(); } catch (const std::exception &e) { message = e.what(); }
        expect(message == "first cleanup failure", "completion did not preserve the first cleanup error");
    };
    expect_first_error([&] { context.query_complete(); });
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.begin_locked();
        context.defer_destruction([&] { ++second; });
        context.submit_locked();
    }
    expect_first_error([&] { context.synchronize_with_flush([] {}); });
    expect(first == 1 && second == 1 && fake.wait_calls == 1,
           "persistent error skipped device wait or subsequent safe cleanup");
    expect_first_error([&] { context.query_complete(); });
    expect_first_error([&] { context.synchronize(); });
    expect(first == 1 && second == 1, "persistent error repeated cleanup");
}

void test_fake_flush_failure_preserves_error_and_safe_cleanup() {
    for (const bool synchronize : {false, true}) {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        int calls = 0;
        context.begin();
        context.defer_destruction([&] { ++calls; throw std::runtime_error("secondary cleanup"); });
        context.submit();
        if (!synchronize) fake.signal_all();
        std::string message;
        const auto failed_flush = [] { throw std::runtime_error("original flush failure"); };
        try {
            if (synchronize) context.synchronize_with_flush(failed_flush);
            else (void)context.query_with_flush(failed_flush);
        } catch (const std::exception &e) { message = e.what(); }
        expect(message == "original flush failure", "boundary replaced original flush error");
        expect(calls == 1, "flush error bypassed other safely completed cleanup");
        expect(synchronize ? fake.wait_calls == 1 : fake.status_calls == 1,
               "flush error bypassed protected device retirement");
        expect_rejected([&] { context.synchronize(); }, "secondary callback error was not persistent");
    }
}

void test_fake_platform_boundary_uses_one_locked_device_phase() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    int flushes = 0;
    int cleanup = 0;
    const auto flush_locked = [&] {
        expect(!other_thread_can_lock(queue_mutex), "boundary flush ran outside queue phase");
        ++flushes;
        if (context.recording()) context.submit_locked();
        fake.signal_all();
    };
    fake.status_hook = [&] {
        expect(!other_thread_can_lock(queue_mutex), "boundary fence poll ran outside queue phase");
    };
    context.begin();
    context.defer_destruction([&] {
        ++cleanup;
        expect(other_thread_can_lock(queue_mutex), "boundary cleanup retained queue lock");
        expect(!context.query_with_flush([&] { throw std::runtime_error("reentrant flush"); }),
               "boundary callback query reported completion");
        bool self_wait = false;
        try {
            context.synchronize_with_flush([&] { throw std::runtime_error("reentrant flush"); });
        } catch (const std::logic_error &) { self_wait = true; }
        expect(self_wait, "boundary callback synchronize did not reject self-wait before flush");
    });
    expect(context.query_with_flush(flush_locked), "boundary did not complete signaled work");
    expect(flushes == 1 && cleanup == 1 && fake.status_calls == 1,
           "boundary did not use one flush/poll/cleanup phase");
    context.begin();
    context.synchronize_with_flush(flush_locked);
    expect(flushes == 2 && fake.wait_calls == 1, "synchronize boundary did not use device wait");
}

void test_fake_teardown_drains_before_services_are_reset() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    bool services_alive = true;
    int calls = 0;
    context.begin();
    context.defer_destruction([&] {
        ++calls;
        expect(services_alive, "teardown callback observed destroyed platform services");
        expect(other_thread_can_lock(queue_mutex), "teardown callback inherited queue lock");
        expect(!context.query_with_flush([] { throw std::runtime_error("teardown flush"); }),
               "teardown callback query reported completion");
        bool self_wait = false;
        try { context.synchronize_with_flush([] {}); }
        catch (const std::logic_error &) { self_wait = true; }
        expect(self_wait, "teardown callback did not reject same-thread synchronize");
        expect_rejected([&] { context.begin(); }, "teardown admitted new recording");
    });
    context.submit();
    context.teardown_with_services([&] {
        expect(calls == 1, "teardown reset services before draining callbacks");
        services_alive = false;
    });
    expect(calls == 1 && !services_alive, "teardown did not drain and then reset services");
    expect_rejected([&] { context.begin(); }, "closed context admitted recording");
}

void test_fake_quarantine_requires_instance_sink() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    VulkanExecutionQuarantineSink missing{};
    std::mutex queue_mutex;
    bool rejected = false;
    try {
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, missing, queue_mutex);
    } catch (const std::invalid_argument &) { rejected = true; }
    expect(rejected, "injected context accepted missing instance quarantine sink");
    expect(fake.next_handle == 10, "invalid sink was rejected only after allocating device objects");
}

void test_fake_faults_retain_whole_payload_in_instance_sink() {
    for (int fault = 0; fault != 4; ++fault) {
        int cleanup = 0;
        int destroyed_owner = 0;
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        {
            VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                           fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
            auto owner = std::shared_ptr<int>(new int(7), [&](int *p) {
                ++destroyed_owner;
                delete p;
            });
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.begin_locked();
                context.defer_destruction([&, owner] { ++cleanup; });
                context.cancel_locked(); // Ready A must be retained alongside uncertain B.
                context.begin_locked();
                context.defer_destruction([&, owner] { ++cleanup; });
            }
            owner.reset();
            if (fault == 0) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                context.submit_locked();
                fake.idle_result = VK_ERROR_UNKNOWN;
            } else if (fault == 1) {
                fake.submit_result = VK_ERROR_DEVICE_LOST;
                expect_rejected([&] { context.submit(); }, "injected submit loss did not fail");
            } else {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    context.submit_locked();
                }
                if (fault == 2) {
                    fake.statuses = {VK_ERROR_DEVICE_LOST};
                    expect_rejected([&] { context.query_complete(); }, "fence query loss did not fail");
                } else {
                    fake.wait_result = VK_ERROR_DEVICE_LOST;
                    expect_rejected([&] { context.synchronize(); }, "fence wait loss did not fail");
                }
            }
        }
        expect(fake.quarantine_calls == 1, "failure did not route exactly once to instance sink");
        expect(cleanup == 0 && destroyed_owner == 0, "uncertain callbacks/owners were released");
        expect(fake.freed_buffers == 0 && fake.destroyed_fences == 0 &&
                   fake.destroyed_semaphores == 0, "uncertain handles were destroyed");
        const auto &payload = *fake.quarantines[0];
        expect(payload.ready_callbacks.size() == 1, "quarantine omitted already-ready callback");
        const auto uncertain = payload.deferred_callbacks.size() +
            payload.record_callbacks[0].size() + payload.record_callbacks[1].size();
        expect(uncertain == 1, "quarantine omitted pending/recording callback");
        expect(payload.command_buffers[0] && payload.command_buffers[1] &&
                   payload.fences[0] && payload.fences[1] &&
                   payload.signal_semaphores[0] && payload.signal_semaphores[1],
               "quarantine omitted execution handles");
        fake.signal_all(); // Declared fake completion; now fixture may reclaim capture ownership.
        fake.quarantines[0].reset();
        expect(cleanup == 0 && destroyed_owner == 1, "fixture did not own last quarantined capture");
    }
}

void test_fake_callback_handoff_is_nonthrowing_and_preserves_owners() {
    static_assert(noexcept(std::declval<VulkanExecutionCallbacks &>().splice(
        std::declval<VulkanExecutionCallbacks &>().end(),
        std::declval<VulkanExecutionCallbacks &>())), "callback handoff must not allocate/throw");
    int cleanup = 0;
    int destroyed_owner = 0;
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    auto owner = std::shared_ptr<int>(new int(1), [&](int *p) { ++destroyed_owner; delete p; });
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (int batch = 0; batch != 2; ++batch) {
            context.begin_locked();
            for (int i = 0; i != 1024; ++i)
                context.defer_destruction([&, owner] { ++cleanup; });
            context.submit_locked();
        }
        owner.reset();
        fake.signal_all();
        context.poll_device_locked();
        expect(cleanup == 0 && destroyed_owner == 0, "retirement destroyed captures under locks");
    }
    context.drain_ready_cleanup();
    expect(cleanup == 2048 && destroyed_owner == 1, "batch splice lost or duplicated ownership");
}

void test_platform_exclusive_staging_reservation(VulkanPlatform &platform) {
    auto &context = platform.execution_context();
    expect_rejected([&] { (void)platform.reserve_transfer(0); },
                    "zero-byte exclusive staging was accepted");
    const uint32_t expected[] = {0x01020304U, 0x0A0B0C0DU};
    std::weak_ptr<VulkanBuffer> weak_first;
    {
        auto first = platform.reserve_transfer(sizeof(expected));
        auto second = platform.reserve_transfer(sizeof(expected));
        expect(first.staging && second.staging && first.staging != second.staging,
               "outstanding transfers did not receive distinct exclusive staging");
        expect(&*first.staging != &platform.staging_buffer(sizeof(expected)) &&
                   &*second.staging != &platform.staging_buffer(sizeof(expected)),
               "exclusive staging aliased the shared synchronous staging buffer");
        first.staging->write(expected, sizeof(expected));
        uint32_t actual[] = {0U, 0U};
        first.staging->read(actual, sizeof(actual));
        expect(actual[0] == expected[0] && actual[1] == expected[1],
               "exclusive staging was not host accessible");
        weak_first = first.staging;
        std::atomic<int> published{0};
        context.begin();
        {
            std::lock_guard<std::mutex> lock(platform.queue_mutex());
            context.defer_transfer(std::move(first.reservation), {first.staging},
                                   [&] { ++published; });
            context.defer_transfer(std::move(second.reservation), {second.staging});
        }
        first.staging.reset();
        second.staging.reset();
        expect(!weak_first.expired(), "staging was released before the transfer completed");
        // Both slots are occupied by an unsubmitted recording: this must flush through the
        // compute callback, wait for the oldest fence, and then grant a slot.
        auto third = platform.reserve_transfer(sizeof(expected));
        expect(third.reservation && third.staging, "flushed capacity pressure did not grant");
        context.synchronize();
        expect(published == 1 && weak_first.expired(),
               "completed transfer did not publish then release its exclusive staging");
    }
    auto a = platform.reserve_transfer(16);
    auto b = platform.reserve_transfer(16);
    expect(a.staging && b.staging, "failed or completed reservations leaked capacity");
}

// Vulkan-executed platform initialization (real device): the per-instance
// staging allocation-failure injection runs inside the production
// reserve-then-allocate wrapper transaction, not a parallel mock.
void test_platform_staging_allocation_rollback() {
    VulkanPlatform platform{false, 1};
    VulkanExecutionContext &context = platform.execution_context();
    const auto creations = platform.timing_snapshot().buffer_creations;

    // A genuinely pending transfer predates the attempted staging allocation.
    int releases = 0;
    std::shared_ptr<int> owner(new int(11),
                               [&](int *value) { delete value; ++releases; });
    int published = 0;
    int legacy_cleanups = 0;
    auto reservation = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(platform.queue_mutex());
        context.defer_transfer(std::move(reservation), {owner}, [&] { ++published; });
        context.defer_destruction([&] { ++legacy_cleanups; });
    }
    owner.reset();
    context.submit();
    const auto pending_before = context.pending_count();
    expect(pending_before != 0, "pending transfer fixture produced no pending work");

    // The wrapper must grant the seat first, then surface the allocator
    // dependency's std::bad_alloc before any Vulkan buffer creation.
    bool surfaced = false;
    try {
        (void)platform.reserve_transfer(64);
    } catch (const std::bad_alloc &) {
        surfaced = true;
    }
    expect(surfaced, "injected staging allocation failure did not reach the caller");
    expect(published == 0, "failed staging allocation published transfer success");
    expect(context.pending_count() == pending_before,
           "failed staging allocation disturbed pending transfer state");
    expect(platform.timing_snapshot().buffer_creations == creations,
           "failed staging allocation created a Vulkan buffer");
    expect(other_thread_can_lock(platform.queue_mutex()),
           "failed staging allocation retained the shared queue lock");

    context.synchronize();
    expect(published == 1 && releases == 1 && legacy_cleanups == 1,
           "preexisting pending transfer or legacy cleanup did not finish exactly once");
    expect(context.pending_count() == 0, "rollback left pending transfer work behind");

    // Capacity recovery is proven by fresh reservations, attachment and fence
    // completion, not by inspecting internal counts.
    auto first = platform.reserve_transfer(64);
    auto second = platform.reserve_transfer(64);
    expect(first.staging && second.staging && first.staging != second.staging,
           "post-rollback reservations did not receive fresh exclusive staging");
    int published_after = 0;
    context.begin();
    {
        std::lock_guard<std::mutex> lock(platform.queue_mutex());
        context.defer_transfer(std::move(first.reservation), {first.staging},
                               [&] { ++published_after; });
        context.defer_transfer(std::move(second.reservation), {second.staging},
                               [&] { ++published_after; });
    }
    first.staging.reset();
    second.staging.reset();
    context.submit();
    context.synchronize();
    expect(published_after == 2,
           "post-rollback attached transfers did not publish exactly once each");
}

void test_nonblocking_completion_query(VulkanPlatform &platform) {
    VulkanExecutionContext &context = platform.execution_context();
    context.wait();
    platform.reset_execution_counters();
    expect(context.query_complete(), "idle execution query reported pending work");
    expect(platform.compute_wait_count() == 0,
           "idle execution query performed a blocking wait");

    int callback_count = 0;
    context.begin();
    context.defer_destruction([&] { ++callback_count; });
    expect(!context.query_complete(), "recording execution query reported complete");
    expect(callback_count == 0, "recording query retired a callback");
    context.submit();
    const bool first_poll = context.query_complete();
    expect(first_poll == (context.pending_count() == 0),
           "submitted query disagreed with pending fence state");
    expect(platform.compute_wait_count() == 0,
           "submitted execution query performed a blocking wait");
    if (first_poll)
        expect(callback_count == 1, "completed query did not retire callback");

    context.wait();
    expect(context.query_complete(), "waited execution query reported pending work");
    expect(callback_count == 1, "submitted query callback did not run exactly once");
    expect(context.query_complete(), "repeat idle query reported pending work");
    expect(callback_count == 1, "repeat query retired callback more than once");

    context.begin();
    context.cancel();
    expect(context.query_complete(), "cancelled recording query reported pending work");
}


void test_timestamp_capacity_query(VulkanPlatform &platform) {
    VulkanExecutionContext &context = platform.execution_context();
    context.wait();
    context.begin();
    expect(!context.timestamp_region_capacity_exhausted(),
           "timestamp capacity was exhausted in a fresh record");
    for (int region = 0; region < 4; ++region) {
        context.begin_timestamp_scope("capacity-test");
        context.end_timestamp_scope();
    }
    expect(context.timestamp_region_capacity_exhausted() ==
               context.timestamp_queries_supported(),
           "timestamp capacity query disagrees with query support");
    context.cancel();
}

void test_wait_for_transfer_flushes_recording(VulkanPlatform &platform) {
    VulkanBuffer buffer(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    platform.reset_execution_counters();
    auto &context = platform.execution_context();
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 64, 0x31415926U);
    platform.wait_for_transfer();
    expect(!context.recording(), "host-visible transfer wait left compute unsubmitted");
    expect(platform.compute_submitted_count() == 1,
           "host-visible transfer wait did not submit pending compute");
    uint32_t values[16]{};
    buffer.read(values, sizeof(values));
    for (uint32_t value : values)
        expect(value == 0x31415926U,
               "host-visible transfer observed data before pending compute");
}

void test_failed_async_recording_surfaces_at_next_boundary(VulkanPlatform &platform) {
    expect(platform.async_execution_enabled(),
           "recording-failure test requires async execution mode");
    VulkanBuffer buffer(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    auto &context = platform.execution_context();
    context.begin("test-prefix");
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 64, 0x1234U);
    expect_rejected([&] { platform.compute().test_inject_recording_failure(); },
                    "injected partial recording failure was not surfaced");
    expect(!context.recording(), "failed partial command buffer was not canceled");
    bool pending_failure = false;
    try {
        platform.wait_for_transfer();
    } catch (const std::exception &error) {
        pending_failure =
            std::string(error.what()).find("pending async eager batch failure") !=
            std::string::npos;
    }
    expect(pending_failure,
           "host-visible boundary did not surface the canceled batch failure");
    expect_rejected([&] { platform.compute().flush_recording(); },
                    "flush cleared a pending async batch failure");
}

void test_async_pre_dispatch_failure_preserves_prefix(VulkanPlatform &platform) {
    expect(platform.async_execution_enabled(),
           "pre-dispatch failure test requires async execution mode");
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VulkanBuffer lhs(platform, 4 * sizeof(float), host);
    VulkanBuffer rhs(platform, 4 * sizeof(float), host);
    VulkanBuffer output(platform, 4 * sizeof(float), host);
    const float lhs_values[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float rhs_values[] = {5.0F, 6.0F, 7.0F, 8.0F};
    lhs.write(lhs_values, sizeof(lhs_values));
    rhs.write(rhs_values, sizeof(rhs_values));
    const pytorch_vulkan::VulkanTensorLayout layout{
        1, {4}, {1}, 0, sizeof(float), 0, 4, 0, sizeof(lhs_values),
        sizeof(lhs_values), pytorch_vulkan::VulkanOverlap::No};

    platform.reset_execution_counters();
    platform.compute().add(lhs.buffer(), layout, rhs.buffer(), layout,
                           output.buffer(), layout);
    expect(platform.compute_submitted_count() == 0,
           "valid eager prefix was submitted before injected setup failure");
    expect_rejected(
        [&] { platform.compute().test_inject_pre_dispatch_failure(); },
        "injected pre-dispatch failure was not surfaced");
    expect(platform.compute_submitted_count() == 1,
           "pre-dispatch failure did not submit the valid eager prefix");
    expect(platform.compute_wait_count() == 0,
           "pre-dispatch failure blocked while preserving the prefix");
    expect(!platform.execution_context().recording(),
           "pre-dispatch failure left an unexpected recording open");

    platform.wait_for_transfer();
    platform.execution_context().synchronize();
    float actual[4]{};
    output.read(actual, sizeof(actual));
    for (int i = 0; i < 4; ++i)
        expect(actual[i] == lhs_values[i] + rhs_values[i],
               "pre-dispatch failure discarded the valid eager prefix");
    platform.compute().flush_recording();
}

void test_async_eager_batch_and_sync_fill_order(VulkanPlatform &platform) {
    expect(platform.async_execution_enabled(),
           "async batch test requires PYTORCH_VULKAN_ASYNC_EXECUTION=1");
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr std::size_t count = 18;
    VulkanBuffer lhs(platform, 16 * sizeof(float), host);
    VulkanBuffer rhs(platform, 16 * sizeof(float), host);
    float lhs_values[16];
    float rhs_values[16];
    for (int i = 0; i < 16; ++i) {
        lhs_values[i] = 1.0F;
        rhs_values[i] = 2.0F;
    }
    lhs.write(lhs_values, sizeof(lhs_values));
    rhs.write(rhs_values, sizeof(rhs_values));
    std::vector<std::unique_ptr<VulkanBuffer>> outputs;
    outputs.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        outputs.push_back(std::make_unique<VulkanBuffer>(platform, sizeof(lhs_values), host));
    const pytorch_vulkan::VulkanTensorLayout layout{1,
                                                    {16},
                                                    {1},
                                                    0,
                                                    sizeof(float),
                                                    0,
                                                    16,
                                                    0,
                                                    sizeof(lhs_values),
                                                    sizeof(lhs_values),
                                                    pytorch_vulkan::VulkanOverlap::No};
    platform.reset_execution_counters();
    for (std::size_t i = 0; i < count; ++i) {
        const VkBuffer previous = i == 0 ? lhs.buffer() : outputs[i - 1]->buffer();
        platform.compute().add(previous, layout, rhs.buffer(), layout,
                               outputs[i]->buffer(), layout);
        if (i == 15) {
            expect(platform.compute_submitted_count() == 1,
                   "16-dispatch async batch was not submitted at its bound");
            expect(platform.compute_wait_count() == 0,
                   "async batch waited after reaching its dispatch bound");
        }
    }
    expect(platform.compute_submitted_count() == 1,
           "async eager chain submitted once per operator");
    expect(platform.compute_wait_count() == 0,
           "async eager chain waited before host readback");
    expect(platform.compute().descriptor_arena_snapshot().pending > 0,
           "async descriptor sets were not retained for submitted work");
    platform.compute().flush_recording();
    platform.execution_context().wait();
    float result[16]{};
    outputs.back()->read(result, sizeof(result));
    for (float value : result)
        expect(std::abs(value - 37.0F) < 1e-5F,
               "async eager chain readback had incorrect queue order");
    expect(platform.compute().descriptor_arena_snapshot().pending == 0,
           "async descriptor sets were not retired after batch completion");

    // A synchronous fill must submit and complete any preceding async compute.
    platform.compute().add(lhs.buffer(), layout, rhs.buffer(), layout,
                           outputs.back()->buffer(), layout);
    platform.fill_buffer_sync(outputs.back()->buffer(), 0, sizeof(result), 0x3f800000U);
    outputs.back()->read(result, sizeof(result));
    for (float value : result)
        expect(value == 1.0F, "synchronous empty-fill boundary reordered pending compute");

    // Exercise both full timestamp groups in distinct records, retaining each
    // record's callback until its fence is retired.
    auto &context = platform.execution_context();
    int retired_callbacks = 0;
    for (int group = 0; group < 2; ++group) {
        context.begin("rnn_backward");
        context.defer_destruction([&] { ++retired_callbacks; });
        for (int region = 0; region < 4; ++region) {
            context.begin_timestamp_scope("rnn_backward_region");
            context.end_timestamp_scope();
        }
        expect(context.timestamp_region_capacity_exhausted() ==
                   context.timestamp_queries_supported(),
               "four-scope timestamp group did not report exact capacity");
        context.submit();
    }
    context.wait();
    expect(retired_callbacks == 2,
           "timestamp rollover did not retain callbacks through fence retirement");
}

void test_scoped_training_batches_device_copy(VulkanPlatform &platform) {
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VulkanBuffer lhs(platform, 4 * sizeof(float), host);
    VulkanBuffer rhs(platform, 4 * sizeof(float), host);
    VulkanBuffer first(platform, 4 * sizeof(float), host);
    VulkanBuffer copied(platform, 4 * sizeof(float), host);
    VulkanBuffer output(platform, 4 * sizeof(float), host);
    const float lhs_values[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float rhs_values[] = {5.0F, 6.0F, 7.0F, 8.0F};
    lhs.write(lhs_values, sizeof(lhs_values));
    rhs.write(rhs_values, sizeof(rhs_values));
    const pytorch_vulkan::VulkanTensorLayout layout{
        1, {4}, {1}, 0, sizeof(float), 0, 4, 0, sizeof(lhs_values),
        sizeof(lhs_values), pytorch_vulkan::VulkanOverlap::No};

    platform.reset_execution_counters();
    platform.compute().begin_training_step();
    platform.compute().add(lhs.buffer(), layout, rhs.buffer(), layout,
                           first.buffer(), layout);
    platform.copy_buffer_sync(first.buffer(), copied.buffer(), sizeof(lhs_values));
    platform.compute().add(copied.buffer(), layout, rhs.buffer(), layout,
                           output.buffer(), layout);
    expect(platform.compute_submitted_count() == 0,
           "device copy split an explicitly scoped training command buffer");
    expect(platform.compute_wait_count() == 0,
           "device copy blocked inside an explicitly scoped training step");
    platform.compute().end_training_step();
    expect(platform.compute_submitted_count() == 1,
           "scoped training step with device copy did not submit once");
    expect(platform.compute_completed_count() == 1 &&
               platform.compute_wait_count() == 1,
           "scoped training copy was not completed by its end boundary");
    float actual[4]{};
    output.read(actual, sizeof(actual));
    for (int i = 0; i < 4; ++i)
        expect(actual[i] == lhs_values[i] + 2.0F * rhs_values[i],
               "scoped device copy did not preserve compute ordering");
}

void test_query_retains_callbacks_until_all_submissions(VulkanPlatform &platform) {
    VulkanExecutionContext &context = platform.execution_context();
    context.wait();
    platform.reset_execution_counters();
    int callback_count = 0;
    context.begin();
    context.defer_destruction([&] { ++callback_count; });
    context.submit();
    context.begin();
    expect(!context.query_complete(),
           "recording query ignored active commands after an earlier submission");
    expect(platform.compute_wait_count() == 0,
           "recording query performed a blocking wait");
    context.submit();
    const bool complete = context.query_complete();
    expect(complete == (context.pending_count() == 0),
           "query did not report both submitted records");
    expect(platform.compute_wait_count() == 0,
           "multi-submission query performed a blocking wait");
    if (complete)
        expect(callback_count == 1, "completed submissions retained callback");
    context.wait();
    expect(context.query_complete(), "query remained incomplete after wait");
    expect(callback_count == 1, "query retired callback more than once");
}

void test_stale_slot_callback_retirement(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.submit();
    context.begin();
    context.cancel();
    context.retain_until_completion([&] { ++callback_count; });
    expect(callback_count == 0, "stale-slot callback ran before completion");
    context.wait();
    expect(callback_count == 1, "stale-slot callback did not run after completion");
    context.wait();
    expect(callback_count == 1, "stale-slot callback ran more than once");
}

void test_callback_waits_for_all_submissions(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    int callback_count = 0;
    context.begin();
    context.retain_until_completion([&] { ++callback_count; });
    context.submit();
    context.begin();
    context.submit();
    context.wait();
    expect(callback_count == 1,
           "callback did not execute exactly once after all submissions");
    context.wait();
    expect(callback_count == 1,
           "callback executed more than once after all submissions");
}

void test_abandoned_callback(VulkanPlatform &platform) {
    bool called = false;
    {
        VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                       platform.command_pool());
        context.begin();
        context.defer_destruction([&] { called = true; });
    }
    expect(called, "abandoned recording callback was discarded");
}

void test_cancelled_callback(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    bool called = false;
    context.begin();
    context.defer_destruction([&] { called = true; });
    context.cancel();
    expect(called, "cancelled recording callback was not invoked");
    expect(!context.recording(), "cancel left context recording");
    context.begin();
    context.cancel();
}

void test_fill(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    VulkanBuffer buffer(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), buffer.buffer(), 0, 64, 0x12345678U);
    context.submit();
    context.wait();

    uint32_t values[16]{};
    buffer.read(values, sizeof(values));
    for (uint32_t value : values) {
        expect(value == 0x12345678U, "vkCmdFillBuffer produced the wrong pattern");
    }
}

void test_retained_descriptor_pool(VulkanPlatform &platform) {
    VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                   platform.command_pool());
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    expect(vkCreateDescriptorSetLayout(platform.device(), &layout_info, nullptr,
                                       &layout) == VK_SUCCESS,
           "could not create retained descriptor layout");
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    expect(vkCreatePipelineLayout(platform.device(), &pipeline_layout_info, nullptr,
                                  &pipeline_layout) == VK_SUCCESS,
           "could not create retained pipeline layout");
    VkDescriptorPool pool = VK_NULL_HANDLE;
    expect(vkCreateDescriptorPool(platform.device(), &pool_info, nullptr, &pool) ==
               VK_SUCCESS,
           "could not create retained descriptor pool");
    auto buffer = std::make_shared<VulkanBuffer>(
        platform, 64,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorSetAllocateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    expect(vkAllocateDescriptorSets(platform.device(), &set_info, &set) == VK_SUCCESS,
           "could not allocate retained descriptor set");
    VkDescriptorBufferInfo buffer_info{buffer->buffer(), 0, 64};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(platform.device(), 1, &write, 0, nullptr);
    context.begin();
    context.retain(pool);
    pool = VK_NULL_HANDLE;
    vkCmdBindDescriptorSets(context.command_buffer(), VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdFillBuffer(context.command_buffer(), buffer->buffer(), 0, 64, 0xdeadbeefU);
    context.defer_destruction([owner = std::move(buffer)] {});
    expect(buffer == nullptr, "recorded buffer owner was not released before submit");
    context.submit();
    expect(context.pending_count() == 1, "retained submission was not pending");
    context.wait();
    expect(context.pending_count() == 0, "retained submission was not retired");
    vkDestroyPipelineLayout(platform.device(), pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(platform.device(), layout, nullptr);

    VulkanBuffer ring_buffer(platform, 64,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), ring_buffer.buffer(), 0, 64, 0xcafebabeU);
    context.submit();
    expect(context.pending_count() == 1, "ring reuse left an incorrect pending count");
    context.begin();
    vkCmdFillBuffer(context.command_buffer(), ring_buffer.buffer(), 0, 64, 0xfeedfaceU);
    context.submit();
    expect(context.pending_count() == 2,
           "ring did not retain both bounded submissions");
    context.begin();
    expect(context.pending_count() == 2,
           "ring reuse did not retire the reused submission before recording");
    context.cancel();
    expect(context.pending_count() == 1,
           "ring reuse lost the remaining submitted slot");
    context.wait();
    VkDescriptorPool replacement = VK_NULL_HANDLE;
    expect(vkCreateDescriptorPool(platform.device(), &pool_info, nullptr,
                                  &replacement) == VK_SUCCESS,
           "descriptor pool could not be reused after retirement");
    vkDestroyDescriptorPool(platform.device(), replacement, nullptr);
}

void test_platform_pending_compute_count(VulkanPlatform &platform) {
    VulkanBuffer lhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer rhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer output(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout layout{1,
                                                    {16},
                                                    {1},
                                                    0,
                                                    sizeof(float),
                                                    0,
                                                    16,
                                                    0,
                                                    64,
                                                    64,
                                                    pytorch_vulkan::VulkanOverlap::No};
    expect(platform.pending_compute_count() == 0,
           "platform reported pending compute work before training step");
    platform.compute().begin_training_step();
    platform.compute().add(lhs.buffer(), layout, rhs.buffer(), layout, output.buffer(),
                           layout);
    expect(platform.pending_compute_count() == 1,
           "platform did not report the active compute recording");
    platform.compute().end_training_step();
    expect(platform.pending_compute_count() == 0,
           "platform retained compute work after training step completion");
}

void test_descriptor_cache_rollover_and_cancellation(VulkanPlatform &platform) {
    VulkanBuffer lhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer rhs(platform, 64,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer output(platform, 64,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout pointwise_layout{
        1,
        {16},
        {1},
        0,
        sizeof(float),
        0,
        16,
        0,
        64,
        64,
        pytorch_vulkan::VulkanOverlap::No};

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 65; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 1,
           "generic descriptor cache did not roll over at pool capacity");

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 64; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    platform.compute().begin_training_step();
    for (int i = 0; i < 64; ++i) {
        platform.compute().add(lhs.buffer(), pointwise_layout, rhs.buffer(),
                               pointwise_layout, output.buffer(), pointwise_layout);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 0,
           "cancelled recording did not reset generic descriptor cursors");

    VulkanBuffer a(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer b(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer c(platform, 4,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer gemm_output(platform, 4,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const pytorch_vulkan::VulkanTensorLayout matrix_layout{
        2,
        {1, 1},
        {1, 1},
        6,
        sizeof(float),
        0,
        1,
        0,
        4,
        4,
        pytorch_vulkan::VulkanOverlap::No};

    platform.compute().reset_descriptor_resource_counters();
    platform.compute().begin_training_step();
    for (int i = 0; i < 4097; ++i) {
        platform.compute().gemm(a.buffer(), matrix_layout, b.buffer(), matrix_layout,
                                c.buffer(), matrix_layout, gemm_output.buffer(),
                                matrix_layout, VK_NULL_HANDLE, matrix_layout, 1, 1, 1,
                                1.0F, 0.0F, false);
    }
    platform.compute().cancel_training_step();
    expect(platform.compute().descriptor_pool_creation_count() == 2,
           "GEMM descriptor cache did not roll over at pool capacity");
}

void test_gemm_transposed_a_and_preflight_rejections(VulkanPlatform &platform) {
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VulkanBuffer a(platform, 6 * sizeof(float), host);
    VulkanBuffer b(platform, 6 * sizeof(float), host);
    VulkanBuffer out(platform, 4 * sizeof(float), host);
    // Logical A is 2x3, stored as a contiguous 3x2 transpose.
    const float av[] = {1, 4, 2, 5, 3, 6};
    const float bv[] = {7, 8, 9, 10, 11, 12};
    a.write(av, sizeof(av));
    b.write(bv, sizeof(bv));
    auto layout = [](std::vector<int64_t> sizes, std::vector<int64_t> strides,
                     uint64_t bytes) {
        return pytorch_vulkan::VulkanTensorLayout{
            2, std::move(sizes), std::move(strides), 6, sizeof(float), 0, 1, 0,
            bytes, bytes, pytorch_vulkan::VulkanOverlap::No};
    };
    const auto at = layout({2, 3}, {1, 2}, sizeof(av));
    const auto row = layout({3, 2}, {2, 1}, sizeof(bv));
    const auto result = layout({2, 2}, {2, 1}, sizeof(float) * 4);
    platform.reset_execution_counters();
    platform.compute().begin_training_step();
    platform.compute().gemm(a.buffer(), at, b.buffer(), row, VK_NULL_HANDLE,
                            result, out.buffer(), result, VK_NULL_HANDLE, result,
                            2, 2, 3, 1.0F, 0.0F, false);
    platform.compute().end_training_step();
    float actual[4]{};
    out.read(actual, sizeof(actual));
    const float expected[] = {58, 64, 139, 154};
    for (int i = 0; i < 4; ++i)
        expect(std::abs(actual[i] - expected[i]) < 1e-4F,
               "transposed-A GEMM numerical mismatch");
    expect(platform.compute_dispatch_count() == 1,
           "transposed-A GEMM did not record exactly one dispatch");

    auto unchanged = [&] {
        expect(platform.compute_dispatch_count() == 0, "invalid GEMM dispatched work");
        expect(platform.vulkan_copy_count() == 0, "invalid GEMM recorded a copy");
        expect(platform.explicit_transfer_count() == 0, "invalid GEMM transferred data");
        expect(platform.execution_counter_snapshot().fallbacks == 0,
               "invalid GEMM used fallback");
    };
    auto rejected = [&](const pytorch_vulkan::VulkanTensorLayout &bad_a,
                        VkBuffer output_buffer, const auto &output_layout) {
        platform.reset_execution_counters();
        platform.compute().begin_training_step();
        expect_rejected([&] {
            platform.compute().gemm(a.buffer(), bad_a, b.buffer(), row,
                                    VK_NULL_HANDLE, result, output_buffer,
                                    output_layout, VK_NULL_HANDLE, result,
                                    2, 2, 3, 1.0F, 0.0F, false);
        }, "malformed GEMM layout was accepted");
        platform.compute().cancel_training_step();
        unchanged();
    };
    rejected(layout({2, 3}, {2, 1}, sizeof(av)), out.buffer(), result); // sliced/noncompact
    rejected(layout({2, 3}, {1, static_cast<int64_t>(UINT32_MAX) + 1}, sizeof(av)),
             out.buffer(), result);
    auto alias_output = layout({2, 2}, {2, 1}, sizeof(float) * 4);
    alias_output.allocation_bytes = sizeof(av);
    rejected(at, a.buffer(), alias_output); // output aliases A
}

void test_mm_preflight_rejects_actual_device_group_limit(VulkanPlatform &platform) {
    using pytorch_vulkan::VulkanOverlap;
    using pytorch_vulkan::VulkanTensorLayout;
    const auto layout = [](int64_t rows, int64_t columns, uint64_t bytes) {
        return VulkanTensorLayout{2,
                                  {rows, columns},
                                  {columns, 1},
                                  6,
                                  sizeof(float),
                                  0,
                                  rows * columns,
                                  0,
                                  bytes,
                                  bytes,
                                  VulkanOverlap::No};
    };
    const auto a = layout(static_cast<int64_t>(UINT32_MAX), 1, sizeof(float));
    const auto b = layout(1, 1, sizeof(float));
    const auto output = layout(static_cast<int64_t>(UINT32_MAX), 1, sizeof(float));

    platform.reset_execution_counters();
    bool rejected_for_device_limits = false;
    try {
        platform.compute().validate_gemm_preflight(a, b, output, UINT32_MAX, 1, 1,
                                                   true);
    } catch (const std::exception &error) {
        rejected_for_device_limits =
            std::string(error.what()).find("device limits") != std::string::npos;
    }
    expect(rejected_for_device_limits,
           "GEMM preflight did not use the active device group limits");
    expect(platform.compute_dispatch_count() == 0,
           "GEMM device-limit preflight dispatched work");
    expect(platform.vulkan_copy_count() == 0,
           "GEMM device-limit preflight recorded a copy");

    const VkDeviceSize over_storage_limit =
        platform.compute().cat_max_storage_buffer_range() + sizeof(float);
    const auto tiny = layout(1, 1, sizeof(float));
    const auto oversized_allocation = layout(1, 1, over_storage_limit);
    bool rejected_for_storage_limit = false;
    try {
        platform.compute().validate_broadcast_preflight(
            oversized_allocation, tiny, 1);
    } catch (const std::exception &error) {
        rejected_for_storage_limit =
            std::string(error.what()).find("device limits") != std::string::npos;
    }
    expect(rejected_for_storage_limit,
           "broadcast preflight did not use the active storage-buffer limit");
    expect(platform.compute_dispatch_count() == 0,
           "broadcast storage-limit preflight dispatched work");

    bool rejected_gemm_storage = false;
    try {
        platform.compute().validate_gemm_preflight(
            oversized_allocation, tiny, tiny, 1, 1, 1);
    } catch (const std::exception &error) {
        rejected_gemm_storage =
            std::string(error.what()).find("storage-buffer limit") != std::string::npos;
    }
    expect(rejected_gemm_storage,
           "GEMM preflight did not use the active storage-buffer limit");
    expect(platform.compute_dispatch_count() == 0,
           "GEMM storage-limit preflight dispatched work");

    const auto batched_layout = [](uint32_t batches, uint32_t batch_stride,
                                   uint64_t bytes) {
        return VulkanTensorLayout{3,
                                  {static_cast<int64_t>(batches), 1, 1},
                                  {static_cast<int64_t>(batch_stride), 1, 1},
                                  6,
                                  sizeof(float),
                                  0,
                                  static_cast<int64_t>(batches),
                                  0,
                                  bytes,
                                  bytes,
                                  VulkanOverlap::No};
    };
    const auto batch_two = batched_layout(2, 1, 2 * sizeof(float));
    platform.compute().validate_gemm_preflight(
        batch_two, batch_two, batch_two, 1, 1, 1, false, 2, 1, 1, 1);
    platform.reset_execution_counters();
    const auto excessive_batch = batched_layout(UINT32_MAX, 1, sizeof(float));
    bool rejected_batch_limit = false;
    try {
        platform.compute().validate_gemm_preflight(
            excessive_batch, excessive_batch, excessive_batch, 1, 1, 1, false,
            UINT32_MAX, 1, 1, 1);
    } catch (const std::exception &) {
        rejected_batch_limit = true;
    }
    expect(rejected_batch_limit,
           "batched GEMM preflight accepted an unrepresentable batch count");
    expect(platform.compute_dispatch_count() == 0,
           "batched GEMM device-limit preflight dispatched work");
}

void test_execution_counters_and_timing(VulkanPlatform &platform) {
    platform.reset_execution_counters();
    platform.reset_timing();
    VulkanExecutionContext &context = platform.execution_context();
    context.begin();
    context.submit();
    expect(platform.compute_submitted_count() == 1,
           "successful submit was not counted independently");
    expect(platform.compute_completed_count() == 0,
           "submission was counted as completed before wait");
    context.wait();
    expect(platform.compute_completed_count() == 1,
           "successful wait was not counted as completion");
    const auto timing = platform.timing_snapshot();
    expect(timing.allocation > 0.0, "execution allocation timing was not recorded");
    expect(timing.recording > 0.0, "execution recording timing was not recorded");
    expect(timing.submit > 0.0, "execution submit timing was not recorded");
    expect(timing.host_fence_wait > 0.0,
           "execution host fence-wait timing was not recorded");
}

void test_bounded_platform_lifetimes() {
    for (int iteration = 0; iteration < 3; ++iteration) {
        VulkanPlatform platform;
        VulkanExecutionContext context(platform.device(), platform.compute_queue(),
                                       platform.command_pool());
        context.begin();
        context.submit();
        context.wait();
        expect(context.pending_count() == 0,
               "repeated platform lifetime retained submitted work");
    }
}

} // namespace

// Generic native-main progress wrapper: flushed START line before the test,
// flushed PASS line with duration after it returns. Uses the actual test
// callable; no skipping, no fake harness, no changed call order.
void run_native(const char *name, const std::function<void()> &test) {
    std::cout << "START " << name << '\n' << std::flush;
    const auto start = std::chrono::steady_clock::now();
    test();
    std::cout << "PASS " << name << ' '
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start).count()
              << "s\n" << std::flush;
}

int main(int argc, char **argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "--fake-only") {
            const std::string selected = argc == 3 ? argv[2] : "all";
            bool found = false;
            const auto run = [&](const char *name, void (*test)()) {
                if (selected != "all" && selected != name) return;
                found = true;
                const auto start = std::chrono::steady_clock::now();
                test();
                std::cout << name << " PASS " << std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count() << "s\n" << std::flush;
            };
            run("retirement", test_controlled_execution_ops_retirement);
            run("errors", test_controlled_cleanup_errors_are_persistent_and_nonblocking_to_cleanup);
            run("transfer-success", test_fake_transfer_record_success_only_publication);
            run("transfer-cancel-errors", test_fake_transfer_cancel_and_failed_submit_never_publish);
            run("transfer-fence-errors", test_fake_transfer_fence_error_recovery_and_quarantine);
            run("transfer-reservations", test_fake_transfer_reservation_capacity_and_shutdown_lifetime);
            run("transfer-capacity-wait", test_fake_transfer_capacity_waits_for_fence_and_capture_release);
            run("transfer-capacity-flush", test_fake_transfer_capacity_flushes_open_recording);
            run("transfer-teardown", test_fake_teardown_after_queue_idle_never_publishes);
            run("transfer-attach-failure", test_fake_transfer_attachment_allocation_failure_is_atomic);
            run("transfer-order", test_fake_transfer_publication_submission_and_operation_order);
            run("transfer-publisher", test_fake_transfer_publisher_and_capture_visibility);
            run("transfer-publisher-quarantine", test_fake_transfer_invalidation_quarantines_active_publisher);
            run("transfer-throw", test_fake_transfer_throw_is_sticky_and_does_not_skip_safe_work);
            run("F1", test_fake_recording_query_and_single_protected_poll);
            run("F2-destructor", test_fake_ready_cleanup_at_destruction);
            run("F2-submit", test_fake_failed_submit_drains_abandoned_cleanup);
            run("F3", test_fake_compute_exit_drains_without_device_poll);
            run("F4", test_fake_platform_boundary_uses_one_locked_device_phase);
            run("F5", test_fake_teardown_drains_before_services_are_reset);
            run("F6", test_fake_quarantine_requires_instance_sink);
            run("F6-faults", test_fake_faults_retain_whole_payload_in_instance_sink);
            run("F7", test_fake_callback_handoff_is_nonthrowing_and_preserves_owners);
            run("paused-cleanup", test_fake_paused_callback_and_last_capture_visibility);
            run("boundary-admission", test_fake_boundary_excludes_producer_until_device_cutoff);
            run("error-next-batch", test_fake_persistent_error_does_not_skip_next_safe_cleanup);
            run("flush-error-cleanup", test_fake_flush_failure_preserves_error_and_safe_cleanup);
            expect(found, "unknown fake-only case");
            std::cout << "Controlled execution-ops tests passed\n";
            return 0;
        }
        std::cout << "START platform-context-initialization\n" << std::flush;
        const auto init_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point destructor_start = init_start;
        {
            VulkanPlatform platform;
            VulkanExecutionContext context(platform.device(),
                                           platform.compute_queue(),
                                           platform.command_pool());
            std::cout << "PASS platform-context-initialization "
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - init_start).count()
                      << "s\n" << std::flush;
            run_native("test_invalid_states",
                       [&] { test_invalid_states(context); });
            run_native("test_noncoherent_mapped_range_calculation",
                       [] { test_noncoherent_mapped_range_calculation(); });
            run_native("test_host_buffer_offset_mapping",
                       [&] { test_host_buffer_offset_mapping(platform); });
            run_native("test_deferred_callback",
                       [&] { test_deferred_callback(platform); });
            run_native("test_submitted_callback_retirement",
                       [&] { test_submitted_callback_retirement(platform); });
            run_native("test_callback_reentrancy_does_not_report_completion",
                       [&] { test_callback_reentrancy_does_not_report_completion(platform); });
            run_native("test_controlled_execution_ops_retirement",
                       [] { test_controlled_execution_ops_retirement(); });
            run_native("test_nonblocking_completion_query",
                       [&] { test_nonblocking_completion_query(platform); });
            run_native("test_timestamp_capacity_query",
                       [&] { test_timestamp_capacity_query(platform); });
            run_native("test_platform_exclusive_staging_reservation",
                       [&] { test_platform_exclusive_staging_reservation(platform); });
            run_native("test_platform_staging_allocation_rollback",
                       [] { test_platform_staging_allocation_rollback(); });
            run_native("test_wait_for_transfer_flushes_recording",
                       [&] { test_wait_for_transfer_flushes_recording(platform); });
            run_native("test_query_retains_callbacks_until_all_submissions",
                       [&] { test_query_retains_callbacks_until_all_submissions(platform); });
            run_native("test_stale_slot_callback_retirement",
                       [&] { test_stale_slot_callback_retirement(platform); });
            run_native("test_callback_waits_for_all_submissions",
                       [&] { test_callback_waits_for_all_submissions(platform); });
            run_native("test_abandoned_callback",
                       [&] { test_abandoned_callback(platform); });
            run_native("test_cancelled_callback",
                       [&] { test_cancelled_callback(platform); });
            run_native("test_fill",
                       [&] { test_fill(platform); });
            run_native("test_retained_descriptor_pool",
                       [&] { test_retained_descriptor_pool(platform); });
            run_native("test_platform_pending_compute_count",
                       [&] { test_platform_pending_compute_count(platform); });
            run_native("test_descriptor_cache_rollover_and_cancellation",
                       [&] { test_descriptor_cache_rollover_and_cancellation(platform); });
            run_native("test_gemm_transposed_a_and_preflight_rejections",
                       [&] { test_gemm_transposed_a_and_preflight_rejections(platform); });
            run_native("test_mm_preflight_rejects_actual_device_group_limit",
                       [&] { test_mm_preflight_rejects_actual_device_group_limit(platform); });
            run_native("test_scoped_training_batches_device_copy",
                       [&] { test_scoped_training_batches_device_copy(platform); });
            if (platform.async_execution_enabled())
                run_native("test_async_eager_batch_and_sync_fill_order",
                           [&] { test_async_eager_batch_and_sync_fill_order(platform); });
            if (platform.async_execution_enabled())
                run_native("test_async_pre_dispatch_failure_preserves_prefix",
                           [&] { test_async_pre_dispatch_failure_preserves_prefix(platform); });
            if (platform.async_execution_enabled())
                run_native("test_failed_async_recording_surfaces_at_next_boundary",
                           [&] { test_failed_async_recording_surfaces_at_next_boundary(platform); });
            run_native("test_execution_counters_and_timing",
                       [&] { test_execution_counters_and_timing(platform); });
            run_native("test_bounded_platform_lifetimes",
                       [] { test_bounded_platform_lifetimes(); });
            std::cout << "START final-destructor-scope\n" << std::flush;
            destructor_start = std::chrono::steady_clock::now();
        }
        std::cout << "PASS final-destructor-scope "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - destructor_start).count()
                  << "s\n" << std::flush;
        std::cout << "Vulkan execution lifecycle tests passed\n";
        return 0;
    } catch (const VulkanUnavailable &) {
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
