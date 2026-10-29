#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"
#include "vulkan/descriptor_arena.h"
#include "vulkan/pipeline_cache.h"
#include "vulkan/shader_registry.h"
#include "vulkan_synchronization.h"
#include "vulkan_tensor_layout.h"
#include "vulkan_lifecycle_seam_test.h"
#include "vulkan_buffer_maintenance_test.h"

#include <cstdint>
#include <cstring>
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
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>

namespace native_allocation_test {
thread_local bool deny = false;
thread_local unsigned attempts = 0;
}

void *operator new(std::size_t size) {
    if (native_allocation_test::deny) {
        ++native_allocation_test::attempts;
        throw std::bad_alloc();
    }
    if (void *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

// Accesses the production routines and their real native table types, not a
// fixture-specific implementation. No VulkanPlatform/instance is constructed.
struct VulkanNativeServiceTestPeer {
    static void destroy_tables(const pytorch_vulkan::VulkanQuiescence &empty,
                               bool invalidated, bool deny_allocations,
                               unsigned &destroyed, unsigned mask = 7,
                               std::size_t pending = 0, std::size_t quarantined = 0) {
        const auto device = reinterpret_cast<VkDevice>(uintptr_t{1});
        VulkanPipelineCache::NativePipelines pipelines;
        VulkanPipelineCache::NativeLayouts layouts;
        VulkanPipelineCache::NativeLru lru;
        PipelineKey key; key.shader_identity = "fake-native-pipeline";
        lru.push_back(key);
        pipelines.emplace(key, VulkanPipelineCache::Entry{
            reinterpret_cast<VkPipeline>(uintptr_t{2}), lru.begin()});
        layouts.emplace(PipelineLayoutKey{}, VulkanPipelineCache::LayoutEntry{
            reinterpret_cast<VkPipelineLayout>(uintptr_t{3})});
        VulkanShaderRegistry::NativeModules modules;
        modules.emplace(ShaderKey{"fake-native-module", 4},
                        reinterpret_cast<VkShaderModule>(uintptr_t{4}));
        std::vector<DescriptorArena::Pool> pools(1);
        pools.front().handle = reinterpret_cast<VkDescriptorPool>(uintptr_t{5});
        const auto previous_deny = native_allocation_test::deny;
        native_allocation_test::deny = deny_allocations;
        try {
            if (mask & 1) VulkanPipelineCache::destroy_empty_native_tables(device, empty, invalidated, pending, 0,
                pipelines, layouts, lru, &destroyed,
                [](void *p, VkDevice, VkPipeline) { ++*static_cast<unsigned *>(p); },
                [](void *p, VkDevice, VkPipelineLayout) { ++*static_cast<unsigned *>(p); });
            if (mask & 2) VulkanShaderRegistry::destroy_empty_native_table(device, empty, invalidated, modules,
                &destroyed, [](void *p, VkDevice, VkShaderModule) { ++*static_cast<unsigned *>(p); });
            if (mask & 4) DescriptorArena::destroy_empty_native_pools(device, empty, invalidated, pending, quarantined, pools,
                &destroyed, [](void *p, VkDevice, VkDescriptorPool) { ++*static_cast<unsigned *>(p); });
        } catch (...) {
            native_allocation_test::deny = previous_deny;
            throw;
        }
        native_allocation_test::deny = previous_deny;
        if (((mask & 1) && (!pipelines.empty() || !layouts.empty() || !lru.empty())) ||
            ((mask & 2) && !modules.empty()) || ((mask & 4) && !pools.empty()))
            throw std::runtime_error("native service tables were not emptied in place");
    }
};

namespace {

struct FakeExecution {
    struct WaitCapture {
        VkSemaphore wait = VK_NULL_HANDLE;
        VkPipelineStageFlags wait_stage = 0;
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<VkFence, bool> fences;
    std::unordered_map<VkFence, VkResult> fence_overrides;
    std::vector<VkFence> submitted_fences;
    std::vector<WaitCapture> wait_captures;
    std::deque<VkResult> statuses;
    VkResult end_result = VK_SUCCESS;
    VkResult submit_result = VK_SUCCESS;
    VkResult idle_result = VK_SUCCESS;
    VkResult wait_result = VK_SUCCESS;
    std::function<void()> status_hook;
    std::function<void()> wait_hook;
    std::function<void()> idle_hook;
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
VkResult fake_submit(void *cookie, VkQueue, uint32_t, const VkSubmitInfo *submit,
                     VkFence fence) {
    auto &fake = *static_cast<FakeExecution *>(cookie);
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.submit_calls;
    fake.submitted_fences.push_back(fence);
    FakeExecution::WaitCapture capture;
    if (submit->waitSemaphoreCount >= 1) {
        capture.wait = submit->pWaitSemaphores[0];
        capture.wait_stage = submit->pWaitDstStageMask[0];
    }
    fake.wait_captures.push_back(capture);
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
    std::function<void()> hook;
    VkResult result;
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        ++fake.idle_calls;
        if (fake.idle_result == VK_SUCCESS)
            for (auto &fence : fake.fences) fence.second = true;
        hook = fake.idle_hook;
        result = fake.idle_result;
    }
    if (hook) hook();
    return result;
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

bool other_thread_can_lock(std::mutex &mutex);

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

// Only the device operations are injected: admission, close and capture retirement
// below all use the production coordinator and execution context.
struct LifecycleExecutionRoot {
    pytorch_vulkan::VulkanHostClock clock;
    pytorch_vulkan::VulkanRuntimeLifecycle runtime;
    std::mutex queue;
    VulkanExecutionQuarantineSink sink;
    VulkanExecutionContext context;
    bool services_destroyed = false;
    bool fail_service_cleanup = false;
    bool forbid_destruction = false;
    ~LifecycleExecutionRoot() { if (forbid_destruction) std::_Exit(98); }
    LifecycleExecutionRoot(FakeExecution &fake, pytorch_vulkan::VulkanHostClock c)
        : clock(c), runtime(c), sink(fake_sink(fake)),
          context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                  fake_handle<VkCommandPool>(3), fake_ops(fake), sink, queue) {}
    pytorch_vulkan::VulkanLifecycleRoot adapter(std::shared_ptr<LifecycleExecutionRoot> owner) {
        return {owner, &runtime,
            [](void *p, pytorch_vulkan::VulkanHostTime deadline) {
                auto &r = *static_cast<LifecycleExecutionRoot *>(p);
                r.context.synchronize_with_flush_until([&] {
                    if (r.context.recording()) r.context.submit_locked();
                }, deadline, r.clock);
            },
            [](void *p) { return static_cast<LifecycleExecutionRoot *>(p)->context.lifecycle_snapshot(); },
            [](void *p, bool closed) { static_cast<LifecycleExecutionRoot *>(p)->context.set_lifecycle_reservations_closed(closed); },
            [](void *p, pytorch_vulkan::VulkanHostTime deadline) {
                auto &r = *static_cast<LifecycleExecutionRoot *>(p);
                r.context.wait_foreign_epilogues_until(deadline, r.clock);
            },
            [](void *p) noexcept { return static_cast<LifecycleExecutionRoot *>(p)->context.callback_active_on_this_thread(); },
            [](void *) noexcept { return false; },
            [](void *p) {
                auto &r = *static_cast<LifecycleExecutionRoot *>(p);
                pytorch_vulkan::VulkanCleanupScope cleanup(r.runtime, pytorch_vulkan::VulkanCleanupKind::AuditedNativeOnly);
                r.context.teardown_empty_native_services([&] {
                    if (r.fail_service_cleanup) {
                        native_allocation_test::deny = true;
                        (void)::operator new(sizeof(int)); // Must fault70, never noexcept terminate/unwind root.
                    }
                    r.services_destroyed = true;
                });
            }};
    }
};

void test_lifecycle_cycle() {
    FakeExecution fake;
    auto clock = pytorch_vulkan::vulkan_host_clock();
    pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
    auto root = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
        auto r = std::make_shared<LifecycleExecutionRoot>(fake, clock);
        return r->adapter(r);
    }));
    std::weak_ptr<LifecycleExecutionRoot> weak = root;
    root->context.begin("lifecycle-cycle");
    root->context.defer_destruction([owner = root] {});
    root.reset();
    coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
    expect(weak.expired(), "production close must retire real execution captures");
    expect(fake.quarantine_calls == 0, "healthy close cannot quarantine");
    expect(fake.submit_calls == 1 && fake.wait_calls == 1,
           "close must submit and wait recording, not cancel it");
    expect(fake.idle_calls == 0 && fake.freed_buffers == 2,
           "empty-native teardown must destroy handles without general destructor drain");
}

void test_lifecycle_snapshot() {
    FakeExecution fake;
    LifecycleExecutionRoot root(fake, pytorch_vulkan::vulkan_host_clock());
    auto &c = root.context;
    expect(c.lifecycle_snapshot().healthy_empty(), "initial snapshot not empty");
    c.begin();
    expect(c.lifecycle_snapshot().recording, "recording absent from snapshot");
    c.defer_destruction([] {});
    expect(c.lifecycle_snapshot().deferred_callbacks == 1, "deferred callback absent");
    c.submit();
    c.retain_until_completion([] {});
    auto s = c.lifecycle_snapshot();
    expect(s.submitted_slots == 1 && s.ring_callbacks == 1 && s.completion_callbacks == 1,
           "submitted/ring/completion snapshot incomplete");
    c.begin(); c.submit();
    expect(c.lifecycle_snapshot().submitted_slots == 2, "snapshot omitted second ring slot");
    { std::lock_guard<std::mutex> q(root.queue); c.wait_device_locked(); }
    s = c.lifecycle_snapshot();
    expect(s.ready_callbacks == 2 && !s.healthy_empty(), "ready callbacks omitted");
    c.drain_ready_cleanup();
    auto reservation = c.reserve_transfer();
    s = c.lifecycle_snapshot();
    expect(c.pending_count() == 0 && s.reservations == 1 && !s.healthy_empty(),
           "pending_count cannot substitute for stable quiescence");
    c.begin();
    std::vector<std::shared_ptr<void>> owners;
    c.defer_transfer(std::move(reservation), std::move(owners));
    expect(c.lifecycle_snapshot().transfer_records == 1, "transfer list absent");
    c.submit(); c.synchronize();
    expect(c.lifecycle_snapshot().healthy_empty(), "successful publication left state");
}

template<class Function> void lifecycle_expect_exit70(Function function);

void test_lifecycle_reservation_close() {
    FakeExecution fake;
    LifecycleExecutionRoot root(fake, pytorch_vulkan::vulkan_host_clock());
    auto first = root.context.reserve_transfer();
    auto second = root.context.reserve_transfer();
    std::atomic<bool> rejected{false};
    std::atomic<bool> capture_released_unlocked{false};
    auto waiter = std::async(std::launch::async, [&] {
        pytorch_vulkan::VulkanOperationScope operation(root.runtime);
        try {
            auto capture = std::shared_ptr<int>(new int(1), [&](int *p) {
                // Would deadlock if closing destroyed waiter captures inside the
                // execution/reservation/queue chain rather than unwinding caller.
                capture_released_unlocked = root.context.lifecycle_snapshot().reservation_closing;
                delete p;
            });
            auto slot = root.context.reserve_transfer();
        }
        catch (const std::logic_error &) { rejected = true; }
        operation.finish();
    });
    { std::unique_lock<std::mutex> lock(fake.mutex);
      fake.cv.wait(lock, [&] { return fake.reservation_waits != 0; }); }
    root.runtime.close_admission();
    root.context.set_lifecycle_reservations_closed(true);
    waiter.get();
    expect(rejected && capture_released_unlocked && root.runtime.outer_permits() == 0,
           "closing failed to wake admitted waiter for unlocked capture unwind");
    auto s = root.context.lifecycle_snapshot();
    expect(s.reservations == 2 && s.reservation_closing && !s.healthy_empty(),
           "unattached leases cannot be ignored or consumed by close");
    first = {}; second = {};
    expect(root.context.lifecycle_snapshot().healthy_empty(), "closed gate is not poison");
    root.context.set_lifecycle_reservations_closed(false);
    auto reopened = root.context.reserve_transfer();
    expect(static_cast<bool>(reopened), "temporary barrier did not reopen reservation gate");
    lifecycle_expect_exit70([] {
        FakeExecution fake;
        auto clock = pytorch_vulkan::vulkan_host_clock();
        pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
        auto r = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
            auto owner = std::make_shared<LifecycleExecutionRoot>(fake, clock); return owner->adapter(owner);
        }));
        auto unattached = r->context.reserve_transfer();
        r.reset();
        coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
    });
}

struct LifecyclePausedDestructor {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    void pause() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true; cv.notify_all();
        cv.wait(lock, [&] { return released; });
    }
    void await() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return entered; }); }
    void release() { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
};
struct LifecycleDestructorCapture {
    LifecyclePausedDestructor &pause;
    ~LifecycleDestructorCapture() { pause.pause(); }
};

struct LifecycleDeadlineClock {
    std::atomic<int64_t> elapsed{0};
    unsigned waits = 0;
    pytorch_vulkan::VulkanHostClock clock() {
        return {this,
            [](void *p) noexcept { return pytorch_vulkan::VulkanHostTime{} + std::chrono::nanoseconds(static_cast<LifecycleDeadlineClock *>(p)->elapsed.load()); },
            [](void *p, std::condition_variable &, std::unique_lock<std::mutex> &,
               pytorch_vulkan::VulkanHostTime deadline) {
                auto &c = *static_cast<LifecycleDeadlineClock *>(p);
                ++c.waits;
                if (deadline != pytorch_vulkan::VulkanHostTime{} + std::chrono::seconds(20)) std::_Exit(98);
                c.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count();
            }};
    }
};

template<class Function> void lifecycle_expect_exit70(Function function) {
    const auto pid = fork();
    expect(pid >= 0, "CPU policy child fork failed");
    if (pid == 0) { function(); std::_Exit(99); }
    int status = 0;
    expect(waitpid(pid, &status, 0) == pid, "CPU policy child wait failed");
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 70, "terminal policy child did not exit70");
}

void test_lifecycle_foreign_pump() {
    FakeExecution fake;
    auto clock = pytorch_vulkan::vulkan_host_clock();
    pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
    auto root = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
        auto r = std::make_shared<LifecycleExecutionRoot>(fake, clock); return r->adapter(r);
    }));
    std::weak_ptr<LifecycleExecutionRoot> weak = root;
    LifecyclePausedDestructor pause;
    auto capture = std::shared_ptr<LifecycleDestructorCapture>(new LifecycleDestructorCapture{pause});
    root->context.begin();
    root->context.defer_destruction([capture, owner = root] {});
    capture.reset();
    root->context.submit();
    auto *context = &root->context;
    { std::lock_guard<std::mutex> queue(root->queue); context->wait_device_locked(); }
    auto pump = std::async(std::launch::async, [&] { context->drain_ready_cleanup(); });
    pause.await();
    const auto paused_snapshot = context->lifecycle_snapshot();
    root.reset();
    auto close = std::async(std::launch::async, [&] { coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly); });
    { std::unique_lock<std::mutex> lock(fake.mutex); fake.cv.wait(lock, [&] { return fake.pump_waits != 0; }); }
    const bool retained_while_paused = !weak.expired();
    pause.release(); pump.get(); close.get();
    expect(paused_snapshot.pump_active && !paused_snapshot.healthy_empty(),
           "capture destructor epilogue must remain visible after invocation");
    expect(retained_while_paused, "root released before foreign destructor epilogue");
    expect(weak.expired(), "root survived completed foreign epilogue close");

    // The bounded synchronization path must also use the supplied clock rather
    // than the legacy unbounded foreign-pump CV wait. This is not a GPU timeout.
    FakeExecution bounded_fake;
    LifecycleDeadlineClock host;
    LifecycleExecutionRoot bounded(bounded_fake, host.clock());
    LifecyclePausedDestructor bounded_pause;
    auto bounded_capture = std::shared_ptr<LifecycleDestructorCapture>(new LifecycleDestructorCapture{bounded_pause});
    bounded.context.begin(); bounded.context.defer_destruction([bounded_capture] {});
    bounded_capture.reset(); bounded.context.submit();
    { std::lock_guard<std::mutex> queue(bounded.queue); bounded.context.wait_device_locked(); }
    auto bounded_pump = std::async(std::launch::async, [&] { bounded.context.drain_ready_cleanup(); });
    bounded_pause.await();
    bool timed_out = false;
    try { bounded.context.synchronize_with_flush_until({}, pytorch_vulkan::VulkanHostTime{} + std::chrono::seconds(20), host.clock()); }
    catch (const pytorch_vulkan::VulkanHostCloseTimeout &) { timed_out = true; }
    bounded_pause.release(); bounded_pump.get();
    expect(timed_out && host.waits == 1, "foreign synchronization renewed or ignored shared host deadline");

    lifecycle_expect_exit70([] {
        FakeExecution fake;
        LifecycleDeadlineClock host;
        pytorch_vulkan::VulkanLifecycleCoordinator coordinator(host.clock());
        auto root = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
            auto r = std::make_shared<LifecycleExecutionRoot>(fake, host.clock()); return r->adapter(r);
        }));
        LifecyclePausedDestructor pause;
        auto capture = std::shared_ptr<LifecycleDestructorCapture>(new LifecycleDestructorCapture{pause});
        root->context.begin(); root->context.defer_destruction([capture, owner = root] {});
        capture.reset(); root->context.submit();
        auto *context = &root->context;
        { std::lock_guard<std::mutex> queue(root->queue); context->wait_device_locked(); }
        auto pump = std::async(std::launch::async, [&] { context->drain_ready_cleanup(); });
        pause.await(); root.reset();
        // Intentional terminal child: root and paused capture must remain intact
        // at exit70, not be unwound or released to make the timeout look clean.
        coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
    });
}

void test_lifecycle_self_close() {
    FakeExecution fake;
    auto clock = pytorch_vulkan::vulkan_host_clock();
    pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
    auto root = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
        auto r = std::make_shared<LifecycleExecutionRoot>(fake, clock); return r->adapter(r);
    }));
    { pytorch_vulkan::VulkanOperationScope operation(root->runtime);
      expect_rejected([&] { coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly); }, "own permit close accepted");
      operation.finish(); }
    root->context.begin();
    root->context.defer_destruction([&] {
        expect(root->context.callback_active_on_this_thread(), "callback identity absent");
        expect(!root->context.query_complete(), "active-pump query must remain false");
        expect(static_cast<bool>(coordinator.observe()), "callback observation deadlocked/lost root");
        expect_rejected([&] { coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly); }, "own callback close accepted");
    });
    root->context.submit(); root->context.synchronize();
    root.reset(); coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
}

void test_lifecycle_sticky() {
    for (int fault = 0; fault != 3; ++fault) lifecycle_expect_exit70([&] {
        FakeExecution fake;
        auto clock = pytorch_vulkan::vulkan_host_clock();
        pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
        auto root = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
            auto r = std::make_shared<LifecycleExecutionRoot>(fake, clock); return r->adapter(r);
        }));
        auto slot = root->context.reserve_transfer();
        root->context.begin();
        std::vector<std::shared_ptr<void>> owners{std::make_shared<int>(42)};
        std::weak_ptr<void> retained = owners.front();
        root->context.defer_transfer(std::move(slot), std::move(owners), [] { throw std::runtime_error("first publication error"); });
        root->context.submit();
        if (fault == 1) fake.wait_result = VK_ERROR_DEVICE_LOST;
        if (fault == 2) { fake.wait_result = VK_TIMEOUT; fake.idle_result = VK_ERROR_UNKNOWN; }
        expect_rejected([&] { root->context.synchronize(); }, "fault was not propagated");
        const auto s = root->context.lifecycle_snapshot();
        expect(!s.healthy_empty(), "sticky error cannot become healthy after safe release");
        if (fault == 0) {
            expect(s.sticky_error && retained.expired(), "safe publication error lost or payload not released");
            try { root->context.synchronize(); expect(false, "sticky error vanished"); }
            catch (const std::runtime_error &e) { expect(std::string(e.what()) == "first publication error", "first error replaced"); }
        } else {
            expect(s.invalidated && s.reservation_poisoned && fake.quarantine_calls == 1 && !retained.expired(),
                   "uncertain payload was not retained intact");
            root->context.set_lifecycle_reservations_closed(true);
            root->context.set_lifecycle_reservations_closed(false);
            expect_rejected([&] { auto reservation = root->context.reserve_transfer(); },
                            "temporary gate reopening revived poisoned execution");
        }
        root.reset(); coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
    });
}

void test_lifecycle_native_empty() {
    FakeExecution fake;
    LifecycleExecutionRoot root(fake, pytorch_vulkan::vulkan_host_clock());
    int destroyed = 0, invoked = 0;
    auto capture = std::shared_ptr<int>(new int(1), [&](int *p) { ++destroyed; delete p; });
    root.context.begin(); root.context.defer_destruction([capture, &invoked] { ++invoked; }); capture.reset();
    bool service_called = false;
    expect_rejected([&] { root.context.teardown_empty_native_services([&] { service_called = true; }); }, "nonempty native-only cleanup accepted");
    expect(destroyed == 0 && invoked == 0 && !service_called && fake.idle_calls == 0 &&
               fake.freed_buffers == 0 && fake.destroyed_fences == 0 && fake.destroyed_semaphores == 0,
           "native precheck touched a service, native handle or arbitrary capture");
    root.context.submit(); root.context.synchronize();
    root.context.teardown_empty_native_services([&] { root.services_destroyed = true; });
    expect(root.services_destroyed && destroyed == 1 && invoked == 1 && fake.idle_calls == 0 && fake.freed_buffers == 2,
           "empty-native teardown did not exclusively destroy native state");
    unsigned native_destroyed = 0;
    const auto attempts_before = native_allocation_test::attempts;
    VulkanNativeServiceTestPeer::destroy_tables(root.context.lifecycle_snapshot(), false, true, native_destroyed);
    expect(native_destroyed == 4 && native_allocation_test::attempts == attempts_before,
           "native table cleanup allocated or skipped a handle");
    native_destroyed = 0;
    auto nonempty = root.context.lifecycle_snapshot(); nonempty.deferred_callbacks = 1;
    for (unsigned mask : {1u, 2u, 4u}) {
        expect_rejected([&] { VulkanNativeServiceTestPeer::destroy_tables(nonempty, false, false, native_destroyed, mask); },
                        "native tables accepted nonempty execution");
        expect(native_destroyed == 0, "failed execution precheck destroyed native handles");
        expect_rejected([&] { VulkanNativeServiceTestPeer::destroy_tables(root.context.lifecycle_snapshot(), true, false, native_destroyed, mask); },
                        "native tables accepted invalidated services");
        expect(native_destroyed == 0, "failed service precheck destroyed native handles");
    }
    for (unsigned mask : {1u, 4u}) {
        expect_rejected([&] { VulkanNativeServiceTestPeer::destroy_tables(root.context.lifecycle_snapshot(), false, false, native_destroyed, mask, 1); },
                        "native tables accepted pending resources");
        expect(native_destroyed == 0, "pending-resource precheck destroyed native handles");
    }
    expect_rejected([&] { VulkanNativeServiceTestPeer::destroy_tables(root.context.lifecycle_snapshot(), false, false, native_destroyed, 4, 0, 1); },
                    "native descriptor tables accepted quarantine");
    expect(native_destroyed == 0, "quarantine precheck destroyed native handles");
    {
        FakeExecution allocation_fake;
        LifecycleExecutionRoot allocation_root(allocation_fake, pytorch_vulkan::vulkan_host_clock());
        std::function<void()> cleanup = [&] { ++native_destroyed; };
        native_allocation_test::deny = true;
        try { allocation_root.context.teardown_empty_native_services(cleanup); }
        catch (...) { native_allocation_test::deny = false; throw; }
        native_allocation_test::deny = false;
        expect(native_allocation_test::attempts == attempts_before && allocation_fake.freed_buffers == 2,
               "empty-native execution teardown allocated");
    }
    lifecycle_expect_exit70([] {
        FakeExecution fake;
        auto clock = pytorch_vulkan::vulkan_host_clock();
        pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
        auto r = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
            auto owner = std::make_shared<LifecycleExecutionRoot>(fake, clock); return owner->adapter(owner);
        }));
        auto capture = std::shared_ptr<int>(new int(1), [](int *) { std::_Exit(98); });
        r->context.begin();
        r->context.defer_destruction([capture] { std::_Exit(97); });
        capture.reset(); r.reset();
        coordinator.close(pytorch_vulkan::VulkanCloseBoundary::PythonNative);
    });
    lifecycle_expect_exit70([] {
        FakeExecution fake;
        auto clock = pytorch_vulkan::vulkan_host_clock();
        pytorch_vulkan::VulkanLifecycleCoordinator coordinator(clock);
        auto r = std::static_pointer_cast<LifecycleExecutionRoot>(coordinator.acquire([&] {
            auto owner = std::make_shared<LifecycleExecutionRoot>(fake, clock); return owner->adapter(owner);
        }));
        r->fail_service_cleanup = true;
        r->forbid_destruction = true;
        r.reset();
        coordinator.close(pytorch_vulkan::VulkanCloseBoundary::NativeEarly);
    });
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

void test_fake_host_access_publishes_work_completed_by_own_flush() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    std::mutex transfer_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(21), fake_handle<VkQueue>(22),
                                   fake_handle<VkCommandPool>(23), ops, sink, queue_mutex);
    fake.status_hook = [&] {
        expect(!other_thread_can_lock(queue_mutex),
               "host-access completion poll escaped shared queue serialization");
    };
    std::atomic<int> owner_released{0};
    std::atomic<int> publications{0};
    auto owner = std::shared_ptr<void>(new int(1), [&](void *value) {
        delete static_cast<int *>(value);
        ++owner_released;
    });
    auto reservation = context.reserve_transfer();
    context.begin();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.defer_transfer(std::move(reservation), {owner}, [&] { ++publications; });
    }
    owner.reset();
    auto guard = context.acquire_host_access(
        [&] { context.submit_locked(); }, [&] { expect(!other_thread_can_lock(queue_mutex), "host wait not serialized"); fake.signal_all(); });
    expect(guard.owns_lock() && publications == 1 && owner_released == 1,
           "host guard returned before its flush publication and owner release");
    expect(other_thread_can_lock(transfer_mutex), "host publication retained transfer lock");
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

void test_fake_host_access_waits_for_foreign_publication_and_capture_release() {
    for (const bool pause_capture : {false, true}) {
        ControlledGate gate;
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        std::mutex transfer_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        JoinedFakeWorkers workers(gate, fake);
        std::atomic<bool> locks_available{false};
        std::atomic<bool> acquired{false};
        std::atomic<int> destroyed{0};
        auto pause = [&] {
            locks_available = other_thread_can_lock(queue_mutex) &&
                              other_thread_can_lock(transfer_mutex);
            gate.pause();
        };
        std::shared_ptr<void> owner(new int(1), [&](void *value) {
            delete static_cast<int *>(value);
            ++destroyed;
            if (pause_capture) pause();
        });
        context.begin();
        context.defer_destruction([&, owner] {
            if (!pause_capture) pause();
        });
        owner.reset();
        context.submit();
        fake.signal_all();
        workers.spawn([&] { context.retire_completed(); });
        gate.wait_entered();
        workers.spawn([&] {
            auto guard = context.acquire_host_access([] {}, [&] { fake.signal_all(); });
            acquired = guard.owns_lock();
        });
        {
            std::unique_lock<std::mutex> lock(fake.mutex);
            fake.cv.wait(lock, [&] { return fake.pump_waits != 0 || fake.worker_failed; });
            expect(!fake.worker_failed, "host access failed before foreign-pump wait");
        }
        expect(!acquired, "host access returned while foreign publication/capture release was paused");
        expect(locks_available, "foreign publication/capture release retained a backend lock");
        gate.open();
        workers.join();
        expect(acquired && destroyed == 1,
               "host access did not wait through last capture release");
    }
}

void test_fake_host_access_rechecks_intervening_producer() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    int flushes = 0;
    context.begin();
    context.defer_destruction([&] {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.begin_locked();
    });
    auto guard = context.acquire_host_access(
        [&] {
            ++flushes;
            if (context.recording()) context.submit_locked();
        }, [&] { fake.signal_all(); });
    expect(guard.owns_lock() && flushes == 2 && !context.recording(),
           "host access returned after a publication-admitted producer without rechecking it");
}

void test_fake_host_access_error_recovery_keeps_queue_serialized() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    context.begin();
    context.submit();
    fake.statuses = {VK_ERROR_OUT_OF_HOST_MEMORY};
    std::atomic<bool> recovery_queue_held{false};
    fake.idle_hook = [&] { recovery_queue_held = !other_thread_can_lock(queue_mutex); };
    bool failed = false;
    try {
        (void)context.acquire_host_access([] {}, [&] { fake.signal_all(); });
    } catch (const std::runtime_error &) {
        failed = true;
    }
    expect(failed && recovery_queue_held,
           "fence-error recovery was not serialized by the shared queue mutex");
}

void test_fake_host_access_reentry_and_sticky_callback_error() {
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        int reentry_rejected = 0;
        context.begin();
        context.defer_destruction([&] {
            try {
                (void)context.acquire_host_access([] {}, [] {});
            } catch (const std::logic_error &) {
                ++reentry_rejected;
            }
        });
        auto guard = context.acquire_host_access(
            [&] { context.submit_locked(); }, [&] { fake.signal_all(); });
        expect(guard.owns_lock() && reentry_rejected == 1,
               "same-thread host-access reentry did not reject without deadlocking");
    }
    {
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
            throw std::runtime_error("sticky host publication error");
        });
        const auto acquire = [&] {
            return context.acquire_host_access(
                [&] { if (context.recording()) context.submit_locked(); },
                [&] { fake.signal_all(); });
        };
        bool first_failed = false;
        try { (void)acquire(); }
        catch (const std::runtime_error &) { first_failed = true; }
        bool repeated_failed = false;
        try { (void)acquire(); }
        catch (const std::runtime_error &) { repeated_failed = true; }
        expect(first_failed && repeated_failed && calls == 1 &&
                   other_thread_can_lock(queue_mutex),
               "host access lost sticky callback error or retained queue on failure");
    }
    {
        FakeExecution fake;
        auto ops = fake_ops(fake);
        auto sink = fake_sink(fake);
        std::mutex queue_mutex;
        VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                       fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
        context.teardown_with_services([] {});
        int callbacks = 0;
        bool closing_rejected = false;
        try {
            (void)context.acquire_host_access([&] { ++callbacks; }, [&] { ++callbacks; });
        } catch (const std::logic_error &) {
            closing_rejected = true;
        }
        expect(closing_rejected && callbacks == 0,
               "host access did not explicitly reject a closing execution context");
    }
}

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
        context.defer_transfer(std::move(reservation), {}, [&order, value] { order.push_back(value); });
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

// The execution-context submission semaphore must wait at every device stage
// that can consume a prior submission on this queue: compute and transfer.
// HOST is deliberately absent; host completion is ordered by the fence wait and
// the host-access exclusion, never by the submit wait mask.
void test_fake_submit_waits_for_compute_and_transfer() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(1), fake_handle<VkQueue>(2),
                                   fake_handle<VkCommandPool>(3), ops, sink, queue_mutex);
    context.begin();
    context.submit();
    context.begin();
    context.submit();
    expect(fake.wait_captures.size() == 2, "fake submit did not capture both submissions");
    expect(fake.wait_captures[0].wait == VK_NULL_HANDLE && fake.wait_captures[0].wait_stage == 0,
           "first submission waited on a nonexistent semaphore");
    const auto &chained = fake.wait_captures[1];
    expect(chained.wait != VK_NULL_HANDLE,
           "second submission did not wait on the prior signal semaphore");
    const VkPipelineStageFlags expected_wait_stages =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    expect(chained.wait_stage == expected_wait_stages,
           "submission wait mask did not cover both compute and transfer consumers");
    expect((chained.wait_stage & VK_PIPELINE_STAGE_HOST_BIT) == 0,
           "submission wait mask included the HOST stage");
}

// Pure descriptor tests: the expected masks are written here as independent
// literal Vulkan requirements (Vulkan 1.3 §7.9 same-queue availability and
// visibility), never derived from the helpers under test. No fake
// VkCommandBuffer is ever passed to real Vulkan.
void test_shared_barrier_descriptors() {
    const VkPipelineStageFlags compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    const VkPipelineStageFlags transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkPipelineStageFlags host = VK_PIPELINE_STAGE_HOST_BIT;
    const VkAccessFlags shader_write = VK_ACCESS_SHADER_WRITE_BIT;
    const VkAccessFlags shader_read = VK_ACCESS_SHADER_READ_BIT;
    const VkAccessFlags shader_rw = shader_read | shader_write;
    const VkAccessFlags transfer_write = VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkAccessFlags transfer_read = VK_ACCESS_TRANSFER_READ_BIT;
    const VkAccessFlags transfer_rw = transfer_read | transfer_write;
    const VkAccessFlags host_read = VK_ACCESS_HOST_READ_BIT;

    const VkPipelineStageFlags expected_wait = compute | transfer;
    expect(vulkan_execution_wait_stages() == expected_wait,
           "execution submit wait stages diverged from literal compute|transfer");
    expect((vulkan_execution_wait_stages() & host) == 0,
           "execution submit wait stages included the HOST stage");
    expect((expected_wait & host) == 0,
           "test literal itself must not admit a HOST wait stage");

    VulkanMemoryBarrierSpec transfer_ready =
        vulkan_transfer_ready_spec();
    const VulkanMemoryBarrierSpec expected_transfer{
        compute | transfer, shader_write | transfer_write, transfer, transfer_rw};
    expect(transfer_ready.src_stage == expected_transfer.src_stage &&
               transfer_ready.src_access == expected_transfer.src_access &&
               transfer_ready.dst_stage == expected_transfer.dst_stage &&
               transfer_ready.dst_access == expected_transfer.dst_access,
           "transfer-ready barrier diverged from literal compute|transfer -> transfer");

    transfer_ready = vulkan_compute_ready_spec();
    const VulkanMemoryBarrierSpec expected_compute{
        compute | transfer, shader_write | transfer_write, compute, shader_rw};
    expect(transfer_ready.src_stage == expected_compute.src_stage &&
               transfer_ready.src_access == expected_compute.src_access &&
               transfer_ready.dst_stage == expected_compute.dst_stage &&
               transfer_ready.dst_access == expected_compute.dst_access,
           "compute-ready barrier diverged from literal compute|transfer -> compute");

    transfer_ready = vulkan_host_read_spec();
    const VulkanMemoryBarrierSpec expected_host{transfer, transfer_write, host, host_read};
    expect(transfer_ready.src_stage == expected_host.src_stage &&
               transfer_ready.src_access == expected_host.src_access &&
               transfer_ready.dst_stage == expected_host.dst_stage &&
               transfer_ready.dst_access == expected_host.dst_access,
           "host-read barrier diverged from literal transfer -> HOST readback");
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

// Audit (i): cancellation of R2 cannot destroy an owner still used by R1.
void test_fake_cancel_preserves_cleanup_for_earlier_submission() {
    FakeExecution fake;
    auto ops = fake_ops(fake);
    auto sink = fake_sink(fake);
    std::mutex queue_mutex;
    VulkanExecutionContext context(fake_handle<VkDevice>(141), fake_handle<VkQueue>(142),
                                   fake_handle<VkCommandPool>(143), ops, sink, queue_mutex);
    int destroyed = 0;
    context.begin();
    context.submit(); // R1 remains genuinely unsignaled in this synthetic fixture.
    context.begin(); // R2
    expect(context.retain_until_completion([&] { ++destroyed; }), "audit callback not retained");
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        context.cancel_locked();
    }
    context.drain_ready_cleanup();
    expect(destroyed == 0, "cancel R2 destroyed an owner while earlier R1 is unsignaled");
    fake.signal_all();
    expect(context.query_complete() && destroyed == 1, "audit cleanup did not retire exactly once");
}

// Task 2 entry witnesses: fake only below the real execution context. Removing
// attachment-before-record, fence gating, or unlocked owner release breaks these.
struct EnqueueFixture {
    FakeExecution fake;
    VulkanExecutionOps ops = fake_ops(fake);
    VulkanExecutionQuarantineSink sink = fake_sink(fake);
    std::mutex queue;
    std::atomic<int> published{0}, released{0};
    std::atomic<bool> release_unlocked{true};
    VulkanExecutionContext context{fake_handle<VkDevice>(151), fake_handle<VkQueue>(152),
                                   fake_handle<VkCommandPool>(153), ops, sink, queue};

    std::shared_ptr<void> owner() {
        return std::shared_ptr<void>(new int(1), [this](void *p) {
            release_unlocked = release_unlocked && other_thread_can_lock(queue);
            (void)context.pending_count(); // also proves the execution mutex is free
            delete static_cast<int *>(p);
            ++released;
        });
    }
    VulkanCompletionId enqueue(bool submit = true,
                              std::function<void(VkCommandBuffer)> record = [](VkCommandBuffer) {},
                              std::function<void()> publish = {}) {
        auto reservation = context.reserve_transfer();
        std::vector<std::shared_ptr<void>> owners{owner()};
        if (!publish) publish = [this] { ++published; };
        std::lock_guard<std::mutex> lock(queue);
        return context.enqueue_transfer_locked(std::move(reservation), std::move(owners),
                                                std::move(publish), record, submit);
    }
    void reusable() {
        expect(released != 0 && release_unlocked, "slot reuse preceded unlocked owner release");
        auto a = context.reserve_transfer();
        auto b = context.reserve_transfer();
        expect(a && b, "safe retirement leaked transfer slots");
    }
    std::string error(const std::function<void()> &operation) {
        try { operation(); } catch (const std::exception &e) { return e.what(); }
        return {};
    }
    void reclaim() {
        // Explicit synthetic completion, never a recovery policy for hardware.
        fake.signal_all();
        fake.quarantines[0].reset();
        expect(released == 1 && published == 0 && release_unlocked,
               "fixture reclaim lost quarantine ownership");
    }
};

void test_fake_enqueue_returns_before_completion_then_publishes_once() {
    EnqueueFixture f;
    std::vector<std::string> events;
    f.enqueue(true, [&](VkCommandBuffer cb) {
        expect(cb != VK_NULL_HANDLE && f.fake.submit_calls == 0, "submit preceded recorder");
        // One open recording + one attached transfer: only Recording can have
        // been attached before commands (no submission exists yet).
        expect(f.context.recording() && f.context.pending_count() == 2,
               "recorder ran without its attached Recording transfer");
        events.push_back("record");
    }, [&] {
        expect(f.context.pending_count() != 0 && !f.context.query_complete(),
               "publication left pending list before callback finished");
        expect(other_thread_can_lock(f.queue), "publication inherited queue lock");
        ++f.published;
        events.push_back("publish");
    });
    expect(events == std::vector<std::string>{"record"} && f.fake.submit_calls == 1,
           "enqueue did not record then submit exactly once");
    expect(!f.context.query_complete() && f.published == 0 && f.released == 0,
           "enqueue returned only after completion or released pending owners");
    f.fake.signal_all();
    expect(f.context.query_complete() && f.published == 1 && f.released == 1,
           "signaled enqueue did not publish then release");
    expect(f.context.query_complete() && events == std::vector<std::string>({"record", "publish"}),
           "publication repeated or preceded recording");
    f.reusable();
}

void test_fake_enqueue_without_submit_joins_open_recording() {
    EnqueueFixture f;
    f.context.begin();
    f.enqueue(false);
    expect(f.fake.submit_calls == 0 && f.context.recording() && f.published == 0,
           "submit_now=false submitted or failed to join recording");
    f.context.submit();
    expect(f.fake.submit_calls == 1 && f.released == 0, "explicit submit lost pending owners");
    f.fake.signal_all();
    expect(f.context.query_complete() && f.published == 1, "explicit submit never published");
    f.reusable();
}

// (a) Cancelled path: caller receives original; later boundaries receive cancel error.
void test_fake_enqueue_record_throw_cancels_and_persists() {
    EnqueueFixture f;
    bool recorded = false;
    const auto original = f.error([&] {
        f.enqueue(true, [&](VkCommandBuffer) {
            recorded = true;
            throw std::runtime_error("original recorder failure");
        });
    });
    expect(recorded && original == "original recorder failure" && f.fake.submit_calls == 0,
           "recorder failure did not propagate original before submission");
    expect(f.released == 0 && f.published == 0, "cancel released owners inside queue phase");
    const auto sticky = f.error([&] { f.context.synchronize_with_flush({}); });
    expect(sticky == "Vulkan transfer recording was cancelled" && f.published == 0 &&
               f.released == 1 && f.fake.quarantine_calls == 0,
           "cancel path did not persist cancel error and safely pump owners");
    expect(f.error([&] { f.context.query_complete(); }) == sticky, "cancel error not sticky");
    f.reusable();
}

// (b) Failed, no idle recovery: end failure occurs before submission attempt.
void test_fake_enqueue_end_failure_without_idle_recovery() {
    EnqueueFixture f;
    f.fake.end_result = VK_ERROR_UNKNOWN;
    expect(!f.error([&] { f.enqueue(); }).empty(), "end failure accepted");
    expect(f.fake.idle_calls == 0 && f.fake.submit_calls == 0 && f.released == 0,
           "end failure attempted idle recovery or released under queue lock");
    expect(!f.error([&] { f.context.synchronize(); }).empty() && f.released == 1 &&
               f.published == 0 && f.fake.quarantine_calls == 0, "end failure did not pump Failed record");
    f.reusable();
}

// (b) Safe submit recovery: queue idle proves release safety, never publication.
void test_fake_enqueue_submit_failure_safe_recovery() {
    EnqueueFixture f;
    f.fake.submit_result = VK_ERROR_UNKNOWN;
    const auto original = f.error([&] { f.enqueue(); });
    expect(!original.empty() && f.fake.idle_calls == 1 && f.fake.quarantine_calls == 0,
           "submit safe recovery policy changed");
    expect(f.error([&] { f.context.synchronize(); }) == original && f.released == 1 &&
               f.published == 0, "submit failure was not sticky or published");
    f.reusable();
}

// (b) Uncertain submit recovery: retain owners and slots in quarantine.
void test_fake_enqueue_submit_failure_uncertain_recovery() {
    EnqueueFixture f;
    f.fake.submit_result = f.fake.idle_result = VK_ERROR_UNKNOWN;
    expect(!f.error([&] { f.enqueue(); }).empty() && f.fake.quarantine_calls == 1 &&
               f.released == 0 && f.published == 0, "uncertain submit did not quarantine");
    expect(f.fake.quarantines[0]->transfer_records.front()->state ==
               VulkanTransferRecord::State::Quarantined, "uncertain record state not Quarantined");
    expect_rejected([&] { f.context.reserve_transfer(); }, "quarantine admitted reservations");
    f.reclaim();
}

// (c) Safe status recovery retains the existing accepted policy.
void test_fake_enqueue_status_failure_safe_recovery() {
    EnqueueFixture f;
    f.enqueue();
    f.fake.statuses = {VK_ERROR_UNKNOWN};
    const auto first = f.error([&] { f.context.query_complete(); });
    expect(!first.empty() && f.fake.idle_calls == 1 && f.fake.quarantine_calls == 0 &&
               f.published == 0 && f.released == 1, "safe status recovery changed");
    expect(f.error([&] { f.context.query_complete(); }) == first, "status failure not sticky");
    f.reusable();
}

// (c) Uncertain status recovery, distinct from the successful idle-recovery case.
void test_fake_enqueue_status_failure_uncertain_recovery() {
    EnqueueFixture f;
    f.enqueue();
    f.fake.statuses = {VK_ERROR_UNKNOWN};
    f.fake.idle_result = VK_ERROR_UNKNOWN;
    expect(!f.error([&] { f.context.query_complete(); }).empty() &&
               f.fake.quarantine_calls == 1 && f.released == 0 && f.published == 0,
           "uncertain status did not retain owners");
    expect_rejected([&] { f.context.reserve_transfer(); }, "uncertain status freed a slot");
    f.reclaim();
}

// (c) Device loss invalidates directly, without safe idle-recovery publication.
void test_fake_enqueue_fence_device_loss_quarantines() {
    EnqueueFixture f;
    f.enqueue();
    f.fake.fence_overrides[f.fake.submitted_fences[0]] = VK_ERROR_DEVICE_LOST;
    expect(!f.error([&] { f.context.query_complete(); }).empty() &&
               f.fake.quarantine_calls == 1 && f.fake.idle_calls == 0 &&
               f.released == 0 && f.published == 0, "device loss released or published");
    expect_rejected([&] { f.context.reserve_transfer(); }, "device loss freed a slot");
    f.reclaim();
}

// (d) Publication fails, destination stays incomplete; safe owner release continues.
void test_fake_enqueue_publication_failure_is_sticky() {
    EnqueueFixture f;
    f.enqueue(true, [](VkCommandBuffer) {}, [&] {
        ++f.published;
        throw std::runtime_error("enqueue publish failed");
    });
    f.fake.signal_all();
    expect(f.error([&] { f.context.query_complete(); }) == "enqueue publish failed" &&
               f.released == 1 && f.published == 1, "publication failure lost safe cleanup");
    expect(f.error([&] { f.context.synchronize(); }) == "enqueue publish failed" &&
               f.error([&] { f.context.query_complete(); }) == "enqueue publish failed" &&
               f.published == 1, "failed destination reported complete or republished");
    f.reusable();
}

// (f) Both genuinely unsignaled fakes hold slots through the last-owner destructor.
void test_fake_enqueue_backpressure_waits_for_owner_release() {
    EnqueueFixture f;
    ControlledGate gate;
    JoinedFakeWorkers workers(gate, f.fake);
    f.fake.block_wait = true;
    auto first = f.context.reserve_transfer();
    std::vector<std::shared_ptr<void>> owners{
        std::shared_ptr<void>(new int(1), [&](void *p) {
            expect(other_thread_can_lock(f.queue), "capacity destructor inherited queue lock");
            gate.pause();
            delete static_cast<int *>(p);
            ++f.released;
        })};
    std::function<void()> action = [&] { ++f.published; };
    {
        std::lock_guard<std::mutex> lock(f.queue);
        f.context.enqueue_transfer_locked(std::move(first), std::move(owners), std::move(action),
                                         [](VkCommandBuffer) {}, true);
    }
    f.enqueue();
    std::atomic<bool> returned{false};
    workers.spawn([&] {
        auto third = f.context.reserve_transfer();
        expect(f.released >= 1, "slot admitted before last-owner destructor returned");
        returned = true;
    });
    {
        std::unique_lock<std::mutex> lock(f.fake.mutex);
        expect(f.fake.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return f.fake.reservation_waits != 0 || f.fake.worker_failed;
        }) && !f.fake.worker_failed, "third enqueue never reached reservation backpressure");
    }
    expect(!returned && f.released == 0 && f.published == 0 && f.fake.submit_calls == 2,
           "backpressure bypassed genuinely unsignaled ring bound");
    f.fake.signal_all();
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        expect(gate.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return gate.entered || gate.failed;
        }) && !gate.failed, "owner destructor did not reach controlled pause within 5s");
    }
    expect(!returned && f.released == 0, "slot released before paused owner destruction");
    gate.open();
    workers.join();
    f.context.synchronize();
    expect(returned && f.released == 2 && f.published == 2 && f.fake.submit_calls == 2,
           "backpressure did not retire bounded enqueues exactly once");
    f.reusable();
}

// (g) Reject callback re-entry even when reservation capacity is available.
void test_fake_enqueue_callback_reentry_rejected() {
    EnqueueFixture f;
    auto spare = f.context.reserve_transfer();
    bool reserve_rejected = false, enqueue_rejected = false;
    f.enqueue(true, [](VkCommandBuffer) {}, [&] {
        try { auto token = f.context.reserve_transfer(); }
        catch (const std::logic_error &) { reserve_rejected = true; }
        std::vector<std::shared_ptr<void>> owners;
        std::function<void()> publish;
        std::lock_guard<std::mutex> lock(f.queue);
        try {
            f.context.enqueue_transfer_locked(std::move(spare), std::move(owners),
                                             std::move(publish), [](VkCommandBuffer) {}, true);
        } catch (const std::logic_error &) { enqueue_rejected = true; }
        ++f.published;
    });
    spare = VulkanTransferReservation{}; // leave free capacity during the callback
    f.fake.signal_all();
    expect(f.context.query_complete() && reserve_rejected && enqueue_rejected,
           "completion callback re-entered enqueue/reservation");
    f.reusable();
}

// (h) Last-owner destruction may retain cleanup through another pending record.
void test_fake_enqueue_owner_destructor_retains_without_recursing() {
    EnqueueFixture f;
    int cleanup = 0;
    bool retained = false, incomplete = false;
    auto reservation = f.context.reserve_transfer();
    std::vector<std::shared_ptr<void>> owners{
        std::shared_ptr<void>(new int(1), [&](void *p) {
            expect(other_thread_can_lock(f.queue), "retaining destructor inherited queue lock");
            retained = f.context.retain_until_completion([&] { ++cleanup; });
            incomplete = !f.context.query_complete();
            f.context.drain_ready_cleanup(); // must not recurse into this active pump
            delete static_cast<int *>(p);
            ++f.released;
        })};
    std::function<void()> action = [&] { ++f.published; };
    {
        std::lock_guard<std::mutex> lock(f.queue);
        f.context.enqueue_transfer_locked(std::move(reservation), std::move(owners),
                                         std::move(action), [](VkCommandBuffer) {}, true);
    }
    f.enqueue();
    f.fake.fences[f.fake.submitted_fences[0]] = true;
    expect(!f.context.query_complete() && retained && incomplete && cleanup == 0 && f.released == 1,
           "last-owner retention recursed or ignored later pending work");
    f.fake.signal_all();
    expect(f.context.query_complete() && cleanup == 1 && f.released == 2,
           "retained destructor cleanup did not drain exactly once");
    f.reusable();
}

void test_fake_enqueue_attachment_failure_preserves_inputs() {
    EnqueueFixture f;
    auto reservation = f.context.reserve_transfer();
    std::vector<std::shared_ptr<void>> owners{f.owner()};
    std::function<void()> action = [&] { ++f.published; };
    bool recorded = false;
    f.fake.fail_attachment_allocations = 1;
    {
        std::lock_guard<std::mutex> lock(f.queue);
        expect_rejected([&] {
            f.context.enqueue_transfer_locked(std::move(reservation), std::move(owners),
                std::move(action), [&](VkCommandBuffer) { recorded = true; }, true);
        }, "forced enqueue attachment failure not surfaced");
    }
    expect(reservation && owners.size() == 1 && action && !recorded && f.released == 0,
           "failed attachment consumed inputs or recorded unowned commands");
    f.context.cancel();
    owners.clear();
    reservation = VulkanTransferReservation{};
    f.reusable();
}

// HS: real tiny copy, but completion observation is synthetic, NOT proof that
// the hardware fence remained unsignaled. Never perform blocking host access
// while held: enqueue, query, drop references, release, then synchronize/read.
void test_platform_completion_hold_and_real_wait_counter(VulkanPlatform &platform) {
    auto &context = platform.execution_context();
    platform.synchronize_execution();
    constexpr auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto source = std::make_shared<VulkanBuffer>(platform, 2 * sizeof(float), host);
    auto destination = std::make_shared<VulkanBuffer>(platform, 2 * sizeof(float), host);
    const float seed[] = {2.0F, 7.0F};
    source->write(seed, sizeof(seed));
    std::weak_ptr<VulkanBuffer> weak = source;
    int published = 0;
    auto reservation = context.reserve_transfer();
    std::vector<std::shared_ptr<void>> owners{source, destination};
    std::function<void()> publish = [&] { ++published; };
    const auto waits = platform.execution_fence_wait_count();
    const auto pressure = platform.transfer_reservation_wait_count();
    const auto start = std::chrono::steady_clock::now();
    platform.set_completion_hold_for_testing(true);
    try {
        {
            std::lock_guard<std::mutex> lock(platform.queue_mutex());
            context.enqueue_transfer_locked(std::move(reservation), std::move(owners),
                std::move(publish), [&](VkCommandBuffer cb) {
                    VkBufferCopy copy{0, 0, sizeof(seed)};
                    vkCmdCopyBuffer(cb, source->buffer(), destination->buffer(), 1, &copy);
                    vulkan_record_memory_barrier(cb, vulkan_host_read_spec());
                }, true);
        }
        source.reset();
        expect(!platform.query_execution() && published == 0 && !weak.expired(),
               "hold did not retain owners and suppress observed completion");
        expect(platform.execution_fence_wait_count() == waits &&
                   platform.transfer_reservation_wait_count() == pressure,
               "query or unpressured enqueue counted a wait");
        expect(std::chrono::steady_clock::now() - start < std::chrono::seconds(5),
               "hold state transition exceeded 5s bound");
    } catch (...) {
        platform.set_completion_hold_for_testing(false);
        throw;
    }
    platform.set_completion_hold_for_testing(false);
    const auto release = std::chrono::steady_clock::now();
    platform.synchronize_execution();
    expect(std::chrono::steady_clock::now() - release < std::chrono::seconds(5),
           "post-release completion exceeded 5s bound");
    expect(platform.execution_fence_wait_count() == waits + 1 && published == 1 && weak.expired(),
           "real wait wrapper failed to count or retire once");
    expect(platform.completion_hold_expiry_count_for_testing() == 0, "completion hold watchdog expired");
    float actual[2]{};
    destination->read(actual, sizeof(actual));
    expect(actual[0] == 2.0F && actual[1] == 7.0F, "held real copy lost CPU seed");

    // H: with hold OFF, real capacity backpressure must count the wait even
    // when the oldest real fence already completed before the CPU reached it.
    for (int i = 0; i < 2; ++i) {
        auto slot = context.reserve_transfer();
        std::vector<std::shared_ptr<void>> retained{destination};
        std::function<void()> done;
        std::lock_guard<std::mutex> lock(platform.queue_mutex());
        context.enqueue_transfer_locked(std::move(slot), std::move(retained), std::move(done),
            [&](VkCommandBuffer cb) {
                vulkan_record_memory_barrier(cb, vulkan_transfer_ready_spec());
                vkCmdFillBuffer(cb, destination->buffer(), 0, sizeof(seed), 0);
            }, true);
    }
    auto third = context.reserve_transfer();
    expect(platform.transfer_reservation_wait_count() == pressure + 1 &&
               platform.execution_fence_wait_count() == waits + 2,
           "real reservation/fence wait counter missed capacity-pressure boundary");
    platform.synchronize_execution();
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

// Host transfer acquisition that precedes any real host map must finish an
// earlier READY publication (including holder release) outside the queue lock
// before returning held. Genuine RED against pre-drain absence: without the
// acquisition pump the publication stays unpumped after device-idle proof.
void test_host_acquisition_publishes_earlier_ready_transfer_outside_queue(
    VulkanPlatform &platform) {
    VulkanExecutionContext &context = platform.execution_context();
    int published = 0;
    int releases = 0;
    std::weak_ptr<VulkanBuffer> weak_staging;
    {
        auto transfer = platform.reserve_transfer(sizeof(uint32_t));
        expect(transfer.staging != nullptr,
               "ready publication fixture got no exclusive staging");
        const uint32_t pattern = 0x21524110U;
        transfer.staging->write(&pattern, sizeof(pattern));
        weak_staging = transfer.staging;
        std::shared_ptr<int> owner(new int(7),
                                   [&](int *value) { delete value; ++releases; });
        context.begin();
        {
            std::lock_guard<std::mutex> lock(platform.queue_mutex());
            context.defer_transfer(std::move(transfer.reservation), {transfer.staging, owner},
                                   [&] {
                                       bool queue_lock_free =
                                           other_thread_can_lock(platform.queue_mutex());
                                       expect(queue_lock_free,
                                              "ready publication ran while the queue lock "
                                              "was held");
                                       ++published;
                                   });
        }
        transfer.staging.reset();
        owner.reset();
        context.submit();
    }
    // Queue idle proves submitted-work completion without pumping publications.
    platform.wait_for_transfer();
    // A second acquisition grants the host guard only after the completed
    // publication has run entirely outside every lock.
    platform.wait_for_transfer();
    expect(published == 1, "host acquisition did not finish an earlier ready "
                           "publication before granting the guard");
    expect(releases == 1, "host acquisition double-released or lost the owner");
    expect(weak_staging.expired(), "host acquisition kept exclusive staging alive "
                                   "after its publication");
    expect(other_thread_can_lock(platform.queue_mutex()),
           "host map guard was not released after acquisition returned");
}

// Deterministic host-map guard witness (design §5): the production formatter
// presentation read holds acquire_host_transfer_lock() THROUGH read/map/unmap.
// Integrated GPUs cannot allocate host-visible tensor storage, so this witness
// drives the same guard/map pairing at the buffer seam: the before-map hook
// joins a foreign thread that try_locks the queue mutex inside the real map
// window; the held guard must exclude it and be free again after unmap.
void test_formatter_presentation_map_guard_excludes_foreign_queue_lock(
    VulkanPlatform &platform) {
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const float seeded[] = {11.0F, 31.0F};
    VulkanBuffer buffer(platform, sizeof(seeded), host);
    buffer.write(seeded, sizeof(seeded));
    bool entered = false;
    bool excluded = false;
    buffer.set_before_map_for_testing([&] {
        entered = true;
        excluded = !other_thread_can_lock(platform.queue_mutex());
    });
    float actual[] = {0.0F, 0.0F};
    {
        auto guard = platform.acquire_host_transfer_lock();
        buffer.read(actual, sizeof(actual));
        // The guard is still held through unmap: prove exclusion again outside
        // the hook, then release exactly where the production read returns.
        expect(excluded && !other_thread_can_lock(platform.queue_mutex()),
               "guarded map window admitted a foreign queue lock before read "
               "returned");
        guard.unlock();
    }
    buffer.set_before_map_for_testing({});
    expect(entered, "before-map observation never ran");
    expect(excluded, "foreign thread took the queue lock inside the held map window");
    expect(actual[0] == seeded[0] && actual[1] == seeded[1],
           "guarded host map did not read the seeded bytes");
    expect(other_thread_can_lock(platform.queue_mutex()),
           "map guard was not released after unmap");
}

// Tiny actual-Vulkan owned-staging COPY witness (not an empty submission):
// CPU-seeded F32 goes through owned upload staging → device buffer, one
// recorded compute→copy→copy→compute command sequence, a distinct
// semaphore-chained transfer submission, and a standalone sync-copy bridge;
// the download staging publishes to the CPU destination only after confirmed
// completion. No intermediate host wait hides the ordering; the two-slot
// reservation cap is exercised by the download grant (it may only be granted
// once the earlier publication releases its slot).
void test_platform_owned_staging_copy_witness(VulkanPlatform &platform) {
    VulkanExecutionContext &context = platform.execution_context();
    context.wait();
    platform.reset_execution_counters();
    const pytorch_vulkan::VulkanTensorLayout layout{
        1, {4}, {1}, 0, sizeof(float), 0, 4, 0, 4 * sizeof(float),
        4 * sizeof(float), pytorch_vulkan::VulkanOverlap::No};
    const float seed_lhs[] = {3.0F, 5.0F, 7.0F, 9.0F};
    const float seed_addend[] = {1.0F, 2.0F, 3.0F, 4.0F};

    VulkanBuffer lhs(platform, sizeof(seed_lhs));   // device-local data plane
    VulkanBuffer addend(platform, sizeof(seed_addend));
    VulkanBuffer sum(platform, sizeof(seed_lhs));
    VulkanBuffer a(platform, sizeof(seed_lhs));
    VulkanBuffer b(platform, sizeof(seed_lhs));
    VulkanBuffer result(platform, sizeof(seed_lhs));

    // Standalone sync-copy bridge in its own fenced command buffer (outside
    // any execution recording): the addend reaches device memory before the
    // recorded sequence is built. The reservation lease is dropped right
    // after the fenced copy: an unattached reservation releases its slot only
    // when the lease is destroyed.
    {
        auto bridge = platform.reserve_transfer(sizeof(seed_addend));
        expect(bridge.staging != nullptr, "witness bridge got no exclusive staging");
        bridge.staging->write(seed_addend, sizeof(seed_addend));
        platform.copy_buffer_sync(bridge.staging->buffer(), addend.buffer(),
                                  sizeof(seed_addend));
    }

    // Slots 1 and 2: owned upload staging attached BEFORE owner-referencing
    // commands in the recorded phases, and released only by publication.
    auto upload_lhs = platform.reserve_transfer(sizeof(seed_lhs));
    auto upload_addend = platform.reserve_transfer(sizeof(seed_addend));
    std::weak_ptr<VulkanBuffer> weak_upload_lhs = upload_lhs.staging;
    std::weak_ptr<VulkanBuffer> weak_upload_addend = upload_addend.staging;
    upload_lhs.staging->write(seed_lhs, sizeof(seed_lhs));
    upload_addend.staging->write(seed_addend, sizeof(seed_addend));

    int upload_publish = 0;
    int download_publish = 0;
    int releases = 0;
    std::weak_ptr<VulkanBuffer> weak_download;
    float witness[4] = {0.0F, 0.0F, 0.0F, 0.0F};

    // Phase 1: one recorded command sequence — compute add → copy A→B →
    // copy B→C → compute add again.
    platform.compute().begin_training_step();
    {
        std::lock_guard<std::mutex> lock(platform.queue_mutex());
        context.defer_transfer(std::move(upload_lhs.reservation), {upload_lhs.staging},
                               [&] { ++upload_publish; });
        context.defer_transfer(std::move(upload_addend.reservation),
                               {upload_addend.staging}, [&] { ++upload_publish; });
    }
    platform.copy_buffers_sync({{upload_lhs.staging->buffer(), lhs.buffer(),
                                 0, 0, sizeof(seed_lhs)}});
    platform.compute().add(lhs.buffer(), layout, addend.buffer(), layout,
                           sum.buffer(), layout);
    platform.copy_buffers_sync(
        {{sum.buffer(), a.buffer(), 0, 0, sizeof(seed_lhs)}});
    platform.copy_buffers_sync(
        {{a.buffer(), b.buffer(), 0, 0, sizeof(seed_lhs)}});
    platform.compute().add(b.buffer(), layout, addend.buffer(), layout,
                           result.buffer(), layout);
    platform.compute().end_training_step();
    upload_lhs.staging.reset();
    upload_addend.staging.reset();

    // Phase 2: distinct semaphore-chained submission. The transfer consumer of
    // compute output only waits the previous signal at COMPUTE_SHADER|TRANSFER
    // stages; the download grant obeys the same two-slot reservation cap the
    // uploads already occupy.
    platform.compute().begin_training_step();
    auto download = platform.reserve_transfer(sizeof(witness));
    weak_download = download.staging;
    auto download_owner = download.staging;
    void *download_mapped = download_owner->map_persistent();
    {
        std::lock_guard<std::mutex> lock(platform.queue_mutex());
        context.defer_transfer(std::move(download.reservation),
                               {download.staging,
                                std::shared_ptr<int>(new int(7),
                                                     [&](int *value) {
                                                         delete value; ++releases;
                                                     })},
                               [&, download_owner, download_mapped] {
                                   bool queue_lock_free =
                                       other_thread_can_lock(platform.queue_mutex());
                                   expect(queue_lock_free,
                                          "witness publication ran while the queue "
                                          "lock was held");
                                   download_owner->invalidate_range(0, sizeof(witness));
                                   std::memcpy(witness, download_mapped, sizeof(witness));
                                   ++download_publish;
                               });
    }
    platform.copy_buffers_sync({{result.buffer(), download.staging->buffer(),
                                 0, 0, sizeof(witness)}});
    platform.compute().end_training_step();
    download.staging.reset();
    download_owner.reset();

    context.synchronize();
    const float expected[] = {seed_lhs[0] + 2.0F * seed_addend[0],
                              seed_lhs[1] + 2.0F * seed_addend[1],
                              seed_lhs[2] + 2.0F * seed_addend[2],
                              seed_lhs[3] + 2.0F * seed_addend[3]};
    for (int i = 0; i < 4; ++i)
        expect(witness[i] == expected[i],
               "owned staging copy witness did not move the seeded values in GPU order");
    expect(upload_publish == 2, "upload publications did not complete exactly once");
    expect(download_publish == 1, "download publication did not complete exactly once");
    expect(releases == 1, "witness transfer owner was not released exactly once");
    expect(weak_upload_lhs.expired() && weak_upload_addend.expired(),
           "upload staging outlived its publication");
    expect(weak_download.expired(), "download staging outlived its publication");
    expect(context.pending_count() == 0, "witness left pending transfer work");
    expect(context.query_complete(), "witness chain did not drain to completion");
    expect(platform.compute_submitted_count() == 2,
           "witness did not exercise exactly the two chained submissions");
    // Five copies move the data: one standalone fenced bridge (addend) plus
    // four recorded copies (lhs upload, sum→a, a→b, result→download staging).
    expect(platform.copy_command_count() == 5,
    "witness copy commands were not executed exactly as designed");
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
        {
            const bool fake_only = argc >= 2 && std::string(argv[1]) == "--fake-only";
            const std::string selected = fake_only && argc == 3 ? argv[2] : "all";
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
            run("lifecycle-cycle", test_lifecycle_cycle);
            run("lifecycle-operation-tail", [] { lifecycle_seam_test::tail(false); });
            run("callback-cache-coherent", [] { buffer_maintenance_test::publication(true); });
            run("callback-cache-noncoherent", [] { buffer_maintenance_test::publication(false); });
            run("callback-cache-error", [] { buffer_maintenance_test::publication(false, true); });
            run("callback-cache-boundaries", buffer_maintenance_test::boundaries);
            run("callback-cache-closing", buffer_maintenance_test::closing_wait);
            run("callback-cache-completion", buffer_maintenance_test::completion);
            run("lifecycle-deferred-entry", lifecycle_seam_test::deferred_entry);
            run("lifecycle-snapshot", test_lifecycle_snapshot);
            run("lifecycle-reservation-close", test_lifecycle_reservation_close);
            run("lifecycle-foreign-pump", test_lifecycle_foreign_pump);
            run("lifecycle-self-close", test_lifecycle_self_close);
            run("lifecycle-sticky", test_lifecycle_sticky);
            run("lifecycle-native-empty", test_lifecycle_native_empty);
            run("submit-wait-mask", test_fake_submit_waits_for_compute_and_transfer);
            run("barrier-specs", test_shared_barrier_descriptors);
            run("errors", test_controlled_cleanup_errors_are_persistent_and_nonblocking_to_cleanup);
            run("transfer-success", test_fake_transfer_record_success_only_publication);
            run("cancel-audit", test_fake_cancel_preserves_cleanup_for_earlier_submission);
            run("enqueue-order", test_fake_enqueue_returns_before_completion_then_publishes_once);
            run("enqueue-open", test_fake_enqueue_without_submit_joins_open_recording);
            run("enqueue-record-throw", test_fake_enqueue_record_throw_cancels_and_persists);
            run("enqueue-end-failure", test_fake_enqueue_end_failure_without_idle_recovery);
            run("enqueue-submit-safe", test_fake_enqueue_submit_failure_safe_recovery);
            run("enqueue-submit-uncertain", test_fake_enqueue_submit_failure_uncertain_recovery);
            run("enqueue-status-safe", test_fake_enqueue_status_failure_safe_recovery);
            run("enqueue-status-uncertain", test_fake_enqueue_status_failure_uncertain_recovery);
            run("enqueue-device-loss", test_fake_enqueue_fence_device_loss_quarantines);
            run("enqueue-publish-throw", test_fake_enqueue_publication_failure_is_sticky);
            run("enqueue-pressure", test_fake_enqueue_backpressure_waits_for_owner_release);
            run("enqueue-reentry", test_fake_enqueue_callback_reentry_rejected);
            run("enqueue-owner-retain", test_fake_enqueue_owner_destructor_retains_without_recursing);
            run("enqueue-attach-failure", test_fake_enqueue_attachment_failure_preserves_inputs);
            run("host-access-publication", test_fake_host_access_publishes_work_completed_by_own_flush);
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
            run("host-access-foreign-pump", test_fake_host_access_waits_for_foreign_publication_and_capture_release);
            run("host-access-recheck", test_fake_host_access_rechecks_intervening_producer);
            run("host-access-recovery-lock", test_fake_host_access_error_recovery_keeps_queue_serialized);
            run("host-access-reentry-errors", test_fake_host_access_reentry_and_sticky_callback_error);
            run("boundary-admission", test_fake_boundary_excludes_producer_until_device_cutoff);
            run("error-next-batch", test_fake_persistent_error_does_not_skip_next_safe_cleanup);
            run("flush-error-cleanup", test_fake_flush_failure_preserves_error_and_safe_cleanup);
            expect(found, "unknown fake-only case");
            std::cout << "Controlled execution-ops tests passed\n";
            if (fake_only) {
                expect(VulkanPlatform::platform_construction_count() == 0 &&
                           VulkanPlatform::availability_probe_count() == 0,
                       "fake-only path constructed or probed a Vulkan platform");
                return 0;
            }
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
            run_native("test_platform_completion_hold_and_real_wait_counter",
                       [&] { test_platform_completion_hold_and_real_wait_counter(platform); });
            run_native("test_platform_staging_allocation_rollback",
                       [] { test_platform_staging_allocation_rollback(); });
            run_native(
                "test_host_acquisition_publishes_earlier_ready_transfer_outside_queue",
                [&] {
                    test_host_acquisition_publishes_earlier_ready_transfer_outside_queue(
                        platform);
                });
            run_native(
                "test_formatter_presentation_map_guard_excludes_foreign_queue_lock",
                [&] {
                    test_formatter_presentation_map_guard_excludes_foreign_queue_lock(
                        platform);
                });
            run_native("test_platform_owned_staging_copy_witness",
                       [&] { test_platform_owned_staging_copy_witness(platform); });
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
