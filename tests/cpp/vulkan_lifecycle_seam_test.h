#pragma once

#include "vulkan_execution.h"
#include "vulkan_lifecycle.h"
#include <atomic>
#include <cxxabi.h>
#include <future>
#include <iostream>
#include <pthread.h>
#include <stdexcept>

// CPU-only device operations; the context, coordinator and completion seam are
// production objects. No fixture-owned drain is allowed at operation return.
namespace lifecycle_seam_test {
using namespace pytorch_vulkan;
inline void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
template<class T> T handle(uintptr_t value) { return reinterpret_cast<T>(value); }
struct Device {
    uintptr_t next = 10;
    bool fail_attachment = false;
    std::function<void()> on_pump_wait;
    VulkanExecutionOps ops() {
        VulkanExecutionOps o{};
        o.cookie = this;
        o.allocate_command_buffers = [](void* p, VkDevice, const VkCommandBufferAllocateInfo* i, VkCommandBuffer* b) {
            for (uint32_t n = 0; n < i->commandBufferCount; ++n) b[n] = handle<VkCommandBuffer>(++static_cast<Device*>(p)->next);
            return VK_SUCCESS;
        };
        o.free_command_buffers = [](void*, VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {};
        o.create_fence = [](void* p, VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* f) {
            *f = handle<VkFence>(++static_cast<Device*>(p)->next); return VK_SUCCESS;
        };
        o.destroy_fence = [](void*, VkDevice, VkFence, const VkAllocationCallbacks*) {};
        o.create_semaphore = [](void* p, VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* s) {
            *s = handle<VkSemaphore>(++static_cast<Device*>(p)->next); return VK_SUCCESS;
        };
        o.destroy_semaphore = [](void*, VkDevice, VkSemaphore, const VkAllocationCallbacks*) {};
        o.begin_command_buffer = [](void*, VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; };
        o.end_command_buffer = [](void*, VkCommandBuffer) { return VK_SUCCESS; };
        o.reset_command_buffer = [](void*, VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; };
        o.reset_fences = [](void*, VkDevice, uint32_t, const VkFence*) { return VK_SUCCESS; };
        o.queue_submit = [](void*, VkQueue, uint32_t, const VkSubmitInfo*, VkFence) { return VK_SUCCESS; };
        o.get_fence_status = [](void*, VkDevice, VkFence) { return VK_NOT_READY; };
        o.wait_for_fences = [](void*, VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) { return VK_SUCCESS; };
        o.queue_wait_idle = [](void*, VkQueue) { return VK_SUCCESS; };
        o.fail_transfer_attachment = [](void* p) noexcept { return static_cast<Device*>(p)->fail_attachment; };
        o.callback_wait_begin = [](void* p) noexcept {
            auto& callback = static_cast<Device*>(p)->on_pump_wait;
            if (callback) callback();
        };
        return o;
    }
};
struct Root {
    Device device;
    VulkanRuntimeLifecycle runtime{vulkan_host_clock()};
    std::mutex queue;
    VulkanExecutionQuarantineSink sink{nullptr, [](void*, VulkanExecutionQuarantine*) noexcept { std::_Exit(97); }};
    VulkanExecutionContext context{handle<VkDevice>(1), handle<VkQueue>(2), handle<VkCommandPool>(3), device.ops(), sink, queue};
    std::atomic<unsigned> completions{0};
    std::function<void(bool)> on_gate;
    VulkanLifecycleRoot adapter(std::shared_ptr<Root> owner) {
        return {owner, &runtime,
            [](void* p, VulkanHostTime deadline) {
                auto& r = *static_cast<Root*>(p);
                ++r.completions;
                r.context.synchronize_with_flush_until([&] {
                    if (r.context.recording()) r.context.submit_locked();
                }, deadline, vulkan_host_clock());
            },
            [](void* p) { return static_cast<Root*>(p)->context.lifecycle_snapshot(); },
            [](void* p, bool closed) {
                auto& r = *static_cast<Root*>(p);
                r.context.set_lifecycle_reservations_closed(closed);
                if (r.on_gate) r.on_gate(closed);
            },
            [](void* p, VulkanHostTime deadline) { static_cast<Root*>(p)->context.wait_foreign_epilogues_until(deadline, vulkan_host_clock()); },
            [](void* p) noexcept { return static_cast<Root*>(p)->context.callback_active_on_this_thread(); },
            [](void*) noexcept { return false; },
            [](void* p) { static_cast<Root*>(p)->context.teardown_empty_native_services({}); }};
    }
};
struct Fixture {
    VulkanLifecycleCoordinator coordinator{vulkan_host_clock()};
    std::shared_ptr<Root> root;
    explicit Fixture(bool transition_before_initialization = false, bool enable_direct_mode = true) {
        if (transition_before_initialization) {
            const auto deadline = coordinator.prepare_python_finalizing_sync();
            coordinator.finish_python_finalizing_sync(deadline);
        }
        root = std::static_pointer_cast<Root>(coordinator.acquire([&] {
            auto r = std::make_shared<Root>(); return r->adapter(r);
        }));
        if (enable_direct_mode && !transition_before_initialization) root->runtime.set_finalizing_sync();
    }
    ~Fixture() { root.reset(); coordinator.close(VulkanCloseBoundary::NativeEarly); }
};
inline void leaf_then_attach(VulkanRuntimeLifecycle& runtime, VulkanExecutionContext& context,
                             std::atomic<unsigned>& retired, bool error, unsigned owners = 1) {
    VulkanOperationScope operation(runtime);
    try {
        [&] {
            { VulkanOperationScope leaf(runtime); context.begin("wrapper-tail"); context.submit(); leaf.finish(); }
            for (unsigned i = 0; i < owners; ++i) {
                auto witness = std::shared_ptr<int>(new int, [&](int* p) { ++retired; delete p; });
                context.retain_until_completion([witness] {});
            }
            if (error) throw std::runtime_error("sentinel");
        }();
        operation.finish();
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { operation.finish_error(std::current_exception()); }
}
inline void tail(bool error, unsigned owners = 1) {
    Fixture f;
    std::atomic<unsigned> retired{0};
    bool original = false;
    try { leaf_then_attach(f.root->runtime, f.root->context, retired, error, owners); }
    catch (const std::runtime_error& e) { original = std::string(e.what()) == "sentinel"; }
    require(original == error && retired == owners, "outer completion must retire post-leaf owners and preserve errors");
    require(f.root->completions == 1 && f.root->context.lifecycle_snapshot().healthy_empty(), "nested scope must not complete separately");
}
[[noreturn]] inline void failed_completion_child() {
    // Intentional retained-root policy child. A swallowed completion failure
    // must not become a false PASS through Fixture's destructor close.
    auto* f = new Fixture;
    try {
        with_vulkan_operation(f->root->runtime, [&] {
            f->root->context.begin("failed-operation-completion");
            f->root->context.defer_destruction([] { throw std::runtime_error("publication failure"); });
            throw std::runtime_error("ordinary sentinel");
        });
    } catch (...) { std::_Exit(99); }
    std::_Exit(98);
}
inline void threads() {
    Fixture f(false, false);
    std::atomic<unsigned> initial_retired{0};
    auto initial = std::shared_ptr<int>(new int, [&](int* p) { ++initial_retired; delete p; });
    f.root->context.begin("before-finalizing-transition");
    f.root->context.defer_destruction([initial] {}); initial.reset();
    const auto transition_deadline = f.coordinator.prepare_python_finalizing_sync();
    f.coordinator.finish_python_finalizing_sync(transition_deadline);
    require(initial_retired == 1 && f.root->runtime.finalizing_sync(), "initialized transition must drain existing captures and enable synchronous mode");
    std::promise<void> held, attempt, release;
    auto released = release.get_future().share();
    std::atomic<bool> overlapped{false};
    std::atomic<unsigned> retired{0};
    auto first = std::async(std::launch::async, [&] {
        VulkanOperationScope operation(f.root->runtime);
        f.root->context.begin("first-postleaf"); f.root->context.submit();
        auto witness = std::shared_ptr<int>(new int, [&](int* p) { ++retired; delete p; });
        f.root->context.retain_until_completion([witness] {}); witness.reset();
        held.set_value(); released.wait();
        operation.finish();
    });
    held.get_future().wait();
    auto second = std::async(std::launch::async, [&] {
        attempt.set_value();
        VulkanOperationScope operation(f.root->runtime);
        // Count the actual permits, not a fixture gate. First's completion may
        // release admission just before its caller clears first_active.
        overlapped = f.root->runtime.outer_permits() > 1;
        f.root->context.begin("second-host-copy-shaped"); f.root->context.submit();
        auto witness = std::shared_ptr<int>(new int, [&](int* p) { ++retired; delete p; });
        f.root->context.retain_until_completion([witness] {}); witness.reset();
        operation.finish();
    });
    attempt.get_future().wait();
    second.wait_for(std::chrono::milliseconds(30));
    release.set_value(); first.get(); second.get();
    require(!overlapped, "finalizing outer admission must serialize through post-leaf epilogue");
    require(retired == 2 && f.root->context.lifecycle_snapshot().healthy_empty(), "two-thread completion left captures");
    require(f.root->runtime.finalizing_sync(), "foreign callers cannot disable finalizing mode");
    Fixture inherited(true);
    require(inherited.root->runtime.finalizing_sync(), "first initialization must inherit coordinator finalizing mode");
    std::thread foreign([&] { with_vulkan_operation(inherited.root->runtime, [] {}); });
    foreign.join();
}
inline void cleanup_self() {
    Fixture f;
    bool rejected = false;
    { VulkanCleanupScope cleanup(f.root->runtime);
      try { f.coordinator.retire_cached(); }
      catch (const VulkanAdmissionClosed&) { rejected = true; } }
    require(rejected, "own cleanup close must reject promptly");
}
inline void nested_closing() {
    Fixture f;
    VulkanOperationScope operation(f.root->runtime);
    std::thread closer([&] {
        f.root->runtime.close_admission();
        f.root->context.set_lifecycle_reservations_closed(true);
    });
    closer.join();
    { VulkanOperationScope nested(f.root->runtime); nested.finish(); }
    bool rejected = false;
    std::thread fresh([&] {
        try { VulkanOperationScope work(f.root->runtime); }
        catch (const VulkanAdmissionClosed&) { rejected = true; }
    });
    fresh.join();
    require(rejected, "Closing must reject fresh outer work but permit admitted nesting");
    f.root->context.begin("nested-closing");
    f.root->context.defer_destruction([&] {
        bool callback_rejected = false;
        try { VulkanOperationScope work(f.root->runtime); }
        catch (const VulkanAdmissionClosed&) { callback_rejected = true; }
        require(callback_rejected, "callback cannot inherit admitted permit");
    });
    operation.finish();
    require(f.root->runtime.outer_permits() == 0, "Closing outer epilogue leaked permit");
}
inline void forced_unwind() {
    Fixture f;
    std::thread worker([&] {
        with_vulkan_operation(f.root->runtime, [] { pthread_exit(nullptr); });
    });
    worker.join();
    require(f.root->runtime.outer_permits() == 0 && f.root->completions == 0,
            "forced unwind must release accounting without completion");
}
inline void closing_admission_waiter() {
    Fixture f;
    VulkanOperationScope outer(f.root->runtime);
    std::promise<void> attempting;
    auto waiter = std::async(std::launch::async, [&] {
        attempting.set_value();
        try { VulkanOperationScope operation(f.root->runtime); operation.finish(); }
        catch (const VulkanAdmissionClosed&) { return true; }
        return false;
    });
    attempting.get_future().wait();
    const bool blocked = waiter.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout;
    f.root->runtime.close_admission();
    const bool rejected = waiter.get();
    { VulkanOperationScope nested(f.root->runtime); nested.finish(); }
    outer.finish();
    require(blocked && rejected && f.root->runtime.outer_permits() == 0,
            "Closing must wake serialized admission waiters without stranding permits or rejecting existing nesting");
}
inline void rnn_attachment_failure() {
    Fixture f;
    std::atomic<unsigned> retired{0};
    bool original = false;
    try {
        with_vulkan_operation(f.root->runtime, [&] {
            std::vector<std::shared_ptr<void>> owners;
            for (unsigned n = 0; n < 7; ++n)
                owners.emplace_back(new int, [&](void* p) { ++retired; delete static_cast<int*>(p); });
            auto reservation = f.root->context.reserve_transfer();
            f.root->context.begin("rnn-attachment-failure");
            f.root->device.fail_attachment = true;
            f.root->context.defer_transfer(std::move(reservation), std::move(owners));
        });
    } catch (const std::bad_alloc&) { original = true; }
    require(original && retired == 7 && f.root->context.lifecycle_snapshot().healthy_empty(),
            "RNN-like attachment failure must release seven caller owners before healthy error return");
}
inline void capture_self() {
    Fixture f;
    std::atomic<unsigned> rejected{0};
    struct Witness {
        Fixture& fixture;
        std::atomic<unsigned>& rejected;
        ~Witness() {
            try { fixture.coordinator.retire_cached(); }
            catch (const VulkanAdmissionClosed&) { ++rejected; }
            try { fixture.coordinator.close(VulkanCloseBoundary::NativeEarly); }
            catch (const VulkanAdmissionClosed&) { ++rejected; }
            try { VulkanOperationScope work(fixture.root->runtime); }
            catch (const VulkanAdmissionClosed&) { ++rejected; }
        }
    };
    with_vulkan_operation(f.root->runtime, [&] {
        auto owner = std::shared_ptr<Witness>(new Witness{f, rejected});
        f.root->context.begin("capture-self");
        f.root->context.defer_destruction([owner] {});
    });
    require(rejected == 3, "last capture destructor must retain callback self-identity");
}
inline void cleanup_tail() {
    cleanup_self();
    capture_self();
    Fixture f;
    std::promise<void> entered, release;
    auto released = release.get_future();
    std::thread worker([&] {
        VulkanCleanupScope cleanup(f.root->runtime);
        entered.set_value(); released.wait(); // after last callback, before owner tail
    });
    entered.get_future().wait();
    auto idle = std::async(std::launch::async, [&] {
        f.root->runtime.wait_idle_until(std::chrono::steady_clock::now() + std::chrono::seconds(20), VulkanIdleWait::AllEpilogues);
    });
    const bool blocked = idle.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout;
    release.set_value(); worker.join(); idle.get();
    require(blocked && f.root->runtime.cleanup_epilogues() == 0, "host idle must include cleanup owner tail");
}
inline void foreign_pump_identity() {
    Fixture f;
    std::promise<void> paused, release, observed;
    auto released = release.get_future();
    struct Witness {
        std::promise<void>& paused;
        std::future<void>& released;
        ~Witness() { paused.set_value(); released.wait(); }
    };
    auto witness = std::shared_ptr<Witness>(new Witness{paused, released});
    f.root->context.begin("foreign-pump-identity");
    f.root->context.defer_destruction([witness] {}); witness.reset();
    f.root->context.submit();
    { std::lock_guard<std::mutex> lock(f.root->queue); f.root->context.wait_device_locked(); }
    std::thread pump([&] { f.root->context.drain_ready_cleanup(); });
    paused.get_future().wait();
    std::atomic<unsigned> rejected{0};
    f.root->device.on_pump_wait = [&] {
        // Production calls this while holding the execution mutex. Correct
        // self-recognition must therefore be TLS-only, including a pump that
        // does not own callbacks_active_'s foreign callback.
        try { f.coordinator.retire_cached(); }
        catch (const VulkanAdmissionClosed&) { ++rejected; }
        try { VulkanOperationScope operation(f.root->runtime); }
        catch (const VulkanAdmissionClosed&) { ++rejected; }
        observed.set_value();
    };
    std::thread waiter([&] {
        f.root->context.synchronize_with_flush_until({}, std::chrono::steady_clock::now() + std::chrono::seconds(20), vulkan_host_clock());
    });
    observed.get_future().wait();
    release.set_value(); pump.join(); waiter.join();
    f.root->device.on_pump_wait = {};
    require(rejected == 2 && f.root->context.lifecycle_snapshot().healthy_empty(), "foreign pump member epilogue lost self identity");
}
inline void gradient_return_initializer(bool scoped) {
    Fixture f;
    std::promise<void> destroying_first, release_first, attempting, foreign_recording, release_foreign;
    auto first_released = release_first.get_future();
    auto foreign_released = release_foreign.get_future();
    std::atomic<unsigned> retired{0};
    std::atomic<bool> original{false}, retired_before_return{false};
    auto first_gradient = [&]() -> std::shared_ptr<void> {
        return with_vulkan_operation(f.root->runtime, [&]() -> std::shared_ptr<void> {
            f.root->context.begin("first-gradient"); f.root->context.submit();
            auto capture = std::shared_ptr<int>(new int, [&](int* p) { ++retired; delete p; });
            return std::shared_ptr<void>(new int, [&, capture](void* p) {
                VulkanCleanupScope cleanup(f.root->runtime);
                destroying_first.set_value(); first_released.wait();
                f.root->context.retain_until_completion([capture] {});
                delete static_cast<int*>(p);
            });
        });
    };
    auto second_gradient = [&]() -> std::shared_ptr<void> {
        return with_vulkan_operation(f.root->runtime, [&]() -> std::shared_ptr<void> {
            f.root->context.begin("second-gradient"); f.root->context.submit();
            throw std::runtime_error("second-gradient-sentinel");
        });
    };
    auto worker = std::async(std::launch::async, [&] {
        try {
            auto body = [&]() -> std::vector<std::shared_ptr<void>> {
                // Same ownership/evaluation boundary as the actual custom
                // Function's two-gradient return initializer in autograd.h.
                return {first_gradient(), second_gradient(), {}};
            };
            if (scoped) (void)with_vulkan_operation(f.root->runtime, body);
            else (void)body(); // Explicit delegated-leaf-only counterexample.
        } catch (const std::runtime_error& e) {
            original = std::string(e.what()) == "second-gradient-sentinel";
            retired_before_return = retired == 1;
        }
    });
    destroying_first.get_future().wait();
    auto foreign = std::async(std::launch::async, [&] {
        attempting.set_value();
        with_vulkan_operation(f.root->runtime, [&] {
            f.root->context.begin("foreign-during-gradient-unwind");
            foreign_recording.set_value(); foreign_released.wait();
        });
    });
    attempting.get_future().wait();
    // The worker intentionally holds its recording once admitted. Use the
    // actual recording promise to distinguish admission from overall finish.
    auto recording = foreign_recording.get_future();
    const bool admitted_while_unwinding = recording.wait_for(std::chrono::milliseconds(30)) == std::future_status::ready;
    release_first.set_value(); worker.get();
    release_foreign.set_value(); foreign.get();
    require(original && retired == 1, "gradient initializer fixture did not finish with original error and retired owner");
    require(admitted_while_unwinding != scoped && retired_before_return == scoped,
            "custom Function outer seam must span first-gradient exceptional destruction");
    std::cout << "gradient-return-initializer scoped=" << scoped
              << " foreign_admitted_during_unwind=" << admitted_while_unwinding
              << " owner_retired_before_error_return=" << retired_before_return << '\n';
}
inline void deferred_entry() {
    VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
    const auto deadline = coordinator.prepare_python_finalizing_sync();
    coordinator.finish_python_finalizing_sync(deadline);
    std::promise<void> allocated, throw_now, foreign_recording, release_foreign;
    auto throwing = throw_now.get_future();
    auto released = release_foreign.get_future();
    std::atomic<unsigned> retired{0}, factories{0};
    Root* raw = nullptr;
    std::weak_ptr<Root> weak;
    std::atomic<bool> original{false}, retired_before_return{false};
    auto worker = std::async(std::launch::async, [&] {
        try {
            with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
                auto root = std::static_pointer_cast<Root>(coordinator.acquire([&] {
                    ++factories;
                    auto r = std::make_shared<Root>(); return r->adapter(r);
                }));
                raw = root.get(); weak = root;
                auto output = with_vulkan_operation(root->runtime, [&] {
                    root->context.begin("deferred-first-allocation"); root->context.submit();
                    auto owner = std::shared_ptr<int>(new int, [&](int* p) { ++retired; delete p; });
                    return std::shared_ptr<int>(new int, [&, owner](int* p) {
                        VulkanCleanupScope cleanup(raw->runtime);
                        raw->context.retain_until_completion([owner] {});
                        delete p;
                    });
                });
                allocated.set_value(); throwing.wait();
                throw std::runtime_error("first-use-copy-sentinel");
            });
        } catch (const std::runtime_error& e) {
            original = std::string(e.what()) == "first-use-copy-sentinel";
            retired_before_return = retired == 1;
        }
    });
    allocated.get_future().wait();
    auto foreign = std::async(std::launch::async, [&] {
        with_vulkan_operation(raw->runtime, [&] {
            raw->context.begin("foreign-between-allocation-and-error");
            foreign_recording.set_value(); released.wait();
        });
    });
    const bool admitted_before_error = foreign_recording.get_future().wait_for(std::chrono::milliseconds(30)) == std::future_status::ready;
    throw_now.set_value(); worker.get();
    release_foreign.set_value(); foreign.get();
    const bool empty = raw->context.lifecycle_snapshot().healthy_empty();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    require(factories == 1 && original && retired == 1 && empty && weak.expired(), "first-use fixture lost error, ownership, or root lifetime");
    require(!admitted_before_error && retired_before_return,
            "deferred first-use outer permit must span allocation through later error and owner destruction");
}
inline void deferred_nesting() {
    Fixture a, b;
    std::atomic<unsigned> retired{0};
    with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
        require(a.root->runtime.can_defer_operation_on_this_thread() && a.root->runtime.outer_permits() == 0,
                "unbound lexical transaction must be recognizable before reservation admission without taking a permit");
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            with_vulkan_operation(a.root->runtime, [&] {
                a.root->context.begin("deferred-A"); a.root->context.submit();
                a.root->context.retain_until_completion([&] {
                    require(b.root->runtime.operation_active_on_this_thread(), "different runtime permit released before all completion passes");
                    ++retired;
                });
            });
        });
        require(a.root->runtime.outer_permits() == 1 && a.root->completions == 0,
                "oldest deferred frame must own nested frames and known-runtime leaves without extra waits");
        with_vulkan_operation(b.root->runtime, [&] {
            b.root->context.begin("deferred-B"); b.root->context.submit();
            b.root->context.retain_until_completion([&] {
                require(a.root->runtime.operation_active_on_this_thread(), "first runtime permit lost during second completion");
                bool rejected = false;
                try { with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] { VulkanOperationScope work(a.root->runtime); }); }
                catch (const VulkanAdmissionClosed&) { rejected = true; }
                require(rejected, "deferred frame cannot turn callback-created work into nesting");
                ++retired;
            });
        });
    });
    require(retired == 2 && a.root->completions == 1 && b.root->completions == 1 &&
                a.root->runtime.outer_permits() == 0 && b.root->runtime.outer_permits() == 0,
            "deferred multi-runtime completion/accounting not balanced");
    // An unregistered stack runtime has no coordinator lifetime guarantee. Its
    // original explicit scope remains lexical, never borrowed by a later frame.
    with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [] {
        VulkanRuntimeLifecycle borrowed(vulkan_host_clock());
        { VulkanOperationScope scope(borrowed); scope.finish(); }
        require(borrowed.outer_permits() == 0, "unregistered runtime lifetime was silently extended");
    });
}
inline void deferred_known_runtime() {
    Fixture a, b;
    with_vulkan_operation(a.root->runtime, [&] {
        a.root->context.begin("known-runtime-A"); a.root->context.submit();
        with_vulkan_operation(b.root->runtime, [&] {
            b.root->context.begin("known-runtime-B"); b.root->context.submit();
        });
        require(a.root->runtime.outer_permits() == 1 && b.root->runtime.outer_permits() == 1 &&
                    a.root->completions == 0 && b.root->completions == 0,
                "known-runtime outer helper must retain different-runtime lexical tails too");
    });
    require(a.root->completions == 1 && b.root->completions == 1 &&
                a.root->runtime.outer_permits() == 0 && b.root->runtime.outer_permits() == 0,
            "known-runtime multi-root epilogue did not complete/release both permits");
}
inline void deferred_closing() {
    Fixture f;
    std::promise<void> frame_entered, attempt;
    auto attempting = attempt.get_future();
    auto waiter = std::async(std::launch::async, [&] {
        return with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            frame_entered.set_value(); attempting.wait();
            try { VulkanOperationScope operation(f.root->runtime); }
            catch (const VulkanAdmissionClosed&) { return true; }
            return false;
        });
    });
    frame_entered.get_future().wait();
    const auto deadline = f.coordinator.prepare_python_finalizing_sync();
    attempt.set_value();
    const bool rejected = waiter.get();
    require(rejected && f.root->runtime.outer_permits() == 0, "first deferred admission ignored temporary Closing");
    f.coordinator.finish_python_finalizing_sync(deadline);
    const auto completions = f.root->completions.load();
    with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
        { VulkanOperationScope nested(f.root->runtime); nested.finish(); }
        require(f.root->runtime.outer_permits() == 1, "deferred admission after reopen did not bind");
    });
    require(f.root->runtime.finalizing_sync() && f.root->completions == completions + 1,
            "deferred admission lost mode inheritance after temporary reopen");
}
inline void deferred_lifetime(bool known_runtime = false) {
    VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
    auto root = std::static_pointer_cast<Root>(coordinator.acquire([&] {
        auto r = std::make_shared<Root>(); return r->adapter(r);
    }));
    root->runtime.set_finalizing_sync();
    auto* raw = root.get();
    std::weak_ptr<Root> weak = root;
    std::promise<void> body_entered, closing, release;
    auto released = release.get_future();
    std::atomic<unsigned> retired{0};
    root->on_gate = [&](bool closed) { if (closed) closing.set_value(); };
    root.reset(); // only the coordinator owns the runtime now
    auto worker = std::async(std::launch::async, [&] {
        auto body = [&] {
            { VulkanOperationScope leaf(raw->runtime);
              raw->context.begin("deferred-root-lifetime"); raw->context.submit();
              raw->context.retain_until_completion([&] { ++retired; }); leaf.finish(); }
            body_entered.set_value(); released.wait();
        };
        if (known_runtime) with_vulkan_operation(raw->runtime, body);
        else with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), body);
    });
    body_entered.get_future().wait();
    auto closer = std::async(std::launch::async, [&] { coordinator.close(VulkanCloseBoundary::NativeEarly); });
    closing.get_future().wait();
    const bool held = !weak.expired() && raw->runtime.outer_permits() == 1 &&
        closer.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
    release.set_value(); worker.get(); closer.get();
    require(held && retired == 1 && weak.expired(), "deferred runtime outlived its root or added a residual owning pin");
}
inline void deferred_forced_unwind() {
    VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
    auto root = std::static_pointer_cast<Root>(coordinator.acquire([&] {
        auto r = std::make_shared<Root>(); return r->adapter(r);
    }));
    root->runtime.set_finalizing_sync();
    std::atomic<unsigned> retired{0};
    std::thread worker([&] {
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            { VulkanOperationScope nested(root->runtime);
              root->context.begin("deferred-forced-unwind"); root->context.submit();
              root->context.retain_until_completion([&] { ++retired; }); nested.finish(); }
            pthread_exit(nullptr);
        });
    });
    worker.join();
    const bool accounting_only = root->runtime.outer_permits() == 0 && root->completions == 0 && retired == 0;
    std::weak_ptr<Root> weak = root;
    root.reset(); coordinator.close(VulkanCloseBoundary::NativeEarly);
    require(accounting_only && retired == 1 && weak.expired(), "deferred forced unwind waited, stranded permit, or lost root before terminal completion");
}
inline void deferred_concurrent_init() {
    VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
    const auto deadline = coordinator.prepare_python_finalizing_sync(); coordinator.finish_python_finalizing_sync(deadline);
    std::atomic<unsigned> factories{0}, admitted{0}, retired{0};
    std::promise<void> factory_entered, publish, first_admitted, release_first;
    auto published = publish.get_future().share(); auto released = release_first.get_future().share();
    std::weak_ptr<Root> weak;
    auto factory = [&] {
        ++factories; factory_entered.set_value(); published.wait();
        auto r = std::make_shared<Root>(); weak = r; return r->adapter(r);
    };
    auto work = [&] {
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            auto r = std::static_pointer_cast<Root>(coordinator.acquire(factory));
            with_vulkan_operation(r->runtime, [&] {
                require(r->runtime.outer_permits() == 1 && r->runtime.finalizing_sync(), "first actual admissions overlapped or missed inherited mode");
                r->context.begin("concurrent-first-root"); r->context.submit();
                r->context.retain_until_completion([&] { ++retired; });
            });
            if (++admitted == 1) { first_admitted.set_value(); released.wait(); }
        });
    };
    auto first = std::async(std::launch::async, work);
    factory_entered.get_future().wait();
    auto second = std::async(std::launch::async, work);
    publish.set_value(); first_admitted.get_future().wait();
    const bool second_held_out = admitted == 1;
    release_first.set_value(); first.get(); second.get();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    require(second_held_out && factories == 1 && admitted == 2 && retired == 2 && weak.expired(),
            "concurrent first creation/admission lost singleton, ownership, or lexical serialization");
}
inline void run(const std::string& name) {
    if (name == "finalizing-threads") { threads(); closing_admission_waiter(); }
    else if (name == "cleanup-epilogue") { cleanup_tail(); foreign_pump_identity(); }
    else if (name == "nested-closing") nested_closing();
    else if (name == "forced-unwind") forced_unwind();
    else if (name == "autograd-exception-tail") { gradient_return_initializer(false); gradient_return_initializer(true); }
    else if (name == "deferred-entry") deferred_entry();
    else if (name == "deferred-nesting") deferred_nesting();
    else if (name == "deferred-known-runtime") deferred_known_runtime();
    else if (name == "deferred-closing") deferred_closing();
    else if (name == "deferred-lifetime") { deferred_lifetime(); deferred_lifetime(true); }
    else if (name == "deferred-forced-unwind") deferred_forced_unwind();
    else if (name == "deferred-concurrent-init") deferred_concurrent_init();
    else if (name == "operation-error") tail(true);
    else if (name == "rnn-scratch-tail") {
        for (unsigned nullable = 0; nullable <= 4; ++nullable) {
            tail(false, 3 + nullable); tail(true, 3 + nullable);
        }
        rnn_attachment_failure();
    }
    else tail(false);
}
} // namespace lifecycle_seam_test
