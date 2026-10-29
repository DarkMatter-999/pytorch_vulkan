#pragma once

// CPU-only shared fixture. No platform header, Vulkan handle, Torch or driver.
// The native pinned main calls retention() BEFORE constructing/probing Vulkan.
#include "vulkan_pinned_host_core.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace vulkan_pinned_core_testing {
using namespace pytorch_vulkan;
inline void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
struct Calls {
    unsigned unmaps = 0, destroys = 0, frees = 0, safety_checks = 0;
    unsigned counter_drops = 0, root_drops = 0, services = 0;
    bool unsafe = false, audited_outer = false;
    void *mapped_allocation = nullptr;
    std::array<char, 3> order{};
    std::weak_ptr<void> weak;
    // Calls outlives every root/resource and stores only static message pointers.
    const char *failure = nullptr;
    void check(bool ok, const char *message) noexcept {
        if (!ok && failure == nullptr) failure = message;
    }
    void checkpoint() const {
        if (failure != nullptr) throw std::runtime_error(failure);
    }
    void cleanup_step(char event) noexcept {
        const auto index = unmaps + destroys + frees;
        check(index < order.size(), "too many fake resource cleanup steps");
        if (index < order.size()) order[index] = event;
    }
};
inline void no_free(const Calls &c) {
    require(c.unmaps == 0 && c.destroys == 0 && c.frees == 0 &&
            c.order == std::array<char, 3>{} && c.mapped_allocation != nullptr &&
            c.safety_checks == 1 && c.counter_drops == 0 && c.root_drops == 0,
            "device-lost guard freed a resource that must be leaked");
}
struct Clock {
    VulkanHostTime time{};
    Calls *retained = nullptr;
    VulkanHostClock clock() {
        return {this, [](void *p) noexcept { return static_cast<Clock *>(p)->time; },
            [](void *, std::condition_variable&, std::unique_lock<std::mutex>&,
               VulkanHostTime) { throw std::runtime_error("unexpected host wait"); },
            [](void *p, VulkanHostTime deadline) {
                auto &c = *static_cast<Clock *>(p);
                require(c.time == VulkanHostTime{} &&
                        deadline == VulkanHostTime{} + std::chrono::seconds(20),
                        "owner proof must use original virtual Closing+20s once");
                require(c.retained != nullptr, "healthy root unexpectedly retained");
                no_free(*c.retained);
                require(c.retained->weak.use_count() == 3, "close must retain cache+local+core");
                require(c.retained->services == 0, "services destroyed before owner proof");
                const char event[] = "virtual20 no-free count3\n";
                (void)::write(STDERR_FILENO, event, sizeof(event)-1);
                c.time = deadline;
            }};
    }
};
struct Root {
    VulkanRuntimeLifecycle runtime;
    Calls &calls;
    unsigned pins = 1;
    Root(Clock &clock, Calls &c) : runtime(clock.clock()), calls(c) {}
    ~Root() noexcept {
        try {
            calls.check(runtime.cleanup_epilogues() == 0, "root died before General accounting left");
        } catch (...) { calls.check(false, "root cleanup observation failed"); }
        ++calls.root_drops;
    }
    VulkanRuntimeLifecycle &runtime_lifecycle() { return runtime; }
    bool device_lost() const { return false; }
    void record_pinned_host_allocation_destroyed() noexcept {
        try {
            auto a = runtime.host_activity();
            calls.check(a.cleanup_epilogues == (calls.audited_outer ? 2U : 1U) &&
                        a.audited_native_cleanup_epilogues == (calls.audited_outer ? 1U : 0U),
                        "pinned cleanup must remain General, including under AuditedNativeOnly");
        } catch (...) { calls.check(false, "counter cleanup observation failed"); }
        calls.check(pins == 1 && calls.weak.use_count() >= 2, "counter must drop with root/member keepalive");
        --pins;
        ++calls.counter_drops;
    }
};
struct Resource {
    Root &root;
    Calls &calls;
    Resource(Root &r, Calls &c) : root(r), calls(c) {}
    ~Resource() noexcept {
        const bool alive = !calls.weak.expired();
        calls.check(alive && root.pins == 0 && calls.counter_drops == 1,
                    "resource destruction must follow counter drop with root alive");
        bool rejected = false;
        if (alive) {
            try {
                auto a = root.runtime.host_activity();
                calls.check(a.cleanup_epilogues == (calls.audited_outer ? 2U : 1U) &&
                            a.audited_native_cleanup_epilogues == (calls.audited_outer ? 1U : 0U),
                            "resource/member tail escaped General cleanup");
            } catch (...) { calls.check(false, "resource cleanup observation failed"); }
            // General rejection precedes the deadline. An incorrect audited tag
            // must fail immediately, not wait on this destructor's own epilogue.
            try { root.runtime.wait_idle_until(VulkanHostTime{}, VulkanIdleWait::AuditedNativeCleanupOnly); }
            catch (const VulkanAdmissionClosed&) { rejected = true; }
            catch (...) { /* The original rejection invariant below still fails. */ }
        }
        calls.check(rejected, "post-interpreter native-only wait accepted General pinned destruction");
        calls.cleanup_step('u'); ++calls.unmaps;
        std::free(calls.mapped_allocation); calls.mapped_allocation = nullptr;
        calls.cleanup_step('d'); ++calls.destroys;
        calls.cleanup_step('f'); ++calls.frees;
    }
};
struct Ops {
    void *cookie = nullptr;
    bool (*should_leak_for_testing)(void *, const Root &) = nullptr;
};
using Core = detail::PinnedHostCore<Root, Resource, Ops>;
inline std::shared_ptr<Core> make_core(const std::shared_ptr<Root> &root, Calls &calls) {
    auto owner = std::make_unique<Core>();
    owner->platform = root;
    calls.mapped_allocation = std::aligned_alloc(64, 256);
    require(calls.mapped_allocation != nullptr, "CPU allocation failed");
    owner->mapped_base = static_cast<std::byte *>(calls.mapped_allocation);
    owner->logical_bytes = 128;
    owner->buffer = std::make_unique<Resource>(*root, calls);
    owner->ops = {&calls, [](void *p, const Root &) {
        auto &c = *static_cast<Calls *>(p);
        ++c.safety_checks;
        return c.unsafe;
    }};
    owner->counted = true;
    return {owner.release(), [](Core *p) noexcept { detail::destroy_pinned_host_core(p); }};
}
inline VulkanLifecycleRoot adapter(Clock &clock, Calls &calls) {
    auto root = std::make_shared<Root>(clock, calls);
    calls.weak = root;
    return {root, &root->runtime,
        [](void *, VulkanHostTime) {},
        [](void *) { return VulkanQuiescence{}; },
        [](void *, bool) {}, [](void *, VulkanHostTime) {},
        [](void *) noexcept { return false; }, [](void *) noexcept { return false; },
        [](void *p) noexcept {
            auto &r = *static_cast<Root *>(p);
            r.calls.check(r.calls.weak.use_count() == 2, "services require exact cache+local ownership");
            r.calls.check(r.pins == 0, "healthy root still has pinned core");
            ++r.calls.services;
        }};
}
inline void release(bool audited_outer = false, bool inject_failure = false) {
    Calls calls;
    calls.audited_outer = audited_outer || inject_failure;
    Clock clock;
    auto root = std::make_shared<Root>(clock, calls);
    calls.weak = root;
    auto core = make_core(root, calls);
    std::shared_ptr<void> lease(core, core.get());
    core.reset();
    require(calls.destroys == 0 && root->pins == 1, "last alias must own actual shared core");
    if (audited_outer) {
        VulkanCleanupScope audited(root->runtime, VulkanCleanupKind::AuditedNativeOnly);
        lease.reset();
        auto a = root->runtime.host_activity();
        require(a.cleanup_epilogues == 1 && a.audited_native_cleanup_epilogues == 1,
                "inner General accounting did not leave independently");
    } else {
        // Force the member and deleter's local keepalive to be the only owners.
        root.reset();
        lease.reset();
    }
    require(calls.unmaps == 1 && calls.destroys == 1 && calls.frees == 1 &&
            calls.order == std::array<char, 3>{'u','d','f'} && calls.safety_checks == 1 &&
            calls.counter_drops == 1 && calls.mapped_allocation == nullptr,
            "normal last alias must unmap/destroy/free exactly once in order");
    root.reset();
    require(calls.weak.expired() && calls.root_drops == 1, "root/member keepalive failed to expire");
    if (inject_failure) {
        // Only fake expectation metadata is wrong; actual cleanup stays safe.
        require(calls.failure != nullptr && std::string_view(calls.failure) ==
                    "pinned cleanup must remain General, including under AuditedNativeOnly",
                "injected destructor/counter invariant was not recorded first");
        const char *first = calls.failure;
        calls.check(false, "later failure must not replace first failure");
        calls.check(true, "successful check must not clear first failure");
        require(calls.failure == first, "first-failure pointer was replaced or cleared");
        std::cout << "controlled invariant recorded; u/d/f once; root expired; first failure preserved\n";
    }
    calls.checkpoint(); // All shared-core/resource/root destruction has finished.
}
inline void reporter_failure() { release(false, true); }
inline void healthy_close() {
    Calls calls;
    Clock clock;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    auto root = std::static_pointer_cast<Root>(coordinator.acquire([&] { return adapter(clock, calls); }));
    auto core = make_core(root, calls);
    core.reset();
    calls.check(root.use_count() == 2, "healthy parent must return to cache+caller count2");
    calls.check(root->runtime.cleanup_epilogues() == 0, "cleanup accounting still active");
    root.reset();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    calls.checkpoint(); // Ordinary errors only AFTER healthy close, never in its callbacks.
    require(calls.weak.expired() && calls.root_drops == 1 && calls.services == 1,
            "healthy parent count2 close must prove weak expiry");
    coordinator.check_late();
}
inline void retained_child() {
    Calls calls;
    calls.unsafe = true;
    Clock clock;
    clock.retained = &calls;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    {
        NativeVulkanTerminalGuard terminal(coordinator);
        auto root = std::static_pointer_cast<Root>(coordinator.acquire([&] { return adapter(clock, calls); }));
        auto core = make_core(root, calls);
        std::weak_ptr<Core> control = core;
        core.reset();
        require(control.expired(), "shared control block must expire, raw core remains retained");
        no_free(calls);
        require(root->pins == 1, "unsafe branch decremented pinned diagnostic counter");
        root.reset();
        require(calls.weak.use_count() == 2 && !calls.weak.expired(), "leaked raw core must retain strong root member");
        const char event[] = "no-free strong-root count2\n";
        (void)::write(STDERR_FILENO, event, sizeof(event)-1);
        // calls/clock/cookie stay alive through guard close until actual _Exit70.
    }
    std::_Exit(93);
}
inline void retention() {
    // Fork only on the CPU path, before a real platform/probe in native main.
    int fds[2];
    require(::pipe(fds) == 0, "CPU fixture pipe failed");
    const auto child = ::fork();
    require(child >= 0, "CPU fixture fork failed");
    if (child == 0) {
        ::close(fds[0]);
        if (::dup2(fds[1], STDERR_FILENO) < 0) std::_Exit(94);
        ::close(fds[1]);
        try { retained_child(); }
        catch (...) { std::_Exit(95); }
    }
    ::close(fds[1]);
    std::string events;
    char bytes[256];
    ssize_t n;
    while ((n = ::read(fds[0], bytes, sizeof(bytes))) > 0) events.append(bytes, n);
    ::close(fds[0]);
    int status = 0;
    require(::waitpid(child, &status, 0) == child, "CPU fixture child not reaped");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 70, "isolated unsafe core must exit70, never success/abort");
    require(events == "no-free strong-root count2\nvirtual20 no-free count3\n"
                      "Vulkan terminal fault: unknown terminal root owner at close deadline\n",
            "isolated child lost exact no-free/original-deadline/retained-owner evidence");
    healthy_close();
    std::cout << "PASS isolated pinned core expected70 virtual20 no-free; healthy parent count2 weak-expired\n";
}
} // namespace vulkan_pinned_core_testing
