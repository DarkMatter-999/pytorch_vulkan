#include "vulkan_lifecycle.h"
#ifdef PYTORCH_VULKAN_LIFECYCLE_EXECUTION_TESTS
#include "vulkan_lifecycle_seam_test.h"
#include "vulkan_platform.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <future>
#include <iostream>
#include <new>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace pytorch_vulkan;
// Test-only allocation failure at the actual production TLS/attachment boundary.
thread_local bool fail_next_allocation = false;
void* operator new(std::size_t bytes) {
    if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc(); }
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
namespace {
void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Evidence {
    std::vector<std::string> events;
    int event_fd = -1;
    void log(const char* event) {
        events.emplace_back(event);
        if (event_fd >= 0) {
            (void)::write(event_fd, event, std::char_traits<char>::length(event));
            (void)::write(event_fd, "\n", 1);
        }
    }
};
struct FakeClock {
    VulkanHostTime time{};
    std::vector<VulkanHostTime> deadlines;
    std::function<void()> tick;
    bool keep_lock = false;
    VulkanHostClock clock() {
        return {this, [](void* p) noexcept { return static_cast<FakeClock*>(p)->time; },
            [](void* p, std::condition_variable&, std::unique_lock<std::mutex>& lock,
               VulkanHostTime deadline) {
                auto& c = *static_cast<FakeClock*>(p);
                c.deadlines.push_back(deadline);
                if (!c.keep_lock) lock.unlock();
                if (c.tick) c.tick(); else c.time = deadline;
                if (!c.keep_lock) lock.lock();
            },
            [](void* p, VulkanHostTime deadline) {
                auto& c = *static_cast<FakeClock*>(p);
                c.deadlines.push_back(deadline);
                if (c.tick) c.tick(); else c.time = deadline;
            }};
    }
};
struct FakeRoot {
    VulkanRuntimeLifecycle runtime;
    Evidence& evidence;
    VulkanLifecycleCoordinator& coordinator;
    std::vector<std::shared_ptr<void>> captures;
    bool callback = false;
    bool private_scope = false;
    bool reenter = false;
    bool gate_closed = false;
    VulkanQuiescence residual;
    std::function<void(VulkanHostTime)> foreign_wait;
    std::function<void()> service_action;
    std::function<void()> final_action;
    FakeRoot(VulkanHostClock c, Evidence& e, VulkanLifecycleCoordinator& owner)
        : runtime(c), evidence(e), coordinator(owner) {}
    ~FakeRoot() { if (final_action) final_action(); evidence.log("root-destroyed"); }
};
VulkanLifecycleRoot make_fake_root(FakeClock& clock, Evidence& evidence,
                                  VulkanLifecycleCoordinator& coordinator,
                                  FakeRoot*& raw) {
    auto owner = std::make_shared<FakeRoot>(clock.clock(), evidence, coordinator);
    raw = owner.get();
    return {owner, &owner->runtime,
        [](void* p, VulkanHostTime) {
            auto& f = *static_cast<FakeRoot*>(p);
            f.evidence.log("complete");
            if (f.reenter) {
                auto observed = f.coordinator.observe();
                expect(observed.get() == &f, "observation must retain existing cache");
                bool rejected = false;
                try { f.coordinator.acquire([]() -> VulkanLifecycleRoot {
                    throw std::runtime_error("closing must not construct");
                }); } catch (const VulkanAdmissionClosed&) { rejected = true; }
                expect(rejected, "closing cache must reject new work");
            }
            f.captures.clear();
            f.evidence.log("captures-released");
        },
        [](void* p) {
            auto& f = *static_cast<FakeRoot*>(p);
            auto s = f.residual;
            s.deferred_callbacks += f.captures.size();
            s.reservation_closing = f.gate_closed;
            return s;
        },
        [](void* p, bool closed) { static_cast<FakeRoot*>(p)->gate_closed = closed; },
        [](void* p, VulkanHostTime deadline) {
            auto& f = *static_cast<FakeRoot*>(p);
            if (f.foreign_wait) f.foreign_wait(deadline);
        },
        [](void* p) noexcept { return static_cast<FakeRoot*>(p)->callback; },
        [](void* p) noexcept { return static_cast<FakeRoot*>(p)->private_scope; },
        [](void* p) {
            auto& f = *static_cast<FakeRoot*>(p);
            expect(f.captures.empty(), "services cannot precede capture retirement");
            if (f.service_action) f.service_action();
            f.evidence.log("services");
        }};
}
void owner_cycle(bool reenter = false) {
    VulkanQuiescence empty;
    empty.reservation_closing = true;
    empty.reservation_generation = 9;
    expect(empty.healthy_empty(), "closed reservation gate alone remains healthy");
    for (auto field : {&VulkanQuiescence::recording, &VulkanQuiescence::pump_active,
                       &VulkanQuiescence::invalidated, &VulkanQuiescence::sticky_error}) {
        auto busy = empty; busy.*field = true;
        expect(!busy.healthy_empty(), "unhealthy execution state must prevent release");
    }
    for (auto field : {&VulkanQuiescence::submitted_slots, &VulkanQuiescence::ring_callbacks,
                       &VulkanQuiescence::deferred_callbacks, &VulkanQuiescence::completion_callbacks,
                       &VulkanQuiescence::ready_callbacks, &VulkanQuiescence::transfer_records,
                       &VulkanQuiescence::reservations}) {
        auto busy = empty; busy.*field = 1;
        expect(!busy.healthy_empty(), "every execution owner count must prevent release");
    }
    auto owned_pump = empty;
    owned_pump.pump_owner = std::this_thread::get_id();
    expect(!owned_pump.healthy_empty(), "residual pump ownership must prevent release");
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot* fake = nullptr;
    auto root = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, fake); });
    auto weak = std::weak_ptr<void>(root);
    {
        VulkanOperationScope operation(fake->runtime);
        bool rejected = false;
        try { coordinator.close(VulkanCloseBoundary::NativeEarly); }
        catch (const VulkanAdmissionClosed&) { rejected = true; }
        expect(rejected, "own-operation close must reject without changing state");
        auto observed = coordinator.acquire({});
        expect(observed.get() == root.get(), "self-close rejection must leave admission running");
        fake->callback = true;
        rejected = false;
        try { VulkanOperationScope nested(fake->runtime); }
        catch (const VulkanAdmissionClosed&) { rejected = true; }
        expect(rejected, "callback-created work is not nested admission");
        fake->callback = false;
    }
    fake->callback = true;
    bool callback_rejected = false;
    try { coordinator.close(VulkanCloseBoundary::NativeEarly); }
    catch (const VulkanAdmissionClosed&) { callback_rejected = true; }
    expect(callback_rejected, "callback self-close must reject without completion");
    fake->callback = false;
    fake->reenter = reenter;
    fake->captures.push_back(root);
    root.reset();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    expect(weak.expired(), "root must drain before release");
    expect(evidence.events == std::vector<std::string>({"complete", "captures-released",
        "services", "root-destroyed"}), "last release follows epilogues");
    coordinator.check_late();
    bool rejected = false;
    try { coordinator.acquire({}); } catch (const VulkanAdmissionClosed&) { rejected = true; }
    expect(rejected, "closed runtime cannot recreate root");
}

// Native-main shape: guard outside the try, every owner local to it. The guard's
// NativeEarly close runs only after the local owner has been destroyed.
struct GuardLocalOwner {
    std::shared_ptr<void> owner;
    Evidence& evidence;
    ~GuardLocalOwner() { evidence.log("local-destroyed"); }
};
int guard_fake_main(const std::string& mode, FakeClock& clock, Evidence& evidence,
                    VulkanLifecycleCoordinator& coordinator, std::shared_ptr<void>& external) {
    NativeVulkanTerminalGuard terminal(coordinator);
    try {
        if (mode == "unused") return 0;
        FakeRoot* fake = nullptr;
        GuardLocalOwner local{coordinator.acquire([&] {
            return make_fake_root(clock, evidence, coordinator, fake);
        }), evidence};
        if (mode == "error") throw std::runtime_error("fake native error");
        if (mode == "skip") return 77;
        if (mode == "residual") external = coordinator.observe();
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
// Child-only: a residual owner makes the guard's terminal close _Exit(70) from
// its destructor, so the child never reaches a clean return.
void guard_child(const std::string& name, FakeClock& clock, int fd) {
    Evidence evidence;
    evidence.event_fd = fd;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    std::shared_ptr<void> external;
    const auto dash = name.rfind('-');
    const int status = guard_fake_main(name.substr(6, dash - 6), clock, evidence, coordinator, external);
    std::_Exit(status);
}
// Each child execs this same CPU-only executable. Actual _Exit(70), never a
// throwing noreturn replacement; the pipe detects forbidden destruction.
std::string executable;
VulkanLifecycleCoordinator* late_fake = nullptr;
void check_fake_late() { late_fake->check_late(); }
void child_expect(const std::string& name, int wanted, const std::string& events = "") {
    int fds[2];
    expect(::pipe(fds) == 0, "pipe");
    auto pid = ::fork();
    expect(pid >= 0, "fork");
    if (pid == 0) {
        ::close(fds[0]);
        auto fd = std::to_string(fds[1]);
        ::execl(executable.c_str(), executable.c_str(), "--child", name.c_str(), fd.c_str(), nullptr);
        std::_Exit(126);
    }
    ::close(fds[1]);
    std::string actual;
    char buffer[256];
    ssize_t n;
    while ((n = ::read(fds[0], buffer, sizeof(buffer))) > 0) actual.append(buffer, n);
    ::close(fds[0]);
    int status = 0;
    expect(::waitpid(pid, &status, 0) == pid, "waitpid");
    expect(WIFEXITED(status) && WEXITSTATUS(status) == wanted, "unexpected child status");
    expect(actual == events, "child emitted forbidden or missing lifecycle events");
    std::cout << "child " << name << " exit=" << wanted << " events="
              << (actual.empty() ? "none" : actual) << " device_calls=0\n";
}
void child_case(const std::string& name, int fd) {
#ifdef PYTORCH_VULKAN_LIFECYCLE_EXECUTION_TESTS
    if (name == "operation-completion-failure") lifecycle_seam_test::failed_completion_child();
#endif
    FakeClock clock;
    if (name.rfind("guard-", 0) == 0) guard_child(name, clock, fd);
    if (name == "cleanup-mixed") {
        VulkanRuntimeLifecycle runtime(clock.clock());
        runtime.close_admission();
        VulkanCleanupScope native(runtime, VulkanCleanupKind::AuditedNativeOnly);
        {
            VulkanCleanupScope general(runtime);
            const auto activity = runtime.host_activity();
            expect(activity.cleanup_epilogues == 2 && activity.audited_native_cleanup_epilogues == 1,
                   "general cleanup cannot inherit audited-native classification");
            bool rejected = false;
            try { runtime.wait_idle_until(clock.time + std::chrono::seconds(20),
                                          VulkanIdleWait::AuditedNativeCleanupOnly); }
            catch (const VulkanAdmissionClosed&) { rejected = true; }
            expect(rejected, "mixed cleanup must reject native-only wait without blocking");
        }
        expect(runtime.cleanup_epilogues() == 1, "mixed cleanup must release its own accounting");
        std::_Exit(0);
    }
    Evidence evidence;
    evidence.event_fd = fd;
    auto coordinator = std::make_unique<VulkanLifecycleCoordinator>(clock.clock());
    FakeRoot* fake = nullptr;
    auto root = coordinator->acquire([&] { return make_fake_root(clock, evidence, *coordinator, fake); });
    if (name == "owner-tail-transient" || name == "owner-tail-native") {
        const bool native = name == "owner-tail-native";
        std::weak_ptr<void> weak = root;
        std::promise<void> accounting_left, release_tail;
        auto released = release_tail.get_future();
        std::thread worker([held = root, &accounting_left, &released, fake, native]() mutable {
            { VulkanCleanupScope cleanup(fake->runtime, native ? VulkanCleanupKind::AuditedNativeOnly
                                                             : VulkanCleanupKind::General); }
            // Real keepalive order: accounting leaves BEFORE the local owner.
            accounting_left.set_value(); released.wait(); held.reset();
        });
        root.reset(); accounting_left.get_future().wait();
        expect(fake->runtime.cleanup_epilogues() == 0, "fixture must pause after accounting drop");
        unsigned retries = 0;
        fake->foreign_wait = [&](VulkanHostTime deadline) {
            expect(deadline == VulkanHostTime{} + std::chrono::seconds(20), "original Closing budget");
            clock.time = VulkanHostTime{} + std::chrono::seconds(7);
        };
        clock.tick = [&] {
            expect(evidence.events == (native ? std::vector<std::string>{}
                                             : std::vector<std::string>{"complete", "captures-released"}),
                   "services or root released while foreign keepalive remains");
            expect(clock.deadlines.back() == VulkanHostTime{} + std::chrono::seconds(20),
                   "owner reconciliation renewed original deadline");
            expect(!coordinator->observe(), "owner reconciliation must seal new observation copies");
            if (++retries == 3) { release_tail.set_value(); worker.join(); clock.time += std::chrono::seconds(1); }
            // First two returns simulate spurious wakeups with no time change.
        };
        fake->service_action = [&] { expect(!coordinator->observe(), "services cannot create observed owners"); };
        fake->final_action = [&] { expect(!coordinator->observe(), "Quiescent cannot introduce observed owners"); };
        coordinator->close(native ? VulkanCloseBoundary::PythonNative : VulkanCloseBoundary::NativeEarly);
        expect(retries == 3 && weak.expired(), "transient foreign keepalive must reconcile before close");
        evidence.log("owner-tail-reconciled");
        std::_Exit(0);
    }
    if (name == "owner-tail-persistent" || name == "owner-tail-persistent-native") {
        const bool native = name == "owner-tail-persistent-native";
        static std::shared_ptr<void> opaque_static;
        thread_local std::shared_ptr<void> opaque_tls;
        if (native) opaque_tls = std::move(root);
        else opaque_static = std::move(root);
        unsigned retries = 0;
        clock.tick = [&] {
            expect(clock.deadlines.back() == VulkanHostTime{} + std::chrono::seconds(20), "owner deadline renewed");
            expect(!coordinator->observe(), "terminal observation remains closed while owner persists");
            expect(evidence.events == (native ? std::vector<std::string>{}
                                             : std::vector<std::string>{"complete", "captures-released"}),
                   "persistent unknown owner must retain root and all native services");
            if (++retries > 2) clock.time += std::chrono::seconds(5);
            if (clock.time == VulkanHostTime{} + std::chrono::seconds(20)) evidence.log("owner-deadline-20");
        };
        coordinator->close(native ? VulkanCloseBoundary::PythonNative : VulkanCloseBoundary::NativeEarly);
        std::_Exit(125);
    }
    if (name == "owner-post-service-persistent") {
        std::weak_ptr<void> weak = root;
        std::shared_ptr<void> tail;
        fake->service_action = [&] { tail = weak.lock(); };
        root.reset();
        clock.tick = [&] {
            expect(clock.deadlines.back() == VulkanHostTime{} + std::chrono::seconds(20), "post-service budget renewed");
            expect(tail && !weak.expired() && !coordinator->observe(), "post-service unknown owner must retain root");
            expect(evidence.events == std::vector<std::string>({"complete", "captures-released", "services"}),
                   "post-service unknown owner cannot be destroyed to force final release");
            clock.time += std::chrono::seconds(5);
            if (clock.time == VulkanHostTime{} + std::chrono::seconds(20)) evidence.log("owner-deadline-20");
        };
        coordinator->close(VulkanCloseBoundary::NativeEarly);
        std::_Exit(125);
    }
    if (name == "retire-cross") {
        std::weak_ptr<void> old = root;
        coordinator->retire_cached();
        FakeRoot* next = nullptr;
        auto newer = coordinator->acquire([&] { return make_fake_root(clock, evidence, *coordinator, next); });
        next->captures.push_back(root);
        root.reset(); newer.reset();
        coordinator->close(VulkanCloseBoundary::NativeEarly);
        expect(old.expired(), "new cache captures must not strand retiring root");
        std::_Exit(0);
    }
    if (name == "unknown-owner") coordinator->close(VulkanCloseBoundary::NativeEarly);
    root.reset();
    if (name == "late-check") {
        late_fake = coordinator.get();
        expect(std::atexit(check_fake_late) == 0, "fake late check registration");
        std::exit(0);
    }
    if (name == "holder-check") coordinator.reset();
    if (name == "native-residual") {
        fake->captures.push_back(coordinator->observe());
        coordinator->close(VulkanCloseBoundary::PythonNative);
    }
    if (name == "host-deadline" || name == "host-deadline-cleanup" || name == "host-deadline-foreign") {
        const auto expected = clock.time + std::chrono::seconds(20);
        clock.tick = [&] {
            expect(clock.deadlines.back() == expected, "deadline must never renew");
            clock.time += std::chrono::seconds(5);
            if (clock.time == expected) evidence.log("deadline-20");
        };
        if (name == "host-deadline-cleanup") {
            fake->runtime.enter_cleanup(VulkanCleanupKind::General);
            coordinator->close(VulkanCloseBoundary::NativeEarly);
            std::_Exit(125);
        }
        if (name == "host-deadline-foreign") {
            fake->foreign_wait = [&](VulkanHostTime deadline) {
                expect(deadline == expected, "foreign pump must receive original deadline");
                std::mutex m;
                std::condition_variable cv;
                std::unique_lock<std::mutex> lock(m);
                while (clock.time < deadline) clock.clock().wait_until(&clock, cv, lock, deadline);
                throw VulkanHostCloseTimeout("fake foreign host wait");
            };
            coordinator->close(VulkanCloseBoundary::NativeEarly);
            std::_Exit(125);
        }
        // Worker owns the permit: own-operation close rejection must not mask timeout.
        std::mutex m;
        std::condition_variable cv;
        bool admitted = false, release = false;
        std::thread worker([&] {
            VulkanOperationScope operation(fake->runtime);
            std::unique_lock<std::mutex> lock(m);
            admitted = true; cv.notify_all();
            cv.wait(lock, [&] { return release; });
        });
        { std::unique_lock<std::mutex> lock(m); cv.wait(lock, [&] { return admitted; }); }
        coordinator->close(VulkanCloseBoundary::NativeEarly);
        std::_Exit(125);
    }
    std::_Exit(124);
}
void retire_live() {
    child_expect("retire-cross", 0, "complete\ncaptures-released\ncomplete\ncaptures-released\n"
                 "complete\ncaptures-released\nservices\nservices\nroot-destroyed\nroot-destroyed\n");
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot* fake = nullptr;
    auto allocation = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, fake); });
    std::weak_ptr<void> weak = allocation;
    coordinator.retire_cached();
    expect(!coordinator.observe(), "retired cache must detach");
    expect(fake->gate_closed && !weak.expired(), "live allocation metadata remains valid");
    bool sealed = false;
    try { fake->runtime.reopen_admission(); }
    catch (const VulkanAdmissionClosed&) { sealed = true; }
    allocation.reset();
    expect(!weak.expired(), "retiring slot must own root");
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    expect(weak.expired(), "retiring root must expire at early close");
    expect(sealed, "retired runtime can never reopen admission");
}
void owner_cross_roots() {
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot *old = nullptr, *current = nullptr;
    auto first = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, old); });
    std::weak_ptr<void> old_weak = first;
    coordinator.retire_cached();
    auto second = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, current); });
    std::weak_ptr<void> new_weak = second;
    std::shared_ptr<void> tail;
    unsigned services = 0, retries = 0;
    old->service_action = [&] {
        expect(!tail && services == 0 && !coordinator.observe(), "first service ownership gate");
        ++services; tail = new_weak.lock();
    };
    current->service_action = [&] {
        expect(!tail && services == 1 && !coordinator.observe(), "cross-root owner must reconcile before second service");
        ++services; tail = old_weak.lock();
    };
    clock.tick = [&] {
        expect(clock.deadlines.back() == VulkanHostTime{} + std::chrono::seconds(20), "cross-root reconciliation renewed deadline");
        expect(tail && !old_weak.expired() && !new_weak.expired(), "no root may release while cross-root tail persists");
        ++retries; clock.time += std::chrono::seconds(3); tail.reset();
    };
    first.reset(); second.reset();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    expect(services == 2 && retries == 2 && old_weak.expired() && new_weak.expired(),
           "post-service/final root ownership gates did not reconcile every root");
}
void owner_observation_race() {
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot* raw = nullptr;
    auto root = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, raw); });
    std::weak_ptr<void> weak = root;
    std::promise<void> observe_now, observed, release;
    auto observing = observe_now.get_future(); auto released = release.get_future();
    std::atomic<bool> later_rejected{false};
    std::thread observer([&] {
        observing.wait();
        auto held = coordinator.observe();
        expect(held != nullptr, "pre-seal Draining observation remains compatible");
        observed.set_value(); released.wait();
        // Main has reached its first post-seal count retry.
        later_rejected = !coordinator.observe(); held.reset();
    });
    raw->foreign_wait = [&](VulkanHostTime) { observe_now.set_value(); observed.get_future().wait(); };
    clock.tick = [&] {
        expect(!coordinator.observe(), "new observation must not race exact count proof");
        release.set_value(); observer.join(); clock.time += std::chrono::seconds(1);
    };
    raw->service_action = [&] { expect(later_rejected && !coordinator.observe(), "late observer/service created owner after seal"); };
    raw->final_action = [&] { expect(!coordinator.observe(), "final gate reopened observation"); };
    root.reset(); coordinator.close(VulkanCloseBoundary::NativeEarly);
    expect(weak.expired() && later_rejected, "racing observation was not reconciled before root release");
}
void init_close() {
    FakeClock clock;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    Evidence evidence;
    FakeRoot* fake = nullptr;
    std::mutex m;
    std::condition_variable cv;
    bool factory_entered = false, publish = false;
    std::atomic<bool> rejected{false};
    std::thread init([&] {
        try {
            coordinator.acquire([&] {
                { std::unique_lock<std::mutex> lock(m); factory_entered = true; cv.notify_all();
                  cv.wait(lock, [&] { return publish; }); }
                return make_fake_root(clock, evidence, coordinator, fake);
            });
        } catch (const VulkanAdmissionClosed&) { rejected = true; }
    });
    { std::unique_lock<std::mutex> lock(m); cv.wait(lock, [&] { return factory_entered; }); }
    clock.tick = [&] {
        // The coordinator has entered Closing and is waiting on initialization.
        bool closed = false;
        try { coordinator.acquire({}); } catch (const VulkanAdmissionClosed&) { closed = true; }
        expect(closed, "close must reject acquisition before factory publication");
        { std::lock_guard<std::mutex> lock(m); publish = true; cv.notify_all(); }
        init.join();
    };
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    expect(rejected && !coordinator.observe(), "initialization cannot publish usable root after closing");
}
void fork_case() {
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot* fake = nullptr;
    auto root = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, fake); });
    root.reset();
    auto permit = std::make_unique<VulkanOperationScope>(fake->runtime);
    // Injected clock holds the actual runtime mutex while signalling parent.
    std::mutex m;
    std::condition_variable cv;
    bool locked = false, released = false;
    clock.tick = [&] {
        std::unique_lock<std::mutex> lock(m);
        locked = true; cv.notify_all(); cv.wait(lock, [&] { return released; });
        clock.time = VulkanHostTime{} + std::chrono::seconds(20);
    };
    clock.keep_lock = true;
    std::thread worker([&] {
        try { fake->runtime.wait_idle_until(VulkanHostTime{} + std::chrono::seconds(20), VulkanIdleWait::AllEpilogues); }
        catch (const VulkanHostCloseTimeout&) {}
    });
    { std::unique_lock<std::mutex> lock(m); cv.wait(lock, [&] { return locked; }); }
    int fds[2];
    expect(::pipe(fds) == 0, "fork evidence pipe");
    evidence.event_fd = fds[1];
    auto pid = ::fork();
    expect(pid >= 0, "fork");
    if (pid == 0) {
        ::close(fds[0]);
        coordinator.abandon_inherited_child();
        coordinator.close(VulkanCloseBoundary::NativeEarly);
        permit.reset();
        coordinator.check_late();
        coordinator.~VulkanLifecycleCoordinator();
        std::_Exit(7);
    }
    ::close(fds[1]);
    evidence.event_fd = -1;
    { std::lock_guard<std::mutex> lock(m); released = true; cv.notify_all(); }
    worker.join();
    int status = 0;
    expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 7,
           "fork child must abandon promptly with exit7");
    char event;
    expect(::read(fds[0], &event, 1) == 0, "fork branch must not destroy inherited root or services");
    ::close(fds[0]);
    permit.reset();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
}
void finalizing_mode() {
    FakeClock clock;
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    const auto deadline = coordinator.prepare_python_finalizing_sync();
    expect(!coordinator.observe(), "unused Python transition must not construct a root");
    coordinator.finish_python_finalizing_sync(deadline);
    FakeRoot* fake = nullptr;
    auto root = coordinator.acquire([&] { return make_fake_root(clock, evidence, coordinator, fake); });
    expect(fake->runtime.finalizing_sync(), "first use must inherit monotonic finalizing mode");
    fake->captures.push_back(root);
    {
        VulkanOperationScope outer(fake->runtime);
        { VulkanOperationScope nested(fake->runtime); nested.finish(); }
        expect(evidence.events.empty(), "nested operation must not synchronize");
        outer.finish();
        outer.finish();
    }
    expect(fake->captures.empty() && fake->runtime.outer_permits() == 0,
           "outer finish must retire captures before releasing permit");
    bool original = false;
    try {
        VulkanOperationScope operation(fake->runtime);
        fake->captures.push_back(root);
        operation.finish_error(std::make_exception_ptr(std::runtime_error("original")));
    } catch (const std::runtime_error& e) { original = std::string(e.what()) == "original"; }
    expect(original && fake->captures.empty() && fake->runtime.outer_permits() == 0,
           "healthy error completion must preserve original exception");
    root.reset();
    coordinator.close(VulkanCloseBoundary::PythonNative);
    expect(evidence.events == std::vector<std::string>({"complete", "captures-released",
        "complete", "captures-released", "services", "root-destroyed"}),
        "post-interpreter native close must not complete arbitrary captures");
    child_expect("native-residual", 70);
}
void admission_allocation() {
    child_expect("cleanup-mixed", 0);
    FakeClock clock;
    VulkanRuntimeLifecycle runtime(clock.clock());
    runtime.close_admission();
    bool rejected = false;
    try { VulkanOperationScope operation(runtime); } catch (const VulkanAdmissionClosed&) { rejected = true; }
    expect(rejected && runtime.outer_permits() == 0, "rejected admission must not count work");
    runtime.reopen_admission();
    std::atomic<bool> allocation_rejected{false};
    std::thread allocation_thread([&] {
        fail_next_allocation = true;
        try { VulkanOperationScope operation(runtime); }
        catch (const std::bad_alloc&) { allocation_rejected = true; }
    });
    allocation_thread.join();
    expect(allocation_rejected && runtime.outer_permits() == 0,
           "TLS allocation failure must leave no permit");
    { VulkanOperationScope outer(runtime);
      runtime.close_admission();
      { VulkanOperationScope nested(runtime); nested.finish(); }
      expect(runtime.outer_permits() == 1, "nested closing scope inherits outer permit"); }
    expect(runtime.outer_permits() == 0, "unwind releases accounting");
    {
        VulkanCleanupScope outer(runtime);
        { VulkanCleanupScope nested(runtime);
          expect(runtime.cleanup_epilogues() == 1, "nested cleanup inherits outer epilogue without relocking"); }
    }
    expect(runtime.cleanup_epilogues() == 0, "outer cleanup releases final accounting");
    Evidence evidence;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    FakeRoot* fake = nullptr;
    int factory_calls = 0;
    const std::function<VulkanLifecycleRoot()> factory = [&] {
        ++factory_calls;
        return make_fake_root(clock, evidence, coordinator, fake);
    };
    fail_next_allocation = true;
    bool attachment_rejected = false;
    try { coordinator.acquire(factory); }
    catch (const std::bad_alloc&) { attachment_rejected = true; }
    expect(attachment_rejected && factory_calls == 0 && !coordinator.observe(),
           "root attachment allocation must precede factory work");
    auto root = coordinator.acquire(factory);
    root.reset();
    coordinator.close(VulkanCloseBoundary::NativeEarly);
    VulkanLifecycleCoordinator failed_coordinator(clock.clock());
    int attempts = 0;
    for (int i = 0; i != 2; ++i) {
        try { failed_coordinator.acquire([&]() -> VulkanLifecycleRoot { ++attempts; throw std::bad_alloc(); }); }
        catch (const std::bad_alloc&) {}
    }
    expect(attempts == 1 && !failed_coordinator.observe(), "failed initialization preserves original exception");
    failed_coordinator.close(VulkanCloseBoundary::NativeEarly);
}
void host_deadline() {
    {
        FakeClock clock;
        VulkanRuntimeLifecycle runtime(clock.clock());
        runtime.set_finalizing_sync();
        std::mutex mutex;
        std::condition_variable cv;
        bool held = false, released = false;
        std::thread worker([&] {
            VulkanOperationScope operation(runtime);
            std::unique_lock<std::mutex> lock(mutex);
            held = true; cv.notify_all(); cv.wait(lock, [&] { return released; });
            // Accounting-only destructor: deliberately no completion adapter.
        });
        { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return held; }); }
        clock.tick = [&] { clock.time += std::chrono::seconds(5); };
        bool timed_out = false;
        try { VulkanOperationScope operation(runtime); }
        catch (const VulkanHostCloseTimeout&) { timed_out = true; }
        const auto remaining = runtime.outer_permits();
        { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
        worker.join();
        expect(timed_out && remaining == 1 && runtime.outer_permits() == 0 &&
                   clock.deadlines == std::vector<VulkanHostTime>(4, VulkanHostTime{} + std::chrono::seconds(20)),
               "Running finalizing admission has one 20s budget and leaves no partially admitted permit");
    }
    {
        FakeClock clock;
        VulkanRuntimeLifecycle runtime(clock.clock());
        const auto operation_budget = clock.time + std::chrono::seconds(20);
        expect(runtime.completion_deadline(operation_budget) == operation_budget,
               "Running completion gets its distinct operation budget");
        const auto closing = clock.time + std::chrono::seconds(7);
        runtime.close_admission(closing);
        clock.time += std::chrono::seconds(3);
        expect(runtime.completion_deadline(clock.time + std::chrono::seconds(20)) == closing,
               "operation epilogue cannot renew Closing budget");
        runtime.close_admission(clock.time + std::chrono::seconds(20));
        expect(runtime.completion_deadline(operation_budget) == closing,
               "repeated close cannot extend Closing deadline");
        runtime.reopen_admission();
        expect(runtime.completion_deadline(operation_budget) == operation_budget,
               "temporary barrier reopening restores only the Running budget");
    }
    child_expect("host-deadline", 70, "deadline-20\n");
    child_expect("host-deadline-cleanup", 70, "deadline-20\n");
    child_expect("host-deadline-foreign", 70, "deadline-20\n");
    FakeClock clock;
    VulkanRuntimeLifecycle runtime(clock.clock());
    runtime.enter_cleanup(VulkanCleanupKind::AuditedNativeOnly);
    clock.tick = [&] {
        expect(clock.deadlines.back() == VulkanHostTime{} + std::chrono::seconds(20), "shared absolute deadline");
        clock.time = VulkanHostTime{} + std::chrono::seconds(19);
        runtime.leave_cleanup(VulkanCleanupKind::AuditedNativeOnly);
    };
    runtime.wait_idle_until(VulkanHostTime{} + std::chrono::seconds(20), VulkanIdleWait::AuditedNativeCleanupOnly);
    expect(runtime.cleanup_epilogues() == 0, "healthy release at19s passes");
    runtime.enter_cleanup(VulkanCleanupKind::AuditedNativeOnly);
    clock.tick = [&] { runtime.enter_cleanup(VulkanCleanupKind::General); };
    bool rejected = false;
    try { runtime.wait_idle_until(VulkanHostTime{} + std::chrono::seconds(20), VulkanIdleWait::AuditedNativeCleanupOnly); }
    catch (const VulkanAdmissionClosed&) { rejected = true; }
    expect(rejected, "native-only wait rejects newly observed general cleanup immediately");
    runtime.leave_cleanup(VulkanCleanupKind::General);
    runtime.leave_cleanup(VulkanCleanupKind::AuditedNativeOnly);
}
#ifdef PYTORCH_VULKAN_LIFECYCLE_EXECUTION_TESTS
void deferred_pure() {
    FakeClock clock;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    fail_next_allocation = true;
    const auto result = with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [] { return 42; });
    const bool value_no_allocation = fail_next_allocation;
    fail_next_allocation = false;
    int rejected = 0;
    for (int validation : {101, 102}) {
        fail_next_allocation = true;
        try {
            with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [validation] {
                throw validation; // dtype/layout-shaped rejection before factory
            });
        } catch (int original) { if (original == validation && fail_next_allocation) ++rejected; }
        fail_next_allocation = false;
    }
    expect(result == 42 && value_no_allocation && rejected == 2 && !coordinator.observe(),
           "deferred metadata or early dtype/layout rejection allocated admission storage or initialized a root");
    coordinator.close(VulkanCloseBoundary::NativeEarly);
}
void deferred_failures() {
    lifecycle_seam_test::Fixture fixture;
    std::atomic<bool> failed{false}, reused{false};
    std::atomic<unsigned> side_effects{0};
    std::thread worker([&] {
        try {
            with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
                fail_next_allocation = true;
                VulkanOperationScope scope(fixture.root->runtime);
                ++side_effects;
                scope.finish();
            });
        } catch (const std::bad_alloc&) { failed = true; }
        fail_next_allocation = false;
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            { VulkanOperationScope scope(fixture.root->runtime); scope.finish(); }
            reused = fixture.root->runtime.outer_permits() == 1;
        });
    });
    worker.join();
    expect(failed && reused && side_effects == 0 && fixture.root->runtime.outer_permits() == 0,
           "deferred admission allocation failure left partial binding, work or TLS state");
    FakeClock clock;
    VulkanLifecycleCoordinator coordinator(clock.clock());
    unsigned factories = 0;
    const std::function<VulkanLifecycleRoot()> factory = [&]() -> VulkanLifecycleRoot {
        ++factories;
        throw std::bad_alloc();
    };
    bool attachment_failed = false, factory_failed = false;
    try {
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] {
            fail_next_allocation = true;
            coordinator.acquire(factory);
        });
    } catch (const std::bad_alloc&) { attachment_failed = true; }
    fail_next_allocation = false;
    expect(attachment_failed && factories == 0 && !coordinator.observe(), "deferred root attachment failure ran factory or published a root");
    try {
        with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), [&] { coordinator.acquire(factory); });
    } catch (const std::bad_alloc&) { factory_failed = true; }
    expect(factory_failed && factories == 1 && !coordinator.observe(), "deferred factory failure changed original error or published a root");
    coordinator.close(VulkanCloseBoundary::NativeEarly);
}
#endif
const std::vector<std::string> guard_labels = {
    "allocator", "transfer_tensor", "pinned_allocator", "dtype", "serialization_probe"};
void guard_selector(const std::string& name) {
    const std::string closed = "local-destroyed\ncomplete\ncaptures-released\nservices\nroot-destroyed\n";
    int status = 0;
    std::string events = closed;
    if (name == "guard-success") status = 0;
    else if (name == "guard-error") status = 1;
    else if (name == "guard-skip") status = 77;
    else if (name == "guard-unused") { events.clear(); }
    else if (name == "guard-residual") {
        status = 70;
        events = "local-destroyed\ncomplete\ncaptures-released\n";
    }
    else throw std::runtime_error("unknown guard case");
    for (const auto& label : guard_labels) child_expect(name + "-" + label, status, events);
}
void run(const std::string& name) {
    if (name == "owner-reconciliation") {
        child_expect("owner-tail-transient", 0,
                     "complete\ncaptures-released\nservices\nroot-destroyed\nowner-tail-reconciled\n");
    }
    else if (name == "owner-reconciliation-native") {
        child_expect("owner-tail-native", 0, "services\nroot-destroyed\nowner-tail-reconciled\n");
    }
    else if (name == "owner-reconciliation-persistent") {
        child_expect("owner-tail-persistent", 70, "complete\ncaptures-released\nowner-deadline-20\n");
        child_expect("owner-tail-persistent-native", 70, "owner-deadline-20\n");
        child_expect("owner-post-service-persistent", 70, "complete\ncaptures-released\nservices\nowner-deadline-20\n");
    }
    else if (name == "owner-reconciliation-cross-roots") owner_cross_roots();
    else if (name == "owner-reconciliation-observe") owner_observation_race();
    else if (name == "owner-cycle") owner_cycle();
    else if (name == "cache-reentry") { owner_cycle(true); finalizing_mode(); }
    else if (name == "unknown-owner") child_expect(name, 70, "complete\ncaptures-released\n");
    else if (name == "late-check" || name == "holder-check") child_expect(name, 70);
    else if (name == "retire-live") retire_live();
    else if (name == "init-close") init_close();
    else if (name == "fork") fork_case();
    else if (name == "host-deadline") host_deadline();
    else if (name == "admission-allocation") admission_allocation();
    else if (name.rfind("guard-", 0) == 0) guard_selector(name);
#ifdef PYTORCH_VULKAN_LIFECYCLE_EXECUTION_TESTS
    else if (name == "deferred-pure") deferred_pure();
    else if (name == "deferred-failures") deferred_failures();
    else if (name == "operation-error") {
        lifecycle_seam_test::run(name);
        child_expect("operation-completion-failure", 70);
    }
    else if (name == "operation-tail" ||
             name == "finalizing-threads" || name == "cleanup-epilogue" ||
             name == "nested-closing" || name == "forced-unwind" ||
             name == "autograd-exception-tail" ||
             name == "deferred-entry" || name == "deferred-nesting" || name == "deferred-closing" ||
             name == "deferred-lifetime" ||
             name == "deferred-known-runtime" ||
             name == "deferred-forced-unwind" || name == "deferred-concurrent-init" ||
             name == "standalone-copy-tail" || name == "wrapper-tail" ||
             name == "rnn-scratch-tail") lifecycle_seam_test::run(name);
#endif
    else throw std::runtime_error("unknown case");
#ifdef PYTORCH_VULKAN_LIFECYCLE_EXECUTION_TESTS
    expect(VulkanPlatform::platform_construction_count() == 0 &&
               VulkanPlatform::availability_probe_count() == 0,
           "lifecycle CPU selector constructed or probed a Vulkan platform");
#endif
    std::cout << "PASS " << name << " device_calls=0\n";
}
} // namespace
int main(int argc, char** argv) {
    executable = argv[0];
    try {
        if (argc == 4 && std::string(argv[1]) == "--child") child_case(argv[2], std::stoi(argv[3]));
        expect(argc == 3 && std::string(argv[1]) == "--case", "usage: --case NAME");
        if (std::string(argv[2]) == "all") {
            for (auto name : {"owner-cycle", "cache-reentry", "unknown-owner", "late-check", "holder-check",
                "init-close", "retire-live", "fork", "host-deadline", "admission-allocation"}) run(name);
        } else run(argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
