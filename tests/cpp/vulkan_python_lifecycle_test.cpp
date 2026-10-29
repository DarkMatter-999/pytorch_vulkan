#include "vulkan_python_lifecycle.h"
#include "vulkan_lifecycle_seam_test.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <future>
#include <iostream>
#include <pthread.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using namespace pytorch_vulkan;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Test-owned threads use an EXPLICIT cleanup seam, never a joining destructor.
// All rendezvous observe worker failure so a thrown worker cannot strand main.
// The caller releases its native gates/holds before join_all drops the GIL.
class FixtureWorkers {
 public:
    template<class Body> void start(std::size_t slot, Body body) {
        threads_[slot] = std::thread([this, slot, body = std::move(body)]() mutable {
            try { body(); }
            catch (const __cxxabiv1::__forced_unwind&) { throw; }
            catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                errors_[slot] = std::current_exception();
                failed_ = true;
                cv_.notify_all();
            }
        });
    }
    void signal(std::atomic<bool>& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        event = true;
        cv_.notify_all();
    }
    template<class Ready> void wait(Ready ready) {
        with_vulkan_python_native([&] {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return ready() || failed_; });
            if (failed_) for (auto& error : errors_) if (error) std::rethrow_exception(error);
        });
    }
    void join_all(bool restore_gil) {
        // A cancellation during pthread_join must not unwind through the other
        // joinable test threads. Delay it until ALL native joins have finished.
        // On an already-forced-unwind route there is no resumed caller/GIL.
        int prior_cancel_state = 0;
        if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &prior_cancel_state) != 0)
            std::_Exit(90);
        auto* saved = PyGILState_Check() ? PyEval_SaveThread() : nullptr;
        try {
            for (auto& thread : threads_) if (thread.joinable()) thread.join();
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            // Cannot assert quiescence or unwind through remaining std::threads.
            // A new occurrence is an uncertain-cleanup STOP, never a test pass.
            constexpr char message[] = "fixture cleanup uncertain: native join failed\n";
            (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
            std::_Exit(90);
        }
        if (restore_gil && saved) PyEval_RestoreThread(saved);
        if (restore_gil && pthread_setcancelstate(prior_cancel_state, nullptr) != 0)
            std::_Exit(90);
    }
    void rethrow_error() {
        for (auto& error : errors_) if (error) std::rethrow_exception(error); // joined
    }
 private:
    std::array<std::thread, 3> threads_;
    std::array<std::exception_ptr, 3> errors_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool failed_ = false;
};
template<class Body> void python_worker(Body body) {
    auto gil = PyGILState_Ensure();
    try { body(); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; } // no GIL work on unwind
    catch (...) {
        if (!PyGILState_Check()) std::_Exit(92); // classify, do not call C API unsafely
        PyGILState_Release(gil);
        throw;
    }
    if (!PyGILState_Check()) std::_Exit(92);
    PyGILState_Release(gil);
}
template<class Release, class AfterJoin, class Body>
void fixture_workers(Release release, AfterJoin after_join, Body body) {
    FixtureWorkers workers; // fully constructed BEFORE any thread starts
    const auto release_native = [&] {
        try { release(workers); }
        catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            constexpr char message[] = "fixture cleanup uncertain: gate release failed\n";
            (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
            std::_Exit(90);
        }
    };
    std::exception_ptr original;
    try { body(workers); }
    catch (const __cxxabiv1::__forced_unwind&) {
        release_native();
        workers.join_all(false); // release gates, join natively, NEVER restore GIL
        after_join();
        throw;
    }
    catch (...) { original = std::current_exception(); }
    // Also covers failure to create the second/third std::thread. No destructor
    // may wait for a worker needing the GIL or a still-closed fixture gate.
    release_native();
    workers.join_all(true);
    try { after_join(); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { if (!original) original = std::current_exception(); }
    if (original) std::rethrow_exception(original);
    workers.rethrow_error();
}
std::string executable;
std::string child_case;
std::vector<std::string> events;
bool interpreter_alive = true;
int factories = 0, native_regs = 0, python_regs = 0, unregisters = 0;
int wait_calls = 0, completions = 0, captures = 0, destroyed = 0;
bool last_completion_gil = false;
bool fake_deadline = false;
VulkanHostTime fake_now{};
std::function<void()> on_wait;
VulkanHostTime first_deadline{};
VulkanHostClock clock_ops() {
    return {nullptr, [](void*) noexcept {
        return fake_deadline ? fake_now : std::chrono::steady_clock::now();
    }, [](void*, std::condition_variable& cv, std::unique_lock<std::mutex>& lock, VulkanHostTime deadline) {
        if (child_case == "native-host-outer" || child_case == "native-host-general" ||
            child_case == "daemon-deadline-outer" || child_case == "native-empty-check") std::_Exit(95);
        ++wait_calls;
        if (!fake_deadline) { cv.wait_until(lock, deadline); return; }
        // Fake-clock children each exercise one injected Closing transaction.
        // Real-clock finalizing operations have independent per-operation budgets.
        if (first_deadline == VulkanHostTime{}) first_deadline = deadline;
        require(first_deadline == deadline, "deadline renewed");
        lock.unlock();
        if (on_wait) on_wait(); else fake_now = deadline;
        lock.lock();
    }, [](void*, VulkanHostTime deadline) {
        ++wait_calls;
        if (on_wait) on_wait(); else fake_now = deadline;
    }};
}
struct Device : lifecycle_seam_test::Device {
    std::function<void()> native_fence;
    VulkanExecutionOps ops() {
        auto o = lifecycle_seam_test::Device::ops();
        o.wait_for_fences = [](void* p, VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) {
            auto& d = *static_cast<Device*>(p);
            if (d.native_fence) d.native_fence();
            return VK_SUCCESS;
        };
        return o;
    }
};
struct Root {
    Device device;
    VulkanRuntimeLifecycle runtime{clock_ops()};
    std::mutex queue;
    VulkanExecutionQuarantineSink sink{nullptr, [](void*, VulkanExecutionQuarantine*) noexcept { std::_Exit(96); }};
    VulkanExecutionContext context{lifecycle_seam_test::handle<VkDevice>(1),
        lifecycle_seam_test::handle<VkQueue>(2), lifecycle_seam_test::handle<VkCommandPool>(3),
        device.ops(), sink, queue};
    bool fail_completion = false;
    std::function<void()> gate;
    std::function<void()> before_complete;
    ~Root() { ++destroyed; events.emplace_back("root-release"); }
    VulkanLifecycleRoot adapter(std::shared_ptr<Root> owner) {
        return {owner, &runtime,
            [](void* p, VulkanHostTime deadline) {
                require(interpreter_alive, "post-interpreter completion");
                auto& r = *static_cast<Root*>(p);
                if (r.before_complete) r.before_complete();
                last_completion_gil = PyGILState_Check();
                ++completions;
                events.emplace_back("transition-complete");
                if (r.fail_completion) throw std::runtime_error("injected completion fault");
                r.context.synchronize_with_flush_until([&] {
                    if (r.context.recording()) r.context.submit_locked();
                }, deadline, clock_ops());
            },
            [](void* p) {
                auto& r = *static_cast<Root*>(p);
                return r.context.lifecycle_snapshot();
            },
            [](void* p, bool closed) {
                auto& r = *static_cast<Root*>(p);
                r.context.set_lifecycle_reservations_closed(closed);
                if (closed && r.gate) r.gate();
            },
            [](void* p, VulkanHostTime deadline) {
                static_cast<Root*>(p)->context.wait_foreign_epilogues_until(deadline, clock_ops());
            },
            [](void* p) noexcept { return static_cast<Root*>(p)->context.callback_active_on_this_thread(); },
            [](void*) noexcept { return false; },
            [](void* p) {
                auto& r = *static_cast<Root*>(p);
                require(r.context.lifecycle_snapshot().healthy_empty(), "services precede empty snapshot");
                events.emplace_back("services");
                r.context.teardown_empty_native_services({});
            }};
    }
};
VulkanLifecycleCoordinator* coordinator;
Root* root = nullptr;
std::shared_ptr<Root> acquire_native() {
    return std::static_pointer_cast<Root>(coordinator->acquire([] {
        ++factories;
        auto r = std::make_shared<Root>();
        root = r.get();
        return r->adapter(r);
    }));
}
std::shared_ptr<Root> acquire_binding() {
    return acquire_vulkan_python_binding_root(*coordinator, acquire_native,
        [](Root& r) -> VulkanRuntimeLifecycle& { return r.runtime; },
        [](Root& r) { return r.context.callback_active_on_this_thread(); });
}
void capture_destructor(PyObject*) {
    if (!interpreter_alive) std::_Exit(91); // NEVER query Python once dead
    if (!PyGILState_Check()) std::_Exit(92);
    ++captures;
}
std::shared_ptr<void> python_capture() {
    auto* value = PyCapsule_New(&captures, "lifecycle-capture", capture_destructor);
    require(value != nullptr, "capsule allocation");
    return std::shared_ptr<void>(value, [](void* p) {
        if (!interpreter_alive) std::_Exit(93);
        if (!PyGILState_Check()) std::_Exit(92); // classify violation, never throw from deleter
        Py_DECREF(static_cast<PyObject*>(p));
    });
}
void enqueue_capture(Root& r) {
    auto capture = python_capture();
    r.context.begin("embedded-capture");
    r.context.defer_destruction([capture] {});
}
void mark_dead() { interpreter_alive = false; events.emplace_back("interpreter-dead"); }
void noop_native() {}
int native_register(void (*function)()) {
    ++native_regs;
    if (child_case == "transaction-native") return -1;
    return Py_AtExit(function);
}
int python_register(PyObject* function) {
    ++python_regs;
    if (python_regs == 2 && child_case == "rollback-stale-hold") {
        if (vulkan_python_hook_ops().python_register(function) != 0) return -1;
        PyErr_SetString(PyExc_RuntimeError, "register added hold then raised");
        return -1;
    }
    if ((python_regs == 1 && (child_case == "transaction-transition" || child_case == "native-preinit" || child_case == "native-preinit-closed")) ||
        (python_regs == 2 && (child_case == "transaction-hold" || child_case == "rollback-stale"))) {
        PyErr_SetString(PyExc_RuntimeError, "injected hook failure");
        return -1;
    }
    return vulkan_python_hook_ops().python_register(function);
}
int python_unregister(PyObject* function) {
    ++unregisters;
    if (child_case == "rollback-stale") {
        PyErr_SetString(PyExc_RuntimeError, "injected rollback failure");
        return -1;
    }
    return vulkan_python_hook_ops().python_unregister(function);
}
void install() {
    register_vulkan_python_lifecycle(*coordinator, {native_register, python_register, python_unregister});
}
void expect_failed_registration() {
    bool failed = false;
    try { install(); } catch (const std::exception&) { failed = true; PyErr_Clear(); }
    require(failed, "registration failure must fail initialization");
    bool blocked = false;
    try { require_vulkan_python_registration_active(); }
    catch (const std::exception&) { blocked = true; PyErr_Clear(); }
    require(blocked, "FAILED must block Python use");
    auto n = native_regs;
    try { install(); } catch (const std::exception&) { PyErr_Clear(); }
    require(native_regs == n, "FAILED reload consumed native slot");
}
PyObject* log_event(PyObject*, PyObject* value) {
    const auto* text = PyUnicode_AsUTF8(value);
    if (!text) return nullptr;
    events.emplace_back(text);
    Py_RETURN_NONE;
}
PyObject* install_builtin(PyObject*, PyObject*) {
    try { install(); Py_RETURN_NONE; }
    catch (const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
PyObject* work_builtin(PyObject*, PyObject*) {
    try {
        const auto before = captures;
        with_vulkan_python_binding([] {
            auto r = acquire_binding();
            require(r->runtime.finalizing_sync(), "before-import finalizer first-use did not inherit mode");
            enqueue_capture(*r);
        });
        require(captures == before + 1, "finalizer returned with capture");
        events.emplace_back("sync-return");
        Py_RETURN_NONE;
    } catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
PyObject* after_work_builtin(PyObject*, PyObject*) {
    try {
        // The after-import user handler owns its synthetic hold release BEFORE
        // blocking, rather than waiting for the later internal LIFO release.
        with_vulkan_python_native([] { release_python_completion_holds(); });
        with_vulkan_python_binding([] {
            auto owner = acquire_binding();
            require(!owner->runtime.finalizing_sync(), "after-import handler reordered past transition");
            enqueue_capture(*owner); // raw ref stays on this explicitly GIL-held path
            owner->context.synchronize_with_flush([&] {
                if (owner->context.recording()) owner->context.submit_locked();
            });
        });
        events.emplace_back("after-sync");
        Py_RETURN_NONE;
    } catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
void add_builtins() {
    static PyMethodDef defs[] = {{"append", log_event, METH_O, nullptr},
        {"install_test_hooks", install_builtin, METH_NOARGS, nullptr},
        {"test_work", work_builtin, METH_NOARGS, nullptr},
        {"test_after_work", after_work_builtin, METH_NOARGS, nullptr}};
    auto* builtin = PyEval_GetBuiltins();
    auto* log = PyModule_New("native_events");
    auto* append = PyCFunction_New(&defs[0], nullptr);
    require(log && append, "event builtins");
    PyObject_SetAttrString(log, "append", append); Py_DECREF(append);
    PyDict_SetItemString(builtin, "events", log); Py_DECREF(log);
    for (int i = 1; i < 4; ++i) {
        auto* f = PyCFunction_New(&defs[i], nullptr);
        require(f != nullptr, "test builtin");
        PyDict_SetItemString(builtin, defs[i].ml_name, f); Py_DECREF(f);
    }
}
void script(const char* source) { require(PyRun_SimpleString(source) == 0, "embedded script failed"); }
void finalize() {
    require(Py_AtExit(mark_dead) == 0, "death marker registration");
    require(Py_FinalizeEx() == 0, "Py_FinalizeEx failed");
}
void early_gil() {
    install();
    auto owner = acquire_native();
    auto* r = owner.get();
    owner.reset();
    // Reference-shaped dispatcher decref: the installed 2.4.0 primary source
    // acquires the main-interpreter GIL before decref. No libtorch is imported.
    auto* value = PyCapsule_New(&captures, "lifecycle-capture", capture_destructor);
    require(value != nullptr, "dispatcher capsule");
    auto capture = std::shared_ptr<void>(value, [](void* p) {
        if (!interpreter_alive) std::_Exit(93);
        const auto gil = PyGILState_Ensure();
        Py_DECREF(static_cast<PyObject*>(p));
        PyGILState_Release(gil);
    });
    std::atomic<bool> pumping{false}, admitted{false}, gate{false};
    r->context.begin("foreign-pump");
    fixture_workers([&](FixtureWorkers& w) { w.signal(gate); }, [&] {
        r->gate = {};
        r->context.synchronize_with_flush([&] { if (r->context.recording()) r->context.submit_locked(); });
    }, [&](FixtureWorkers& w) {
        r->context.defer_destruction([capture, &pumping, &gate, &w] {
            w.signal(pumping);
            w.wait([&] { return gate.load(); }); // paused until prepare closes gate
        });
        capture.reset();
        w.start(0, [&] {
            r->context.synchronize_with_flush([&] { if (r->context.recording()) r->context.submit_locked(); });
        });
        w.wait([&] { return pumping.load(); });
        w.start(1, [&] {
            VulkanOperationScope operation(r->runtime);
            w.signal(admitted);
            w.wait([&] { return gate.load(); });
            python_worker([&] { events.emplace_back("worker-gil"); });
            operation.finish();
        });
        w.wait([&] { return admitted.load(); });
        r->gate = [&] { w.signal(gate); };
        vulkan_python_early_transition();
    });
    require(r->runtime.finalizing_sync(), "early mode missing");
    require(captures == 1, "foreign epilogue did not finish during host-only prepare");
}
void host_wait_cases() {
    install();
    auto owner = acquire_native(); auto* r = owner.get(); owner.reset();
    vulkan_python_early_transition();
    for (const char* label : {"getter", "first-use"}) {
        (void)label;
        std::atomic<bool> held{false}, waiting{false};
        fixture_workers([&](FixtureWorkers& w) { w.signal(waiting); }, [] {}, [&](FixtureWorkers& w) {
            w.start(0, [&] {
                VulkanOperationScope operation(r->runtime);
                w.signal(held);
                w.wait([&] { return waiting.load(); });
                python_worker([&] { events.emplace_back("lane-worker-gil"); });
                operation.finish();
            });
            w.wait([&] { return held.load(); });
            w.signal(waiting);
            with_vulkan_python_binding([] { auto r = acquire_binding(); require(r->runtime.finalizing_sync(), "getter lost mode"); });
        });
    }
}
void host_protocol() {
    // Behavioral RED target: replacing vulkan_host_wait with wait() must fail
    // without hanging. release mutates the predicate in the first unlock window.
    install();
    const auto python = vulkan_python_host_wait_hooks();
    std::mutex mutex;
    bool pending = true;
    int released = 0, restored = 0, waits = 0;
    struct Probe {
        std::mutex* mutex;
        bool* pending;
        int* released;
        int* restored;
        VulkanHostWaitHooks python;
    } probe{&mutex, &pending, &released, &restored, python};
    VulkanHostWaitHooks hooks{&probe, [](void* p) -> void* {
        auto& v = *static_cast<Probe*>(p);
        require(v.mutex->try_lock(), "release hook entered under native lock");
        *v.pending = false;
        v.mutex->unlock(); ++*v.released;
        return v.python.release(v.python.cookie);
    }, [](void* p, void* token) {
        auto& v = *static_cast<Probe*>(p);
        require(v.mutex->try_lock(), "restore hook entered under native lock");
        v.mutex->unlock(); ++*v.restored;
        v.python.restore(v.python.cookie, token);
    }};
    {
        VulkanHostWaitHookScope outer(python);
        { VulkanHostWaitHookScope inner(hooks);
          std::unique_lock<std::mutex> lock(mutex);
          vulkan_host_wait(lock, [&] { return pending; }, [&] { ++waits; });
          require(lock.owns_lock() && !pending, "unlock state change not revalidated"); }
        require(vulkan_host_wait_hooks() == &python, "nested hook context not restored");
    }
    require(!vulkan_host_wait_hooks() && released == 1 && restored == 1 && waits == 0, "host hook protocol missing");
    pending = true;
    {
        VulkanHostWaitHookScope installed(python);
        std::unique_lock<std::mutex> lock(mutex);
        bool caught = false;
        try { vulkan_host_wait(lock, [&] { return pending; }, [] { throw std::runtime_error("wait sentinel"); }); }
        catch (const std::runtime_error&) { caught = true; }
        require(caught && lock.owns_lock() && PyGILState_Check(), "wait exception did not restore GIL/lock");
    }
    std::atomic<int> forced_cleanup{0};
    fixture_workers([](FixtureWorkers&) {}, [] {}, [&](FixtureWorkers& w) {
        w.start(0, [&] {
            struct Last { std::atomic<int>& counter; ~Last() { if (!vulkan_host_wait_hooks()) ++counter; } } last{forced_cleanup};
            VulkanHostWaitHooks native{nullptr, [](void*) -> void* { return nullptr; }, [](void*, void*) { std::_Exit(97); }};
            VulkanHostWaitHookScope installed(native);
            std::unique_lock<std::mutex> lock(mutex);
            vulkan_host_wait(lock, [] { return true; }, [] { pthread_exit(nullptr); });
        });
    });
    require(forced_cleanup == 1 && !vulkan_host_wait_hooks(), "forced unwind lost prior native TLS context");
}
void mixed_waits() {
    install();
    auto owner = acquire_native(); auto* r = owner.get(); owner.reset();
    for (const auto* shape : {"sync", "end", "cancel"}) {
        std::atomic<bool> pumping{false}, gate{false};
        auto* value = PyCapsule_New(&captures, "lifecycle-capture", capture_destructor);
        require(value != nullptr, "foreign capsule");
        auto capture = std::shared_ptr<void>(value, [](void* p) {
            if (!interpreter_alive) std::_Exit(93);
            auto gil = PyGILState_Ensure();
            Py_DECREF(static_cast<PyObject*>(p));
            PyGILState_Release(gil);
        });
        r->context.begin(shape);
        fixture_workers([&](FixtureWorkers& w) { w.signal(gate); }, [&] {
            r->context.synchronize_with_flush([&] { if (r->context.recording()) r->context.submit_locked(); });
        }, [&](FixtureWorkers& w) {
            r->context.defer_destruction([capture, &pumping, &gate, &w] {
                w.signal(pumping); w.wait([&] { return gate.load(); });
            });
            capture.reset();
            w.start(0, [&] {
                r->context.synchronize_with_flush([&] { if (r->context.recording()) r->context.submit_locked(); });
            });
            w.wait([&] { return pumping.load(); });
            w.signal(gate); // foreign capture will need our GIL to leave pump
            with_vulkan_python_binding([&] {
                auto admitted = acquire_binding();
                // Same execution queue/device-wait shapes as compute end/cancel;
                // no real Compute is constructed in this CPU fixture.
                if (std::string(shape) == "sync") admitted->context.wait();
                else {
                    { std::lock_guard<std::mutex> queue(admitted->queue);
                      if (std::string(shape) == "cancel") admitted->context.cancel_locked();
                      else admitted->context.wait_device_locked(); }
                    admitted->context.wait_foreign_epilogues_until(std::chrono::steady_clock::now() + std::chrono::seconds(20), vulkan_host_clock());
                }
            });
        });
    }
    // Native device completion does not acquire the GIL. Keep both original
    // locks and GIL while waiting; test-only worker touches native state only.
    std::atomic<bool> completed{false};
    fixture_workers([](FixtureWorkers&) {}, [&] { r->device.native_fence = {}; }, [&](FixtureWorkers& w) {
        w.start(0, [&] { completed = true; });
        r->device.native_fence = [&] {
            require(PyGILState_Check(), "device wait unexpectedly released binding GIL");
            auto observer = std::async(std::launch::async, [&] {
                const bool queue_available = r->queue.try_lock();
                if (queue_available) r->queue.unlock();
                return queue_available;
            });
            require(!observer.get(), "device wait dropped caller queue lock");
            while (!completed.load()) std::this_thread::yield();
        };
        with_vulkan_python_binding([&] { auto a = acquire_binding(); a->context.begin("native-device"); a->context.submit(); a->context.wait(); });
    });
    require(captures == 3, "mixed wait lost dispatcher capture");
}
void native_body_contract() {
    install();
    acquire_native().reset();
    vulkan_python_early_transition();
    auto capture = python_capture(); // raw Python reference created under GIL
    with_vulkan_python_binding([&] {
        with_vulkan_python_native([&] {
            require(!PyGILState_Check(), "native body retained GIL");
            with_vulkan_python_native_admission([&] {
                require(!PyGILState_Check(), "nested native helper changed GIL");
                auto r = acquire_binding();
                r->context.begin("native-body-tail");
                // Explicit invariant: this raw reference is queued ONLY for the
                // controlled GIL-held outer tail, never synchronously pumped.
                r->context.defer_destruction([capture] {});
                capture.reset();
            });
        });
        require(PyGILState_Check() && captures == 0, "body restored after tail or released raw capture early");
    });
    require(captures == 1 && last_completion_gil, "outer tail did not hold GIL");
    bool original = false;
    try {
        with_vulkan_python_binding([&] {
            with_vulkan_python_native([] {
                with_vulkan_python_native([] { throw std::runtime_error("native sentinel"); });
            });
        });
    } catch (const std::runtime_error& e) { original = std::string(e.what()) == "native sentinel"; }
    require(original && PyGILState_Check() && !vulkan_host_wait_hooks(), "native error/nesting did not restore GIL/TLS");
    // Classification, not unsafe destruction: raw Py_DECREF without acquiring
    // the GIL is forbidden in the native body; do not deliberately invoke UB.
    bool raw_release_allowed = true;
    with_vulkan_python_native([&] { raw_release_allowed = interpreter_alive && PyGILState_Check(); });
    require(!raw_release_allowed, "raw reference release was not classified as forbidden in native body");
}
void native_hold_contract(bool finalizing) {
    install();
    auto owner = acquire_native(); auto* r = owner.get(); owner.reset();
    if (finalizing) vulkan_python_early_transition();
    struct Hold {
        std::mutex mutex;
        std::condition_variable cv;
        bool held = true, parked = false;
        unsigned expiries = 0;
        void release() { std::lock_guard<std::mutex> lock(mutex); held = false; cv.notify_all(); }
    } hold;
    // Same independent host-hold + caller queue/execution lock shape as the
    // production counting ops; no watchdog or artificial expiry exists here.
    auto token = std::shared_ptr<void>(&hold, [](void*) {});
    track_python_completion_hold(token, [](void* p) noexcept { static_cast<Hold*>(p)->release(); });
    std::atomic<bool> parked{false}, entered{false};
    fixture_workers([&](FixtureWorkers&) { hold.release(); }, [&] { r->device.native_fence = {}; }, [&](FixtureWorkers& w) {
        r->device.native_fence = [&] {
            require(!PyGILState_Check(), "held driver wrapper entered with GIL");
            std::unique_lock<std::mutex> lock(hold.mutex);
            hold.parked = true; w.signal(parked);
            hold.cv.wait(lock, [&] { return !hold.held; });
        };
        w.start(0, [&] { python_worker([&] {
            with_vulkan_python_binding([&] { with_vulkan_python_native([&] {
                auto a = acquire_binding(); a->context.begin("held"); a->context.submit(); a->context.wait();
            }); });
            require(PyGILState_Check(), "first binding failed to restore GIL");
        }); });
        w.wait([&] { return parked.load(); });
        w.start(1, [&] { python_worker([&] {
            with_vulkan_python_binding([&] { with_vulkan_python_native([&] {
                w.signal(entered);
                auto a = acquire_binding(); (void)a->context.lifecycle_snapshot();
            }); });
        }); });
        w.wait([&] { return entered.load(); });
        const auto before = factories;
        with_vulkan_python_binding([] { with_vulkan_python_native([] { release_python_completion_holds(); }); });
        require(factories == before, "hold release initialized a root");
    });
    require(!hold.held && hold.expiries == 0, "cross-thread hold did not release without expiry");
}
void restore_barrier_contract() {
    install();
    acquire_native().reset();
    auto* r = root;
    std::atomic<bool> admitted{false}, return_body{false}, closing{false}, restored{false}, finish_started{false};
    fixture_workers([&](FixtureWorkers& w) { w.signal(return_body); }, [&] {
        r->gate = {}; r->before_complete = {};
    }, [&](FixtureWorkers& w) {
        w.start(0, [&] { python_worker([&] {
            with_vulkan_python_binding([&] {
                with_vulkan_python_native([&] {
                    auto owner = acquire_binding();
                    w.signal(admitted);
                    w.wait([&] { return return_body.load(); });
                }); // actual production RestoreThread, while prepare waits
                require(PyGILState_Check() && closing && !finish_started,
                        "worker did not restore GIL during host-only prepare");
                restored = true;
            }); // drop the actual deferred outer permit BEFORE prepare returns
        }); });
        w.wait([&] { return admitted.load(); });
        r->gate = [&] { closing = true; w.signal(return_body); };
        r->before_complete = [&] {
            require(restored && !r->runtime.host_activity().outer_permits,
                    "finish raced worker restoration/permit epilogue");
            finish_started = true;
        };
        vulkan_python_early_transition();
    });
    require(restored && finish_started && r->runtime.finalizing_sync(), "split barrier completion missing");
}
void native_forced_unwind_contract() {
    install(); acquire_native().reset();
    std::atomic<unsigned> accounted{0};
    fixture_workers([](FixtureWorkers&) {}, [] {}, [&](FixtureWorkers& w) {
        w.start(0, [&] {
            struct Last {
                std::atomic<unsigned>& count;
                ~Last() {
                    if (!vulkan_host_wait_hooks() && !root->runtime.operation_active_on_this_thread()) ++count;
                }
            } last{accounted};
            python_worker([] {
                with_vulkan_python_binding([] { with_vulkan_python_native([] {
                    auto owner = acquire_binding();
                    pthread_exit(nullptr);
                }); });
            });
        });
    });
    require(accounted == 1 && root->runtime.host_activity().outer_permits == 0 && completions == 0,
            "forced unwind restored GIL/pumped or leaked native accounting/TLS");
}
void fixture_error_contract() {
    install(); acquire_native().reset();
    for (bool worker_throws : {false, true}) {
        std::atomic<bool> ready{false}, gate{false}, joined_body{false};
        bool caught = false;
        try {
            fixture_workers([&](FixtureWorkers& w) { w.signal(gate); }, [] {}, [&](FixtureWorkers& w) {
                w.start(0, [&] { python_worker([&] {
                    with_vulkan_python_binding([&] { with_vulkan_python_native([&] {
                        auto owner = acquire_binding();
                        w.signal(ready); w.wait([&] { return gate.load(); });
                        if (worker_throws) throw std::runtime_error("worker cleanup sentinel");
                    }); });
                    require(PyGILState_Check(), "cleanup worker did not restore GIL");
                    joined_body = true;
                }); });
                w.wait([&] { return ready.load(); });
                if (!worker_throws) throw std::runtime_error("main cleanup sentinel");
                w.signal(gate);
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == (worker_throws ? "worker cleanup sentinel" : "main cleanup sentinel");
        }
        require(caught && gate && (worker_throws || joined_body) && PyGILState_Check() &&
                !root->runtime.host_activity().outer_permits && !vulkan_host_wait_hooks(),
                "fixture error escaped before gate/GIL/native-thread quiescence");
    }
}
void weak_hold_repeat_contract() {
    install();
    struct NativeHold { bool held = true; unsigned releases = 0; };
    auto hold = std::make_shared<NativeHold>();
    track_python_completion_hold(hold, [](void* p) noexcept {
        auto& h = *static_cast<NativeHold*>(p); h.held = false; ++h.releases;
    });
    with_vulkan_python_binding([] { with_vulkan_python_native([] { release_python_completion_holds(); }); });
    require(!hold->held && hold->releases == 1 && factories == 0, "weak hold release initialized runtime");
    // An enable may follow an independent False setter. The weak exit tracking
    // must survive that False, not disappear before a concurrent enable commits.
    hold->held = true;
    with_vulkan_python_binding([] { with_vulkan_python_native([] { release_python_completion_holds(); }); });
    require(!hold->held && hold->releases == 2 && factories == 0, "False consumed weak tracking before later hold enable");
    std::weak_ptr<void> weak = hold;
    hold.reset();
    require(weak.expired(), "weak hold tracker retained native root");
    release_python_completion_holds();
}
void late_hold_probe() { release_python_completion_holds(); } // native-only probe, no Python APIs
void weak_hold_seal_contract() {
    // Register FIRST so this native probe runs AFTER the real native close.
    require(Py_AtExit(late_hold_probe) == 0, "late native probe registration");
    install();
    auto token = std::make_shared<int>(0);
    track_python_completion_hold(token, [](void* p) noexcept {
        if (!interpreter_alive) std::_Exit(95); // classify forbidden late callback without Python query
        ++*static_cast<int*>(p);
    });
    finalize();
    require(*token == 1 && factories == 0, "native close did not seal weak control promotion");
}
void callback_observation_contract() {
    install(); acquire_native().reset();
    bool observed = false, new_work_rejected = false;
    root->context.begin("callback-observation");
    root->context.defer_destruction([&] {
        // Runs in the real early-completion pump with coordinator Draining.
        with_vulkan_python_binding([&] { with_vulkan_python_native([&] {
            auto owner = acquire_binding();
            observed = owner.get() == root;
            require(!owner->context.query_with_flush({}), "callback query did not report active pump false");
            try { VulkanOperationScope forbidden(owner->runtime); }
            catch (const VulkanAdmissionClosed&) { new_work_rejected = true; }
        }); });
    });
    vulkan_python_early_transition();
    require(observed && new_work_rejected && factories == 1,
            "binding observation initialized, admitted work or rejected existing callback root");
}
void child(const std::string& name) {
    child_case = name;
    fake_deadline = name.find("deadline") != std::string::npos || name.find("native-host") != std::string::npos;
    coordinator = new VulkanLifecycleCoordinator(clock_ops());
    // The uv/standalone libpython is relocatable; its compiled /install prefix
    // is not this machine's stdlib. Supply the exact configured interpreter,
    // rather than relying on the harness executable's unrelated argv[0].
    static const wchar_t program[] = L"" PYTORCH_VULKAN_TEST_PYTHON;
    Py_SetProgramName(program);
    Py_Initialize();
    require(Py_IsInitialized(), "Py_Initialize failed");
    add_builtins();
    if (name.rfind("transaction-", 0) == 0 || name == "rollback-stale" || name == "rollback-stale-hold") {
        expect_failed_registration();
        require(factories == 0, "failed import created runtime");
        require(unregisters == ((name == "transaction-hold" || name == "rollback-stale" || name == "rollback-stale-hold") ? 1 : 0), "rollback count");
        vulkan_python_early_transition(); // stale guard precedes lookup/runtime
    } else if (name == "capacity32") {
        unsigned count = 0;
        while (Py_AtExit(noop_native) == 0) ++count;
        require(count == 32, "actual CPython low-level table capacity changed");
        expect_failed_registration();
        require(factories == 0, "full table created runtime");
        require(Py_FinalizeEx() == 0, "capacity finalization");
        delete coordinator; return;
    } else if (name == "non-main") {
        auto* main_state = PyThreadState_Get();
        auto* sub = Py_NewInterpreter();
        require(sub != nullptr, "subinterpreter creation");
        bool rejected = false;
        try { install(); } catch (const std::exception&) { rejected = true; PyErr_Clear(); }
        // Always clean up the actual subinterpreter before testing assertions.
        Py_EndInterpreter(sub); PyThreadState_Swap(main_state);
        require(rejected && native_regs == 0 && python_regs == 0 && factories == 0, "non-main setup was not rejected first");
    } else if (name == "native-preinit" || name == "native-preinit-closed") {
        acquire_native().reset();
        expect_failed_registration();
        if (name == "native-preinit-closed") coordinator->close(VulkanCloseBoundary::NativeEarly);
    } else if (name == "reload") {
        install(); install(); install();
        require(native_regs == 1 && python_regs == 2 && factories == 0, "ACTIVE reload duplicated hooks");
    } else if (name == "lifo" || name == "unused-first-use") {
        script("import atexit\natexit.register(lambda: (events.append('before-import'), test_work()))\ninstall_test_hooks()\n");
        script(name == "lifo"
            ? "atexit.register(lambda: (events.append('after-import'), test_after_work()))\n"
            : "atexit.register(lambda: events.append('after-import'))\n");
        struct Hold { bool held = true; unsigned releases = 0; bool self; };
        auto hold = std::make_shared<Hold>(Hold{true, 0, name == "lifo"});
        track_python_completion_hold(hold, [](void* p) noexcept {
            auto& h = *static_cast<Hold*>(p);
            h.held = false;
            events.emplace_back(h.self && h.releases++ == 0 ? "self-hold-release" : "hold-release");
        });
        if (name == "lifo") {
            // No wait/expiry is performed in this fake: an unreleased hold is a
            // plain invariant failure, not an intentional watchdog replay.
            acquire_native()->device.native_fence = [hold] { require(!hold->held, "handler blocked before self hold release"); };
        }
        // weak tracking only: the synthetic owner stays alive through finalization
        finalize();
        auto pos = [](const char* s) { return std::find(events.begin(), events.end(), s) - events.begin(); };
        require(pos("after-import") < pos("hold-release") && pos("hold-release") < pos("before-import") &&
            pos("before-import") < pos("sync-return"), "Python atexit LIFO ordering");
        if (name == "lifo") require(pos("after-import") < pos("self-hold-release") &&
            pos("self-hold-release") < pos("after-sync") && pos("after-sync") < pos("hold-release") &&
            pos("hold-release") < pos("transition-complete") && pos("transition-complete") < pos("before-import"),
            "self-release/transition LIFO ordering");
        require(factories == 1 && captures == (name == "lifo" ? 2 : 1) && destroyed == 1,
                "unused transition/first-use completion");
        delete coordinator; return;
    } else if (name == "early-gil") {
        early_gil();
    } else if (name == "capture-lifetime") {
        install();
        auto r = acquire_native(); enqueue_capture(*r); r.reset();
        vulkan_python_early_transition();
        require(captures == 1, "transition did not release capture");
        with_vulkan_python_binding([] { auto r = acquire_binding(); enqueue_capture(*r); });
        require(captures == 2, "binding epilogue did not release capture");
    } else if (name == "swallowed-atexit-fault") {
        install(); auto r = acquire_native(); r->fail_completion = true; r.reset();
    } else if (name == "native-empty-check") {
        install(); acquire_native().reset(); vulkan_python_early_transition();
        // Inject an ACTUAL execution-owned Python capture after the early seam.
        // Native close must fault70 without destroying it; its deleter would
        // classify a forbidden late release as91/93 without a Python query.
        enqueue_capture(*root);
    } else if (name.rfind("native-host-", 0) == 0 || name.rfind("daemon-deadline", 0) == 0) {
        install(); acquire_native().reset();
        if (name == "daemon-deadline-early") {
            fixture_workers([](FixtureWorkers&) {}, [] {}, [](FixtureWorkers& w) {
                w.start(0, [] { root->runtime.enter_operation(); });
            }); // intentional stranded count, not a daemon abort
            vulkan_python_early_transition();
            throw std::runtime_error("stranded early permit returned");
        }
        vulkan_python_early_transition();
        if (name == "native-host-outer" || name == "daemon-deadline-outer") {
            fixture_workers([](FixtureWorkers&) {}, [] {}, [](FixtureWorkers& w) {
                w.start(0, [] { root->runtime.enter_operation(); });
            }); // foreign stranded permit: exercise host precheck, not TLS self-close
        } else {
            auto kind = name == "native-host-general" ? VulkanCleanupKind::General : VulkanCleanupKind::AuditedNativeOnly;
            root->runtime.enter_cleanup(kind);
            if (name == "native-host-audited") on_wait = [kind] { root->runtime.leave_cleanup(kind); };
        }
    } else if (name == "host-hooks") {
        host_wait_cases();
    } else if (name == "host-protocol") {
        host_protocol();
    } else if (name == "mixed-waits") {
        mixed_waits();
    } else if (name == "native-body") {
        native_body_contract();
    } else if (name == "native-hold") {
        native_hold_contract(false);
    } else if (name == "native-hold-finalizing") {
        native_hold_contract(true);
    } else if (name == "restore-barrier") {
        restore_barrier_contract();
    } else if (name == "native-forced-unwind") {
        native_forced_unwind_contract();
    } else if (name == "fixture-errors") {
        fixture_error_contract();
    } else if (name == "weak-hold-repeat") {
        weak_hold_repeat_contract();
    } else if (name == "weak-hold-seal") {
        weak_hold_seal_contract(); delete coordinator; return; // already finalized
    } else if (name == "callback-observation") {
        callback_observation_contract();
    } else throw std::runtime_error("unknown child selector");
    finalize();
    require(factories == 0 || destroyed == 1, "root not released at native boundary");
    delete coordinator;
}
void run_child(const std::string& name, int wanted) {
    auto pid = fork(); require(pid >= 0, "fork");
    if (!pid) {
        execl(executable.c_str(), executable.c_str(), "--child", name.c_str(), nullptr);
        std::_Exit(98);
    }
    int status;
    require(waitpid(pid, &status, 0) == pid, "waitpid");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != wanted) {
        std::cerr << name << " status=" << status << " expected=" << wanted << '\n';
        throw std::runtime_error("child status mismatch");
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        executable = argv[0];
        require(argc == 3, "use --case selector or --child selector");
        std::string name = argv[2];
        if (std::string(argv[1]) == "--child") { child(name); return 0; }
        require(std::string(argv[1]) == "--case", "selector flag");
        if (name == "native-preinit") { run_child(name, 70); run_child("native-preinit-closed", 0); }
        else if (name == "rollback-stale") { run_child(name, 0); run_child("rollback-stale-hold", 0); }
        else if (name == "native-host-wait") {
            run_child("native-host-outer", 70); run_child("native-host-general", 70); run_child("native-host-audited", 0);
        } else if (name == "daemon-deadline") {
            run_child("daemon-deadline-early", 70);
            run_child("daemon-deadline-audited", 70); run_child("daemon-deadline-outer", 70);
        } else if (name == "swallowed-atexit-fault" || name == "native-empty-check") run_child(name, 70);
        else {
            run_child(name, 0);
            if (name == "capture-lifetime") {
                run_child("host-protocol", 0); run_child("host-hooks", 0); run_child("mixed-waits", 0);
                run_child("native-body", 0); run_child("native-hold", 0); run_child("native-hold-finalizing", 0);
                run_child("restore-barrier", 0); run_child("native-forced-unwind", 0); run_child("fixture-errors", 0);
                run_child("weak-hold-repeat", 0);
                run_child("weak-hold-seal", 0);
                run_child("callback-observation", 0);
            }
        }
        std::cout << "PASS " << name << '\n';
        return 0;
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        std::_Exit(2); // never abort/unsafe fixture unwinding through CPython
    }
}
