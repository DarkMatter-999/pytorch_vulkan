#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <cxxabi.h>

namespace pytorch_vulkan {
enum class VulkanLifecycleState { Uninitialized, Running, Closing, Draining,
                                  Quiescent, Closed, ShutdownFault };
enum class VulkanCloseBoundary { NativeEarly, PythonNative };
using VulkanHostTime = std::chrono::steady_clock::time_point;
struct VulkanHostClock {
    void* cookie;
    VulkanHostTime (*now)(void*) noexcept;
    void (*wait_until)(void*, std::condition_variable&,
                       std::unique_lock<std::mutex>&, VulkanHostTime);
    // Owner drops have no CV notification. Host-only bounded backoff receives
    // the ORIGINAL close deadline on every retry (never a renewed budget).
    // Optional for older injected clocks: wait_until then supplies 1ms slices.
    void (*wait_owner_retry_until)(void*, VulkanHostTime) = nullptr;
};
VulkanHostClock vulkan_host_clock() noexcept;
// Optional live-Python binding adapter. Native/dispatcher threads have no hooks.
// Neither hook is noexcept: CPython GIL restoration can force-unwind a worker.
struct VulkanHostWaitHooks {
    void* cookie;
    void* (*release)(void*);
    void (*restore)(void*, void*);
};
class VulkanHostWaitHookScope {
 public:
    explicit VulkanHostWaitHookScope(const VulkanHostWaitHooks&) noexcept;
    ~VulkanHostWaitHookScope() noexcept;
    VulkanHostWaitHookScope(const VulkanHostWaitHookScope&) = delete;
    VulkanHostWaitHookScope& operator=(const VulkanHostWaitHookScope&) = delete;
 private:
    const VulkanHostWaitHooks* previous_;
};
const VulkanHostWaitHooks* vulkan_host_wait_hooks() noexcept;
// ONLY for waits whose complete lock set is this one unique_lock. Release it
// before either adapter call; reacquire and recheck after BOTH unlock windows.
// No ring/queue transaction or native handle reference may span this helper.
template<class Pending, class Wait>
void vulkan_host_wait(std::unique_lock<std::mutex>& lock, Pending&& pending, Wait&& wait) {
    const auto* hooks = vulkan_host_wait_hooks();
    if (!hooks) { if (pending()) wait(); return; }
    lock.unlock();
    void* saved = hooks->release(hooks->cookie);
    try {
        lock.lock();
        if (pending()) wait();
        lock.unlock();
    } catch (const __cxxabiv1::__forced_unwind&) {
        // No GIL reacquisition or callbacks on a forced-unwind cleanup path.
        if (lock.owns_lock()) lock.unlock();
        throw;
    } catch (...) {
        if (lock.owns_lock()) lock.unlock();
        hooks->restore(hooks->cookie, saved);
        lock.lock();
        throw;
    }
    hooks->restore(hooks->cookie, saved);
    lock.lock(); // caller MUST revalidate its loop predicate after restoration
}
struct VulkanQuiescence {
    bool recording = false, pump_active = false, invalidated = false;
    bool sticky_error = false, reservation_closing = false;
    bool reservation_poisoned = false, private_scope_active = false;
    bool service_invalidated = false;
    std::size_t standalone_resources = 0;
    std::size_t submitted_slots = 0, ring_callbacks = 0, deferred_callbacks = 0;
    std::size_t completion_callbacks = 0, ready_callbacks = 0;
    std::size_t transfer_records = 0, reservations = 0;
    uint64_t reservation_generation = 0;
    std::thread::id pump_owner{};
    bool healthy_empty() const noexcept;
};
enum class VulkanCleanupKind { General, AuditedNativeOnly };
enum class VulkanIdleWait { AllEpilogues, AuditedNativeCleanupOnly };
// Observation is not admission. This allocation-free tag also lets terminal
// self-rejection run before touching any coordinator/queue/execution mutex.
class VulkanCallbackObservationScope {
 public:
    VulkanCallbackObservationScope() noexcept;
    ~VulkanCallbackObservationScope() noexcept;
    VulkanCallbackObservationScope(const VulkanCallbackObservationScope&) = delete;
    VulkanCallbackObservationScope& operator=(const VulkanCallbackObservationScope&) = delete;
};
bool vulkan_callback_observation_active() noexcept;
class VulkanAdmissionClosed : public std::runtime_error {
 public: using std::runtime_error::runtime_error;
};
class VulkanHostCloseTimeout : public std::runtime_error {
 public: using std::runtime_error::runtime_error;
};
struct VulkanHostActivity {
    std::size_t outer_permits = 0;
    std::size_t cleanup_epilogues = 0;
    std::size_t audited_native_cleanup_epilogues = 0;
};
class VulkanDeferredOperationFrame;
class VulkanRuntimeLifecycle {
 public:
    explicit VulkanRuntimeLifecycle(VulkanHostClock);
    void install_completion(void* cookie, void (*complete)(void*),
                            void (*fault)(void*, const char*) noexcept);
    // Installed before publication; callback preflight precedes even nested TLS.
    void install_callback_check(void*, bool (*)(void*) noexcept);
    void set_finalizing_sync() noexcept;
    bool finalizing_sync() const noexcept;
    void close_admission(VulkanHostTime deadline = VulkanHostTime::max());
    void close_terminal_admission(VulkanHostTime deadline = VulkanHostTime::max()); // coordinator-only
    VulkanHostTime completion_deadline(VulkanHostTime fallback) const;
    void reopen_admission();
    bool enter_operation();
    void leave_operation(bool outer) noexcept;
    void complete_outer_if_required();
    bool operation_active_on_this_thread() const noexcept;
    bool can_defer_operation_on_this_thread() const noexcept;
    bool cleanup_active_on_this_thread() const noexcept;
    static bool host_context_active_on_this_thread() noexcept;
    void enter_cleanup(VulkanCleanupKind) noexcept;
    void leave_cleanup(VulkanCleanupKind) noexcept;
    VulkanHostActivity host_activity() const;
    std::size_t outer_permits() const;
    std::size_t cleanup_epilogues() const;
    void wait_idle_until(VulkanHostTime absolute_deadline, VulkanIdleWait);
 private:
    friend class VulkanLifecycleCoordinator;
    bool defer_to_outer_frame() noexcept;
    bool inherited() const noexcept;
    const int process_;
    VulkanHostClock clock_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    VulkanHostActivity activity_;
    bool admission_closed_ = false;
    bool terminal_closed_ = false;
    // Set only by the coordinator before root publication. Such a root remains
    // strongly owned until this runtime's admitted permits/epilogues are zero.
    bool coordinator_owned_ = false;
    VulkanHostTime closing_deadline_ = VulkanHostTime::max();
    std::atomic<bool> finalizing_sync_{false};
    void* completion_cookie_ = nullptr;
    void (*complete_)(void*) = nullptr;
    void (*fault_)(void*, const char*) noexcept = nullptr;
    void* callback_cookie_ = nullptr;
    bool (*in_callback_)(void*) noexcept = nullptr;
};
struct VulkanLifecycleRoot {
    std::shared_ptr<void> owner;
    VulkanRuntimeLifecycle* runtime;
    void (*complete)(void*, VulkanHostTime);
    VulkanQuiescence (*snapshot)(void*);
    void (*close_reservations)(void*, bool);
    void (*wait_foreign_epilogues)(void*, VulkanHostTime);
    bool (*in_callback)(void*) noexcept;
    bool (*private_scope_active)(void*) noexcept;
    void (*destroy_native_services)(void*);
};
class VulkanLifecycleCoordinator {
 public:
    explicit VulkanLifecycleCoordinator(VulkanHostClock);
    ~VulkanLifecycleCoordinator() noexcept;
    std::shared_ptr<void> acquire(const std::function<VulkanLifecycleRoot()>& factory);
    std::shared_ptr<void> observe() const;
    void retire_cached();
    VulkanHostTime prepare_python_finalizing_sync();
    void finish_python_finalizing_sync(VulkanHostTime);
    [[noreturn]] void terminal_fault(const char*) noexcept;
    void close(VulkanCloseBoundary);
    void check_late() const noexcept;
    bool ever_initialized() const noexcept;
    void abandon_inherited_child() noexcept;
    VulkanLifecycleCoordinator(const VulkanLifecycleCoordinator&) = delete;
    VulkanLifecycleCoordinator& operator=(const VulkanLifecycleCoordinator&) = delete;
 private:
    struct Box;
    struct Entry;
    bool inherited() const noexcept;
    void reject_self_close();
    void wait_initialization(VulkanHostTime);
    void prepare_roots(VulkanHostTime, VulkanIdleWait);
    const int process_;
    VulkanHostClock clock_;
    // Separately allocated owning box is deliberately abandoned in a fork child.
    Box* box_;
};
// Native-main terminal guard: nonowning, creates no runtime, and performs the
// single NativeEarly terminal close after main-scope locals have unwound.
class NativeVulkanTerminalGuard {
 public:
    explicit NativeVulkanTerminalGuard(VulkanLifecycleCoordinator&) noexcept;
    ~NativeVulkanTerminalGuard() noexcept;
    NativeVulkanTerminalGuard(const NativeVulkanTerminalGuard&) = delete;
    NativeVulkanTerminalGuard& operator=(const NativeVulkanTerminalGuard&) = delete;
 private:
    VulkanLifecycleCoordinator& coordinator_;
};
class VulkanOperationScope {
 public:
    explicit VulkanOperationScope(VulkanRuntimeLifecycle&);
    ~VulkanOperationScope() noexcept;
    void finish();
    void finish_error(std::exception_ptr original_error);
    VulkanOperationScope(const VulkanOperationScope&) = delete;
    VulkanOperationScope& operator=(const VulkanOperationScope&) = delete;
 private:
    VulkanRuntimeLifecycle& runtime_;
    bool outer_;
    bool finished_ = false;
};
// Stack/TLS only: does not initialize a root or allocate admission storage.
// The first actual admission transfers its outer accounting depth to the
// oldest active frame. Multiple runtimes use their existing TLS permit entries,
// not a new owning list or scheduler. Only coordinator-backed lifetimes bind.
class VulkanDeferredOperationFrame {
 public:
    VulkanDeferredOperationFrame() noexcept;
    ~VulkanDeferredOperationFrame() noexcept;
    void finish();
    void finish_error(std::exception_ptr);
    VulkanDeferredOperationFrame(const VulkanDeferredOperationFrame&) = delete;
    VulkanDeferredOperationFrame& operator=(const VulkanDeferredOperationFrame&) = delete;
 private:
    friend class VulkanRuntimeLifecycle;
    void release_accounting() noexcept;
    const int process_;
    VulkanDeferredOperationFrame* previous_;
    bool active_ = true;
    bool has_bindings_ = false;
    bool finished_ = false;
    static thread_local VulkanDeferredOperationFrame* current_;
};
class VulkanCleanupScope {
 public:
    explicit VulkanCleanupScope(VulkanRuntimeLifecycle&,
                                VulkanCleanupKind = VulkanCleanupKind::General) noexcept;
    ~VulkanCleanupScope() noexcept;
    VulkanCleanupScope(const VulkanCleanupScope&) = delete;
    VulkanCleanupScope& operator=(const VulkanCleanupScope&) = delete;
  private:
    friend class VulkanRuntimeLifecycle;
    VulkanRuntimeLifecycle& runtime_;
    VulkanCleanupKind kind_;
    VulkanCleanupScope* previous_;
    bool counted_ = false;
    static thread_local VulkanCleanupScope* current_;
};
// Explicit operation epilogue shared by value/reference/void boundaries. The
// immediately invoked body destroys all its locals before finish; its return
// object stays alive across completion. Never use around a returned lock/lease
// whose protected transaction continues in the caller.
template<class Body>
decltype(auto) with_vulkan_operation(VulkanRuntimeLifecycle& runtime, Body&& body) {
    VulkanDeferredOperationFrame frame;
    VulkanOperationScope operation(runtime);
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Body>>) {
            std::forward<Body>(body)();
            operation.finish();
            frame.finish();
            return;
        } else {
            decltype(auto) output = std::forward<Body>(body)();
            operation.finish();
            frame.finish();
            return output;
        }
    } catch (const __cxxabiv1::__forced_unwind&) {
        throw;
    } catch (...) {
        auto original = std::current_exception();
        operation.finish();
        frame.finish_error(original);
        throw;
    }
}
// A null runtime may precede first-use initialization. Keep a lexical frame
// without initializing/allocating anything until an actual inner admission.
template<class Body>
decltype(auto) with_vulkan_operation(VulkanRuntimeLifecycle* runtime, Body&& body) {
    if (runtime) return with_vulkan_operation(*runtime, std::forward<Body>(body));
    VulkanDeferredOperationFrame operation;
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Body>>) {
            std::forward<Body>(body)();
            operation.finish();
            return;
        } else {
            decltype(auto) output = std::forward<Body>(body)();
            operation.finish();
            return output;
        }
    } catch (const __cxxabiv1::__forced_unwind&) {
        throw;
    } catch (...) {
        operation.finish_error(std::current_exception());
        throw;
    }
}
[[noreturn]] void vulkan_terminal_fault(const char* literal_reason) noexcept;
} // namespace pytorch_vulkan
