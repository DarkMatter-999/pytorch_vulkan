#include "vulkan_lifecycle.h"

#include <algorithm>
#include <cxxabi.h>
#include <cstdlib>
#include <unistd.h>
#include <vector>

namespace pytorch_vulkan {
namespace {
struct ThreadPermit {
    VulkanRuntimeLifecycle* runtime;
    std::size_t depth;
    VulkanDeferredOperationFrame* frame = nullptr;
};
thread_local std::vector<ThreadPermit> permits;
thread_local unsigned callback_observation_depth = 0;
thread_local const VulkanHostWaitHooks* host_wait_hooks = nullptr;
auto find_permit(VulkanRuntimeLifecycle* runtime) noexcept {
    return std::find_if(permits.begin(), permits.end(),
                        [runtime](const ThreadPermit& p) { return p.runtime == runtime; });
}
constexpr auto host_close_budget = std::chrono::seconds(20);
constexpr auto owner_retry_interval = std::chrono::milliseconds(1);
} // namespace
VulkanHostWaitHookScope::VulkanHostWaitHookScope(const VulkanHostWaitHooks& hooks) noexcept
    : previous_(host_wait_hooks) { host_wait_hooks = &hooks; }
VulkanHostWaitHookScope::~VulkanHostWaitHookScope() noexcept { host_wait_hooks = previous_; }
const VulkanHostWaitHooks* vulkan_host_wait_hooks() noexcept { return host_wait_hooks; }
VulkanCallbackObservationScope::VulkanCallbackObservationScope() noexcept { ++callback_observation_depth; }
VulkanCallbackObservationScope::~VulkanCallbackObservationScope() noexcept { --callback_observation_depth; }
bool vulkan_callback_observation_active() noexcept { return callback_observation_depth != 0; }

VulkanHostClock vulkan_host_clock() noexcept {
    return {nullptr, [](void*) noexcept { return std::chrono::steady_clock::now(); },
        [](void*, std::condition_variable& cv, std::unique_lock<std::mutex>& lock,
           VulkanHostTime deadline) { cv.wait_until(lock, deadline); },
        [](void*, VulkanHostTime deadline) {
            std::this_thread::sleep_until(std::min(deadline,
                std::chrono::steady_clock::now() + owner_retry_interval));
        }};
}
bool VulkanQuiescence::healthy_empty() const noexcept {
    return !recording && !pump_active && !invalidated && !sticky_error &&
        submitted_slots == 0 && ring_callbacks == 0 && deferred_callbacks == 0 &&
        completion_callbacks == 0 && ready_callbacks == 0 && transfer_records == 0 &&
        reservations == 0 && pump_owner == std::thread::id{} &&
        !reservation_poisoned && !private_scope_active && !service_invalidated &&
        standalone_resources == 0;
}
[[noreturn]] void vulkan_terminal_fault(const char* reason) noexcept {
    constexpr char prefix[] = "Vulkan terminal fault: ";
    (void)::write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    std::size_t n = 0;
    while (n < 192 && reason[n] != '\0') ++n;
    (void)::write(STDERR_FILENO, reason, n);
    (void)::write(STDERR_FILENO, "\n", 1);
    std::_Exit(70);
}

VulkanRuntimeLifecycle::VulkanRuntimeLifecycle(VulkanHostClock clock)
    : process_(::getpid()), clock_(clock) {}
bool VulkanRuntimeLifecycle::inherited() const noexcept { return process_ != ::getpid(); }
void VulkanRuntimeLifecycle::install_completion(void* cookie, void (*complete)(void*),
                                               void (*fault)(void*, const char*) noexcept) {
    completion_cookie_ = cookie;
    complete_ = complete;
    fault_ = fault;
}
void VulkanRuntimeLifecycle::install_callback_check(void* cookie, bool (*check)(void*) noexcept) {
    callback_cookie_ = cookie;
    in_callback_ = check;
}
void VulkanRuntimeLifecycle::set_finalizing_sync() noexcept {
    if (!inherited()) finalizing_sync_.store(true, std::memory_order_release);
}
bool VulkanRuntimeLifecycle::finalizing_sync() const noexcept {
    return !inherited() && finalizing_sync_.load(std::memory_order_acquire);
}
void VulkanRuntimeLifecycle::close_admission(VulkanHostTime deadline) {
    if (inherited()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    admission_closed_ = true;
    closing_deadline_ = std::min(closing_deadline_, deadline);
    cv_.notify_all();
}
void VulkanRuntimeLifecycle::reopen_admission() {
    if (inherited()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (terminal_closed_) throw VulkanAdmissionClosed("terminal Vulkan runtime cannot reopen");
    admission_closed_ = false;
    closing_deadline_ = VulkanHostTime::max();
}
void VulkanRuntimeLifecycle::close_terminal_admission(VulkanHostTime deadline) {
    if (inherited()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    terminal_closed_ = true;
    admission_closed_ = true;
    closing_deadline_ = std::min(closing_deadline_, deadline);
    cv_.notify_all();
}
VulkanHostTime VulkanRuntimeLifecycle::completion_deadline(VulkanHostTime fallback) const {
    if (inherited()) return fallback;
    std::lock_guard<std::mutex> lock(mutex_);
    return std::min(fallback, closing_deadline_);
}
bool VulkanRuntimeLifecycle::enter_operation() {
    if (inherited()) throw VulkanAdmissionClosed("Vulkan operation inherited across fork");
    if (vulkan_callback_observation_active())
        throw VulkanAdmissionClosed("Vulkan callback cannot create work");
    if (in_callback_ && in_callback_(callback_cookie_))
        throw VulkanAdmissionClosed("Vulkan callback cannot create work");
    if (cleanup_active_on_this_thread())
        throw VulkanAdmissionClosed("Vulkan cleanup cannot create work");
    auto found = find_permit(this);
    if (found != permits.end()) { ++found->depth; return false; }
    // Allocate TLS before admitting/counting anything. A bad_alloc has no partial
    // permit and, crucially, precedes the caller's first device side effect.
    permits.push_back({this, 1});
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        // Only finalizing-sync serializes outer callers. An admitted outer owns
        // this lane through local destruction, attachment and full completion.
        // No lifecycle mutex survives admission; nested work takes TLS above.
        const auto deadline = clock_.now(clock_.cookie) + host_close_budget;
        while (!admission_closed_ && finalizing_sync() && activity_.outer_permits) {
            if (clock_.now(clock_.cookie) >= deadline)
                throw VulkanHostCloseTimeout("Vulkan finalizing admission deadline");
            vulkan_host_wait(lock,
                [&] { return !admission_closed_ && finalizing_sync() && activity_.outer_permits; },
                [&] { clock_.wait_until(clock_.cookie, cv_, lock, deadline); });
        }
        if (admission_closed_) throw VulkanAdmissionClosed("Vulkan admission closed");
        ++activity_.outer_permits;
    } catch (...) {
        permits.pop_back();
        throw;
    }
    return !defer_to_outer_frame();
}
void VulkanRuntimeLifecycle::leave_operation(bool outer) noexcept {
    if (inherited()) return;
    auto found = find_permit(this);
    if (found == permits.end() || found->depth == 0)
        vulkan_terminal_fault("unbalanced operation accounting");
    if (--found->depth != 0) {
        if (outer) vulkan_terminal_fault("outer operation left before nested scope");
        return;
    }
    permits.erase(found);
    if (!outer) vulkan_terminal_fault("nested operation outlived outer scope");
    std::lock_guard<std::mutex> lock(mutex_);
    --activity_.outer_permits;
    cv_.notify_all();
}
bool VulkanRuntimeLifecycle::operation_active_on_this_thread() const noexcept {
    return !inherited() && find_permit(const_cast<VulkanRuntimeLifecycle*>(this)) != permits.end();
}
bool VulkanRuntimeLifecycle::cleanup_active_on_this_thread() const noexcept {
    if (inherited()) return false;
    for (auto* scope = VulkanCleanupScope::current_; scope; scope = scope->previous_)
        if (&scope->runtime_ == this) return true;
    return false;
}
bool VulkanRuntimeLifecycle::host_context_active_on_this_thread() noexcept {
    return !permits.empty() || VulkanCleanupScope::current_ != nullptr;
}
void VulkanRuntimeLifecycle::complete_outer_if_required() {
    if (inherited() || !finalizing_sync()) return;
    if (!complete_ || !fault_) vulkan_terminal_fault("missing finalizing completion adapter");
    try { complete_(completion_cookie_); }
    catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        fault_(completion_cookie_, "finalizing operation completion failed");
        vulkan_terminal_fault("returning terminal fault adapter");
    }
}
void VulkanRuntimeLifecycle::enter_cleanup(VulkanCleanupKind kind) noexcept {
    if (inherited()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    ++activity_.cleanup_epilogues;
    if (kind == VulkanCleanupKind::AuditedNativeOnly)
        ++activity_.audited_native_cleanup_epilogues;
    cv_.notify_all();
}
void VulkanRuntimeLifecycle::leave_cleanup(VulkanCleanupKind kind) noexcept {
    if (inherited()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!activity_.cleanup_epilogues ||
        (kind == VulkanCleanupKind::AuditedNativeOnly && !activity_.audited_native_cleanup_epilogues))
        vulkan_terminal_fault("unbalanced cleanup accounting");
    --activity_.cleanup_epilogues;
    if (kind == VulkanCleanupKind::AuditedNativeOnly)
        --activity_.audited_native_cleanup_epilogues;
    cv_.notify_all();
}
VulkanHostActivity VulkanRuntimeLifecycle::host_activity() const {
    if (inherited()) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    return activity_;
}
std::size_t VulkanRuntimeLifecycle::outer_permits() const { return host_activity().outer_permits; }
std::size_t VulkanRuntimeLifecycle::cleanup_epilogues() const { return host_activity().cleanup_epilogues; }
void VulkanRuntimeLifecycle::wait_idle_until(VulkanHostTime deadline, VulkanIdleWait mode) {
    if (inherited()) return;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        if (mode == VulkanIdleWait::AuditedNativeCleanupOnly &&
            (activity_.outer_permits ||
             activity_.cleanup_epilogues != activity_.audited_native_cleanup_epilogues))
            throw VulkanAdmissionClosed("unsafe activity at native Python close");
        if (!activity_.outer_permits && !activity_.cleanup_epilogues) return;
        if (clock_.now(clock_.cookie) >= deadline)
            throw VulkanHostCloseTimeout("Vulkan host close deadline");
        vulkan_host_wait(lock,
            [&] { return activity_.outer_permits || activity_.cleanup_epilogues; },
            [&] { clock_.wait_until(clock_.cookie, cv_, lock, deadline); });
    }
}
NativeVulkanTerminalGuard::NativeVulkanTerminalGuard(VulkanLifecycleCoordinator& coordinator) noexcept
    : coordinator_(coordinator) {}
NativeVulkanTerminalGuard::~NativeVulkanTerminalGuard() noexcept {
    try { coordinator_.close(VulkanCloseBoundary::NativeEarly); }
    catch (...) { vulkan_terminal_fault("native terminal close failed"); }
}
VulkanOperationScope::VulkanOperationScope(VulkanRuntimeLifecycle& runtime)
    : runtime_(runtime), outer_(runtime.enter_operation()) {}
VulkanOperationScope::~VulkanOperationScope() noexcept {
    if (!finished_) runtime_.leave_operation(outer_);
}
void VulkanOperationScope::finish() {
    if (finished_) return;
    if (outer_) runtime_.complete_outer_if_required();
    runtime_.leave_operation(outer_);
    finished_ = true;
}
void VulkanOperationScope::finish_error(std::exception_ptr original_error) {
    finish();
    std::rethrow_exception(original_error);
}
thread_local VulkanDeferredOperationFrame* VulkanDeferredOperationFrame::current_ = nullptr;
VulkanDeferredOperationFrame::VulkanDeferredOperationFrame() noexcept
    : process_(::getpid()), previous_(current_) { current_ = this; }
bool VulkanRuntimeLifecycle::can_defer_operation_on_this_thread() const noexcept {
    if (inherited() || !coordinator_owned_) return false;
    for (auto* f = VulkanDeferredOperationFrame::current_; f; f = f->previous_)
        if (f->active_ && f->process_ == process_) return true;
    return false;
}
bool VulkanRuntimeLifecycle::defer_to_outer_frame() noexcept {
    if (!coordinator_owned_) return false;
    VulkanDeferredOperationFrame* owner = nullptr;
    for (auto* f = VulkanDeferredOperationFrame::current_; f; f = f->previous_)
        if (f->active_ && f->process_ == process_) owner = f;
    if (!owner) return false;
    auto permit = find_permit(this);
    if (permit == permits.end() || permit->depth != 1 || permit->frame)
        vulkan_terminal_fault("invalid deferred admission transfer");
    permit->frame = owner;
    ++permit->depth; // frame keeps outer depth; the actual inner scope is nested
    owner->has_bindings_ = true;
    return true;
}
void VulkanDeferredOperationFrame::release_accounting() noexcept {
    if (!has_bindings_) return;
    for (;;) {
        auto permit = std::find_if(permits.begin(), permits.end(),
            [this](const ThreadPermit& p) { return p.frame == this; });
        if (permit == permits.end()) break;
        if (permit->depth != 1) vulkan_terminal_fault("deferred frame outlived nested scope");
        // The coordinator cannot release its root while this count is nonzero.
        // leave erases the TLS pointer before its final count/mutex release; no
        // runtime/root member is accessed afterwards, and no owning pin is added.
        auto* runtime = permit->runtime;
        runtime->leave_operation(true);
    }
    has_bindings_ = false;
}
VulkanDeferredOperationFrame::~VulkanDeferredOperationFrame() noexcept {
    if (current_ != this) vulkan_terminal_fault("nonlexical deferred frame destruction");
    current_ = previous_;
    if (process_ != ::getpid()) return; // inherited TLS is nonowning; no locks/drops
    if (!finished_) release_accounting(); // forced unwind: no wait/callback/device
}
void VulkanDeferredOperationFrame::finish() {
    if (finished_) return;
    if (process_ != ::getpid()) { finished_ = true; return; }
    active_ = false; // completion cannot bind fresh work to a finished body
    if (has_bindings_) {
        // Keep ALL bound permits until every root's completion/capture tail has
        // finished. Re-scan by index, retaining no iterator across adapter calls.
        for (std::size_t i = 0; i < permits.size(); ++i) {
            if (permits[i].frame != this) continue;
            if (permits[i].depth != 1) vulkan_terminal_fault("deferred completion with nested scope");
            auto* runtime = permits[i].runtime;
            runtime->complete_outer_if_required();
        }
        release_accounting();
    }
    finished_ = true;
}
void VulkanDeferredOperationFrame::finish_error(std::exception_ptr original) {
    finish();
    std::rethrow_exception(original);
}
thread_local VulkanCleanupScope* VulkanCleanupScope::current_ = nullptr;
VulkanCleanupScope::VulkanCleanupScope(VulkanRuntimeLifecycle& runtime, VulkanCleanupKind kind) noexcept
    : runtime_(runtime), kind_(kind), previous_(current_) {
    // An outer operation or cleanup scope already covers the full lexical tail.
    // This intrusive TLS chain neither allocates nor reacquires runtime locks.
    bool covered = runtime_.operation_active_on_this_thread();
    for (auto* scope = current_; scope; scope = scope->previous_) {
        if (&scope->runtime_ != &runtime_) continue;
        // Never relabel arbitrary cleanup as audited native. This cross-kind
        // epilogue is separately counted (and makes native-only waits reject).
        // Audited service scopes are entered outside queue/execution locks;
        // ordinary nested General cleanup still takes the lock-free TLS path.
        covered = scope->kind_ != VulkanCleanupKind::AuditedNativeOnly || kind_ != VulkanCleanupKind::General;
        break;
    }
    if (!covered) { runtime_.enter_cleanup(kind_); counted_ = true; }
    current_ = this;
}
VulkanCleanupScope::~VulkanCleanupScope() noexcept {
    current_ = previous_;
    if (counted_) runtime_.leave_cleanup(kind_);
}

struct VulkanLifecycleCoordinator::Entry {
    VulkanLifecycleRoot root{};
    VulkanLifecycleCoordinator* coordinator = nullptr;
    static void complete(void* cookie) {
        auto& e = *static_cast<Entry*>(cookie);
        const auto deadline = e.root.runtime->completion_deadline(
            e.coordinator->clock_.now(e.coordinator->clock_.cookie) + host_close_budget);
        e.root.complete(e.root.owner.get(), deadline);
        if (!e.root.snapshot(e.root.owner.get()).healthy_empty())
            e.coordinator->terminal_fault("nonempty finalizing operation");
    }
    static void fault(void* cookie, const char* reason) noexcept {
        static_cast<Entry*>(cookie)->coordinator->terminal_fault(reason);
    }
};
struct VulkanLifecycleCoordinator::Box {
    mutable std::mutex mutex;
    std::condition_variable cv;
    VulkanLifecycleState state = VulkanLifecycleState::Uninitialized;
    bool initializing = false;
    bool administering = false;
    bool finalizing_sync = false;
    bool ever_initialized = false;
    bool terminal_observation_closed = false;
    std::exception_ptr initialization_error;
    std::unique_ptr<Entry> cached;
    std::vector<std::unique_ptr<Entry>> retiring;
    // Retiring entries precede the cache; no adapter/shared_ptr copies in visits.
    template<class F> void each(F&& function) {
        for (auto& entry : retiring) function(*entry);
        if (cached) function(*cached);
    }
};
VulkanLifecycleCoordinator::VulkanLifecycleCoordinator(VulkanHostClock clock)
    : process_(::getpid()), clock_(clock), box_(new Box) {}
bool VulkanLifecycleCoordinator::inherited() const noexcept { return process_ != ::getpid(); }
VulkanLifecycleCoordinator::~VulkanLifecycleCoordinator() noexcept {
    if (inherited()) { abandon_inherited_child(); return; }
    check_late(); // Must precede *any* owning box/member destruction.
    delete box_;
}
std::shared_ptr<void> VulkanLifecycleCoordinator::acquire(
    const std::function<VulkanLifecycleRoot()>& factory) {
    if (inherited() || !box_) throw VulkanAdmissionClosed("Vulkan cache inherited across fork");
    std::unique_lock<std::mutex> lock(box_->mutex);
    for (;;) {
        if (box_->state != VulkanLifecycleState::Uninitialized && box_->state != VulkanLifecycleState::Running)
            throw VulkanAdmissionClosed("Vulkan cache admission closed");
        if (box_->cached) return box_->cached->root.owner;
        if (box_->initialization_error) std::rethrow_exception(box_->initialization_error);
        if (!box_->initializing) break;
        vulkan_host_wait(lock, [&] { return box_->initializing; }, [&] { box_->cv.wait(lock); });
    }
    // Preflight attachment allocation before running the factory.
    auto entry = std::make_unique<Entry>();
    entry->coordinator = this;
    box_->initializing = true;
    lock.unlock();
    try {
        entry->root = factory();
        { std::lock_guard<std::mutex> initialized(box_->mutex); box_->ever_initialized = true; }
        auto& r = entry->root;
        if (!r.owner || !r.runtime || !r.complete || !r.snapshot || !r.close_reservations ||
            !r.wait_foreign_epilogues || !r.in_callback || !r.private_scope_active || !r.destroy_native_services)
            terminal_fault("incomplete lifecycle root adapter");
        r.runtime->install_completion(entry.get(), &Entry::complete, &Entry::fault);
        r.runtime->install_callback_check(r.owner.get(), r.in_callback);
        r.runtime->coordinator_owned_ = true;
        // Mode is read under the coordinator lock, applied unlocked, then checked
        // again at publication. A transition waits initializing before setting it.
        lock.lock();
        const bool finalizing = box_->finalizing_sync;
        lock.unlock();
        if (finalizing) r.runtime->set_finalizing_sync();
    } catch (const __cxxabiv1::__forced_unwind&) {
        // Cancellation cannot silently destroy a successfully constructed root.
        if (entry->root.owner) terminal_fault("initialization forced unwind with root");
        lock.lock();
        box_->initializing = false;
        box_->initialization_error = std::current_exception();
        box_->cv.notify_all();
        throw;
    } catch (...) {
        auto error = std::current_exception();
        lock.lock();
        box_->initialization_error = error;
        box_->initializing = false;
        box_->cv.notify_all();
        lock.unlock();
        std::rethrow_exception(error);
    }
    lock.lock();
    box_->cached = std::move(entry);
    box_->initializing = false;
    box_->cv.notify_all();
    if (box_->state != VulkanLifecycleState::Uninitialized && box_->state != VulkanLifecycleState::Running)
        throw VulkanAdmissionClosed("Vulkan initialization raced terminal close");
    box_->state = VulkanLifecycleState::Running;
    return box_->cached->root.owner;
}
std::shared_ptr<void> VulkanLifecycleCoordinator::observe() const {
    if (inherited() || !box_) return {};
    std::lock_guard<std::mutex> lock(box_->mutex);
    if (box_->terminal_observation_closed || box_->state == VulkanLifecycleState::Quiescent ||
        box_->state == VulkanLifecycleState::Closed)
        return {};
    return box_->cached ? box_->cached->root.owner : std::shared_ptr<void>{};
}
void VulkanLifecycleCoordinator::reject_self_close() {
    if (vulkan_callback_observation_active() || VulkanRuntimeLifecycle::host_context_active_on_this_thread())
        throw VulkanAdmissionClosed("cannot close Vulkan from operation, cleanup or callback");
    // Pin administrative access without holding any lock across adapter calls.
    // Entry addresses are stable; initialization may add a new cached entry but
    // cannot remove these entries while administering is set.
    std::vector<Entry*> entries;
    {
        std::lock_guard<std::mutex> lock(box_->mutex);
        if (box_->administering) throw VulkanAdmissionClosed("concurrent Vulkan lifecycle boundary");
        box_->each([&](Entry& e) { entries.push_back(&e); });
        box_->administering = true;
    }
    for (auto* e : entries) {
        if (e->root.runtime->operation_active_on_this_thread() ||
            e->root.runtime->cleanup_active_on_this_thread() || e->root.in_callback(e->root.owner.get())) {
            std::lock_guard<std::mutex> lock(box_->mutex);
            box_->administering = false;
            throw VulkanAdmissionClosed("cannot close Vulkan from operation, cleanup or callback");
        }
    }
}
void VulkanLifecycleCoordinator::wait_initialization(VulkanHostTime deadline) {
    std::unique_lock<std::mutex> lock(box_->mutex);
    while (box_->initializing) {
        if (clock_.now(clock_.cookie) >= deadline)
            throw VulkanHostCloseTimeout("Vulkan initialization close deadline");
        vulkan_host_wait(lock, [&] { return box_->initializing; },
            [&] { clock_.wait_until(clock_.cookie, box_->cv, lock, deadline); });
    }
}
void VulkanLifecycleCoordinator::prepare_roots(VulkanHostTime deadline, VulkanIdleWait mode) {
    box_->each([&](Entry& e) { e.root.runtime->close_admission(deadline); });
    box_->each([](Entry& e) { e.root.close_reservations(e.root.owner.get(), true); });
    box_->each([&](Entry& e) {
        if (e.root.private_scope_active(e.root.owner.get()))
            terminal_fault("private scope active at close");
        e.root.runtime->wait_idle_until(deadline, mode);
        e.root.wait_foreign_epilogues(e.root.owner.get(), deadline);
    });
}
void VulkanLifecycleCoordinator::retire_cached() {
    if (inherited() || !box_) return;
    reject_self_close();
    VulkanHostTime deadline;
    {
        std::lock_guard<std::mutex> lock(box_->mutex);
        if (box_->state == VulkanLifecycleState::Uninitialized && !box_->initializing) {
            box_->administering = false;
            return;
        }
        if (box_->state != VulkanLifecycleState::Running && box_->state != VulkanLifecycleState::Uninitialized)
        {
            box_->administering = false;
            throw VulkanAdmissionClosed("cannot retire closing cache");
        }
        try { box_->retiring.reserve(box_->retiring.size() + 1); }
        catch (...) { box_->administering = false; throw; }
        box_->state = VulkanLifecycleState::Closing;
        deadline = clock_.now(clock_.cookie) + host_close_budget;
    }
    try {
        wait_initialization(deadline);
        if (box_->cached) {
            auto& r = box_->cached->root;
            const auto local = r.owner;
            r.runtime->close_terminal_admission(deadline);
            r.close_reservations(local.get(), true);
            r.runtime->wait_idle_until(deadline, VulkanIdleWait::AllEpilogues);
            r.wait_foreign_epilogues(local.get(), deadline);
            r.complete(local.get(), deadline);
            r.runtime->wait_idle_until(deadline, VulkanIdleWait::AllEpilogues);
            if (!r.snapshot(local.get()).healthy_empty()) terminal_fault("retired root not empty");
        }
        std::lock_guard<std::mutex> lock(box_->mutex);
        if (box_->cached) box_->retiring.push_back(std::move(box_->cached));
        box_->state = VulkanLifecycleState::Uninitialized;
        box_->administering = false;
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { terminal_fault("cache retirement failed"); }
}
VulkanHostTime VulkanLifecycleCoordinator::prepare_python_finalizing_sync() {
    if (inherited() || !box_) return clock_.now(clock_.cookie);
    reject_self_close();
    VulkanHostTime deadline;
    {
        std::lock_guard<std::mutex> lock(box_->mutex);
        if (box_->state != VulkanLifecycleState::Uninitialized && box_->state != VulkanLifecycleState::Running)
        {
            box_->administering = false;
            throw VulkanAdmissionClosed("Python barrier on closing Vulkan runtime");
        }
        box_->state = VulkanLifecycleState::Closing;
        deadline = clock_.now(clock_.cookie) + host_close_budget;
    }
    try {
        wait_initialization(deadline);
        prepare_roots(deadline, VulkanIdleWait::AllEpilogues);
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { terminal_fault("Python host barrier failed"); }
    return deadline;
}
void VulkanLifecycleCoordinator::finish_python_finalizing_sync(VulkanHostTime deadline) {
    if (inherited() || !box_) return;
    try {
        {
            std::lock_guard<std::mutex> lock(box_->mutex);
            if (box_->state != VulkanLifecycleState::Closing || box_->initializing)
                throw VulkanAdmissionClosed("Python finish without prepared barrier");
            box_->state = VulkanLifecycleState::Draining;
        }
        box_->each([&](Entry& e) {
            auto& r = e.root;
            const auto local = r.owner;
            const auto activity = r.runtime->host_activity();
            const auto before = r.snapshot(local.get());
            if (activity.outer_permits || activity.cleanup_epilogues || before.pump_active ||
                before.pump_owner != std::thread::id{})
                terminal_fault("foreign activity after prepared Python barrier");
            r.complete(local.get(), deadline);
            const auto after = r.runtime->host_activity();
            if (after.outer_permits || after.cleanup_epilogues)
                terminal_fault("host activity during Python capture completion");
            if (!r.snapshot(local.get()).healthy_empty()) terminal_fault("Python barrier not empty");
            r.runtime->set_finalizing_sync();
        });
        // Retiring roots stay closed; only the public cached root may reopen.
        if (box_->cached) {
            auto& r = box_->cached->root;
            r.close_reservations(r.owner.get(), false);
            r.runtime->reopen_admission();
        }
        std::lock_guard<std::mutex> lock(box_->mutex);
        box_->finalizing_sync = true;
        box_->state = box_->cached ? VulkanLifecycleState::Running : VulkanLifecycleState::Uninitialized;
        box_->administering = false;
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { terminal_fault("Python completion barrier failed"); }
}
[[noreturn]] void VulkanLifecycleCoordinator::terminal_fault(const char* reason) noexcept {
    if (!inherited() && box_) {
        std::lock_guard<std::mutex> lock(box_->mutex);
        box_->state = VulkanLifecycleState::ShutdownFault;
    }
    vulkan_terminal_fault(reason);
}
void VulkanLifecycleCoordinator::close(VulkanCloseBoundary boundary) {
    if (inherited() || !box_) return;
    reject_self_close();
    VulkanHostTime deadline;
    {
        std::lock_guard<std::mutex> lock(box_->mutex);
        if (box_->state == VulkanLifecycleState::Closed) { box_->administering = false; return; }
        if (box_->state != VulkanLifecycleState::Uninitialized && box_->state != VulkanLifecycleState::Running)
        {
            box_->administering = false;
            throw VulkanAdmissionClosed("concurrent Vulkan terminal close");
        }
        if (boundary == VulkanCloseBoundary::PythonNative && box_->initializing) {
            box_->administering = false;
            throw VulkanAdmissionClosed("initialization active at Python native boundary");
        }
        box_->state = VulkanLifecycleState::Closing;
        deadline = clock_.now(clock_.cookie) + host_close_budget;
    }
    try {
        if (boundary == VulkanCloseBoundary::NativeEarly) wait_initialization(deadline);
        std::vector<std::shared_ptr<void>> locals;
        box_->each([&](Entry& e) { locals.push_back(e.root.owner); });
        if (boundary == VulkanCloseBoundary::PythonNative) {
            // Absolutely no pump/capture destruction at this post-interpreter boundary.
            box_->each([&](Entry& e) {
                e.root.runtime->close_terminal_admission(deadline);
                e.root.close_reservations(e.root.owner.get(), true);
                const auto activity = e.root.runtime->host_activity();
                if (activity.outer_permits || activity.cleanup_epilogues != activity.audited_native_cleanup_epilogues ||
                    !e.root.snapshot(e.root.owner.get()).healthy_empty())
                    terminal_fault("nonempty Python native boundary");
            });
        }
        const auto mode = boundary == VulkanCloseBoundary::PythonNative
            ? VulkanIdleWait::AuditedNativeCleanupOnly : VulkanIdleWait::AllEpilogues;
        box_->each([&](Entry& e) { e.root.runtime->close_terminal_admission(deadline); });
        if (boundary == VulkanCloseBoundary::NativeEarly) prepare_roots(deadline, mode);
        else box_->each([&](Entry& e) {
            // No foreign-pump wait/private-scope adapter/general completion at
            // this post-interpreter boundary. Snapshot was empty BEFORE waiting.
            e.root.runtime->wait_idle_until(deadline, mode);
            const auto activity = e.root.runtime->host_activity();
            if (activity.outer_permits || activity.cleanup_epilogues ||
                !e.root.snapshot(e.root.owner.get()).healthy_empty())
                terminal_fault("unsafe activity after Python native host wait");
        });
        {
            std::lock_guard<std::mutex> lock(box_->mutex);
            box_->state = VulkanLifecycleState::Draining;
        }
        std::size_t index = 0;
        box_->each([&](Entry& e) {
            auto& r = e.root;
            // Keep the cache plus this single local root through every epilogue.
            auto& local = locals[index++];
            if (boundary == VulkanCloseBoundary::NativeEarly) r.complete(local.get(), deadline);
            r.runtime->wait_idle_until(deadline, mode);
            if (!r.snapshot(local.get()).healthy_empty()) terminal_fault("terminal root not empty");
        });
        // Cross-root captures can retain a retiring platform from the new cache.
        // Complete *every* root before treating any residual owner as unknown.
        {
            std::lock_guard<std::mutex> lock(box_->mutex);
            // Linearize against observe's owner copy under this same lock.
            // Earlier observations are reconciled below; later ones get no
            // owner. The temporary Python barrier does NOT seal observation.
            box_->terminal_observation_closed = true;
        }
        const auto reconcile_owners = [&] {
            const auto exclusive = [&] {
                return std::all_of(locals.begin(), locals.end(),
                    [](const std::shared_ptr<void>& p) { return !p || p.use_count() == 2; });
            };
            for (;;) {
                if (exclusive()) return;
                if (clock_.now(clock_.cookie) >= deadline) {
                    // Recheck after sampling time: an owner may have dropped
                    // between the first predicate and the deadline observation.
                    if (exclusive()) return;
                    terminal_fault("unknown terminal root owner at close deadline");
                }
                // Accounting may have left while its necessary local keepalive
                // still exists. Never remove/reorder that keepalive: resample
                // without pumping or destroying an opaque owner to force proof.
                // No coordinator/runtime/queue lock is held across this wait.
                if (clock_.wait_owner_retry_until) {
                    clock_.wait_owner_retry_until(clock_.cookie, deadline);
                } else {
                    std::mutex retry_mutex;
                    std::condition_variable retry_cv;
                    std::unique_lock<std::mutex> retry_lock(retry_mutex);
                    clock_.wait_until(clock_.cookie, retry_cv, retry_lock,
                        std::min(deadline, clock_.now(clock_.cookie) + owner_retry_interval));
                }
            }
        };
        reconcile_owners();
        index = 0;
        box_->each([&](Entry& e) {
            auto& r = e.root;
            auto& local = locals[index++];
            r.destroy_native_services(local.get());
            r.runtime->wait_idle_until(deadline, mode);
            if (!r.snapshot(local.get()).healthy_empty())
                terminal_fault("terminal epilogue changed ownership");
            // A service epilogue can affect another root. Reconcile every live
            // root before any further services or root release, on SAME budget.
            reconcile_owners();
        });
        reconcile_owners();
        {
            std::lock_guard<std::mutex> lock(box_->mutex);
            box_->state = VulkanLifecycleState::Quiescent;
        }
        index = 0;
        box_->each([&](Entry& e) {
            auto& local = locals[index++];
            std::weak_ptr<void> weak = local;
            reconcile_owners();
            e.root.owner.reset();
            local.reset(); // Outside every lifecycle/queue lock.
            if (!weak.expired()) terminal_fault("terminal root did not expire");
        });
        std::lock_guard<std::mutex> lock(box_->mutex);
        box_->cached.reset();
        box_->retiring.clear();
        box_->state = VulkanLifecycleState::Closed;
        box_->administering = false;
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { terminal_fault("terminal close failed"); }
}
void VulkanLifecycleCoordinator::check_late() const noexcept {
    if (inherited() || !box_) return;
    bool safe;
    {
        std::lock_guard<std::mutex> lock(box_->mutex);
        safe = !box_->initializing && !box_->cached && box_->retiring.empty() &&
            (box_->state == VulkanLifecycleState::Uninitialized || box_->state == VulkanLifecycleState::Closed);
    }
    if (!safe) const_cast<VulkanLifecycleCoordinator*>(this)->terminal_fault("missed early terminal boundary");
}
bool VulkanLifecycleCoordinator::ever_initialized() const noexcept {
    if (inherited() || !box_) return false;
    std::lock_guard<std::mutex> lock(box_->mutex);
    return box_->ever_initialized;
}
void VulkanLifecycleCoordinator::abandon_inherited_child() noexcept {
    if (inherited()) box_ = nullptr; // No decrement, allocation, lock or destructor.
}
} // namespace pytorch_vulkan
