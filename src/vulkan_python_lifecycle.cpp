#include "vulkan_python_lifecycle.h"

#include <atomic>
#include <vector>

namespace pytorch_vulkan {
namespace {
// Native-only storage. Python atexit owns the C-function wrappers; no PyObject
// reference, pybind object or GIL-dependent destructor survives in this holder.
std::atomic<VulkanPythonRegistration> registration{VulkanPythonRegistration::Unregistered};
VulkanLifecycleCoordinator* registered_coordinator = nullptr;
bool transition_entered = false;
bool transition_completed = false;
struct Hold { std::weak_ptr<void> root; void (*release)(void*) noexcept; };
std::mutex holds_mutex;
std::vector<Hold> holds;
bool holds_closed = false; // guarded by holds_mutex; seals weak promotion at native close
bool python_hook_registration_should_run() noexcept;

int python_atexit_call(const char* name, PyObject* function) {
    PyObject* module = PyImport_ImportModule("atexit");
    if (!module) return -1;
    PyObject* result = PyObject_CallMethod(module, name, "O", function);
    Py_DECREF(module);
    if (!result) return -1;
    Py_DECREF(result);
    return 0;
}
PyObject* transition_wrapper(PyObject*, PyObject*) {
    vulkan_python_early_transition();
    Py_RETURN_NONE;
}
PyObject* hold_wrapper(PyObject*, PyObject*) {
    if (!python_hook_registration_should_run()) Py_RETURN_NONE;
    // No strong late holder: lock a weak root only while interpreter/GIL live.
    // Release outside the tracking mutex; records remain weak across re-enables.
    try { release_python_completion_holds(); }
    catch (...) { vulkan_terminal_fault("Python completion hold release failed"); }
    Py_RETURN_NONE;
}
PyMethodDef transition_def{"_vulkan_early_transition", transition_wrapper, METH_NOARGS, nullptr};
PyMethodDef hold_def{"_vulkan_release_completion_holds", hold_wrapper, METH_NOARGS, nullptr};
[[noreturn]] void registration_error(const char* message) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, message);
    throw std::runtime_error(message);
}
bool python_hook_registration_should_run() noexcept {
    const bool active = registration.load(std::memory_order_acquire) == VulkanPythonRegistration::Active;
    if (!active) {
        if (!transition_entered && (!registered_coordinator || !registered_coordinator->ever_initialized())) return false;
        vulkan_terminal_fault("inactive Python lifecycle with initialized runtime");
    }
    if (!registered_coordinator) vulkan_terminal_fault("active Python lifecycle missing coordinator");
    return true;
}
bool python_early_hook_should_run() noexcept {
    return python_hook_registration_should_run() && !transition_completed;
}
VulkanLifecycleCoordinator& coordinator_for_python() { return *registered_coordinator; }
} // namespace
VulkanPythonHookOps vulkan_python_hook_ops() noexcept {
    return {&Py_AtExit,
        [](PyObject* f) { return python_atexit_call("register", f); },
        [](PyObject* f) { return python_atexit_call("unregister", f); }};
}
void register_vulkan_python_lifecycle(VulkanLifecycleCoordinator& coordinator, const VulkanPythonHookOps& ops) {
    // CPython main identity check precedes even the transaction pointer/setup.
    if (PyThreadState_GetInterpreter(PyThreadState_Get()) != PyInterpreterState_Main())
        registration_error("Vulkan supports only the CPython main interpreter");
    const auto state = registration.load(std::memory_order_acquire);
    if (state == VulkanPythonRegistration::Active) {
        if (registered_coordinator != &coordinator) registration_error("Vulkan lifecycle coordinator changed on reload");
        return;
    }
    if (state != VulkanPythonRegistration::Unregistered)
        registration_error("Vulkan Python lifecycle registration previously failed or is in progress");
    registered_coordinator = &coordinator;
    registration.store(VulkanPythonRegistration::Registering, std::memory_order_release);
    PyObject* transition = nullptr;
    PyObject* hold = nullptr;
    bool transition_added = false;
    try {
        if (!ops.native_register || !ops.python_register || !ops.python_unregister ||
            ops.native_register(&vulkan_python_native_exit) != 0)
            registration_error("Could not register Vulkan native exit callback");
        transition = PyCFunction_New(&transition_def, nullptr);
        if (!transition || ops.python_register(transition) != 0)
            registration_error("Could not register Vulkan Python transition");
        transition_added = true;
        hold = PyCFunction_New(&hold_def, nullptr);
        if (!hold || ops.python_register(hold) != 0)
            registration_error("Could not register Vulkan completion hold release");
        registration.store(VulkanPythonRegistration::Active, std::memory_order_release);
        Py_DECREF(hold); Py_DECREF(transition);
    } catch (...) {
        registration.store(VulkanPythonRegistration::Failed, std::memory_order_release);
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        if (transition_added) { (void)ops.python_unregister(transition); PyErr_Clear(); }
        Py_XDECREF(hold); Py_XDECREF(transition);
        PyErr_Restore(type, value, traceback);
        throw;
    }
}
void require_vulkan_python_registration_active() {
    if (registration.load(std::memory_order_acquire) != VulkanPythonRegistration::Active || !registered_coordinator)
        throw std::runtime_error("Vulkan Python lifecycle is not active");
}
void vulkan_python_early_transition() noexcept {
    if (!python_early_hook_should_run()) return;
    auto& coordinator = coordinator_for_python();
    transition_entered = true;
    VulkanHostTime deadline;
    auto* saved = PyEval_SaveThread();
    try { deadline = coordinator.prepare_python_finalizing_sync(); }
    catch (...) { coordinator.terminal_fault("Python early admission close failed"); }
    PyEval_RestoreThread(saved);
    try { coordinator.finish_python_finalizing_sync(deadline); }
    catch (...) { coordinator.terminal_fault("Python early capture completion failed"); }
    transition_completed = true;
}
void vulkan_python_native_exit() noexcept {
    // NO Python API/interpreter queries/GIL/general callbacks on this route.
    auto* coordinator = registered_coordinator;
    if (!coordinator) {
        if (!transition_entered && registration.load() != VulkanPythonRegistration::Active) return;
        vulkan_terminal_fault("Python native exit missing coordinator");
    }
    if (registration.load(std::memory_order_acquire) != VulkanPythonRegistration::Active) {
        if (!transition_entered && !coordinator->ever_initialized()) return;
        coordinator->check_late(); // permits independently safely closed native use
        return;
    }
    if (coordinator->ever_initialized() && !transition_completed)
        coordinator->terminal_fault("missed Python early completion boundary");
    // Linearize against the ONLY permitted weak-root control promotion. A
    // pre-seal native setter already owns its root and reconciles on the same
    // close deadline; a later setter cannot recreate an owner after owner proof.
    { std::lock_guard<std::mutex> lock(holds_mutex); holds_closed = true; }
    try { coordinator->close(VulkanCloseBoundary::PythonNative); }
    catch (...) { coordinator->terminal_fault("Python native close failed"); }
}
void track_python_completion_hold(std::weak_ptr<void> root, void (*release)(void*) noexcept) {
    require_vulkan_python_registration_active();
    std::lock_guard<std::mutex> lock(holds_mutex);
    if (holds_closed) throw VulkanAdmissionClosed("Python completion hold tracking closed");
    // One record per live root; weak bookkeeping never extends its lifetime.
    for (auto& hold : holds) {
        if (!hold.root.owner_before(root) && !root.owner_before(hold.root)) {
            hold.release = release;
            return;
        }
    }
    holds.push_back({std::move(root), release});
}
void release_python_completion_holds() {
    // Also callable without the GIL by an active binding; do not enter the work
    // lane which may be occupied by the very waiter this control must release.
    require_vulkan_python_registration_active();
    // A False setter must not consume exit tracking: a concurrent/next True may
    // commit after that setter. Keep weak records, copying one native control
    // at a time without allocation or invoking its setter under holds_mutex.
    std::size_t index = 0;
    for (;;) {
        std::shared_ptr<void> owner;
        void (*release)(void*) noexcept = nullptr;
        {
            std::lock_guard<std::mutex> lock(holds_mutex);
            if (holds_closed || index == holds.size()) break;
            const auto& hold = holds[index++];
            owner = hold.root.lock();
            release = hold.release;
        }
        if (owner) release(owner.get());
    }
}
VulkanHostWaitHooks vulkan_python_host_wait_hooks() noexcept {
    return {nullptr, [](void*) -> void* {
        // Only installed by a binding while interpreter is alive. Admission may
        // already have released the GIL; never SaveThread twice in that case.
        // PyGILState_Check assumes a live interpreter: it can return true after
        // interpreter shutdown and is not a post-interpreter callback API.
        return PyGILState_Check() ? PyEval_SaveThread() : nullptr;
    }, [](void*, void* saved) {
        if (saved) PyEval_RestoreThread(static_cast<PyThreadState*>(saved));
    }};
}
} // namespace pytorch_vulkan
