#pragma once

#include <Python.h>
#include "vulkan_lifecycle.h"

namespace pytorch_vulkan {
enum class VulkanPythonRegistration { Unregistered, Registering, Active, Failed };
struct VulkanPythonHookOps {
    int (*native_register)(void (*)());
    int (*python_register)(PyObject*);
    int (*python_unregister)(PyObject*);
};
VulkanPythonHookOps vulkan_python_hook_ops() noexcept;
void register_vulkan_python_lifecycle(VulkanLifecycleCoordinator&, const VulkanPythonHookOps&);
void require_vulkan_python_registration_active();
void vulkan_python_early_transition() noexcept;
void vulkan_python_native_exit() noexcept;
void track_python_completion_hold(std::weak_ptr<void>, void (*)(void*) noexcept);
// Native non-work control: weak roots only, no factory/queue/lane admission.
void release_python_completion_holds();
VulkanHostWaitHooks vulkan_python_host_wait_hooks() noexcept;

// Audited native body only: no Python APIs/pybind objects or raw Python-reference
// destruction. Tensor-owned interpreter references use PyTorch's self-acquiring
// PyInterpreter route while the interpreter is alive. Restore explicitly after
// all body locks/local native tails, BEFORE the outer deferred frame completes.
// Idempotent when nested in an already GIL-free native body.
template<class Body>
decltype(auto) with_vulkan_python_native(Body&& body) {
    PyThreadState* saved = PyGILState_Check() ? PyEval_SaveThread() : nullptr;
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Body>>) {
            std::forward<Body>(body)();
            if (saved) PyEval_RestoreThread(saved);
            return;
        } else {
            decltype(auto) output = std::forward<Body>(body)();
            if (saved) PyEval_RestoreThread(saved);
            return output;
        }
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) { if (saved) PyEval_RestoreThread(saved); throw; }
}
template<class Body>
decltype(auto) with_vulkan_python_native_admission(Body&& body) {
    return with_vulkan_python_native(std::forward<Body>(body));
}
// Observation inside an existing completion callback is NOT work admission.
// In particular a query must retain its active-pump false result during Closing/
// Draining, without constructing a root or waiting for its own finalizing lane.
template<class Acquire, class Runtime, class InCallback>
auto acquire_vulkan_python_binding_root(VulkanLifecycleCoordinator& coordinator,
                                       Acquire acquire, Runtime runtime, InCallback in_callback) {
    using Owner = std::invoke_result_t<Acquire>;
    using Root = typename Owner::element_type;
    return with_vulkan_python_native_admission([&] {
        auto existing = std::static_pointer_cast<Root>(coordinator.observe());
        if (existing && in_callback(*existing)) return existing;
        existing.reset(); // no observer pin outlives the subsequent admission
        auto owner = acquire();
        VulkanOperationScope operation(runtime(*owner));
        operation.finish(); // the binding's oldest lexical frame keeps admission
        return owner;
    });
}
template<class Body>
decltype(auto) with_vulkan_python_binding(Body&& body) {
    require_vulkan_python_registration_active();
    const auto hooks = vulkan_python_host_wait_hooks();
    VulkanHostWaitHookScope installed(hooks);
    return with_vulkan_operation(static_cast<VulkanRuntimeLifecycle*>(nullptr), std::forward<Body>(body));
}
} // namespace pytorch_vulkan
