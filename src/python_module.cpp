#include "vulkan_python_lifecycle.h" // Python feature macros must precede system headers.
#include "vulkan/descriptor_arena.h"
#include "vulkan/pipeline_cache.h"
#include "vulkan/shader_registry.h"
#include "vulkan_allocator.h"
#include "vulkan_compute.h"
#include "vulkan_device_guard.h"
#include "vulkan_execution.h"
#include "vulkan_platform.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
template<class Signature> struct BindingSignature;
template<class C, class R, class... Args>
struct BindingSignature<R(C::*)(Args...) const> {
    template<bool Native, class Body> static auto wrap(Body body) {
        return std::function<R(Args...)>([body = std::move(body)](Args... args) -> R {
            return pytorch_vulkan::with_vulkan_python_binding([&]() -> R {
                if constexpr (Native) return pytorch_vulkan::with_vulkan_python_native([&]() -> R { return body(args...); });
                else return body(args...);
            });
        });
    }
};
template<bool Native, class R, class... Args>
auto guarded_binding(R (*body)(Args...)) {
    return std::function<R(Args...)>([body](Args... args) -> R {
        return pytorch_vulkan::with_vulkan_python_binding([&]() -> R {
            if constexpr (Native) return pytorch_vulkan::with_vulkan_python_native([&]() -> R { return body(args...); });
            else return body(args...);
        });
    });
}
template<bool Native, class R, class... Args>
auto guarded_binding(R (*body)(Args...) noexcept) {
    return guarded_binding<Native>(static_cast<R (*)(Args...)>(body));
}
template<bool Native, class Body> auto guarded_binding(Body body) {
    return BindingSignature<decltype(&Body::operator())>::template wrap<Native>(std::move(body));
}
struct GuardedPythonModule {
    py::module_& underlying;
    decltype(auto) doc() { return underlying.doc(); }
    template<class Body> void def(const char* name, Body body) {
        underlying.def(name, guarded_binding<true>(std::move(body)));
    }
    // Mixed bindings explicitly copy native values under native(), then build
    // Python containers with the GIL. No implicit blanket-release exception.
    template<class Body> void def_python(const char* name, Body body) {
        underlying.def(name, guarded_binding<false>(std::move(body)));
    }
};
// Called at exactly the old platform lookup, not at binding entry: invalid or
// metadata-only calls keep their existing no-initialization/validation behavior.
std::shared_ptr<VulkanPlatform> python_platform() {
    return pytorch_vulkan::acquire_vulkan_python_binding_root(
        pytorch_vulkan::platform_lifecycle(), [] { return pytorch_vulkan::platform(); },
        [](VulkanPlatform& owner) -> pytorch_vulkan::VulkanRuntimeLifecycle& { return owner.runtime_lifecycle(); },
        [](VulkanPlatform& owner) { return owner.execution_context().callback_active_on_this_thread(); });
}
} // namespace

PYBIND11_MODULE(_C, python_module) {
    if (PyThreadState_GetInterpreter(PyThreadState_Get()) != PyInterpreterState_Main())
        throw std::runtime_error("Vulkan supports only the CPython main interpreter");
    try {
        pytorch_vulkan::register_vulkan_python_lifecycle(
            pytorch_vulkan::platform_lifecycle(), pytorch_vulkan::vulkan_python_hook_ops());
    } catch (...) {
        if (PyErr_Occurred()) throw py::error_already_set();
        throw;
    }
    GuardedPythonModule module{python_module};
    (void)vulkan_allocator_instance();
    module.doc() = "Minimal Vulkan runtime interface";
    module.def("is_available", &VulkanPlatform::is_available);
    module.def("device_count", []() { return VulkanPlatform::is_available() ? 1 : 0; });
    module.def("platform_construction_count",
               [] { return VulkanPlatform::platform_construction_count(); });
    module.def("availability_probe_count",
               [] { return VulkanPlatform::availability_probe_count(); });
    module.def("current_device",
               []() { return pytorch_vulkan::current_device().index(); });
    module.def("set_device", &pytorch_vulkan::set_device);
    module.def("formatter_double_supported",
               [] { return python_platform()->supports_formatter_double(); });
    module.def("reset_execution_counters",
               [] { python_platform()->reset_execution_counters(); });
    module.def("compute_dispatch_count",
               [] { return python_platform()->compute_dispatch_count(); });
    module.def("compute_submitted_count",
               [] { return python_platform()->compute_submitted_count(); });
    module.def("compute_completed_count",
               [] { return python_platform()->compute_completed_count(); });
    module.def("compute_wait_count",
               [] { return python_platform()->compute_wait_count(); });
    // Compatibility name: this historically reported completed submissions.
    module.def("compute_submission_count",
               [] { return python_platform()->compute_completed_count(); });
    module.def("descriptor_pool_creation_count", [] {
        return python_platform()->compute().descriptor_pool_creation_count();
    });
    module.def("descriptor_set_allocation_count", [] {
        return python_platform()->compute().descriptor_set_allocation_count();
    });
    module.def("descriptor_set_reuse_count", [] {
        return python_platform()->compute().descriptor_set_reuse_count();
    });
    module.def("reset_descriptor_resource_counters", [] {
        python_platform()->compute().reset_descriptor_resource_counters();
    });
    module.def_python("execution_counter_snapshot", [] {
        const auto snapshot = pytorch_vulkan::with_vulkan_python_native([] { return python_platform()->execution_counter_snapshot(); });
        return py::make_tuple(snapshot.dispatches, snapshot.vulkan_copies,
                              snapshot.explicit_transfers, snapshot.fallbacks);
    });
    module.def_python("live_resource_snapshot", [] {
        const auto snapshot = pytorch_vulkan::with_vulkan_python_native([] { return python_platform()->live_resource_snapshot(); });
        return py::make_tuple(snapshot.descriptor_pools, snapshot.descriptor_sets,
                              snapshot.pipelines, snapshot.shader_modules,
                              snapshot.pending_transfers, snapshot.pending_compute,
                              snapshot.allocations, snapshot.pinned_host_allocations);
    });
    // Narrow diagnostic seam for migration tests; this does not expose runtime
    // ownership or control outside the existing test module.
    module.def_python("shared_service_snapshot", [] {
        const auto native = pytorch_vulkan::with_vulkan_python_native([] {
            const auto owner = python_platform();
            return std::make_tuple(owner->pipeline_cache_snapshot(), owner->shader_registry().snapshot(),
                                   owner->compute().descriptor_arena_snapshot());
        });
        const auto& pipeline = std::get<0>(native);
        const auto& shader = std::get<1>(native);
        const auto& descriptor = std::get<2>(native);
        py::dict result;
        result["pipeline_entries"] = pipeline.entry_count;
        result["pipeline_count"] = pipeline.pipeline_count;
        result["pipeline_hits"] = pipeline.hits;
        result["pipeline_misses"] = pipeline.misses;
        result["pipeline_evictions"] = pipeline.evictions;
        result["pipeline_pending_destructions"] = pipeline.pending_destructions;
        result["pipeline_invalidated"] = pipeline.invalidated;
        result["shader_modules"] = shader.module_count;
        result["shader_hits"] = shader.cache_hits;
        result["shader_misses"] = shader.cache_misses;
        result["descriptor_pools"] = descriptor.pool_count;
        result["descriptor_pool_limit"] = descriptor.pool_limit;
        result["descriptor_pool_creations"] = descriptor.pool_creations;
        result["descriptor_sets"] = descriptor.live_sets;
        result["descriptor_allocations"] = descriptor.allocations;
        result["descriptor_reuses"] = descriptor.reuses;
        result["descriptor_rollovers"] = descriptor.rollovers;
        result["descriptor_pending"] = descriptor.pending;
        result["descriptor_quarantined"] = descriptor.quarantined;
        result["descriptor_invalidated"] = descriptor.invalidated;
        return result;
    });
    module.def("begin_training_step",
               [] { python_platform()->compute().begin_training_step(); });
    module.def("execution_mode", [] {
        return python_platform()->async_execution_enabled() ? "async" : "sync";
    });
    module.def("synchronize", [] {
        python_platform()->synchronize_execution();
    });
    module.def("query", [] {
        return python_platform()->query_execution();
    });
    module.def("end_training_step",
               [] { python_platform()->compute().end_training_step(); });
    module.def("cancel_training_step",
               [] { python_platform()->compute().cancel_training_step(); });
    module.def("test_hold_completion", [](bool hold) {
        if (!hold) {
            pytorch_vulkan::release_python_completion_holds();
            return; // weak-root non-work control: never initializes/admit/waits
        }
        auto owner = python_platform();
        pytorch_vulkan::track_python_completion_hold(owner, [](void* p) noexcept {
            static_cast<VulkanPlatform*>(p)->set_completion_hold_for_testing(false);
        });
        owner->set_completion_hold_for_testing(hold);
    });
    module.def("completion_hold_expiry_count", [] {
        return python_platform()->completion_hold_expiry_count_for_testing();
    });
    module.def("execution_fence_wait_count", [] {
        return python_platform()->execution_fence_wait_count();
    });
    module.def("transfer_reservation_wait_count", [] {
        return python_platform()->transfer_reservation_wait_count();
    });
    module.def("training_step_active", [] {
        return python_platform()->compute().training_step_active();
    });
    module.def("explicit_transfer_count",
               [] { return python_platform()->explicit_transfer_count(); });
    module.def("fallback_count", [] {
        return python_platform()->execution_counter_snapshot().fallbacks;
    });
    module.def("set_strict_mode", [](bool enabled) {
        python_platform()->set_strict_mode(enabled);
    });
    module.def("strict_mode", [] { return python_platform()->strict_mode(); });
    // Test-only seam: records an attempted fallback without moving payloads or
    // invoking a CPU implementation. Real fallback boundaries must call the
    // same centralized policy instead of incrementing counters directly.
    module.def("test_inject_fallback",
               [] { python_platform()->record_fallback(); });
    module.def("pending_compute_count",
               [] { return python_platform()->pending_compute_count(); });
    module.def("vulkan_copy_count",
               [] { return python_platform()->vulkan_copy_count(); });
    module.def("copy_command_count",
               [] { return python_platform()->copy_command_count(); });
    module.def("transfer_operation_count",
               [] { return python_platform()->transfer_operation_count(); });
    module.def("transfer_submission_count",
               [] { return python_platform()->transfer_submission_count(); });
    module.def("transfer_completion_count",
               [] { return python_platform()->transfer_completion_count(); });
    module.def("transfer_wait_count",
               [] { return python_platform()->transfer_wait_count(); });
    module.def("reset_timing", [] { python_platform()->reset_timing(); });
    module.def_python("timing_snapshot", [] {
        const auto timing = pytorch_vulkan::with_vulkan_python_native([] { return python_platform()->timing_snapshot(); });
        return py::make_tuple(timing.allocation, timing.recording, timing.submit,
                              timing.host_fence_wait, timing.total);
    });
    module.def_python("timing_breakdown", [] {
        const auto timing = pytorch_vulkan::with_vulkan_python_native([] { return python_platform()->timing_snapshot(); });
        py::dict result;
        result["allocation_seconds"] = timing.allocation;
        result["recording_seconds"] = timing.recording;
        result["submit_seconds"] = timing.submit;
        result["host_fence_wait_seconds"] = timing.host_fence_wait;
        result["buffer_create_seconds"] = timing.buffer_create;
        result["buffer_destroy_seconds"] = timing.buffer_destroy;
        result["buffer_map_write_seconds"] = timing.buffer_map_write;
        result["buffer_map_read_seconds"] = timing.buffer_map_read;
        result["descriptor_update_seconds"] = timing.descriptor_update;
        result["command_record_seconds"] = timing.command_record;
        result["dispatch_host_seconds"] = timing.dispatch_host;
        result["buffer_creations"] = timing.buffer_creations;
        result["buffer_requested_bytes"] = timing.buffer_requested_bytes;
        return result;
    });
    module.def("timestamp_queries_supported", [] {
        return python_platform()->timestamp_queries_supported();
    });
    module.def("timestamp_query_support_reason", [] {
        return python_platform()->timestamp_query_support_reason();
    });
    module.def("reset_gpu_timing",
               [] { python_platform()->reset_timestamp_samples(); });
    module.def_python("gpu_timing_snapshot", [] {
        const auto native = pytorch_vulkan::with_vulkan_python_native([] { return python_platform()->timestamp_samples(); });
        py::list samples;
        for (const auto &sample : native) {
            py::dict value;
            value["gpu_time_ns"] = sample.gpu_time_ns;
            value["available"] = sample.available;
            value["submission_id"] = sample.submission_id;
            value["scope"] = sample.scope;
            samples.append(value);
        }
        return samples;
    });
    module.def_python("timestamp_query_snapshot", [] {
        const auto native = pytorch_vulkan::with_vulkan_python_native([] {
            const auto owner = python_platform();
            return std::make_tuple(owner->timestamp_query_capacity(), owner->timestamp_query_in_use(),
                                   owner->timestamp_queries_supported(), owner->timestamp_query_quarantined());
        });
        return py::make_tuple(std::get<0>(native), std::get<1>(native), std::get<2>(native), std::get<3>(native));
    });
    module.def("test_inject_device_loss", [] {
        auto owner = python_platform();
        owner->mark_device_lost(VK_ERROR_DEVICE_LOST);
        try {
            if (owner->execution_context().recording())
                owner->execution_context().submit();
            else
                owner->execution_context().begin("device-loss-test");
        } catch (const VulkanDeviceLost &) {
        }
    });
}
