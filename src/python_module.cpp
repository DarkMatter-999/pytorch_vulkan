#include "vulkan_allocator.h"
#include "vulkan_compute.h"
#include "vulkan_device_guard.h"
#include "vulkan_execution.h"
#include "vulkan/descriptor_arena.h"
#include "vulkan/pipeline_cache.h"
#include "vulkan/shader_registry.h"
#include "vulkan_platform.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, module) {
    (void)vulkan_allocator_instance();
    module.doc() = "Minimal Vulkan runtime interface";
    module.def("is_available", &VulkanPlatform::is_available);
    module.def("device_count", []() { return VulkanPlatform::is_available() ? 1 : 0; });
    module.def("current_device",
               []() { return pytorch_vulkan::current_device().index(); });
    module.def("set_device", &pytorch_vulkan::set_device);
    module.def("formatter_double_supported",
               [] { return pytorch_vulkan::formatter_double_supported(); });
    module.def("reset_execution_counters",
               [] { pytorch_vulkan::platform()->reset_execution_counters(); });
    module.def("compute_dispatch_count",
               [] { return pytorch_vulkan::platform()->compute_dispatch_count(); });
    module.def("compute_submitted_count",
               [] { return pytorch_vulkan::platform()->compute_submitted_count(); });
    module.def("compute_completed_count",
               [] { return pytorch_vulkan::platform()->compute_completed_count(); });
    module.def("compute_wait_count",
               [] { return pytorch_vulkan::platform()->compute_wait_count(); });
    // Compatibility name: this historically reported completed submissions.
    module.def("compute_submission_count",
               [] { return pytorch_vulkan::platform()->compute_completed_count(); });
    module.def("descriptor_pool_creation_count", [] {
        return pytorch_vulkan::platform()->compute().descriptor_pool_creation_count();
    });
    module.def("descriptor_set_allocation_count", [] {
        return pytorch_vulkan::platform()->compute().descriptor_set_allocation_count();
    });
    module.def("descriptor_set_reuse_count", [] {
        return pytorch_vulkan::platform()->compute().descriptor_set_reuse_count();
    });
    module.def("reset_descriptor_resource_counters", [] {
        pytorch_vulkan::platform()->compute().reset_descriptor_resource_counters();
    });
    module.def("execution_counter_snapshot", [] {
        const auto snapshot = pytorch_vulkan::platform()->execution_counter_snapshot();
        return py::make_tuple(snapshot.dispatches, snapshot.vulkan_copies,
                               snapshot.explicit_transfers, snapshot.fallbacks);
    });
    module.def("live_resource_snapshot", [] {
        const auto snapshot = pytorch_vulkan::platform()->live_resource_snapshot();
        return py::make_tuple(snapshot.descriptor_pools, snapshot.descriptor_sets,
                              snapshot.pipelines, snapshot.shader_modules,
                              snapshot.pending_transfers, snapshot.pending_compute,
                               snapshot.allocations);
    });
    // Narrow diagnostic seam for migration tests; this does not expose runtime
    // ownership or control outside the existing test module.
    module.def("shared_service_snapshot", [] {
        const auto pipeline = pytorch_vulkan::platform()->pipeline_cache_snapshot();
        const auto shader = pytorch_vulkan::platform()->shader_registry().snapshot();
        const auto descriptor =
            pytorch_vulkan::platform()->compute().descriptor_arena_snapshot();
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
               [] { pytorch_vulkan::platform()->compute().begin_training_step(); });
    module.def("end_training_step",
               [] { pytorch_vulkan::platform()->compute().end_training_step(); });
    module.def("cancel_training_step",
               [] { pytorch_vulkan::platform()->compute().cancel_training_step(); });
    module.def("training_step_active", [] {
        return pytorch_vulkan::platform()->compute().training_step_active();
    });
    module.def("explicit_transfer_count",
               [] { return pytorch_vulkan::platform()->explicit_transfer_count(); });
    module.def("fallback_count", [] {
        return pytorch_vulkan::platform()->execution_counter_snapshot().fallbacks;
    });
    module.def("set_strict_mode", [](bool enabled) {
        pytorch_vulkan::platform()->set_strict_mode(enabled);
    });
    module.def("strict_mode", [] { return pytorch_vulkan::platform()->strict_mode(); });
    // Test-only seam: records an attempted fallback without moving payloads or
    // invoking a CPU implementation. Real fallback boundaries must call the
    // same centralized policy instead of incrementing counters directly.
    module.def("test_inject_fallback",
               [] { pytorch_vulkan::platform()->record_fallback(); });
    module.def("pending_compute_count",
               [] { return pytorch_vulkan::platform()->pending_compute_count(); });
    module.def("vulkan_copy_count",
               [] { return pytorch_vulkan::platform()->vulkan_copy_count(); });
    module.def("copy_command_count",
               [] { return pytorch_vulkan::platform()->copy_command_count(); });
    module.def("transfer_operation_count", [] {
        return pytorch_vulkan::platform()->transfer_operation_count();
    });
    module.def("transfer_submission_count", [] {
        return pytorch_vulkan::platform()->transfer_submission_count();
    });
    module.def("transfer_completion_count", [] {
        return pytorch_vulkan::platform()->transfer_completion_count();
    });
    module.def("transfer_wait_count", [] {
        return pytorch_vulkan::platform()->transfer_wait_count();
    });
    module.def("reset_timing", [] { pytorch_vulkan::platform()->reset_timing(); });
    module.def("timing_snapshot", [] {
        const auto timing = pytorch_vulkan::platform()->timing_snapshot();
        return py::make_tuple(timing.allocation, timing.recording, timing.submit,
                              timing.host_fence_wait, timing.total);
    });
    module.def("timestamp_queries_supported",
               [] { return pytorch_vulkan::platform()->timestamp_queries_supported(); });
    module.def("timestamp_query_support_reason", [] {
        return pytorch_vulkan::platform()->timestamp_query_support_reason();
    });
    module.def("reset_gpu_timing",
               [] { pytorch_vulkan::platform()->reset_timestamp_samples(); });
    module.def("gpu_timing_snapshot", [] {
        py::list samples;
        for (const auto &sample : pytorch_vulkan::platform()->timestamp_samples()) {
            py::dict value;
            value["gpu_time_ns"] = sample.gpu_time_ns;
            value["available"] = sample.available;
            value["submission_id"] = sample.submission_id;
            value["scope"] = sample.scope;
            samples.append(value);
        }
        return samples;
    });
    module.def("timestamp_query_snapshot", [] {
        const auto owner = pytorch_vulkan::platform();
        return py::make_tuple(owner->timestamp_query_capacity(),
                              owner->timestamp_query_in_use(),
                              owner->timestamp_queries_supported(),
                              owner->timestamp_query_quarantined());
    });
    module.def("test_inject_device_loss", [] {
        auto owner = pytorch_vulkan::platform();
        owner->mark_device_lost(VK_ERROR_DEVICE_LOST);
        try {
            if (owner->execution_context().recording())
                owner->execution_context().submit();
            else
                owner->execution_context().begin("device-loss-test");
        } catch (const VulkanDeviceLost &) {
        }
    });
    py::module_::import("atexit").attr("register")(
        py::cpp_function(&pytorch_vulkan::shutdown_platform));
}
