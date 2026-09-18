#include "vulkan_allocator.h"
#include "vulkan_compute.h"
#include "vulkan_device_guard.h"
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
    module.def("execution_counter_snapshot", [] {
        const auto snapshot = pytorch_vulkan::platform()->execution_counter_snapshot();
        return py::make_tuple(snapshot.dispatches, snapshot.vulkan_copies,
                              snapshot.explicit_transfers, snapshot.fallbacks);
    });
    module.def("begin_training_step",
               [] { pytorch_vulkan::platform()->compute().begin_training_step(); });
    module.def("end_training_step",
               [] { pytorch_vulkan::platform()->compute().end_training_step(); });
    module.def("cancel_training_step",
               [] { pytorch_vulkan::platform()->compute().cancel_training_step(); });
    module.def("training_step_active",
               [] { return pytorch_vulkan::platform()->compute().training_step_active(); });
    module.def("explicit_transfer_count",
               [] { return pytorch_vulkan::platform()->explicit_transfer_count(); });
    module.def("fallback_count",
               [] { return pytorch_vulkan::platform()->execution_counter_snapshot().fallbacks; });
    module.def("set_strict_mode",
               [](bool enabled) { pytorch_vulkan::platform()->set_strict_mode(enabled); });
    module.def("strict_mode",
               [] { return pytorch_vulkan::platform()->strict_mode(); });
    // Test-only seam: records an attempted fallback without moving payloads or
    // invoking a CPU implementation. Real fallback boundaries must call the
    // same centralized policy instead of incrementing counters directly.
    module.def("test_inject_fallback", [] {
        pytorch_vulkan::platform()->record_fallback();
    });
    module.def("pending_compute_count",
               [] { return pytorch_vulkan::platform()->pending_compute_count(); });
    module.def("vulkan_copy_count",
               [] { return pytorch_vulkan::platform()->vulkan_copy_count(); });
    module.def("copy_command_count",
               [] { return pytorch_vulkan::platform()->copy_command_count(); });
    module.def("reset_timing",
               [] { pytorch_vulkan::platform()->reset_timing(); });
    module.def("timing_snapshot", [] {
        const auto timing = pytorch_vulkan::platform()->timing_snapshot();
        return py::make_tuple(timing.allocation, timing.recording,
                              timing.submit_wait, timing.compute, timing.total);
    });
    py::module_::import("atexit").attr("register")(
        py::cpp_function(&pytorch_vulkan::shutdown_platform));
}
