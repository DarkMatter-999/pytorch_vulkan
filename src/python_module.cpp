#include "vulkan_platform.h"
#include "vulkan_device_guard.h"
#include "vulkan_allocator.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, module) {
    (void)vulkan_allocator_instance();
    module.doc() = "Minimal Vulkan runtime interface";
    module.def("is_available", &VulkanPlatform::is_available);
    module.def("device_count", []() { return VulkanPlatform::is_available() ? 1 : 0; });
    module.def("current_device", []() { return pytorch_vulkan::current_device().index(); });
    module.def("set_device", &pytorch_vulkan::set_device);
    py::module_::import("atexit").attr("register")(
        py::cpp_function(&pytorch_vulkan::shutdown_platform));
}
