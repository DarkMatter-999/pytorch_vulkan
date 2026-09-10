#include "vulkan_platform.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, module) {
    module.doc() = "Minimal Vulkan runtime interface";
    module.def("is_available", &VulkanPlatform::is_available);
    module.def("device_count", []() { return VulkanPlatform::is_available() ? 1 : 0; });
}
