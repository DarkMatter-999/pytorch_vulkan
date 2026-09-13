#include "convolution.h"
#include <c10/util/Exception.h>
#include <torch/library.h>
namespace pytorch_vulkan {
at::Tensor convolution_overrideable(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &,
                                    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t) {
    TORCH_CHECK(false, "Vulkan convolution is not declared in the implemented model slice");
}
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("convolution_overrideable", &pytorch_vulkan::convolution_overrideable); }
