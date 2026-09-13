#include "pooling.h"
#include <c10/util/Exception.h>
#include <torch/library.h>
namespace pytorch_vulkan {
std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                                                           at::IntArrayRef, at::IntArrayRef, bool) {
    TORCH_CHECK(false, "Vulkan pooling is not declared in the implemented model slice");
}
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("max_pool2d_with_indices", &pytorch_vulkan::max_pool2d_with_indices); }
