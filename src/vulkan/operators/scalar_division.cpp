#include "out.h"

#include <torch/library.h>

namespace pytorch_vulkan {
// Numerical leaf only. Generated Autograd and ADInplaceOrView retain ownership
// of derivative history, leaf-write checks, aliases, and version increments.
at::Tensor &div_scalar_inplace(at::Tensor &self, const at::Scalar &other) {
    return dispatch_tensor_scalar_alias(self, other, PointwiseOperation::Div,
                                        "div_.Scalar");
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("div_.Scalar", &pytorch_vulkan::div_scalar_inplace);
}
