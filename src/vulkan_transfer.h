#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,
                        bool non_blocking);

// Internal final-value presentation path used by tensor formatting.
at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source);

} // namespace pytorch_vulkan
