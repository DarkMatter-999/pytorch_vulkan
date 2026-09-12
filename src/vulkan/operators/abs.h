#pragma once

#include <ATen/core/Tensor.h>

namespace pytorch_vulkan {

at::Tensor &abs_out(const at::Tensor &self, at::Tensor &out);

} // namespace pytorch_vulkan
