#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {

void copy_tensor(at::Tensor &destination, const at::Tensor &source,
                 bool non_blocking);

} // namespace pytorch_vulkan
