#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

namespace pytorch_vulkan {

at::Tensor formatter_view(const at::Tensor &tensor, c10::SymIntArrayRef size);

} // namespace pytorch_vulkan
