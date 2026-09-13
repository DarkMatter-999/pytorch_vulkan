#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias);
}
