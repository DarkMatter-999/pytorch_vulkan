#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
at::Tensor mse_loss(const at::Tensor &, const at::Tensor &, int64_t reduction);
at::Tensor mse_loss_backward(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t reduction);
at::Tensor autograd_mse_loss(const at::Tensor &, const at::Tensor &, int64_t reduction);
} // namespace pytorch_vulkan
