#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm(const at::Tensor &, const c10::optional<at::Tensor> &,
                  const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                  const c10::optional<at::Tensor> &, bool, double, double);
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &,
    const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
    const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, bool, double,
    std::array<bool, 3>);
} // namespace pytorch_vulkan
