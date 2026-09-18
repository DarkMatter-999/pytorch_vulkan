#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
std::tuple<at::Tensor, at::Tensor> nll_loss_forward(const at::Tensor &,
                                                    const at::Tensor &,
                                                    const c10::optional<at::Tensor> &,
                                                    int64_t, c10::SymInt);
at::Tensor nll_loss_backward(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                             const c10::optional<at::Tensor> &, int64_t, c10::SymInt,
                             const at::Tensor &);
} // namespace pytorch_vulkan
