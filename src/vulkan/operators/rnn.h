#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
at::Tensor rnn_sequence(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                        const at::Tensor &);
} // namespace pytorch_vulkan
