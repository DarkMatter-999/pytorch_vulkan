#pragma once

#include <ATen/ATen.h>

namespace pytorch_vulkan {
at::Tensor sum_tensor(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                      c10::optional<at::ScalarType>);
at::Tensor mean_tensor(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                       c10::optional<at::ScalarType>);
at::Tensor &sum_out(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                    c10::optional<at::ScalarType>, at::Tensor &);
at::Tensor &mean_out(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                     c10::optional<at::ScalarType>, at::Tensor &);
at::Tensor autograd_sum(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                        c10::optional<at::ScalarType>);
at::Tensor autograd_mean(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                         c10::optional<at::ScalarType>);
} // namespace pytorch_vulkan
