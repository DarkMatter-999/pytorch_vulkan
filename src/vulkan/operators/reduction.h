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
at::Tensor amax_tensor(const at::Tensor &, c10::ArrayRef<int64_t>, bool);
at::Tensor amin_tensor(const at::Tensor &, c10::ArrayRef<int64_t>, bool);
at::Tensor &amax_out(const at::Tensor &, c10::ArrayRef<int64_t>, bool, at::Tensor &);
at::Tensor &amin_out(const at::Tensor &, c10::ArrayRef<int64_t>, bool, at::Tensor &);
at::Tensor prod_tensor(const at::Tensor &, c10::OptionalArrayRef<int64_t>, bool,
                       c10::optional<at::ScalarType>);
at::Tensor prod_dim_tensor(const at::Tensor &, int64_t, bool, c10::optional<at::ScalarType>);
at::Tensor &prod_int_out(const at::Tensor &, int64_t, bool, c10::optional<at::ScalarType>, at::Tensor &);
at::Tensor softmax_tensor(const at::Tensor &, int64_t, bool);
at::Tensor log_softmax_tensor(const at::Tensor &, int64_t, bool);
at::Tensor &softmax_out(const at::Tensor &, int64_t, bool, at::Tensor &);
at::Tensor &log_softmax_out(const at::Tensor &, int64_t, bool, at::Tensor &);
at::Tensor autograd_amax(const at::Tensor &, c10::ArrayRef<int64_t>, bool);
at::Tensor autograd_amin(const at::Tensor &, c10::ArrayRef<int64_t>, bool);
at::Tensor autograd_prod_dim(const at::Tensor &, int64_t, bool, c10::optional<at::ScalarType>);
at::Tensor autograd_softmax(const at::Tensor &, int64_t, bool);
at::Tensor autograd_log_softmax(const at::Tensor &, int64_t, bool);
at::Tensor &softmax_backward_out(const at::Tensor &, const at::Tensor &, int64_t,
                                 c10::ScalarType, at::Tensor &);
at::Tensor &log_softmax_backward_out(const at::Tensor &, const at::Tensor &, int64_t,
                                     c10::ScalarType, at::Tensor &);
} // namespace pytorch_vulkan
