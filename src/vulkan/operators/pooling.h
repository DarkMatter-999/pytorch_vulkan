#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
at::Tensor adaptive_avg_pool2d(const at::Tensor &, at::IntArrayRef);
at::Tensor adaptive_avg_pool2d_backward(const at::Tensor &, const at::Tensor &);
at::Tensor avg_pool2d(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                      at::IntArrayRef, bool, bool, c10::optional<int64_t>);
at::Tensor avg_pool2d_backward(const at::Tensor &, const at::Tensor &, at::IntArrayRef,
                               at::IntArrayRef, at::IntArrayRef, bool, bool,
                               c10::optional<int64_t>);
std::tuple<at::Tensor, at::Tensor>
max_pool2d_with_indices(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                        at::IntArrayRef, at::IntArrayRef, bool);
at::Tensor max_pool2d_with_indices_backward(const at::Tensor &, const at::Tensor &,
                                            at::IntArrayRef, at::IntArrayRef,
                                            at::IntArrayRef, at::IntArrayRef, bool,
                                            const at::Tensor &);
} // namespace pytorch_vulkan
