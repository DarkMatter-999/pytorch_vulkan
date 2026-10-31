#pragma once
#include <ATen/ATen.h>
namespace pytorch_vulkan {
at::Tensor upsample_nearest2d(const at::Tensor &, at::IntArrayRef, c10::optional<double>,
                              c10::optional<double>);
at::Tensor upsample_nearest2d_backward(const at::Tensor &, at::IntArrayRef,
                                       at::IntArrayRef, c10::optional<double>,
                                       c10::optional<double>);
at::Tensor upsample_nearest_exact2d(const at::Tensor &, at::IntArrayRef,
                                    c10::optional<double>, c10::optional<double>);
at::Tensor upsample_nearest_exact2d_backward(const at::Tensor &, at::IntArrayRef,
                                             at::IntArrayRef, c10::optional<double>,
                                             c10::optional<double>);
at::Tensor upsample_bilinear2d(const at::Tensor &, at::IntArrayRef, bool,
                               c10::optional<double>, c10::optional<double>);
at::Tensor upsample_bilinear2d_backward(const at::Tensor &, at::IntArrayRef,
                                        at::IntArrayRef, bool, c10::optional<double>,
                                        c10::optional<double>);
} // namespace pytorch_vulkan
