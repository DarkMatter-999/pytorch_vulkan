#pragma once
#include <ATen/ATen.h>
#include <array>
#include <tuple>
namespace pytorch_vulkan {
struct ConvolutionGeometry {
    int64_t stride_height = 1;
    int64_t stride_width = 1;
    int64_t padding_height = 1;
    int64_t padding_width = 1;
    int64_t dilation_height = 1;
    int64_t dilation_width = 1;
};
at::Tensor convolution(const at::Tensor &, const at::Tensor &,
                       const c10::optional<at::Tensor> &, at::IntArrayRef,
                       at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef,
                       int64_t);
at::Tensor convolution_backward_input(const at::Tensor &, const at::Tensor &,
                                      const ConvolutionGeometry &, int64_t input_height,
                                      int64_t input_width);
at::Tensor convolution_backward_weight(const at::Tensor &, const at::Tensor &,
                                       const ConvolutionGeometry &, int64_t kernel_height,
                                       int64_t kernel_width);
at::Tensor convolution_backward_bias(const at::Tensor &);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                     c10::OptionalArrayRef<int64_t>, at::IntArrayRef, at::IntArrayRef,
                     at::IntArrayRef, bool, at::IntArrayRef, int64_t,
                     std::array<bool, 3>);
at::Tensor convolution_overrideable(const at::Tensor &, const at::Tensor &,
                                    const c10::optional<at::Tensor> &, at::IntArrayRef,
                                    at::IntArrayRef, at::IntArrayRef, bool,
                                    at::IntArrayRef, int64_t);
} // namespace pytorch_vulkan
