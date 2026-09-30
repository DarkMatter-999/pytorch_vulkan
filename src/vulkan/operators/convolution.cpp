#include "convolution.h"
#include "autograd.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include <c10/util/Exception.h>
#include <limits>
#include <torch/library.h>
namespace pytorch_vulkan {
namespace {
VkDeviceSize bytes(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.numel() >= 0 &&
                    static_cast<uint64_t>(t.numel()) <=
                        std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan convolution ", name, " byte count overflows");
    const uint64_t value = static_cast<uint64_t>(t.numel()) * sizeof(float);
    TORCH_CHECK(value <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan convolution ", name, " byte count exceeds range");
    return static_cast<VkDeviceSize>(value);
}
void validate(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 &&
                    t.device().index() == 0,
                "Vulkan convolution ", name, " requires vk:0");
    TORCH_CHECK(t.scalar_type() == at::kFloat, "Vulkan convolution ", name,
                " requires float32");
}
void validate_rank4(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.dim() == 4, "Vulkan convolution ", name,
                " requires rank 4 NCHW input");
    TORCH_CHECK(t.numel() > 0, "Vulkan convolution ", name, " rejects empty tensors");
}
// PyTorch computes floor((H + 2p - d*(k-1) - 1) / s) + 1. Rejecting a negative
// numerator first is what makes C++ truncation toward zero agree with floor:
// CPU rejects any input whose padded extent is below the dilated kernel span,
// so on every input CPU accepts the numerator is non-negative and the two
// operations coincide. CPU accepts the rejected case by auto-padding the input
// up to the kernel span, which this backend does not implement.
void validate_geometry_axis(int64_t stride, int64_t padding, int64_t dilation,
                            int64_t extent, int64_t kernel, const char *name) {
    constexpr int64_t shader_int_max = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(stride >= 1, "Vulkan convolution ", name, " requires stride >= 1, got ", stride);
    TORCH_CHECK(dilation >= 1, "Vulkan convolution ", name, " requires dilation >= 1, got ", dilation);
    TORCH_CHECK(padding >= 0, "Vulkan convolution ", name, " requires padding >= 0, got ", padding);
    TORCH_CHECK(stride <= shader_int_max && padding <= shader_int_max &&
                    dilation <= shader_int_max,
                "Vulkan convolution ", name,
                " geometry values must fit signed 32-bit shader arithmetic");
    // Shader coordinates combine output/input coordinates, padding and the
    // dilated kernel offset in signed int. Bounding the padded extent covers
    // both the forward mapping and op 1's inverse mapping (see below).
    const int64_t padded = extent + 2 * padding;
    TORCH_CHECK(padded <= shader_int_max,
                "Vulkan convolution ", name,
                " padded extent must fit signed 32-bit shader arithmetic");
    const int64_t span = dilation * (kernel - 1) + 1;
    TORCH_CHECK(span <= padded,
                "Vulkan convolution ", name,
                " kernel span ", span, " exceeds padded input extent ", padded,
                " (extent ", extent, ", kernel ", kernel, ", padding ", padding,
                ", dilation ", dilation, ")");
}
int64_t output_extent(int64_t extent, int64_t kernel, int64_t stride, int64_t padding,
                      int64_t dilation, const char *name) {
    validate_geometry_axis(stride, padding, dilation, extent, kernel, name);
    // These bounds make the following int64_t extent arithmetic safe: tensor
    // dimensions are uint32_t, and geometry is at most INT32_MAX.
    const int64_t span = dilation * (kernel - 1) + 1;
    const int64_t padded = extent + 2 * padding;
    TORCH_CHECK(padded >= span, "Vulkan convolution ", name,
                " kernel span ", span, " exceeds padded input extent ", padded,
                " (extent ", extent, ", padding ", padding, ", dilation ", dilation,
                "); CPU auto-pads this case, which Vulkan convolution does not");
    return (padded - span) / stride + 1;
}
at::Tensor run(const at::Tensor &input, const at::Tensor &weight,
               const at::Tensor &bias, uint32_t operation,
               const ConvolutionGeometry &geometry = {},
               int64_t grad_input_height = 0, int64_t grad_input_width = 0,
               int64_t kernel_height = 0, int64_t kernel_width = 0) {
    validate(input, "input");
    validate(weight, "weight");
    validate(bias, "bias");
    validate_rank4(input, "input");
    if (operation != 3)
        validate_rank4(weight, "weight");
    TORCH_CHECK(input.device() == weight.device() && input.device() == bias.device(),
                "Vulkan convolution tensors require one device");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "convolution input");
    const auto weight_layout =
        inspect_vulkan_tensor_layout(weight, "convolution weight");
    const auto bias_layout = inspect_vulkan_tensor_layout(bias, "convolution bias");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan convolution rejects overlapping input layouts");
    TORCH_CHECK(weight_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan convolution rejects overlapping weight layouts");
    TORCH_CHECK(bias_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan convolution rejects overlapping bias layouts");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                    bias.scalar_type() == at::kFloat,
                "Vulkan convolution requires float32 tensors");
    if (operation == 0) {
        TORCH_CHECK(bias.dim() == 1, "Vulkan convolution bias requires rank 1");
        TORCH_CHECK(input.size(1) % geometry.groups == 0 &&
                        weight.size(0) % geometry.groups == 0 &&
                        weight.size(1) == input.size(1) / geometry.groups,
                    "Vulkan convolution grouped channel dimensions do not match");
        TORCH_CHECK(bias.size(0) == weight.size(0),
                    "Vulkan convolution bias must have weight.size(0) = ",
                    weight.size(0), " elements, got ", bias.size(0));
    }
    // Convolution shader indexing uses tensor metadata strides, preserving
    // transposed views without introducing Vulkan copies.
    const int64_t kernel_h = operation == 2 ? kernel_height : weight.size(2);
    const int64_t kernel_w = operation == 2 ? kernel_width : weight.size(3);
    const int64_t checked_extent_h =
        operation == 1 ? grad_input_height : operation == 2 ? weight.size(2) : input.size(2);
    const int64_t checked_extent_w =
        operation == 1 ? grad_input_width : operation == 2 ? weight.size(3) : input.size(3);
    validate_geometry_axis(geometry.stride_height, geometry.padding_height,
                           geometry.dilation_height, checked_extent_h, kernel_h, "height");
    validate_geometry_axis(geometry.stride_width, geometry.padding_width,
                           geometry.dilation_width, checked_extent_w, kernel_w, "width");
    const int64_t out_h =
        operation == 0 ? output_extent(input.size(2), kernel_h, geometry.stride_height,
                                       geometry.padding_height, geometry.dilation_height,
                                       "forward height")
        : operation == 1 ? grad_input_height : input.size(2) + 3 - kernel_h;
    const int64_t out_w =
        operation == 0 ? output_extent(input.size(3), kernel_w, geometry.stride_width,
                                       geometry.padding_width, geometry.dilation_width,
                                       "forward width")
        : operation == 1 ? grad_input_width : input.size(3) + 3 - kernel_w;
    if (operation == 1)
        TORCH_CHECK(grad_input_height > 0 && grad_input_width > 0,
                    "Vulkan convolution backward grad_input extent must be supplied by the caller");
    at::Tensor output =
        operation == 0
            ? at::empty({input.size(0), weight.size(0), out_h, out_w},
                        input.options())
        : operation == 1
            ? at::empty({input.size(0), weight.size(1) * geometry.groups, out_h, out_w},
                        input.options())
        : operation == 2
            ? at::empty({input.size(1), weight.size(1) / geometry.groups, kernel_h, kernel_w}, input.options())
            : at::empty({input.size(1)}, input.options());
    const auto &in_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &bias_data = bias.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    const auto output_layout =
        inspect_vulkan_tensor_layout(output, "convolution output");
    const auto &platform = allocation_platform(in_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(bias_data) &&
                    &platform == &allocation_platform(out_data),
                "Vulkan convolution requires Vulkan allocations");
    validate_allocation(in_data, bytes(input, "input"), "convolution input");
    validate_allocation(weight_data, bytes(weight, "weight"), "convolution weight");
    validate_allocation(bias_data, bytes(bias, "bias"), "convolution bias");
    validate_allocation(out_data, bytes(output, "output"), "convolution output");
    platform.compute().convolution(
        allocation_buffer(in_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(bias_data).buffer(), allocation_buffer(out_data).buffer(),
        input_layout, weight_layout, bias_layout, output_layout, operation,
        static_cast<uint32_t>(kernel_h),
        static_cast<uint32_t>(kernel_w),
        VulkanConvolutionGeometry{static_cast<uint32_t>(geometry.stride_height),
                                  static_cast<uint32_t>(geometry.stride_width),
                                  static_cast<uint32_t>(geometry.padding_height),
         static_cast<uint32_t>(geometry.padding_width),
                                  static_cast<uint32_t>(geometry.dilation_height),
                                  static_cast<uint32_t>(geometry.dilation_width),
                                  static_cast<uint32_t>(geometry.groups)});
    return output;
}
} // namespace
at::Tensor convolution(const at::Tensor &input, const at::Tensor &weight,
                       const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       bool transposed, at::IntArrayRef output_padding,
                       int64_t groups) {
    TORCH_CHECK(bias.has_value(), "Vulkan convolution requires bias");
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Vulkan convolution requires 2-D stride, padding and dilation");
    TORCH_CHECK(stride[0] >= 1 && stride[1] >= 1 && dilation[0] >= 1 && dilation[1] >= 1 &&
                    padding[0] >= 0 && padding[1] >= 0,
                "Vulkan convolution requires stride and dilation >= 1 and padding >= 0");
    TORCH_CHECK(groups >= 1, "Vulkan convolution requires groups >= 1, got ", groups);
    TORCH_CHECK(input.size(1) % groups == 0 && weight.size(0) % groups == 0,
                "Vulkan convolution requires groups ", groups,
                " to divide both input channels (", input.size(1), ") and output channels (",
                weight.size(0), ")");
    TORCH_CHECK(weight.size(1) == input.size(1) / groups,
                "Vulkan convolution grouped weight shape expects ", input.size(1) / groups,
                " input channels per group, got ", weight.size(1));
    TORCH_CHECK(output_padding.equals({0, 0}) && !transposed,
                "Vulkan convolution supports only non-transposed convolutions with zero "
                "output_padding");
    return run(input, weight, *bias, 0,
               {stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups});
}
at::Tensor convolution_backward_input(const at::Tensor &grad, const at::Tensor &weight,
                                       const ConvolutionGeometry &geometry,
                                       int64_t input_height, int64_t input_width) {
    const int64_t expected_height =
        output_extent(input_height, weight.size(2), geometry.stride_height,
                      geometry.padding_height, geometry.dilation_height,
                      "backward height");
    const int64_t expected_width =
        output_extent(input_width, weight.size(3), geometry.stride_width,
                      geometry.padding_width, geometry.dilation_width,
                      "backward width");
    TORCH_CHECK(grad.size(2) == expected_height && grad.size(3) == expected_width,
                "Vulkan convolution backward grad_output spatial shape expected (",
                expected_height, ", ", expected_width, "), actual (", grad.size(2),
                ", ", grad.size(3), ")");
    return run(grad, weight, grad, 1, geometry, input_height, input_width);
}
at::Tensor convolution_backward_weight(const at::Tensor &grad,
                                       const at::Tensor &input,
                                       const ConvolutionGeometry &geometry,
                                       int64_t kernel_height, int64_t kernel_width) {
    return run(grad, input, grad, 2, geometry, 0, 0, kernel_height, kernel_width);
}
at::Tensor convolution_backward_bias(const at::Tensor &grad) {
    return run(grad, grad, grad, 3);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &weight,
    c10::OptionalArrayRef<int64_t> bias_sizes, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
    validate(grad_output, "backward grad_output");
    validate(input, "backward input");
    validate(weight, "backward weight");
    validate_rank4(grad_output, "backward grad_output");
    validate_rank4(input, "backward input");
    validate_rank4(weight, "backward weight");
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Vulkan convolution backward requires 2-D stride, padding and dilation");
    TORCH_CHECK(stride[0] >= 1 && stride[1] >= 1 && dilation[0] >= 1 && dilation[1] >= 1 &&
                    padding[0] >= 0 && padding[1] >= 0,
                "Vulkan convolution backward requires stride and dilation >= 1 and padding >= 0");
    TORCH_CHECK(output_padding.equals({0, 0}) && !transposed,
                "Vulkan convolution backward supports only non-transposed convolutions "
                "with zero output_padding");
    TORCH_CHECK(output_mask[0] && output_mask[1] && output_mask[2],
                "Vulkan convolution backward requires output_mask [true, true, true]");
    TORCH_CHECK(groups >= 1, "Vulkan convolution backward requires groups >= 1, got ", groups);
    TORCH_CHECK(input.size(1) % groups == 0 && weight.size(0) % groups == 0,
                "Vulkan convolution backward requires groups ", groups,
                " to divide both input channels (", input.size(1), ") and output channels (",
                weight.size(0), ")");
    TORCH_CHECK(weight.size(1) == input.size(1) / groups,
                "Vulkan convolution backward grouped weight shape expects ", input.size(1) / groups,
                " input channels per group, got ", weight.size(1));
    TORCH_CHECK(grad_output.size(0) == input.size(0),
                "Vulkan convolution backward grad_output batch size must match input batch size: ",
                grad_output.size(0), " vs ", input.size(0));
    TORCH_CHECK(grad_output.size(1) == weight.size(0),
                "Vulkan convolution backward grad_output channels must match weight output channels");
    TORCH_CHECK(bias_sizes.has_value() && bias_sizes->equals({weight.size(0)}),
                "Vulkan convolution backward bias shape does not match weight");
    const auto grad_layout =
        inspect_vulkan_tensor_layout(grad_output, "convolution backward grad_output");
    const auto input_layout =
        inspect_vulkan_tensor_layout(input, "convolution backward input");
    const auto weight_layout =
        inspect_vulkan_tensor_layout(weight, "convolution backward weight");
    TORCH_CHECK(grad_layout.internal_overlap == VulkanOverlap::No &&
                    input_layout.internal_overlap == VulkanOverlap::No &&
                    weight_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan convolution backward rejects overlapping layouts");
    TORCH_CHECK(grad_output.numel() > 0 && input.numel() > 0 && weight.numel() > 0,
                "Vulkan convolution backward rejects empty tensors");
    const ConvolutionGeometry geometry{stride[0], stride[1], padding[0], padding[1],
                                       dilation[0], dilation[1], groups};
    return {convolution_backward_input(grad_output, weight, geometry, input.size(2),
                                       input.size(3)),
            convolution_backward_weight(grad_output, input, geometry, weight.size(2),
                                       weight.size(3)),
            convolution_backward_bias(grad_output)};
}
} // namespace pytorch_vulkan
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("convolution", &pytorch_vulkan::convolution);
    m.impl("convolution_overrideable", &pytorch_vulkan::convolution);
    m.impl("convolution_backward", &pytorch_vulkan::convolution_backward);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("convolution", &pytorch_vulkan::autograd_convolution);
    m.impl("convolution_overrideable", &pytorch_vulkan::autograd_convolution);
}
