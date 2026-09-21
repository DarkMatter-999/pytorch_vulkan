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
at::Tensor run(const at::Tensor &input, const at::Tensor &weight,
               const at::Tensor &bias, uint32_t operation) {
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
    bool cnn = false;
    if (operation == 0) {
        cnn = input.size(1) == 3 && input.size(2) == 32 && input.size(3) == 32 &&
              weight.sizes().equals({8, 3, 3, 3}) && bias.sizes().equals({8});
        bool legacy = input.sizes().equals({2, 1, 8, 8}) &&
                      weight.sizes().equals({4, 1, 3, 3}) && bias.sizes().equals({4});
        TORCH_CHECK(cnn || legacy,
                    "Vulkan convolution input has an unsupported fixed shape");
    } else if (operation == 1) {
        cnn = input.size(1) == 8 && input.size(2) == 32 && input.size(3) == 32 &&
              weight.sizes().equals({8, 3, 3, 3});
        bool legacy =
            input.sizes().equals({2, 4, 8, 8}) && weight.sizes().equals({4, 1, 3, 3});
        TORCH_CHECK(cnn || legacy,
                    "Vulkan convolution backward grad has unsupported fixed shape");
    } else if (operation == 2) {
        cnn = input.size(1) == 8 && input.size(2) == 32 && input.size(3) == 32 &&
              weight.sizes().equals({input.size(0), 3, 32, 32});
        bool legacy =
            input.sizes().equals({2, 4, 8, 8}) && weight.sizes().equals({2, 1, 8, 8});
        TORCH_CHECK(cnn || legacy,
                    "Vulkan convolution backward input has unsupported fixed shape");
    } else {
        cnn = input.size(1) == 8 && input.size(2) == 32 && input.size(3) == 32;
        bool legacy = input.sizes().equals({2, 4, 8, 8});
        TORCH_CHECK(cnn || legacy,
                    "Vulkan convolution backward grad has unsupported fixed shape");
    }
    if (cnn)
        TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                        bias.is_contiguous(),
                    "Vulkan convolution CNN schema requires contiguous layout");
    int64_t channels = cnn ? 8 : 4;
    if (operation == 0)
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == channels,
                    "Vulkan convolution bias has a fixed shape");
    at::Tensor output =
        operation == 0
            ? at::empty({input.size(0), weight.size(0), input.size(2), input.size(3)},
                        input.options())
        : operation == 1
            ? at::empty({input.size(0), weight.size(1), input.size(2), input.size(3)},
                        input.options())
        : operation == 2
            ? at::empty({input.size(1), weight.size(1), 3, 3}, input.options())
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
        input_layout, weight_layout, bias_layout, output_layout, operation);
    return output;
}
} // namespace
at::Tensor convolution(const at::Tensor &input, const at::Tensor &weight,
                       const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       bool transposed, at::IntArrayRef output_padding,
                       int64_t groups) {
    TORCH_CHECK(bias.has_value(), "Vulkan convolution requires bias");
    TORCH_CHECK(stride.equals({1, 1}) && padding.equals({1, 1}) &&
                    dilation.equals({1, 1}) && output_padding.equals({0, 0}) &&
                    !transposed && groups == 1,
                "Vulkan convolution supports only fixed stride, padding, dilation, "
                "groups, and non-transposed parameters");
    return run(input, weight, *bias, 0);
}
at::Tensor convolution_backward_input(const at::Tensor &grad,
                                      const at::Tensor &weight) {
    return run(grad, weight, grad, 1);
}
at::Tensor convolution_backward_weight(const at::Tensor &grad,
                                       const at::Tensor &input) {
    return run(grad, input, grad, 2);
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
    TORCH_CHECK(stride.equals({1, 1}) && padding.equals({1, 1}) &&
                    dilation.equals({1, 1}) && output_padding.equals({0, 0}) &&
                    !transposed && groups == 1,
                "Vulkan convolution backward supports only fixed stride, padding, "
                "dilation, groups, and non-transposed parameters");
    TORCH_CHECK(output_mask[0] && output_mask[1] && output_mask[2],
                "Vulkan convolution backward requires output_mask [true, true, true]");
    bool cnn = grad_output.dim() == 4 && input.dim() == 4 && weight.dim() == 4 &&
               grad_output.size(0) == input.size(0) &&
               grad_output.sizes().equals({input.size(0), 8, 32, 32}) &&
               input.sizes().equals({input.size(0), 3, 32, 32}) &&
               weight.sizes().equals({8, 3, 3, 3});
    bool legacy = grad_output.sizes().equals({2, 4, 8, 8}) &&
                  input.sizes().equals({2, 1, 8, 8}) &&
                  weight.sizes().equals({4, 1, 3, 3});
    TORCH_CHECK(cnn || legacy,
                "Vulkan convolution backward has unsupported fixed shape");
    TORCH_CHECK(bias_sizes.has_value() &&
                    (cnn ? bias_sizes->equals({8}) : bias_sizes->equals({4})),
                "Vulkan convolution backward bias shape does not match schema");
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
    if (cnn)
        TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous() &&
                        weight.is_contiguous(),
                    "Vulkan convolution CNN schema requires contiguous layout");
    TORCH_CHECK(grad_output.numel() > 0 && input.numel() > 0 && weight.numel() > 0,
                "Vulkan convolution backward rejects empty tensors");
    return {convolution_backward_input(grad_output, weight),
            convolution_backward_weight(grad_output, input),
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
