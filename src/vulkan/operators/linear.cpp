#include "linear.h"

#include "autograd.h"
#include "capability.h"
#include "fake_tensor.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "unary.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <ATen/ops/linear.h>
#include <torch/library.h>

#include <limits>
#include <vector>

namespace pytorch_vulkan {
namespace {
VkDeviceSize checked_bytes(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.numel() >= 0, "Vulkan ", name, " has a negative element count");
    const uint64_t elements = static_cast<uint64_t>(tensor.numel());
    TORCH_CHECK(elements <= std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan ", name, " byte count overflows uint64_t");
    const uint64_t bytes = elements * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() &&
                    bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan ", name, " byte count exceeds supported size");
    return static_cast<VkDeviceSize>(bytes);
}

bool layouts_overlap(const at::Tensor &lhs, const VulkanTensorLayout &lhs_layout,
                     const at::Tensor &rhs, const VulkanTensorLayout &rhs_layout) {
    if (lhs.storage().data_ptr().get_context() != rhs.storage().data_ptr().get_context() ||
        lhs_layout.byte_range == 0 || rhs_layout.byte_range == 0)
        return false;
    const uint64_t lhs_end = static_cast<uint64_t>(lhs_layout.byte_offset) +
                             static_cast<uint64_t>(lhs_layout.byte_range);
    const uint64_t rhs_end = static_cast<uint64_t>(rhs_layout.byte_offset) +
                             static_cast<uint64_t>(rhs_layout.byte_range);
    return static_cast<uint64_t>(lhs_layout.byte_offset) < rhs_end &&
           static_cast<uint64_t>(rhs_layout.byte_offset) < lhs_end;
}
} // namespace

at::Tensor lower_linear(const at::Tensor &input, const at::Tensor &weight,
                        const c10::optional<at::Tensor> &bias,
                        bool transposed_weight = false, at::Tensor *out = nullptr,
                        uint32_t operation = 0) {
    if (pytorch_vulkan::is_fake_tensor(input)) {
        return at::_ops::linear::redispatch(
            c10::DispatchKeySet(c10::DispatchKey::Meta), input, weight, bias);
    }
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.device().index() == 0,
                "Vulkan linear requires Vulkan device index 0");
    TORCH_CHECK(weight.device() == input.device(),
                "Vulkan linear requires weight on the input device");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "Vulkan linear supports only float32");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 &&
                    input.size(1) ==
                        (transposed_weight ? weight.size(0) : weight.size(1)),
                "Vulkan linear supports 2-D input and weight with matching features");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "linear input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "linear weight");
    at::Tensor b = bias.has_value() ? *bias : at::empty({1}, input.options());
    TORCH_CHECK(
        !bias.has_value() ||
            (b.device() == input.device() && b.scalar_type() == at::kFloat &&
             b.dim() == 1 &&
             b.size(0) == (transposed_weight ? weight.size(1) : weight.size(0)) &&
              b.layout() == at::kStrided),
         "Vulkan linear requires a strided float32 bias with out_features elements");
    const auto bias_layout = inspect_vulkan_tensor_layout(b, "linear bias");
    const int64_t outputs = transposed_weight ? weight.size(1) : weight.size(0);
    at::Tensor output =
        out ? *out : at::empty({input.size(0), outputs}, input.options());
    TORCH_CHECK(output.device() == input.device() &&
                    output.scalar_type() == at::kFloat &&
                    output.sizes().equals({input.size(0), outputs}) &&
                    output.layout() == at::kStrided,
                 "Vulkan linear output requires matching strided float32 metadata");
    const auto output_layout = inspect_vulkan_tensor_layout(output, "linear output");
    TORCH_CHECK(output_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan linear output has unsupported overlap");
    TORCH_CHECK(!layouts_overlap(input, input_layout, output, output_layout) &&
                    !layouts_overlap(weight, weight_layout, output, output_layout) &&
                    !layouts_overlap(b, bias_layout, output, output_layout),
                "Vulkan linear output may not alias input, weight, or bias");
    const auto &in_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &bias_data = b.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(in_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(bias_data) &&
                    &platform == &allocation_platform(out_data),
                "Vulkan linear requires one Vulkan platform");
    const VkDeviceSize input_bytes = checked_bytes(input, "linear input");
    const VkDeviceSize weight_bytes = checked_bytes(weight, "linear weight");
    const VkDeviceSize bias_bytes = checked_bytes(b, "linear bias");
    const VkDeviceSize output_bytes = checked_bytes(output, "linear output");
    validate_allocation(in_data, input_bytes, "linear input");
    validate_allocation(weight_data, weight_bytes, "linear weight");
    validate_allocation(bias_data, bias_bytes, "linear bias");
    validate_allocation(out_data, output_bytes, "linear output");
    TORCH_CHECK(input.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    input.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    outputs <= std::numeric_limits<uint32_t>::max(),
                "Vulkan linear dimensions exceed dispatch limits");
    platform.compute().linear(
        allocation_buffer(in_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(bias_data).buffer(), allocation_buffer(out_data).buffer(),
        input_layout, weight_layout, bias_layout, output_layout,
        static_cast<uint32_t>(input.size(0)), static_cast<uint32_t>(input.size(1)),
        static_cast<uint32_t>(outputs), transposed_weight,
        bias.has_value(), operation);
    return output;
}

at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias) {
    return lower_linear(input, weight, bias);
}

at::Tensor linear_relu(const at::Tensor &input, const at::Tensor &weight,
                       const at::Tensor &bias) {
    TORCH_CHECK(bias.defined(), "Vulkan fused linear_relu requires a bias");
    return lower_linear(input, weight, c10::optional<at::Tensor>(bias), false, nullptr, 3);
}

namespace {
void validate_gradient_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan linear ", name, " requires Vulkan device index 0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.dim() == 2,
                "Vulkan linear ", name, " requires a float32 2-D tensor");
}

at::Tensor linear_gradient(const at::Tensor &input, const at::Tensor &weight,
                           int64_t rows, int64_t features, int64_t outputs,
                           uint32_t operation) {
    validate_gradient_tensor(input, "gradient input");
    validate_gradient_tensor(weight, "gradient weight");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "linear gradient input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "linear gradient weight");
    if (operation == 1) {
        TORCH_CHECK(input.size(0) == rows && input.size(1) == features &&
                        weight.size(0) == features && weight.size(1) == outputs,
                    "Vulkan linear gradient dimensions do not match");
    } else {
        TORCH_CHECK(input.size(0) == features && input.size(1) == rows &&
                        weight.size(0) == features && weight.size(1) == outputs,
                    "Vulkan linear gradient dimensions do not match");
    }
    at::Tensor output = at::empty({rows, outputs}, input.options());
    at::Tensor dummy_bias = at::empty({1}, input.options());
    const auto bias_layout = inspect_vulkan_tensor_layout(dummy_bias, "linear gradient bias");
    const auto output_layout = inspect_vulkan_tensor_layout(output, "linear gradient output");
    const auto &input_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &bias_data = dummy_bias.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(bias_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan linear gradients require one Vulkan platform");
    validate_allocation(input_data, checked_bytes(input, "gradient input"),
                        "linear gradient input");
    validate_allocation(weight_data, checked_bytes(weight, "gradient weight"),
                        "linear gradient weight");
    validate_allocation(bias_data, checked_bytes(dummy_bias, "gradient bias"),
                        "linear gradient bias");
    validate_allocation(output_data, checked_bytes(output, "gradient output"),
                        "linear gradient output");
    platform.compute().linear(
        allocation_buffer(input_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(bias_data).buffer(), allocation_buffer(output_data).buffer(),
        input_layout, weight_layout, bias_layout, output_layout,
        static_cast<uint32_t>(rows), static_cast<uint32_t>(features),
        static_cast<uint32_t>(outputs), false, false, operation);
    return output;
}

at::Tensor fused_gradient(const at::Tensor &grad_output, const at::Tensor &rhs,
                          const at::Tensor &activation, std::vector<int64_t> shape,
                          uint32_t operation) {
    validate_gradient_tensor(grad_output, "fused gradient output");
    validate_gradient_tensor(rhs, "fused gradient operand");
    validate_gradient_tensor(activation, "fused activation");
    auto go_layout = inspect_vulkan_tensor_layout(grad_output, "fused gradient output");
    auto rhs_layout = inspect_vulkan_tensor_layout(rhs, "fused gradient operand");
    auto activation_layout = inspect_vulkan_tensor_layout(activation, "fused activation");
    at::Tensor output = at::empty(shape, grad_output.options());
    auto output_layout = inspect_vulkan_tensor_layout(output, "fused gradient result");
    const auto &go_data = grad_output.storage().data_ptr();
    const auto &rhs_data = rhs.storage().data_ptr();
    const auto &activation_data = activation.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    auto &platform = allocation_platform(go_data);
    TORCH_CHECK(&platform == &allocation_platform(rhs_data) &&
                    &platform == &allocation_platform(activation_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan fused gradients require one Vulkan platform");
    validate_allocation(go_data, go_layout.allocation_bytes, "fused gradient output");
    validate_allocation(rhs_data, rhs_layout.allocation_bytes, "fused gradient operand");
    validate_allocation(activation_data, activation_layout.allocation_bytes, "fused activation");
    validate_allocation(output_data, output_layout.allocation_bytes, "fused gradient result");
    const uint32_t rows = static_cast<uint32_t>(grad_output.size(0));
    if (operation == 4) {
        platform.compute().linear_relu_backward_input(
            allocation_buffer(go_data).buffer(), allocation_buffer(rhs_data).buffer(),
            allocation_buffer(activation_data).buffer(), allocation_buffer(output_data).buffer(),
            go_layout, rhs_layout, activation_layout, output_layout, rows,
            static_cast<uint32_t>(grad_output.size(1)), static_cast<uint32_t>(rhs.size(1)));
    } else if (operation == 5) {
        platform.compute().linear_relu_backward_weight(
            allocation_buffer(go_data).buffer(), allocation_buffer(rhs_data).buffer(),
            allocation_buffer(activation_data).buffer(), allocation_buffer(output_data).buffer(),
            go_layout, rhs_layout, activation_layout, output_layout, rows,
            static_cast<uint32_t>(rhs.size(1)), static_cast<uint32_t>(grad_output.size(1)));
    } else {
        platform.compute().linear_relu_backward_bias(
            allocation_buffer(go_data).buffer(), allocation_buffer(activation_data).buffer(),
            allocation_buffer(output_data).buffer(), go_layout, activation_layout,
            output_layout, rows, static_cast<uint32_t>(grad_output.size(1)));
    }
    return output;
}
} // namespace

at::Tensor linear_backward_input(const at::Tensor &grad_output,
                                 const at::Tensor &weight, const at::Tensor &input) {
    (void)input;
    return linear_gradient(grad_output, weight, grad_output.size(0),
                           grad_output.size(1), weight.size(1), 1);
}

at::Tensor linear_backward_weight(const at::Tensor &grad_output,
                                  const at::Tensor &input) {
    return linear_gradient(grad_output, input, grad_output.size(1), grad_output.size(0),
                           input.size(1), 2);
}

at::Tensor linear_backward_bias(const at::Tensor &grad_output) {
    return at::sum(grad_output, {0});
}

at::Tensor linear_relu_backward_input(const at::Tensor &grad_output,
                                      const at::Tensor &weight,
                                      const at::Tensor &activation) {
    return fused_gradient(grad_output, weight, activation,
                          {grad_output.size(0), weight.size(1)}, 4);
}
at::Tensor linear_relu_backward_weight(const at::Tensor &grad_output,
                                       const at::Tensor &input,
                                       const at::Tensor &activation) {
    return fused_gradient(grad_output, input, activation,
                          {grad_output.size(1), input.size(1)}, 5);
}
at::Tensor linear_relu_backward_bias(const at::Tensor &grad_output,
                                      const at::Tensor &activation) {
    auto masked = relu_backward_tensor(activation, grad_output);
    return at::sum(masked, {0});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_relu_backward(
    const at::Tensor &grad_output, const at::Tensor &input,
    const at::Tensor &weight, const at::Tensor &activation) {
    validate_gradient_tensor(grad_output, "backward gradient output");
    validate_gradient_tensor(input, "backward input");
    validate_gradient_tensor(weight, "backward weight");
    validate_gradient_tensor(activation, "backward activation");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous() &&
                    weight.is_contiguous() && activation.is_contiguous(),
                "Vulkan fused backward requires contiguous float32 tensors");
    TORCH_CHECK(grad_output.sizes().equals(activation.sizes()) &&
                    input.dim() == 2 && weight.dim() == 2 &&
                    grad_output.size(0) == input.size(0) &&
                    grad_output.size(1) == weight.size(0) &&
                    input.size(1) == weight.size(1),
                "Vulkan fused backward dimensions do not match");
    const int64_t rows = input.size(0);
    const int64_t features = input.size(1);
    const int64_t outputs = weight.size(0);
    TORCH_CHECK(rows <= std::numeric_limits<uint32_t>::max() &&
                    features <= std::numeric_limits<uint32_t>::max() &&
                    outputs <= std::numeric_limits<uint32_t>::max(),
                "Vulkan fused backward dimensions exceed dispatch limits");
    at::Tensor d_input = at::empty({rows, features}, input.options());
    at::Tensor d_weight = at::empty({outputs, features}, input.options());
    at::Tensor d_bias = at::empty({outputs}, input.options());
    const auto go_layout = inspect_vulkan_tensor_layout(grad_output, "backward gradient output");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "backward input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "backward weight");
    const auto activation_layout = inspect_vulkan_tensor_layout(activation, "backward activation");
    const auto d_input_layout = inspect_vulkan_tensor_layout(d_input, "backward dInput");
    const auto d_weight_layout = inspect_vulkan_tensor_layout(d_weight, "backward dWeight");
    const auto d_bias_layout = inspect_vulkan_tensor_layout(d_bias, "backward dBias");
    const auto &go_data = grad_output.storage().data_ptr();
    const auto &input_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &activation_data = activation.storage().data_ptr();
    const auto &d_input_data = d_input.storage().data_ptr();
    const auto &d_weight_data = d_weight.storage().data_ptr();
    const auto &d_bias_data = d_bias.storage().data_ptr();
    const auto &platform = allocation_platform(go_data);
    TORCH_CHECK(&platform == &allocation_platform(input_data) &&
                    &platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(activation_data) &&
                    &platform == &allocation_platform(d_input_data) &&
                    &platform == &allocation_platform(d_weight_data) &&
                    &platform == &allocation_platform(d_bias_data),
                "Vulkan fused backward requires one Vulkan platform");
    validate_allocation(go_data, go_layout.allocation_bytes, "backward gradient output");
    validate_allocation(input_data, input_layout.allocation_bytes, "backward input");
    validate_allocation(weight_data, weight_layout.allocation_bytes, "backward weight");
    validate_allocation(activation_data, activation_layout.allocation_bytes, "backward activation");
    validate_allocation(d_input_data, d_input_layout.allocation_bytes, "backward dInput");
    validate_allocation(d_weight_data, d_weight_layout.allocation_bytes, "backward dWeight");
    validate_allocation(d_bias_data, d_bias_layout.allocation_bytes, "backward dBias");
    auto &compute = platform.compute();
    compute.linear_relu_backward(
        &allocation_buffer(go_data), go_layout, &allocation_buffer(input_data), input_layout,
        &allocation_buffer(weight_data), weight_layout, &allocation_buffer(activation_data),
        activation_layout, &allocation_buffer(d_input_data), d_input_layout,
        &allocation_buffer(d_weight_data), d_weight_layout, &allocation_buffer(d_bias_data),
        d_bias_layout, static_cast<uint32_t>(rows), static_cast<uint32_t>(features),
        static_cast<uint32_t>(outputs));
    return {d_input, d_weight, d_bias};
}

at::Tensor addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
                 const at::Scalar &beta, const at::Scalar &alpha) {
    TORCH_CHECK(beta.toDouble() == 1.0 && alpha.toDouble() == 1.0,
                "Vulkan addmm supports only beta == 1 and alpha == 1");
    TORCH_CHECK(self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2 &&
                    mat1.size(1) == mat2.size(0) && self.size(0) == mat2.size(1),
                "Vulkan addmm supports a 1-D bias and matching 2-D matrices");
    return lower_linear(mat1, mat2, self, true);
}

at::Tensor &addmm_out(const at::Tensor &self, const at::Tensor &mat1,
                      const at::Tensor &mat2, const at::Scalar &beta,
                      const at::Scalar &alpha, at::Tensor &out) {
    TORCH_CHECK(beta.toDouble() == 1.0 && alpha.toDouble() == 1.0,
                "Vulkan addmm supports only beta == 1 and alpha == 1");
    lower_linear(mat1, mat2, self, true, &out);
    return out;
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("linear", &pytorch_vulkan::linear);
    m.impl("addmm", &pytorch_vulkan::addmm);
    m.impl("addmm.out", &pytorch_vulkan::addmm_out);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("linear", &pytorch_vulkan::autograd_linear);
}

TORCH_LIBRARY(pytorch_vulkan, m) {
    m.def("linear_relu(Tensor input, Tensor weight, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(pytorch_vulkan, PrivateUse1, m) {
    m.impl("linear_relu", &pytorch_vulkan::linear_relu);
}

TORCH_LIBRARY_IMPL(pytorch_vulkan, AutogradPrivateUse1, m) {
    m.impl("linear_relu", &pytorch_vulkan::autograd_linear_relu);
}
