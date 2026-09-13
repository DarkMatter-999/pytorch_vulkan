#include "linear.h"

#include "autograd.h"
#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

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
} // namespace

at::Tensor lower_linear(const at::Tensor &input, const at::Tensor &weight,
                        const c10::optional<at::Tensor> &bias,
                        bool transposed_weight = false, at::Tensor *out = nullptr) {
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
                "Vulkan linear supports contiguous 2-D input and weight with matching "
                "features");
    const bool packed_transpose = transposed_weight && weight.stride(0) == 1 &&
                                  weight.stride(1) == weight.size(0);
    TORCH_CHECK(input.is_contiguous() && (weight.is_contiguous() || packed_transpose) &&
                    input.storage_offset() == 0 && weight.storage_offset() == 0,
                "Vulkan linear requires contiguous tensors with zero storage offset");
    at::Tensor b = bias.has_value() ? *bias : at::empty({1}, input.options());
    TORCH_CHECK(
        !bias.has_value() ||
            (b.device() == input.device() && b.scalar_type() == at::kFloat &&
             b.dim() == 1 &&
             b.size(0) == (transposed_weight ? weight.size(1) : weight.size(0)) &&
             b.is_contiguous() && b.storage_offset() == 0),
        "Vulkan linear requires a contiguous float32 bias with out_features elements");
    const int64_t outputs = transposed_weight ? weight.size(1) : weight.size(0);
    at::Tensor output =
        out ? *out : at::empty({input.size(0), outputs}, input.options());
    TORCH_CHECK(output.device() == input.device() &&
                    output.scalar_type() == at::kFloat &&
                    output.sizes().equals({input.size(0), outputs}) &&
                    output.is_contiguous() && output.storage_offset() == 0,
                "Vulkan linear output requires matching contiguous float32 metadata");
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
        static_cast<uint32_t>(input.size(0)), static_cast<uint32_t>(input.size(1)),
        static_cast<uint32_t>(outputs), transposed_weight && weight.is_contiguous(),
        bias.has_value());
    return output;
}

at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias) {
    return lower_linear(input, weight, bias);
}

namespace {
void validate_gradient_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan linear ", name, " requires Vulkan device index 0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.dim() == 2 &&
                    tensor.is_contiguous() && tensor.storage_offset() == 0,
                "Vulkan linear ", name, " requires a contiguous float32 2-D tensor");
}

at::Tensor linear_gradient(const at::Tensor &input, const at::Tensor &weight,
                           int64_t rows, int64_t features, int64_t outputs,
                           uint32_t operation) {
    validate_gradient_tensor(input, "gradient input");
    validate_gradient_tensor(weight, "gradient weight");
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
        static_cast<uint32_t>(rows), static_cast<uint32_t>(features),
        static_cast<uint32_t>(outputs), false, false, operation);
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
