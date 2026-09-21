#include "linear.h"

#include "autograd.h"
#include "binary.h"
#include "capability.h"
#include "fake_tensor.h"
#include "unary.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <ATen/ops/linear.h>
#include <ATen/ops/mm.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cmath>
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

float checked_scalar(const at::Scalar &scalar, const char *name) {
    TORCH_CHECK(!scalar.isComplex(), "Vulkan ", name, " requires a real scalar");
    const double value = scalar.toDouble();
    const float result = static_cast<float>(value);
    TORCH_CHECK(std::isfinite(value) && std::isfinite(result), "Vulkan ", name,
                " must be finite and representable as float32");
    return result;
}

bool layouts_overlap(const at::Tensor &lhs, const VulkanTensorLayout &lhs_layout,
                     const at::Tensor &rhs, const VulkanTensorLayout &rhs_layout) {
    if (lhs.storage().data_ptr().get_context() !=
            rhs.storage().data_ptr().get_context() ||
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
        return at::_ops::linear::redispatch(c10::DispatchKeySet(c10::DispatchKey::Meta),
                                            input, weight, bias);
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
        static_cast<uint32_t>(outputs), transposed_weight, bias.has_value(), operation);
    return output;
}

at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias) {
    if (!is_fake_tensor(input) && input.dim() == 3) {
        TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                        input.device().index() == 0 && weight.device() == input.device() &&
                        input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                        input.layout() == at::kStrided && weight.layout() == at::kStrided &&
                        input.is_contiguous() && weight.dim() == 2 && weight.is_contiguous() &&
                        input.size(2) == weight.size(1),
                    "Vulkan 3-D linear requires matching contiguous float32 Vulkan tensors");
        auto input_layout = inspect_vulkan_tensor_layout(input, "3-D linear input");
        auto weight_layout = inspect_vulkan_tensor_layout(weight, "3-D linear weight");
        TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No &&
                        weight_layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan 3-D linear rejects overlapping inputs");
        validate_allocation(input.storage().data_ptr(), input_layout.byte_range,
                            "3-D linear input");
        validate_allocation(weight.storage().data_ptr(), weight_layout.byte_range,
                            "3-D linear weight");
        if (bias.has_value()) {
            TORCH_CHECK(bias->device() == input.device() && bias->scalar_type() == at::kFloat &&
                            bias->layout() == at::kStrided && bias->dim() == 1 &&
                            bias->is_contiguous() && bias->size(0) == weight.size(0),
                        "Vulkan 3-D linear requires a matching contiguous bias");
        }
        auto flattened = input.reshape({input.size(0) * input.size(1), input.size(2)});
        return pytorch_vulkan::linear(flattened, weight, bias).reshape(
            {input.size(0), input.size(1), weight.size(0)});
    }
    if (!is_fake_tensor(input) && input.layout() == at::kStrided &&
        weight.layout() == at::kStrided && input.scalar_type() == at::kFloat &&
        weight.scalar_type() == at::kFloat && input.dim() == 2 && weight.dim() == 2 &&
        input.is_contiguous() && weight.is_contiguous() &&
        input.device() == weight.device() && input.device().index() == 0 &&
        (!bias.has_value() ||
         (bias->layout() == at::kStrided && bias->scalar_type() == at::kFloat &&
          bias->dim() == 1 && bias->is_contiguous() &&
          bias->size(0) == weight.size(0) && bias->device() == input.device()))) {
        const int64_t m = input.size(0);
        const int64_t k = input.size(1);
        const int64_t n = weight.size(0);
        if (weight.size(1) == k && m > 0 && n > 0 && k > 0) {
            at::Tensor output = at::empty({m, n}, input.options());
            at::Tensor b = bias.has_value() ? *bias : at::empty({1}, input.options());
            // Materialize the transposed weight with a single strided-gather
            // broadcast dispatch. A plain `weight.t().contiguous()` routes
            // through the generic tensor copy, which issues one synchronous
            // Vulkan submit per element (~130us each, ~3.4s for a 32x784
            // weight). This path is below autograd (see
            // LinearAutogradFunction, which saves the original input/weight),
            // so a raw dispatch keeps the gradient contract intact.
            at::Tensor b_matrix = at::empty({k, n}, input.options());
            const auto a_layout = validate_gemm_2d(input, {m, k}, "linear input");
            const auto b_layout = validate_gemm_2d(b_matrix, {k, n}, "linear weight");
            const auto c_layout = validate_gemm_2d(output, {m, n}, "linear output");
            const auto bias_layout =
                bias.has_value()
                    ? inspect_vulkan_tensor_layout(b, "linear bias")
                    : inspect_vulkan_tensor_layout(b, "linear ignored bias");
            const auto &platform = allocation_platform(input.storage().data_ptr());
            validate_allocation(b.storage().data_ptr(), bias_layout.byte_range,
                                "linear bias");
            validate_allocation(b_matrix.storage().data_ptr(), b_layout.byte_range,
                                "linear weight");
            validate_allocation(output.storage().data_ptr(), c_layout.byte_range,
                                "linear output");
            TORCH_CHECK(
                &platform == &allocation_platform(b_matrix.storage().data_ptr()) &&
                    &platform == &allocation_platform(b.storage().data_ptr()) &&
                    &platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan linear requires one Vulkan platform");
            TORCH_CHECK(!layouts_overlap(input, a_layout, output, c_layout) &&
                            !layouts_overlap(b_matrix, b_layout, output, c_layout) &&
                            !layouts_overlap(b, bias_layout, output, c_layout),
                        "Vulkan linear output may not alias a GEMM input");
            {
                at::Tensor weight_t = weight.t();
                const auto src_layout =
                    inspect_vulkan_tensor_layout(weight_t, "linear weight transpose");
                TORCH_CHECK(src_layout.numel == k * n,
                            "Vulkan linear transpose element count mismatch");
                const uint64_t elements =
                    static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
                TORCH_CHECK(elements <= std::numeric_limits<uint32_t>::max(),
                            "Vulkan linear transpose exceeds dispatch limits");
                TORCH_CHECK(&platform ==
                                &allocation_platform(weight.storage().data_ptr()),
                            "Vulkan linear requires one Vulkan platform");
                platform.compute().broadcast(
                    allocation_buffer(weight.storage().data_ptr()).buffer(), src_layout,
                    allocation_buffer(b_matrix.storage().data_ptr()).buffer(), b_layout,
                    static_cast<uint32_t>(elements), 1.0F);
            }
            platform.compute().gemm(
                allocation_buffer(input.storage().data_ptr()).buffer(), a_layout,
                allocation_buffer(b_matrix.storage().data_ptr()).buffer(), b_layout,
                VK_NULL_HANDLE, c_layout,
                allocation_buffer(output.storage().data_ptr()).buffer(), c_layout,
                bias.has_value() ? allocation_buffer(b.storage().data_ptr()).buffer()
                                 : VK_NULL_HANDLE,
                bias.has_value() ? bias_layout : c_layout, static_cast<uint32_t>(m),
                static_cast<uint32_t>(n), static_cast<uint32_t>(k), 1.0F, 0.0F,
                bias.has_value());
            return output;
        }
    }
    return lower_linear(input, weight, bias);
}

at::Tensor mm(const at::Tensor &mat1, const at::Tensor &mat2) {
    if (is_fake_tensor(mat1))
        return at::_ops::mm::redispatch(c10::DispatchKeySet(c10::DispatchKey::Meta),
                                        mat1, mat2);
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2 && mat1.size(1) == mat2.size(0),
                "Vulkan mm requires matching 2-D matrices");
    TORCH_CHECK(mat1.device() == mat2.device() && mat1.scalar_type() == at::kFloat &&
                    mat2.scalar_type() == at::kFloat && mat1.layout() == at::kStrided &&
                    mat2.layout() == at::kStrided && mat1.is_contiguous() &&
                    mat2.is_contiguous(),
                "Vulkan mm requires matching contiguous float32 Vulkan matrices");
    const int64_t m = mat1.size(0), k = mat1.size(1), n = mat2.size(1);
    at::Tensor output = at::empty({m, n}, mat1.options());
    if (m == 0 || n == 0)
        return output;
    if (k == 0) {
        const auto output_layout = inspect_vulkan_tensor_layout(output, "mm output");
        const auto &platform = allocation_platform(output.storage().data_ptr());
        validate_allocation(output.storage().data_ptr(), output_layout.allocation_bytes,
                            "mm output");
        platform.compute().fill_alias(
            allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
            allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
            0.0F);
        return output;
    }
    const auto a_layout = validate_gemm_2d(mat1, {m, k}, "mm mat1");
    const auto b_layout = validate_gemm_2d(mat2, {k, n}, "mm mat2");
    const auto out_layout = validate_gemm_2d(output, {m, n}, "mm output");
    const auto &platform = allocation_platform(mat1.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()) &&
                    &platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan mm requires one Vulkan platform");
    TORCH_CHECK(!layouts_overlap(mat1, a_layout, output, out_layout) &&
                    !layouts_overlap(mat2, b_layout, output, out_layout),
                "Vulkan mm output may not alias an input");
    platform.compute().gemm(
        allocation_buffer(mat1.storage().data_ptr()).buffer(), a_layout,
        allocation_buffer(mat2.storage().data_ptr()).buffer(), b_layout, VK_NULL_HANDLE,
        out_layout, allocation_buffer(output.storage().data_ptr()).buffer(), out_layout,
        VK_NULL_HANDLE, out_layout, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(k));
    return output;
}

at::Tensor bmm(const at::Tensor &mat1, const at::Tensor &mat2) {
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3 && mat1.size(0) == mat2.size(0) &&
                    mat1.size(2) == mat2.size(1),
                "Vulkan bmm requires matching 3-D batch matrices");
    TORCH_CHECK(mat1.scalar_type() == at::kFloat && mat2.scalar_type() == at::kFloat &&
                    mat1.device() == mat2.device() && mat1.device().index() == 0,
                "Vulkan bmm requires matching float32 Vulkan matrices");
    TORCH_CHECK(mat1.size(0) > 0 && mat1.size(1) > 0 && mat1.size(2) > 0 &&
                    mat2.size(2) > 0,
                "Vulkan bmm requires non-empty matrices");
    TORCH_CHECK(mat1.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    mat1.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    mat1.size(2) <= std::numeric_limits<uint32_t>::max() &&
                    mat2.size(2) <= std::numeric_limits<uint32_t>::max(),
                "Vulkan bmm dimensions exceed dispatch limits");
    TORCH_CHECK(mat1.layout() == at::kStrided && mat2.layout() == at::kStrided,
                "Vulkan bmm requires strided matrices");
    const auto mat1_layout = inspect_vulkan_tensor_layout(mat1, "bmm mat1");
    const auto mat2_layout = inspect_vulkan_tensor_layout(mat2, "bmm mat2");
    TORCH_CHECK(mat1_layout.internal_overlap == VulkanOverlap::No &&
                    mat2_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan bmm rejects overlapping inputs");
    validate_allocation(mat1.storage().data_ptr(), mat1_layout.byte_range, "bmm mat1");
    validate_allocation(mat2.storage().data_ptr(), mat2_layout.byte_range, "bmm mat2");
    auto output = at::empty({mat1.size(0), mat1.size(1), mat2.size(2)}, mat1.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "bmm output");
    validate_allocation(output.storage().data_ptr(), output_layout.byte_range, "bmm output");
    auto validate_batched_matrix = [](const VulkanTensorLayout &layout, int64_t rows,
                                      int64_t cols, const char *name) {
        TORCH_CHECK(layout.rank == 3 && layout.sizes[1] == rows && layout.sizes[2] == cols &&
                        layout.strides[1] >= 0 && layout.strides[2] >= 0 &&
                        ((layout.strides[1] == cols && layout.strides[2] == 1) ||
                         (layout.strides[1] == 1 && layout.strides[2] == rows)),
                    "Vulkan ", name,
                    " requires a contiguous or transposed-contiguous batched layout");
        TORCH_CHECK(layout.strides[0] >= 0 &&
                        static_cast<uint64_t>(layout.strides[0]) <=
                            std::numeric_limits<uint32_t>::max() &&
                        static_cast<uint64_t>(layout.strides[1]) <=
                            std::numeric_limits<uint32_t>::max() &&
                        static_cast<uint64_t>(layout.strides[2]) <=
                            std::numeric_limits<uint32_t>::max(),
                    "Vulkan ", name, " strides exceed dispatch limits");
        TORCH_CHECK(layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan ", name, " has unsupported internal overlap");
    };
    validate_batched_matrix(mat1_layout, mat1.size(1), mat1.size(2), "bmm lhs");
    validate_batched_matrix(mat2_layout, mat2.size(1), mat2.size(2), "bmm rhs");
    validate_batched_matrix(output_layout, mat1.size(1), mat2.size(2), "bmm output");
    const auto &platform = allocation_platform(mat1.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()) &&
                    &platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan bmm requires one Vulkan platform");
    platform.compute().gemm(
        allocation_buffer(mat1.storage().data_ptr()).buffer(), mat1_layout,
        allocation_buffer(mat2.storage().data_ptr()).buffer(), mat2_layout, VK_NULL_HANDLE,
        output_layout, allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
        VK_NULL_HANDLE, output_layout, static_cast<uint32_t>(mat1.size(1)),
        static_cast<uint32_t>(mat2.size(2)), static_cast<uint32_t>(mat1.size(2)),
        1.0F, 0.0F, false, static_cast<uint32_t>(mat1.size(0)),
        static_cast<uint32_t>(mat1_layout.strides[0]),
        static_cast<uint32_t>(mat2_layout.strides[0]),
        static_cast<uint32_t>(output_layout.strides[0]),
        static_cast<uint32_t>(output_layout.strides[0]));
    return output;
}

at::Tensor transpose_contiguous_2d(const at::Tensor &input) {
    TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kFloat &&
                    input.layout() == at::kStrided && input.is_contiguous(),
                "Vulkan GEMM transpose requires a contiguous float32 2-D tensor");
    at::Tensor output = at::empty({input.size(1), input.size(0)}, input.options());
    if (input.numel() == 0)
        return output;

    const at::Tensor input_transposed = input.t();
    const auto source =
        inspect_vulkan_tensor_layout(input_transposed, "GEMM transpose input");
    const auto destination = validate_gemm_2d(output, {input.size(1), input.size(0)},
                                              "GEMM transpose output");
    const auto &platform = allocation_platform(input.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan GEMM transpose requires one Vulkan platform");
    validate_allocation(input.storage().data_ptr(), source.allocation_bytes,
                        "GEMM transpose input");
    validate_allocation(output.storage().data_ptr(), destination.allocation_bytes,
                        "GEMM transpose output");
    TORCH_CHECK(input.numel() <= std::numeric_limits<uint32_t>::max(),
                "Vulkan GEMM transpose exceeds dispatch limits");
    platform.compute().broadcast(
        allocation_buffer(input.storage().data_ptr()).buffer(), source,
        allocation_buffer(output.storage().data_ptr()).buffer(), destination,
        static_cast<uint32_t>(input.numel()), 1.0F);
    return output;
}

at::Tensor linear_relu(const at::Tensor &input, const at::Tensor &weight,
                       const at::Tensor &bias) {
    TORCH_CHECK(bias.defined(), "Vulkan fused linear_relu requires a bias");
    return lower_linear(input, weight, c10::optional<at::Tensor>(bias), false, nullptr,
                        3);
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
    const auto input_layout =
        inspect_vulkan_tensor_layout(input, "linear gradient input");
    const auto weight_layout =
        inspect_vulkan_tensor_layout(weight, "linear gradient weight");
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
    const auto bias_layout =
        inspect_vulkan_tensor_layout(dummy_bias, "linear gradient bias");
    const auto output_layout =
        inspect_vulkan_tensor_layout(output, "linear gradient output");
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
    auto activation_layout =
        inspect_vulkan_tensor_layout(activation, "fused activation");
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
    validate_allocation(rhs_data, rhs_layout.allocation_bytes,
                        "fused gradient operand");
    validate_allocation(activation_data, activation_layout.allocation_bytes,
                        "fused activation");
    validate_allocation(output_data, output_layout.allocation_bytes,
                        "fused gradient result");
    const uint32_t rows = static_cast<uint32_t>(grad_output.size(0));
    if (operation == 4) {
        platform.compute().linear_relu_backward_input(
            allocation_buffer(go_data).buffer(), allocation_buffer(rhs_data).buffer(),
            allocation_buffer(activation_data).buffer(),
            allocation_buffer(output_data).buffer(), go_layout, rhs_layout,
            activation_layout, output_layout, rows,
            static_cast<uint32_t>(grad_output.size(1)),
            static_cast<uint32_t>(rhs.size(1)));
    } else if (operation == 5) {
        platform.compute().linear_relu_backward_weight(
            allocation_buffer(go_data).buffer(), allocation_buffer(rhs_data).buffer(),
            allocation_buffer(activation_data).buffer(),
            allocation_buffer(output_data).buffer(), go_layout, rhs_layout,
            activation_layout, output_layout, rows, static_cast<uint32_t>(rhs.size(1)),
            static_cast<uint32_t>(grad_output.size(1)));
    } else {
        platform.compute().linear_relu_backward_bias(
            allocation_buffer(go_data).buffer(),
            allocation_buffer(activation_data).buffer(),
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
linear_relu_backward(const at::Tensor &grad_output, const at::Tensor &input,
                     const at::Tensor &weight, const at::Tensor &activation) {
    validate_gradient_tensor(grad_output, "backward gradient output");
    validate_gradient_tensor(input, "backward input");
    validate_gradient_tensor(weight, "backward weight");
    validate_gradient_tensor(activation, "backward activation");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous() &&
                    weight.is_contiguous() && activation.is_contiguous(),
                "Vulkan fused backward requires contiguous float32 tensors");
    TORCH_CHECK(grad_output.sizes().equals(activation.sizes()) && input.dim() == 2 &&
                    weight.dim() == 2 && grad_output.size(0) == input.size(0) &&
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
    const auto go_layout =
        inspect_vulkan_tensor_layout(grad_output, "backward gradient output");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "backward input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "backward weight");
    const auto activation_layout =
        inspect_vulkan_tensor_layout(activation, "backward activation");
    const auto d_input_layout =
        inspect_vulkan_tensor_layout(d_input, "backward dInput");
    const auto d_weight_layout =
        inspect_vulkan_tensor_layout(d_weight, "backward dWeight");
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
    validate_allocation(go_data, go_layout.allocation_bytes,
                        "backward gradient output");
    validate_allocation(input_data, input_layout.allocation_bytes, "backward input");
    validate_allocation(weight_data, weight_layout.allocation_bytes, "backward weight");
    validate_allocation(activation_data, activation_layout.allocation_bytes,
                        "backward activation");
    validate_allocation(d_input_data, d_input_layout.allocation_bytes,
                        "backward dInput");
    validate_allocation(d_weight_data, d_weight_layout.allocation_bytes,
                        "backward dWeight");
    validate_allocation(d_bias_data, d_bias_layout.allocation_bytes, "backward dBias");
    auto &compute = platform.compute();
    compute.linear_relu_backward(
        &allocation_buffer(go_data), go_layout, &allocation_buffer(input_data),
        input_layout, &allocation_buffer(weight_data), weight_layout,
        &allocation_buffer(activation_data), activation_layout,
        &allocation_buffer(d_input_data), d_input_layout,
        &allocation_buffer(d_weight_data), d_weight_layout,
        &allocation_buffer(d_bias_data), d_bias_layout, static_cast<uint32_t>(rows),
        static_cast<uint32_t>(features), static_cast<uint32_t>(outputs));
    return {d_input, d_weight, d_bias};
}

at::Tensor addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
                 const at::Scalar &beta, const at::Scalar &alpha) {
    if (!is_fake_tensor(mat1) && self.dim() == 2 && mat1.dim() == 2 &&
        mat2.dim() == 2 && self.size(0) == mat1.size(0) &&
        self.size(1) == mat2.size(1) && mat1.size(1) == mat2.size(0) &&
        self.is_contiguous() && mat1.is_contiguous() && mat2.is_contiguous() &&
        self.scalar_type() == at::kFloat && mat1.scalar_type() == at::kFloat &&
        mat2.scalar_type() == at::kFloat && self.device() == mat1.device() &&
        mat1.device() == mat2.device()) {
        const int64_t m = mat1.size(0), n = mat2.size(1), k = mat1.size(1);
        const float beta_value = checked_scalar(beta, "addmm beta");
        const float alpha_value = checked_scalar(alpha, "addmm alpha");
        at::Tensor output = at::empty({m, n}, self.options());
        if (m == 0 || n == 0)
            return output;
        if (k == 0) {
            const auto self_layout = inspect_vulkan_tensor_layout(self, "addmm self");
            const auto output_layout =
                inspect_vulkan_tensor_layout(output, "addmm output");
            const auto &platform = allocation_platform(self.storage().data_ptr());
            TORCH_CHECK(&platform == &allocation_platform(output.storage().data_ptr()),
                        "Vulkan addmm requires one Vulkan platform");
            validate_allocation(self.storage().data_ptr(), self_layout.allocation_bytes,
                                "addmm self");
            validate_allocation(output.storage().data_ptr(),
                                output_layout.allocation_bytes, "addmm output");
            if (beta_value == 0.0F) {
                platform.compute().fill_alias(
                    allocation_buffer(output.storage().data_ptr()).buffer(),
                    output_layout,
                    allocation_buffer(output.storage().data_ptr()).buffer(),
                    output_layout, 0.0F);
            } else {
                platform.compute().tensor_scalar(
                    allocation_buffer(self.storage().data_ptr()).buffer(), self_layout,
                    allocation_buffer(output.storage().data_ptr()).buffer(),
                    output_layout, beta_value, 2);
            }
            return output;
        }
        const auto a_layout = validate_gemm_2d(mat1, {m, k}, "addmm mat1");
        const auto b_layout = validate_gemm_2d(mat2, {k, n}, "addmm mat2");
        const auto c_layout = validate_gemm_2d(self, {m, n}, "addmm self");
        const auto out_layout = validate_gemm_2d(output, {m, n}, "addmm output");
        const auto &platform = allocation_platform(mat1.storage().data_ptr());
        TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()) &&
                        &platform == &allocation_platform(self.storage().data_ptr()) &&
                        &platform == &allocation_platform(output.storage().data_ptr()),
                    "Vulkan addmm requires one Vulkan platform");
        TORCH_CHECK(!layouts_overlap(output, out_layout, mat1, a_layout) &&
                        !layouts_overlap(output, out_layout, mat2, b_layout) &&
                        !layouts_overlap(output, out_layout, self, c_layout),
                    "Vulkan addmm output may not alias an input");
        platform.compute().gemm(
            allocation_buffer(mat1.storage().data_ptr()).buffer(), a_layout,
            allocation_buffer(mat2.storage().data_ptr()).buffer(), b_layout,
            allocation_buffer(self.storage().data_ptr()).buffer(), c_layout,
            allocation_buffer(output.storage().data_ptr()).buffer(), out_layout,
            VK_NULL_HANDLE, out_layout, static_cast<uint32_t>(m),
            static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha_value, beta_value,
            false);
        return output;
    }
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1 &&
                    self.device().index() == 0 && mat1.device() == self.device() &&
                    mat2.device() == self.device(),
                "Vulkan addmm requires all operands on Vulkan device index 0");
    TORCH_CHECK(self.layout() == at::kStrided && mat1.layout() == at::kStrided &&
                    mat2.layout() == at::kStrided,
                "Vulkan addmm requires strided operands");
    TORCH_CHECK(self.scalar_type() == at::kFloat && mat1.scalar_type() == at::kFloat &&
                    mat2.scalar_type() == at::kFloat,
                "Vulkan addmm supports only float32 operands");
    TORCH_CHECK(self.dim() == 1 && mat1.dim() == 2 && mat2.dim() == 2 &&
                    mat1.size(1) == mat2.size(0) && self.size(0) == mat2.size(1),
                "Vulkan addmm supports a 1-D bias and matching 2-D matrices");
    const float beta_value = checked_scalar(beta, "addmm beta");
    const float alpha_value = checked_scalar(alpha, "addmm alpha");
    const int64_t m = mat1.size(0), n = mat2.size(1);
    if (m == 0 || n == 0)
        return at::empty({m, n}, self.options());
    if (mat1.size(1) == 0) {
        at::Tensor output = at::empty({m, n}, self.options());
        if (m == 0 || n == 0)
            return output;
        const auto bias_matrix = self.expand({m, n});
        const auto bias_layout =
            inspect_vulkan_tensor_layout(bias_matrix, "addmm bias");
        const auto output_layout = inspect_vulkan_tensor_layout(output, "addmm output");
        const auto &platform = allocation_platform(self.storage().data_ptr());
        TORCH_CHECK(&platform == &allocation_platform(output.storage().data_ptr()),
                    "Vulkan addmm requires one Vulkan platform");
        validate_allocation(self.storage().data_ptr(), bias_layout.allocation_bytes,
                            "addmm bias");
        validate_allocation(output.storage().data_ptr(), output_layout.allocation_bytes,
                            "addmm output");
        platform.compute().tensor_scalar(
            allocation_buffer(bias_matrix.storage().data_ptr()).buffer(), bias_layout,
            allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
            beta_value, 2);
        return output;
    }
    TORCH_CHECK(
        beta_value == 1.0F && alpha_value == 1.0F,
        "Vulkan addmm supports non-default scalars only for contiguous 2-D self");
    return lower_linear(mat1, mat2, self, true);
}

at::Tensor &addmm_out(const at::Tensor &self, const at::Tensor &mat1,
                      const at::Tensor &mat2, const at::Scalar &beta,
                      const at::Scalar &alpha, at::Tensor &out) {
    if (self.dim() == 2 && mat1.dim() == 2 && mat2.dim() == 2 && out.dim() == 2 &&
        self.sizes().equals({mat1.size(0), mat2.size(1)}) &&
        mat1.size(1) == mat2.size(0) && out.sizes().equals(self.sizes()) &&
        self.is_contiguous() && mat1.is_contiguous() && mat2.is_contiguous() &&
        out.is_contiguous() && self.scalar_type() == at::kFloat &&
        mat1.scalar_type() == at::kFloat && mat2.scalar_type() == at::kFloat &&
        out.scalar_type() == at::kFloat) {
        const int64_t m = mat1.size(0), n = mat2.size(1), k = mat1.size(1);
        const auto a_layout = validate_gemm_2d(mat1, {m, k}, "addmm.out mat1");
        const auto b_layout = validate_gemm_2d(mat2, {k, n}, "addmm.out mat2");
        const auto c_layout = validate_gemm_2d(self, {m, n}, "addmm.out self");
        const auto out_layout = validate_gemm_2d(out, {m, n}, "addmm.out output");
        const auto &platform = allocation_platform(mat1.storage().data_ptr());
        TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()) &&
                        &platform == &allocation_platform(self.storage().data_ptr()) &&
                        &platform == &allocation_platform(out.storage().data_ptr()),
                    "Vulkan addmm.out requires one Vulkan platform");
        TORCH_CHECK(!layouts_overlap(out, out_layout, mat1, a_layout) &&
                        !layouts_overlap(out, out_layout, mat2, b_layout) &&
                        !layouts_overlap(out, out_layout, self, c_layout),
                    "Vulkan addmm.out output may not alias an input");
        platform.compute().gemm(
            allocation_buffer(mat1.storage().data_ptr()).buffer(), a_layout,
            allocation_buffer(mat2.storage().data_ptr()).buffer(), b_layout,
            allocation_buffer(self.storage().data_ptr()).buffer(), c_layout,
            allocation_buffer(out.storage().data_ptr()).buffer(), out_layout,
            VK_NULL_HANDLE, out_layout, static_cast<uint32_t>(m),
            static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            checked_scalar(alpha, "addmm.out alpha"),
            checked_scalar(beta, "addmm.out beta"), false);
        return out;
    }
    TORCH_CHECK(
        beta.toDouble() == 1.0 && alpha.toDouble() == 1.0,
        "Vulkan addmm.out supports non-default scalars only for the allocating form");
    lower_linear(mat1, mat2, self, true, &out);
    return out;
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mm", &pytorch_vulkan::mm);
    m.impl("bmm", &pytorch_vulkan::bmm);
    m.impl("linear", &pytorch_vulkan::linear);
    m.impl("addmm", &pytorch_vulkan::addmm);
    m.impl("addmm.out", &pytorch_vulkan::addmm_out);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("linear", &pytorch_vulkan::autograd_linear);
    m.impl("mm", &pytorch_vulkan::autograd_mm);
    m.impl("bmm", &pytorch_vulkan::autograd_bmm);
    m.impl("addmm", &pytorch_vulkan::autograd_addmm);
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
