#include "linear.h"

#include "autograd.h"
#include "binary.h"
#include "capability.h"
#include "fake_tensor.h"
#include "unary.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <ATen/ops/linear.h>
#include <ATen/ops/mm.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <algorithm>
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

float checked_allocating_addmm_scalar(const at::Scalar &scalar, const char *name) {
    TORCH_CHECK(!scalar.isComplex(), "Vulkan ", name, " requires a real scalar");
    const double value = scalar.toDouble();
    const float result = static_cast<float>(value);
    TORCH_CHECK(!std::isfinite(value) || std::isfinite(result), "Vulkan ", name,
                " value cannot be converted to type float without overflow");
    return result;
}

float converting_k0_addmm_beta(const at::Scalar &scalar) {
    TORCH_CHECK(!scalar.isComplex(), "Vulkan addmm beta requires a real scalar");
    return static_cast<float>(scalar.toDouble());
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

struct MatrixReadPlan {
    at::Tensor source;
    VulkanTensorLayout source_layout;
    std::vector<int64_t> dense_shape;
    bool needs_materialization;
};

VulkanTensorLayout dense_matrix_layout(at::IntArrayRef shape,
                                       const char *operation) {
    TORCH_CHECK((shape.size() == 2 || shape.size() == 3) &&
                    std::all_of(shape.begin(), shape.end(), [](int64_t value) {
                        return value > 0;
                    }),
                "Vulkan ", operation,
                " dense GEMM layout requires non-empty matrix dimensions");
    uint64_t elements = 1;
    for (const int64_t extent : shape) {
        TORCH_CHECK(elements <= std::numeric_limits<uint64_t>::max() /
                                    static_cast<uint64_t>(extent),
                    "Vulkan ", operation, " dense matrix element count overflows");
        elements *= static_cast<uint64_t>(extent);
    }
    TORCH_CHECK(elements <= std::numeric_limits<uint32_t>::max() &&
                    elements <= std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan ", operation, " dense matrix exceeds shader limits");
    const uint64_t bytes = elements * sizeof(float);
    std::vector<int64_t> sizes(shape.begin(), shape.end());
    std::vector<int64_t> strides(shape.size());
    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * sizes[i + 1];
    return {static_cast<uint32_t>(shape.size()),
            std::move(sizes),
            std::move(strides),
            static_cast<int>(at::kFloat),
            sizeof(float),
            0,
            static_cast<int64_t>(elements),
            0,
            bytes,
            bytes,
            VulkanOverlap::No};
}

bool canonical_gemm_read_layout(const VulkanTensorLayout &layout,
                                at::IntArrayRef dense_shape) {
    if ((dense_shape.size() != 2 && dense_shape.size() != 3) ||
        layout.rank != dense_shape.size() || layout.sizes.size() != dense_shape.size() ||
        layout.strides.size() != dense_shape.size() ||
        layout.internal_overlap != VulkanOverlap::No)
        return false;
    const size_t base = dense_shape.size() - 2;
    for (size_t i = 0; i < dense_shape.size(); ++i)
        if (layout.sizes[i] != dense_shape[i])
            return false;
    return layout.strides[base] == dense_shape[base + 1] &&
           layout.strides[base + 1] == 1;
}

MatrixReadPlan plan_matrix_read(const at::Tensor &source,
                                at::IntArrayRef dense_shape,
                                const char *operation) {
    TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1 &&
                    source.device().index() == 0,
                "Vulkan ", operation, " requires Vulkan device index 0");
    TORCH_CHECK(source.layout() == at::kStrided && source.scalar_type() == at::kFloat,
                "Vulkan ", operation, " requires a strided float32 tensor");
    TORCH_CHECK(source.dim() <= 8 && dense_shape.size() <= 8 &&
                    source.dim() <= static_cast<int64_t>(dense_shape.size()),
                "Vulkan ", operation, " supports broadcastable ranks up to 8");

    uint64_t dense_numel = 1;
    bool dense_empty = false;
    for (const int64_t extent : dense_shape) {
        TORCH_CHECK(extent >= 0 &&
                        static_cast<uint64_t>(extent) <=
                            std::numeric_limits<uint32_t>::max(),
                    "Vulkan ", operation, " shape exceeds shader dimension limits");
        if (extent == 0) {
            dense_empty = true;
            dense_numel = 0;
        } else if (!dense_empty) {
            TORCH_CHECK(dense_numel <=
                            std::numeric_limits<uint32_t>::max() /
                                static_cast<uint64_t>(extent),
                        "Vulkan ", operation, " element count exceeds shader limits");
            dense_numel *= static_cast<uint64_t>(extent);
        }
    }
    auto expanded = source.expand(dense_shape);
    auto layout = inspect_vulkan_tensor_layout(expanded, operation);
    TORCH_CHECK(static_cast<uint64_t>(layout.numel) == dense_numel,
                "Vulkan ", operation, " expanded element count is inconsistent");
    if (layout.numel != 0)
        validate_allocation(source.storage().data_ptr(), layout.allocation_bytes, operation);
    const bool direct = layout.numel != 0 &&
                        canonical_gemm_read_layout(layout, dense_shape);
    if (direct && dense_shape.size() == 2)
        (void)validate_gemm_2d(expanded, layout, dense_shape, operation);
    return {std::move(expanded), std::move(layout), dense_shape.vec(), !direct};
}

at::Tensor materialize_matrix_read(const MatrixReadPlan &plan,
                                   const char *operation) {
    if (!plan.needs_materialization || plan.source_layout.numel == 0)
        return plan.source;
    TORCH_CHECK(plan.source_layout.rank <= 8 &&
                    static_cast<uint64_t>(plan.source_layout.numel) <=
                        std::numeric_limits<uint32_t>::max(),
                "Vulkan ", operation, " gather exceeds shader limits");
    const auto dense_preflight = dense_matrix_layout(plan.dense_shape, operation);
    const auto &preflight_platform =
        allocation_platform(plan.source.storage().data_ptr());
    preflight_platform.compute().validate_broadcast_preflight(
        plan.source_layout, dense_preflight,
        static_cast<uint32_t>(plan.source_layout.numel));
    at::Tensor dense = at::empty(plan.dense_shape, plan.source.options());
    auto dense_layout = inspect_vulkan_tensor_layout(dense, operation);
    if (plan.dense_shape.size() == 2)
        (void)validate_gemm_2d(dense, dense_layout, plan.dense_shape, operation);
    const auto &source_data = plan.source.storage().data_ptr();
    const auto &dense_data = dense.storage().data_ptr();
    const auto &platform = allocation_platform(source_data);
    TORCH_CHECK(&platform == &allocation_platform(dense_data), "Vulkan ", operation,
                " requires one Vulkan platform");
    validate_allocation(source_data, plan.source_layout.allocation_bytes, operation);
    validate_allocation(dense_data, dense_layout.allocation_bytes, operation);
    platform.compute().broadcast(
        allocation_buffer(source_data).buffer(), plan.source_layout,
        allocation_buffer(dense_data).buffer(), dense_layout,
        static_cast<uint32_t>(plan.source_layout.numel), 1.0F);
    return dense;
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

at::Tensor mm(const at::Tensor &mat1, const at::Tensor &mat2) {
    if (is_fake_tensor(mat1))
        return at::_ops::mm::redispatch(c10::DispatchKeySet(c10::DispatchKey::Meta),
                                        mat1, mat2);
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2 && mat1.size(1) == mat2.size(0),
                "Vulkan mm requires matching 2-D matrices");
    TORCH_CHECK(mat1.device().type() == c10::DeviceType::PrivateUse1 &&
                    mat1.device().index() == 0 && mat2.device() == mat1.device(),
                "Vulkan mm requires matching Vulkan device index 0 matrices");
    TORCH_CHECK(mat1.scalar_type() == at::kFloat && mat2.scalar_type() == at::kFloat &&
                    mat1.layout() == at::kStrided && mat2.layout() == at::kStrided,
                "Vulkan mm requires matching strided float32 matrices");
    const int64_t m = mat1.size(0), k = mat1.size(1), n = mat2.size(1);
    TORCH_CHECK(m <= std::numeric_limits<uint32_t>::max() &&
                    n <= std::numeric_limits<uint32_t>::max() &&
                    k <= std::numeric_limits<uint32_t>::max(),
                "Vulkan mm dimensions exceed dispatch limits");
    TORCH_CHECK(m == 0 || n == 0 ||
                    static_cast<uint64_t>(m) <=
                        std::numeric_limits<uint32_t>::max() /
                            static_cast<uint64_t>(n),
                "Vulkan mm output element count exceeds shader limits");
    const auto a_plan = plan_matrix_read(mat1, {m, k}, "mm mat1");
    const auto b_plan = plan_matrix_read(mat2, {k, n}, "mm mat2");
    const VulkanPlatform *matrix_platform = nullptr;
    if (m > 0 && n > 0 && k > 0) {
        const auto &platform = allocation_platform(mat1.storage().data_ptr());
        TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()),
                    "Vulkan mm requires one Vulkan platform");
        const auto a_gemm_layout =
            a_plan.needs_materialization
                ? dense_matrix_layout(a_plan.dense_shape, "mm mat1")
                : a_plan.source_layout;
        const auto b_gemm_layout =
            b_plan.needs_materialization
                ? dense_matrix_layout(b_plan.dense_shape, "mm mat2")
                : b_plan.source_layout;
        const auto output_gemm_layout = dense_matrix_layout({m, n}, "mm output");
        if (a_plan.needs_materialization)
            platform.compute().validate_broadcast_preflight(
                a_plan.source_layout, dense_matrix_layout(a_plan.dense_shape, "mm mat1"),
                static_cast<uint32_t>(a_plan.source_layout.numel));
        if (b_plan.needs_materialization)
            platform.compute().validate_broadcast_preflight(
                b_plan.source_layout, dense_matrix_layout(b_plan.dense_shape, "mm mat2"),
                static_cast<uint32_t>(b_plan.source_layout.numel));
        platform.compute().validate_gemm_preflight(
            a_gemm_layout, b_gemm_layout, output_gemm_layout,
            static_cast<uint32_t>(m), static_cast<uint32_t>(n),
            static_cast<uint32_t>(k), true);
        matrix_platform = &platform;
    }
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
    const auto &platform = *matrix_platform;
    at::Tensor a = materialize_matrix_read(a_plan, "mm mat1");
    at::Tensor b = materialize_matrix_read(b_plan, "mm mat2");
    const auto a_layout = validate_gemm_2d(a, {m, k}, "mm mat1");
    const auto b_layout = validate_gemm_2d(b, {k, n}, "mm mat2");
    const auto out_layout = validate_gemm_2d(output, {m, n}, "mm output");
    TORCH_CHECK(&platform == &allocation_platform(a.storage().data_ptr()) &&
                    &platform == &allocation_platform(b.storage().data_ptr()) &&
                    &platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan mm requires one Vulkan platform");
    TORCH_CHECK(!layouts_overlap(a, a_layout, output, out_layout) &&
                    !layouts_overlap(b, b_layout, output, out_layout),
                "Vulkan mm output may not alias an input");
    platform.compute().gemm(
        allocation_buffer(a.storage().data_ptr()).buffer(), a_layout,
        allocation_buffer(b.storage().data_ptr()).buffer(), b_layout, VK_NULL_HANDLE,
        out_layout, allocation_buffer(output.storage().data_ptr()).buffer(), out_layout,
        VK_NULL_HANDLE, out_layout, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(k));
    platform.execution_context().retain_until_completion(
        [mat1, mat2, a, b, output] {});
    return output;
}

at::Tensor dot(const at::Tensor &lhs, const at::Tensor &rhs) {
    TORCH_CHECK(lhs.dim() == 1 && rhs.dim() == 1 && lhs.size(0) == rhs.size(0),
                "Vulkan dot requires equal-length 1-D vectors");
    TORCH_CHECK(lhs.device().type() == c10::DeviceType::PrivateUse1 &&
                    lhs.device().index() == 0 && rhs.device() == lhs.device(),
                "Vulkan dot requires matching Vulkan device index 0 vectors");
    TORCH_CHECK(lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kFloat &&
                    lhs.layout() == at::kStrided && rhs.layout() == at::kStrided,
                "Vulkan dot requires matching strided float32 vectors");

    // Retain the original read layouts and offsets in the promoted views. The
    // shared checked mm planner materializes only layouts its GEMM kernel cannot
    // read directly, and its K=0 path produces the scalar additive identity.
    return pytorch_vulkan::mm(lhs.unsqueeze(0), rhs.unsqueeze(1))
        .squeeze(0).squeeze(0);
}

at::Tensor mv(const at::Tensor &matrix, const at::Tensor &vector) {
    TORCH_CHECK(matrix.dim() == 2 && vector.dim() == 1 &&
                    matrix.size(1) == vector.size(0),
                "Vulkan mv requires a 2-D matrix and matching 1-D vector");
    TORCH_CHECK(matrix.device().type() == c10::DeviceType::PrivateUse1 &&
                    matrix.device().index() == 0 && vector.device() == matrix.device(),
                "Vulkan mv requires matching Vulkan device index 0 operands");
    TORCH_CHECK(matrix.scalar_type() == at::kFloat &&
                    vector.scalar_type() == at::kFloat &&
                    matrix.layout() == at::kStrided && vector.layout() == at::kStrided,
                "Vulkan mv requires matching strided float32 operands");

    // Promotion to a column keeps vector storage offset/stride intact; mm's
    // planner validates and materializes readable noncanonical views.
    return pytorch_vulkan::mm(matrix, vector.unsqueeze(1)).squeeze(1);
}

at::Tensor bmm(const at::Tensor &mat1, const at::Tensor &mat2) {
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3,
                "Vulkan bmm requires matching 3-D batch matrices");
    return bmm_out(mat1, mat2, at::Tensor());
}

at::Tensor bmm_out(const at::Tensor &mat1, const at::Tensor &mat2,
                   const at::Tensor &output_arg) {
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3 && mat1.size(0) == mat2.size(0) &&
                    mat1.size(2) == mat2.size(1),
                "Vulkan bmm requires matching 3-D batch matrices");
    TORCH_CHECK(mat1.scalar_type() == at::kFloat && mat2.scalar_type() == at::kFloat &&
                    mat1.device() == mat2.device() && mat1.device().index() == 0,
                "Vulkan bmm requires matching float32 Vulkan matrices");
    TORCH_CHECK(mat1.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    mat1.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    mat1.size(2) <= std::numeric_limits<uint32_t>::max() &&
                    mat2.size(2) <= std::numeric_limits<uint32_t>::max(),
                "Vulkan bmm dimensions exceed dispatch limits");
    TORCH_CHECK(mat1.layout() == at::kStrided && mat2.layout() == at::kStrided,
                "Vulkan bmm requires strided matrices");
    TORCH_CHECK(!output_arg.defined() ||
                    (output_arg.sizes() ==
                         at::IntArrayRef({mat1.size(0), mat1.size(1), mat2.size(2)}) &&
                     output_arg.scalar_type() == mat1.scalar_type() &&
                     output_arg.device() == mat1.device() &&
                     output_arg.layout() == at::kStrided),
                "Vulkan bmm output must match the result shape, dtype, device, and layout");
    const std::vector<int64_t> a_shape{mat1.size(0), mat1.size(1), mat1.size(2)};
    const std::vector<int64_t> b_shape{mat2.size(0), mat2.size(1), mat2.size(2)};
    const auto a_plan = plan_matrix_read(mat1, a_shape, "bmm mat1");
    const auto b_plan = plan_matrix_read(mat2, b_shape, "bmm mat2");
    const int64_t batch = mat1.size(0), m = mat1.size(1), k = mat1.size(2),
                  n = mat2.size(2);
    TORCH_CHECK(batch == 0 || m == 0 || n == 0 ||
                    static_cast<uint64_t>(batch) <=
                        std::numeric_limits<uint32_t>::max() /
                            (static_cast<uint64_t>(m) * static_cast<uint64_t>(n)),
                "Vulkan bmm output element count exceeds shader limits");
    const std::vector<int64_t> output_shape{batch, m, n};
    at::Tensor output = output_arg;
    if (batch == 0 || m == 0 || n == 0) {
        if (!output.defined())
            output = at::empty(output_shape, mat1.options());
        return output;
    }
    if (k == 0) {
        if (!output.defined())
            output = at::empty(output_shape, mat1.options());
        const auto output_layout = inspect_vulkan_tensor_layout(output, "bmm output");
        const auto &platform = allocation_platform(output.storage().data_ptr());
        platform.compute().fill_alias(
            allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
            allocation_buffer(output.storage().data_ptr()).buffer(), output_layout, 0.0F);
        return output;
    }
    const auto &platform = allocation_platform(mat1.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()),
                "Vulkan bmm requires one Vulkan platform");
    const auto a_gemm_layout = a_plan.needs_materialization
                                   ? dense_matrix_layout(a_plan.dense_shape, "bmm mat1")
                                   : a_plan.source_layout;
    const auto b_gemm_layout = b_plan.needs_materialization
                                   ? dense_matrix_layout(b_plan.dense_shape, "bmm mat2")
                                   : b_plan.source_layout;
    const auto output_preflight = dense_matrix_layout(output_shape, "bmm output");
    const auto a_batch_stride = static_cast<uint32_t>(a_gemm_layout.strides[0]);
    const auto b_batch_stride = static_cast<uint32_t>(b_gemm_layout.strides[0]);
    const auto output_preflight_layout = output.defined()
                                             ? inspect_vulkan_tensor_layout(output,
                                                                            "bmm output")
                                             : output_preflight;
    TORCH_CHECK(output_preflight_layout.strides[0] >= 0 &&
                    static_cast<uint64_t>(output_preflight_layout.strides[0]) <=
                        std::numeric_limits<uint32_t>::max(),
                "Vulkan bmm output batch stride exceeds dispatch limits");
    const auto output_batch_stride =
        static_cast<uint32_t>(output_preflight_layout.strides[0]);
    if (a_plan.needs_materialization)
        platform.compute().validate_broadcast_preflight(
            a_plan.source_layout, a_gemm_layout,
            static_cast<uint32_t>(a_plan.source_layout.numel));
    if (b_plan.needs_materialization)
        platform.compute().validate_broadcast_preflight(
            b_plan.source_layout, b_gemm_layout,
            static_cast<uint32_t>(b_plan.source_layout.numel));
    platform.compute().validate_gemm_preflight(
        a_gemm_layout, b_gemm_layout, output_preflight_layout, static_cast<uint32_t>(m),
        static_cast<uint32_t>(n), static_cast<uint32_t>(k), false,
        static_cast<uint32_t>(batch), a_batch_stride, b_batch_stride,
        output_batch_stride);
    if (!output.defined())
        output = at::empty(output_shape, mat1.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "bmm output");
    const auto &output_platform = allocation_platform(output.storage().data_ptr());
    TORCH_CHECK(&platform == &output_platform,
                "Vulkan bmm requires one Vulkan platform");
    TORCH_CHECK(!layouts_overlap(mat1, a_plan.source_layout, output, output_layout) &&
                    !layouts_overlap(mat2, b_plan.source_layout, output, output_layout),
                "Vulkan bmm output may not alias an input");
    at::Tensor a = materialize_matrix_read(a_plan, "bmm mat1");
    at::Tensor b = materialize_matrix_read(b_plan, "bmm mat2");
    const auto a_layout = inspect_vulkan_tensor_layout(a, "bmm mat1 prepared");
    const auto b_layout = inspect_vulkan_tensor_layout(b, "bmm mat2 prepared");
    const auto out_layout = inspect_vulkan_tensor_layout(output, "bmm output");
    validate_allocation(output.storage().data_ptr(), out_layout.allocation_bytes,
                        "bmm output");
    TORCH_CHECK(&platform == &allocation_platform(a.storage().data_ptr()) &&
                    &platform == &allocation_platform(b.storage().data_ptr()) &&
                    &platform == &allocation_platform(output.storage().data_ptr()),
                "Vulkan bmm requires one Vulkan platform");
    TORCH_CHECK(!layouts_overlap(a, a_layout, output, out_layout) &&
                    !layouts_overlap(b, b_layout, output, out_layout),
                "Vulkan bmm output may not alias an input");
    platform.compute().gemm(
        allocation_buffer(a.storage().data_ptr()).buffer(), a_layout,
        allocation_buffer(b.storage().data_ptr()).buffer(), b_layout,
        VK_NULL_HANDLE, output_layout,
        allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
        VK_NULL_HANDLE, output_layout, static_cast<uint32_t>(m),
        static_cast<uint32_t>(n), static_cast<uint32_t>(k), 1.0F, 0.0F, false,
        static_cast<uint32_t>(batch), static_cast<uint32_t>(a_layout.strides[0]),
        static_cast<uint32_t>(b_layout.strides[0]),
        static_cast<uint32_t>(out_layout.strides[0]),
        static_cast<uint32_t>(out_layout.strides[0]));
    platform.execution_context().retain_until_completion([mat1, mat2, a, b, output] {});
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
    if (is_fake_tensor(mat1))
        return at::_ops::addmm::redispatch(c10::DispatchKeySet(c10::DispatchKey::Meta),
                                           self, mat1, mat2, beta, alpha);
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
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2 && mat1.size(1) == mat2.size(0),
                "Vulkan addmm requires matching 2-D matrices");
    const int64_t m = mat1.size(0), n = mat2.size(1), k = mat1.size(1);
    TORCH_CHECK(m <= std::numeric_limits<uint32_t>::max() &&
                    n <= std::numeric_limits<uint32_t>::max() &&
                    k <= std::numeric_limits<uint32_t>::max(),
                "Vulkan addmm dimensions exceed dispatch limits");
    TORCH_CHECK(m == 0 || n == 0 ||
                    static_cast<uint64_t>(m) <=
                        std::numeric_limits<uint32_t>::max() /
                            static_cast<uint64_t>(n),
                "Vulkan addmm output element count exceeds shader limits");
    const std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, c_shape{m, n};
    const auto a_plan = plan_matrix_read(mat1, a_shape, "addmm mat1");
    const auto b_plan = plan_matrix_read(mat2, b_shape, "addmm mat2");
    // expand() is the same authoritative broadcasting rule used by PyTorch.
    const auto c_plan = plan_matrix_read(self, c_shape, "addmm self");
    if (m == 0 || n == 0)
        return at::empty({m, n}, self.options());
    const auto dense_output = dense_matrix_layout(c_shape, "addmm output");
    if (k == 0) {
        const bool beta_is_zero = beta.toComplexDouble() == 0.0;
        const float beta_value = converting_k0_addmm_beta(beta);
        const auto &platform = allocation_platform(self.storage().data_ptr());
        if (beta_is_zero)
            platform.compute().validate_broadcast_preflight(
                dense_output, dense_output,
                static_cast<uint32_t>(dense_output.numel));
        else
            platform.compute().validate_broadcast_preflight(
                c_plan.source_layout, dense_output,
                static_cast<uint32_t>(dense_output.numel));
        at::Tensor output = at::empty({m, n}, self.options());
        const auto output_layout = inspect_vulkan_tensor_layout(output, "addmm output");
        TORCH_CHECK(&platform == &allocation_platform(output.storage().data_ptr()),
                    "Vulkan addmm requires one Vulkan platform");
        if (beta_is_zero) {
            platform.compute().fill_alias(
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                0.0F);
        } else {
            platform.compute().tensor_scalar(
                allocation_buffer(c_plan.source.storage().data_ptr()).buffer(),
                c_plan.source_layout,
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                beta_value, 2);
            platform.execution_context().retain_until_completion(
                [self, output, c_source = c_plan.source] {});
        }
        return output;
    }
    const float beta_value = checked_allocating_addmm_scalar(beta, "addmm beta");
    const float alpha_value = checked_allocating_addmm_scalar(alpha, "addmm alpha");
    const auto &platform = allocation_platform(mat1.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(mat2.storage().data_ptr()) &&
                    &platform == &allocation_platform(self.storage().data_ptr()),
                "Vulkan addmm requires one Vulkan platform");
    if (alpha_value == 0.0F) {
        if (beta_value == 0.0F)
            platform.compute().validate_broadcast_preflight(
                dense_output, dense_output,
                static_cast<uint32_t>(dense_output.numel));
        else
            platform.compute().validate_broadcast_preflight(
                c_plan.source_layout, dense_output,
                static_cast<uint32_t>(dense_output.numel));
        at::Tensor output = at::empty({m, n}, self.options());
        const auto output_layout = inspect_vulkan_tensor_layout(output, "addmm output");
        if (beta_value == 0.0F) {
            platform.compute().fill_alias(
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                0.0F);
        } else {
            const auto &self_read = c_plan.source;
            const auto &self_layout = c_plan.source_layout;
            platform.compute().tensor_scalar(
                allocation_buffer(self_read.storage().data_ptr()).buffer(), self_layout,
                allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
                beta_value, 2);
            platform.execution_context().retain_until_completion(
                [self, self_read, output] {});
        }
        return output;
    }
    const auto dense_a = dense_matrix_layout(a_shape, "addmm mat1");
    const auto dense_b = dense_matrix_layout(b_shape, "addmm mat2");
    const auto dense_c = dense_matrix_layout(c_shape, "addmm self");
    if (alpha_value != 0.0F) {
        if (a_plan.needs_materialization)
            platform.compute().validate_broadcast_preflight(
                a_plan.source_layout, dense_a,
                static_cast<uint32_t>(a_plan.source_layout.numel));
        if (b_plan.needs_materialization)
            platform.compute().validate_broadcast_preflight(
                b_plan.source_layout, dense_b,
                static_cast<uint32_t>(b_plan.source_layout.numel));
    }
    if (beta_value != 0.0F && c_plan.needs_materialization)
        platform.compute().validate_broadcast_preflight(
            c_plan.source_layout, dense_c,
            static_cast<uint32_t>(c_plan.source_layout.numel));
    const auto &a_preflight = alpha_value == 0.0F ? dense_a :
        (a_plan.needs_materialization ? dense_a : a_plan.source_layout);
    const auto &b_preflight = alpha_value == 0.0F ? dense_b :
        (b_plan.needs_materialization ? dense_b : b_plan.source_layout);
    const auto &c_preflight = c_plan.needs_materialization ? dense_c :
        c_plan.source_layout;
    platform.compute().validate_gemm_preflight(
        a_preflight, b_preflight, dense_output, static_cast<uint32_t>(m),
        static_cast<uint32_t>(n), static_cast<uint32_t>(k), true, 0, 0, 0, 0,
        beta_value == 0.0F ? nullptr : &c_preflight);
    at::Tensor output = at::empty({m, n}, self.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "addmm output");
    TORCH_CHECK(!layouts_overlap(output, output_layout, mat1, a_plan.source_layout) &&
                    !layouts_overlap(output, output_layout, mat2, b_plan.source_layout) &&
                    !layouts_overlap(output, output_layout, self, c_plan.source_layout),
                "Vulkan addmm output may not alias an input");
    at::Tensor a = alpha_value == 0.0F ? output :
        materialize_matrix_read(a_plan, "addmm mat1");
    at::Tensor b = alpha_value == 0.0F ? output :
        materialize_matrix_read(b_plan, "addmm mat2");
    at::Tensor c = beta_value == 0.0F ? output :
        materialize_matrix_read(c_plan, "addmm self");
    const auto a_layout = alpha_value == 0.0F ? output_layout :
        inspect_vulkan_tensor_layout(a, "addmm mat1 prepared");
    const auto b_layout = alpha_value == 0.0F ? output_layout :
        inspect_vulkan_tensor_layout(b, "addmm mat2 prepared");
    const auto c_layout = beta_value == 0.0F ? output_layout :
        inspect_vulkan_tensor_layout(c, "addmm self prepared");
    platform.compute().gemm(
        allocation_buffer(a.storage().data_ptr()).buffer(), a_layout,
        allocation_buffer(b.storage().data_ptr()).buffer(), b_layout,
        beta_value == 0.0F ? VK_NULL_HANDLE
                           : allocation_buffer(c.storage().data_ptr()).buffer(),
        c_layout, allocation_buffer(output.storage().data_ptr()).buffer(), output_layout,
        VK_NULL_HANDLE, output_layout, static_cast<uint32_t>(m),
        static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha_value, beta_value,
        false, 0, 0, 0, 0, 0);
    platform.execution_context().retain_until_completion(
        [self, mat1, mat2, a, b, c, output] {});
    return output;
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
             checked_scalar(beta, "addmm.out beta"), false, 0, 0, 0, 0, 0);
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
    m.impl("dot", &pytorch_vulkan::dot);
    m.impl("mv", &pytorch_vulkan::mv);
    m.impl("mm", &pytorch_vulkan::mm);
    m.impl("bmm", &pytorch_vulkan::bmm);
    m.impl("addmm", &pytorch_vulkan::addmm);
    m.impl("addmm.out", &pytorch_vulkan::addmm_out);
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
