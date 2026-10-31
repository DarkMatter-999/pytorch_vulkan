#include "normalization.h"

#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <ATen/native/layer_norm.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>

namespace pytorch_vulkan {
namespace {
void validate(const at::Tensor &tensor, const char *name, bool channel,
              int64_t channels = 0) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0),
                "Vulkan native_batch_norm ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan native_batch_norm ", name, " requires strided float32");
    TORCH_CHECK(tensor.is_contiguous(), "Vulkan native_batch_norm ", name,
                " requires contiguous tensors");
    if (channel)
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == channels,
                    "Vulkan native_batch_norm ", name, " requires ", channels,
                    " channels, got ", tensor.size(0));
}
void validate_input(const at::Tensor &input) {
    validate(input, "input", false);
    TORCH_CHECK(input.dim() == 2 || input.dim() == 4,
                "Vulkan native_batch_norm requires 2-D (N,C) or 4-D (N,C,H,W) "
                "input, got ", input.dim(), " dimensions");
}
void validate_common(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                     const c10::optional<at::Tensor> &bias,
                     const c10::optional<at::Tensor> &running_mean,
                     const c10::optional<at::Tensor> &running_var) {
    validate_input(input);
    const int64_t channels = input.size(1);
    TORCH_CHECK(weight.has_value() == bias.has_value(),
                "Vulkan native_batch_norm requires both affine tensors or neither");
    TORCH_CHECK(
        weight.has_value() && running_mean.has_value() && running_var.has_value(),
        "Vulkan native_batch_norm requires affine and running-statistics tensors");
    validate(*weight, "weight", true, channels);
    validate(*bias, "bias", true, channels);
    validate(*running_mean, "running_mean", true, channels);
    validate(*running_var, "running_var", true, channels);
}
void run(const at::Tensor &primary, const at::Tensor &weight, const at::Tensor &bias,
         const at::Tensor &running_mean, const at::Tensor &running_var,
         const at::Tensor *secondary, const at::Tensor *save_mean,
         const at::Tensor *save_inv, at::Tensor &out0, at::Tensor &out1,
         at::Tensor &out2, uint32_t mode, double momentum, double eps) {
    const auto &data = primary.storage().data_ptr();
    const auto &platform = allocation_platform(data);
    const at::Tensor *inputs[] = {&primary,     &weight,   &bias,   &running_mean,
                                  &running_var, save_mean, save_inv};
    const at::Tensor *outputs[] = {&out0, &out1, &out2};
    const uint32_t input_count = mode == 0 ? 5 : 7;
    const int64_t rank = primary.dim();
    TORCH_CHECK(rank == 2 || rank == 4,
                "Vulkan native_batch_norm requires 2-D (N,C) or 4-D (N,C,H,W) "
                "input, got ", rank, " dimensions");
    const uint32_t batch = static_cast<uint32_t>(primary.size(0));
    const uint32_t channels = static_cast<uint32_t>(primary.size(1));
    uint32_t spatial = 1U;
    for (int64_t dim = 2; dim < rank; ++dim)
        spatial *= static_cast<uint32_t>(primary.size(dim));
    const uint64_t elements = static_cast<uint64_t>(batch) * channels * spatial;
    TORCH_CHECK(elements <= std::numeric_limits<uint32_t>::max(),
                "Vulkan native_batch_norm element count ", elements,
                " exceeds the uint32_t shader limit");
    const uint32_t invocations = std::max(channels, static_cast<uint32_t>(elements));
    if (mode != 0)
        inputs[1] = secondary;
    for (uint32_t i = 0; i < input_count; ++i) {
        const bool channel = mode == 0 ? (i > 0 && i < 5) : (i >= 2 && i < 5);
        validate(*inputs[i], "tensor", channel, primary.size(1));
    }
    for (const auto *output : outputs)
        validate(*output, "output", false);
    const auto layout = [](const at::Tensor &tensor) {
        return inspect_vulkan_tensor_layout(tensor, "native_batch_norm");
    };
    VulkanTensorLayout input_layouts[7];
    for (uint32_t i = 0; i < input_count; ++i)
        input_layouts[i] = layout(*inputs[i]);
    const VulkanTensorLayout *input_layout_ptrs[7];
    for (uint32_t i = 0; i < input_count; ++i)
        input_layout_ptrs[i] = &input_layouts[i];
    VulkanTensorLayout output_layouts[3];
    for (uint32_t i = 0; i < 3; ++i)
        output_layouts[i] = layout(*outputs[i]);
    const VulkanTensorLayout *output_layout_ptrs[3];
    for (uint32_t i = 0; i < 3; ++i)
        output_layout_ptrs[i] = &output_layouts[i];
    const VulkanBuffer *input_buffers[7];
    for (uint32_t i = 0; i < input_count; ++i)
        input_buffers[i] = &allocation_buffer(inputs[i]->storage().data_ptr());
    const VulkanBuffer *output_buffers[3];
    for (uint32_t i = 0; i < 3; ++i)
        output_buffers[i] = &allocation_buffer(outputs[i]->storage().data_ptr());
    struct Params {
        uint32_t mode, batch, channels, spatial, classes;
        int32_t ignore;
        uint32_t p0, p1;
        float momentum, eps;
    } params{mode,
             batch,
             channels,
             spatial,
             0,
             0,
             0,
             0,
             static_cast<float>(momentum),
             static_cast<float>(eps)};
    platform.compute().compute_multi_output(
        input_buffers, input_layout_ptrs, output_buffers, output_layout_ptrs,
        input_count, 3, invocations, &params, sizeof(params), false);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                  const c10::optional<at::Tensor> &bias,
                  const c10::optional<at::Tensor> &running_mean,
                  const c10::optional<at::Tensor> &running_var, bool training,
                  double momentum, double eps) {
    TORCH_CHECK(training, "Vulkan native_batch_norm supports training=True only");
    TORCH_CHECK(momentum == 0.1, "Vulkan native_batch_norm supports momentum=0.1 only");
    TORCH_CHECK(eps == 1e-5, "Vulkan native_batch_norm supports eps=1e-5 only");
    validate_common(input, weight, bias, running_mean, running_var);
    return with_vulkan_operation(allocation_runtime(weight->storage().data_ptr()), [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
    auto output = at::empty_like(input);
    auto mean = at::empty_like(*weight);
    auto invstd = at::empty_like(*weight);
    run(input, *weight, *bias, *running_mean, *running_var, nullptr, nullptr, nullptr,
        output, mean, invstd, 0, momentum, eps);
    return {output, mean, invstd};
    });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm_backward(const at::Tensor &grad_out, const at::Tensor &input,
                           const c10::optional<at::Tensor> &weight,
                           const c10::optional<at::Tensor> &running_mean,
                           const c10::optional<at::Tensor> &running_var,
                           const c10::optional<at::Tensor> &save_mean,
                           const c10::optional<at::Tensor> &save_invstd, bool train,
                           double eps, std::array<bool, 3> mask) {
    TORCH_CHECK(train && mask[0] && mask[1] && mask[2],
                "Vulkan native_batch_norm_backward requires training and "
                "output_mask=[true,true,true]");
    TORCH_CHECK(eps == 1e-5,
                "Vulkan native_batch_norm_backward supports eps=1e-5 only");
    TORCH_CHECK(weight && running_mean && running_var && save_mean && save_invstd,
                "Vulkan native_batch_norm_backward requires all saved tensors");
    validate_common(input, weight, weight, running_mean, running_var);
    TORCH_CHECK(grad_out.device() == c10::Device(c10::DeviceType::PrivateUse1, 0),
                "Vulkan native_batch_norm grad_out requires vk:0");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat &&
                    grad_out.layout() == at::kStrided,
                "Vulkan native_batch_norm grad_out requires strided float32");
    TORCH_CHECK(grad_out.sizes().equals(input.sizes()),
                "Vulkan native_batch_norm_backward shape mismatch");
    const auto grad_layout = inspect_vulkan_tensor_layout(grad_out, "grad_out");
    TORCH_CHECK(grad_layout.internal_overlap == VulkanOverlap::No ||
                    is_non_overlapping_except_broadcast_dims(grad_layout),
                "Vulkan native_batch_norm grad_out has unsupported overlapping layout");
    validate(*save_mean, "save_mean", true, input.size(1));
    validate(*save_invstd, "save_invstd", true, input.size(1));
    return with_vulkan_operation(allocation_runtime(weight->storage().data_ptr()), [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
    const at::Tensor readable_grad =
        grad_out.is_contiguous() ? grad_out : grad_out.contiguous();
    validate(readable_grad, "grad_out", false);
    auto dx = at::empty_like(input);
    auto dw = at::empty_like(*weight);
    auto db = at::empty_like(*weight);
    run(readable_grad, *weight, *weight, *running_mean, *running_var, &input,
        save_mean.operator->(), save_invstd.operator->(), dx, dw, db, 1, 0.1, eps);
    return {dx, dw, db};
    });
}

namespace {
// LayerNorm (shader modes 2-6 of normalization.comp). The kernels index raw
// contiguous storage from offset 0, so every operand is made dense first, as
// the CPU kernel does with .contiguous().
constexpr uint32_t kLayerNormStats = 2, kLayerNormApply = 3, kLayerNormRows = 4,
                   kLayerNormColumns = 5, kLayerNormInput = 6;

void check_layer_norm_tensor(const at::Tensor &tensor, const char *op, const char *name) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0),
                "Vulkan ", op, " ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan ", op, " ", name, " requires strided float32");
}

// at::layer_norm forwards an absent weight/bias as an undefined tensor wrapped
// in an optional that reports has_value(); treat that as absent.
c10::optional<at::Tensor> present(const c10::optional<at::Tensor> &tensor) {
    if (tensor.has_value() && tensor->defined())
        return tensor;
    return c10::nullopt;
}

at::Tensor dense(const at::Tensor &tensor) {
    if (tensor.numel() == 0)
        return tensor;
    if (!tensor.is_contiguous())
        return tensor.contiguous();
    if (tensor.storage_offset() != 0)
        return tensor.clone(at::MemoryFormat::Contiguous);
    return tensor;
}

struct LayerNormShape {
    int64_t rows, cols;
    std::vector<int64_t> stats;
};

LayerNormShape check_layer_norm(const char *op, const at::Tensor &input,
                                at::IntArrayRef normalized_shape,
                                const c10::optional<at::Tensor> &weight,
                                const c10::optional<at::Tensor> &bias) {
    check_layer_norm_tensor(input, op, "input");
    TORCH_CHECK(input.dim() <= 8, "Vulkan ", op, " supports ranks up to 8, got ",
                input.dim());
    if (weight)
        check_layer_norm_tensor(*weight, op, "weight");
    if (bias)
        check_layer_norm_tensor(*bias, op, "bias");
    const auto [rows, cols] = at::native::_check_layer_norm_inputs(
        input, normalized_shape, weight.value_or(at::Tensor()),
        bias.value_or(at::Tensor()));
    TORCH_CHECK(static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) <=
                    std::numeric_limits<uint32_t>::max(),
                "Vulkan ", op, " element count exceeds the uint32_t shader limit");
    LayerNormShape result{rows, cols, {}};
    const int64_t lead = input.dim() - static_cast<int64_t>(normalized_shape.size());
    for (int64_t i = 0; i < input.dim(); ++i)
        result.stats.push_back(i < lead ? input.size(i) : 1);
    return result;
}

void layer_norm_dispatch(uint32_t mode, std::initializer_list<const at::Tensor *> inputs,
                         std::initializer_list<const at::Tensor *> outputs,
                         uint32_t invocations, uint32_t rows, uint32_t cols,
                         uint32_t flags, double eps) {
    const auto &platform = allocation_platform((*outputs.begin())->storage().data_ptr());
    const VulkanBuffer *input_buffers[7];
    const VulkanBuffer *output_buffers[3];
    VulkanTensorLayout input_layouts[7], output_layouts[3];
    const VulkanTensorLayout *input_layout_ptrs[7], *output_layout_ptrs[3];
    uint32_t count = 0;
    for (const auto *tensor : inputs) {
        input_buffers[count] = &allocation_buffer(tensor->storage().data_ptr());
        input_layouts[count] = inspect_vulkan_tensor_layout(*tensor, "layer_norm");
        input_layout_ptrs[count] = &input_layouts[count];
        ++count;
    }
    const uint32_t input_count = count;
    count = 0;
    for (const auto *tensor : outputs) {
        output_buffers[count] = &allocation_buffer(tensor->storage().data_ptr());
        output_layouts[count] = inspect_vulkan_tensor_layout(*tensor, "layer_norm");
        output_layout_ptrs[count] = &output_layouts[count];
        ++count;
    }
    struct Params {
        uint32_t mode, batch, channels, spatial, classes;
        int32_t ignore;
        uint32_t p0, p1;
        float momentum, eps;
    } params{mode, rows, cols, flags, 0, 0, 0, 0, 0.0F, static_cast<float>(eps)};
    platform.compute().compute_multi_output(
        input_buffers, input_layout_ptrs, output_buffers, output_layout_ptrs,
        input_count, count, invocations, &params, sizeof(params), false);
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                  const c10::optional<at::Tensor> &weight_arg,
                  const c10::optional<at::Tensor> &bias_arg, double eps) {
    const auto weight = present(weight_arg);
    const auto bias = present(bias_arg);
    const auto shape = check_layer_norm("native_layer_norm", input, normalized_shape,
                                        weight, bias);
    at::Tensor runtime_tensor = input.numel() > 0 ? input : (weight ? *weight : at::Tensor());
    auto *runtime = runtime_tensor.defined() && runtime_tensor.numel() > 0
        ? allocation_runtime(runtime_tensor.storage().data_ptr())
        : nullptr;
    return with_vulkan_operation(runtime, [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
    auto output = at::empty(input.sizes(), input.options());
    auto mean = at::empty(shape.stats, input.options());
    auto rstd = at::empty(shape.stats, input.options());
    if (shape.rows == 0)
        return {output, mean, rstd};
    const uint32_t rows = static_cast<uint32_t>(shape.rows);
    const uint32_t cols = static_cast<uint32_t>(shape.cols);
    const at::Tensor x = shape.cols == 0 ? at::empty({1}, input.options()) : dense(input);
    layer_norm_dispatch(kLayerNormStats, {&x}, {&mean, &rstd}, rows, rows, cols, 0, eps);
    if (shape.cols == 0)
        return {output, mean, rstd};
    const at::Tensor w = weight ? dense(*weight) : mean;
    const at::Tensor b = bias ? dense(*bias) : mean;
    layer_norm_dispatch(kLayerNormApply, {&x, &mean, &rstd, &w, &b}, {&output},
                        rows * cols, rows, cols,
                        (weight ? 1U : 0U) | (bias ? 2U : 0U), eps);
    return {output, mean, rstd};
    });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor &grad_out, const at::Tensor &input, at::IntArrayRef normalized_shape,
    const at::Tensor &mean, const at::Tensor &rstd,
    const c10::optional<at::Tensor> &weight_arg,
    const c10::optional<at::Tensor> &bias_arg, std::array<bool, 3> mask) {
    const auto weight = present(weight_arg);
    const auto bias = present(bias_arg);
    const auto shape = check_layer_norm("native_layer_norm_backward", input,
                                        normalized_shape, weight, bias);
    check_layer_norm_tensor(grad_out, "native_layer_norm_backward", "grad_out");
    check_layer_norm_tensor(mean, "native_layer_norm_backward", "mean");
    check_layer_norm_tensor(rstd, "native_layer_norm_backward", "rstd");
    TORCH_CHECK(grad_out.sizes().equals(input.sizes()),
                "Vulkan native_layer_norm_backward grad_out shape ", grad_out.sizes(),
                " does not match input shape ", input.sizes());
    TORCH_CHECK(mean.numel() == shape.rows && rstd.numel() == shape.rows,
                "Vulkan native_layer_norm_backward mean and rstd must have ", shape.rows,
                " elements, got ", mean.numel(), " and ", rstd.numel());
    TORCH_CHECK(!mask[1] || weight, "Vulkan native_layer_norm_backward output_mask[1] "
                                    "requires weight");
    TORCH_CHECK(!mask[2] || bias, "Vulkan native_layer_norm_backward output_mask[2] "
                                  "requires bias");
    at::Tensor runtime_tensor = input.numel() > 0 ? input : mean;
    auto *runtime = runtime_tensor.numel() > 0
        ? allocation_runtime(runtime_tensor.storage().data_ptr())
        : nullptr;
    return with_vulkan_operation(runtime, [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
    at::Tensor dx, dw, db;
    const bool parameters = mask[1] || mask[2];
    if (mask[0])
        dx = at::empty(input.sizes(), input.options());
    if (parameters) {
        dw = shape.rows == 0 ? at::zeros(normalized_shape, input.options())
                             : at::empty(normalized_shape, input.options());
        db = shape.rows == 0 ? at::zeros(normalized_shape, input.options())
                             : at::empty(normalized_shape, input.options());
    }
    const auto finish = [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
        return {mask[0] ? dx : at::Tensor(), mask[1] ? dw : at::Tensor(),
                mask[2] ? db : at::Tensor()};
    };
    if (shape.rows == 0 || shape.cols == 0)
        return finish();
    const uint32_t rows = static_cast<uint32_t>(shape.rows);
    const uint32_t cols = static_cast<uint32_t>(shape.cols);
    const at::Tensor dy = dense(grad_out), x = dense(input), m = dense(mean),
                     r = dense(rstd);
    if (parameters)
        layer_norm_dispatch(kLayerNormColumns, {&dy, &x, &m, &r}, {&dw, &db}, cols,
                            rows, cols, 0, 0.0);
    if (mask[0]) {
        const at::Tensor w = weight ? dense(*weight) : m;
        auto ds = at::empty({shape.rows}, input.options());
        auto dbias = at::empty({shape.rows}, input.options());
        const uint32_t flags = weight ? 1U : 0U;
        layer_norm_dispatch(kLayerNormRows, {&dy, &x, &m, &r, &w}, {&ds, &dbias}, rows,
                            rows, cols, flags, 0.0);
        layer_norm_dispatch(kLayerNormInput, {&dy, &x, &m, &r, &w, &ds, &dbias}, {&dx},
                            rows * cols, rows, cols, flags, 0.0);
    }
    return finish();
    });
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("native_batch_norm", &pytorch_vulkan::native_batch_norm);
    m.impl("native_batch_norm_backward", &pytorch_vulkan::native_batch_norm_backward);
    m.impl("native_layer_norm", &pytorch_vulkan::native_layer_norm);
    m.impl("native_layer_norm_backward", &pytorch_vulkan::native_layer_norm_backward);
}
