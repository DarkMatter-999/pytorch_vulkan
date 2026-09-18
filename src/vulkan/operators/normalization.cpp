#include "normalization.h"

#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

namespace pytorch_vulkan {
namespace {
void validate(const at::Tensor &tensor, const char *name, bool channel) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0),
                "Vulkan native_batch_norm ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan native_batch_norm ", name, " requires strided float32");
    TORCH_CHECK(tensor.is_contiguous(), "Vulkan native_batch_norm ", name,
                " requires contiguous tensors");
    if (channel)
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == 4,
                    "Vulkan native_batch_norm ", name, " requires four channels");
}
void validate_input(const at::Tensor &input) {
    validate(input, "input", false);
    TORCH_CHECK((input.dim() == 2 && input.sizes().equals({2, 4})) ||
                    (input.dim() == 4 && input.sizes().equals({2, 4, 2, 2})),
                "Vulkan native_batch_norm supports only fixed (2,4) or (2,4,2,2)");
}
void validate_common(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                     const c10::optional<at::Tensor> &bias,
                     const c10::optional<at::Tensor> &running_mean,
                     const c10::optional<at::Tensor> &running_var) {
    validate_input(input);
    TORCH_CHECK(weight.has_value() == bias.has_value(),
                "Vulkan native_batch_norm requires both affine tensors or neither");
    TORCH_CHECK(
        weight.has_value() && running_mean.has_value() && running_var.has_value(),
        "Vulkan native_batch_norm requires affine and running-statistics tensors");
    validate(*weight, "weight", true);
    validate(*bias, "bias", true);
    validate(*running_mean, "running_mean", true);
    validate(*running_var, "running_var", true);
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
    const auto spatial = primary.dim() == 2 ? 1U : 4U;
    const uint32_t batch = 2, channels = 4;
    if (mode != 0)
        inputs[1] = secondary;
    for (uint32_t i = 0; i < input_count; ++i)
        validate(*inputs[i], "tensor",
                 mode == 0 ? (i > 0 && i < 5) : (i >= 2 && i < 5));
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
        input_count, 3, 8, &params, sizeof(params), false);
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
    auto output = at::empty_like(input);
    auto mean = at::empty_like(*weight);
    auto invstd = at::empty_like(*weight);
    run(input, *weight, *bias, *running_mean, *running_var, nullptr, nullptr, nullptr,
        output, mean, invstd, 0, momentum, eps);
    return {output, mean, invstd};
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
    validate(grad_out, "grad_out", false);
    validate(*save_mean, "save_mean", true);
    validate(*save_invstd, "save_invstd", true);
    TORCH_CHECK(grad_out.sizes().equals(input.sizes()),
                "Vulkan native_batch_norm_backward shape mismatch");
    auto dx = at::empty_like(input);
    auto dw = at::empty_like(*weight);
    auto db = at::empty_like(*weight);
    run(grad_out, *weight, *weight, *running_mean, *running_var, &input,
        save_mean.operator->(), save_invstd.operator->(), dx, dw, db, 1, 0.1, eps);
    return {dx, dw, db};
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("native_batch_norm", &pytorch_vulkan::native_batch_norm);
    m.impl("native_batch_norm_backward", &pytorch_vulkan::native_batch_norm_backward);
}
