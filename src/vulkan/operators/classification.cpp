#include "classification.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

namespace pytorch_vulkan {
namespace {
void validate_float(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0),
                "Vulkan nll_loss ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided &&
                    tensor.is_contiguous(),
                "Vulkan nll_loss ", name,
                " requires a contiguous Vulkan float32 tensor");
}
void validate_logits(const at::Tensor &logits) {
    validate_float(logits, "logits");
    TORCH_CHECK(logits.dim() == 2 && logits.sizes().equals({2, 3}),
                "Vulkan nll_loss supports logits shape (2, 3) only");
}
void validate_labels(const at::Tensor &labels) {
    TORCH_CHECK(labels.device() == c10::Device(c10::DeviceType::PrivateUse1, 0) &&
                    labels.scalar_type() == at::kLong &&
                    labels.layout() == at::kStrided && labels.is_contiguous() &&
                    labels.sizes().equals({2}),
                "Vulkan nll_loss labels require contiguous vk:0 int64 shape (2)");
}
void run(const at::Tensor *inputs[], uint32_t input_count, const at::Tensor *outputs[],
         uint32_t output_count, uint32_t mode, int64_t ignore_index) {
    const auto &platform = allocation_platform(inputs[0]->storage().data_ptr());
    VulkanTensorLayout in_layouts[4];
    const VulkanBuffer *in_buffers[4];
    for (uint32_t i = 0; i < input_count; ++i) {
        in_layouts[i] = inspect_vulkan_tensor_layout(*inputs[i], "nll_loss input");
        in_buffers[i] = &allocation_buffer(inputs[i]->storage().data_ptr());
    }
    VulkanTensorLayout out_layouts[2];
    const VulkanBuffer *out_buffers[2];
    for (uint32_t i = 0; i < output_count; ++i) {
        out_layouts[i] = inspect_vulkan_tensor_layout(*outputs[i], "nll_loss output");
        out_buffers[i] = &allocation_buffer(outputs[i]->storage().data_ptr());
    }
    const VulkanTensorLayout *in_layout_ptrs[4];
    for (uint32_t i = 0; i < input_count; ++i)
        in_layout_ptrs[i] = &in_layouts[i];
    const VulkanTensorLayout *out_layout_ptrs[2];
    for (uint32_t i = 0; i < output_count; ++i)
        out_layout_ptrs[i] = &out_layouts[i];
    struct Params {
        uint32_t mode, batch, channels, spatial, classes;
        int32_t ignore;
        uint32_t p0, p1;
        float momentum, eps;
    } params{mode, 2, 1, 1, 3, static_cast<int32_t>(ignore_index), 0, 0, 0.0F, 0.0F};
    platform.compute().compute_multi_output(
        in_buffers, in_layout_ptrs, out_buffers, out_layout_ptrs, input_count,
        output_count, mode == 2 ? 1 : 6, &params, sizeof(params), true);
}
} // namespace

std::tuple<at::Tensor, at::Tensor>
nll_loss_forward(const at::Tensor &self, const at::Tensor &target,
                 const c10::optional<at::Tensor> &weight, int64_t reduction,
                 c10::SymInt ignore_index) {
    validate_logits(self);
    validate_labels(target);
    TORCH_CHECK(!weight.has_value() || !weight->defined() || weight->numel() == 0,
                "Vulkan nll_loss does not support class weights");
    TORCH_CHECK(reduction == 1, "Vulkan nll_loss supports reduction=mean only");
    TORCH_CHECK(ignore_index == -100,
                "Vulkan nll_loss supports ignore_index=-100 only");
    auto loss = at::empty({}, self.options());
    auto total = at::empty({}, self.options());
    auto scratch = at::empty_like(self);
    const at::Tensor *inputs[] = {&scratch, &self, &target, &scratch};
    const at::Tensor *outputs[] = {&loss, &total};
    run(inputs, 4, outputs, 2, 2, ignore_index.expect_int());
    return {loss, total};
}

at::Tensor nll_loss_backward(const at::Tensor &grad_output, const at::Tensor &self,
                             const at::Tensor &target,
                             const c10::optional<at::Tensor> &weight, int64_t reduction,
                             c10::SymInt ignore_index, const at::Tensor &total_weight) {
    validate_float(grad_output, "grad_output");
    validate_logits(self);
    validate_labels(target);
    validate_float(total_weight, "total_weight");
    TORCH_CHECK(grad_output.dim() == 0 && total_weight.dim() == 0 &&
                    (!weight.has_value() || !weight->defined() || weight->numel() == 0),
                "Vulkan nll_loss backward requires scalar mean inputs and no weights");
    TORCH_CHECK(
        reduction == 1 && ignore_index == -100,
        "Vulkan nll_loss backward supports mean reduction and ignore_index=-100 only");
    auto output = at::empty_like(self);
    const at::Tensor *inputs[] = {&grad_output, &self, &target, &total_weight};
    const at::Tensor *outputs[] = {&output};
    run(inputs, 4, outputs, 1, 3, ignore_index.expect_int());
    return output;
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("nll_loss_forward", &pytorch_vulkan::nll_loss_forward);
    m.impl("nll_loss_backward", &pytorch_vulkan::nll_loss_backward);
}
