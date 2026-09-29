#include "classification.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

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
    TORCH_CHECK(logits.dim() == 2, "Vulkan nll_loss requires 2-D logits, got ",
                logits.dim(), " dimensions");
    TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0,
                "Vulkan nll_loss requires a non-empty logits shape, got (",
                logits.size(0), ", ", logits.size(1), ")");
}
void validate_labels(const at::Tensor &labels, int64_t batch) {
    TORCH_CHECK(labels.device() == c10::Device(c10::DeviceType::PrivateUse1, 0) &&
                    labels.scalar_type() == at::kLong &&
                    labels.layout() == at::kStrided && labels.is_contiguous() &&
                    labels.sizes().equals({batch}),
                "Vulkan nll_loss labels require contiguous vk:0 int64 shape (",
                batch, ")");
    TORCH_CHECK(is_validated_label_allocation(labels.storage().data_ptr()),
                "Vulkan nll_loss labels have no validated CPU provenance");
}
void run(const at::Tensor *inputs[], uint32_t input_count, const at::Tensor *outputs[],
         uint32_t output_count, uint32_t mode, int32_t ignore_index, uint32_t batch,
         uint32_t classes) {
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
    } params{mode, batch, 1, 1, classes, ignore_index, 0, 0, 0.0F, 0.0F};
    platform.compute().compute_multi_output(
        in_buffers, in_layout_ptrs, out_buffers, out_layout_ptrs, input_count,
        output_count, (mode == 2 || mode == 4) ? 1 :
            (mode == 5 ? batch : batch * classes), &params, sizeof(params), true);
}
} // namespace

std::tuple<at::Tensor, at::Tensor>
nll_loss_forward(const at::Tensor &self, const at::Tensor &target,
                 const c10::optional<at::Tensor> &weight, int64_t reduction,
                 c10::SymInt ignore_index) {
    validate_logits(self);
    validate_labels(target, self.size(0));
    const int64_t batch = self.size(0);
    const int64_t classes = self.size(1);
    TORCH_CHECK(!weight.has_value() || !weight->defined() || weight->numel() == 0,
                "Vulkan nll_loss does not support class weights");
    TORCH_CHECK(reduction >= 0 && reduction <= 2,
                "Vulkan nll_loss supports reduction none, mean, or sum");
    TORCH_CHECK(ignore_index.expect_int() >= std::numeric_limits<int32_t>::min() &&
                    ignore_index.expect_int() <= std::numeric_limits<int32_t>::max(),
                "Vulkan nll_loss ignore_index does not fit in int32");
    auto loss = reduction == 0 ? at::empty({batch}, self.options())
                               : at::empty({}, self.options());
    auto total = at::empty({}, self.options());
    auto scratch = at::empty_like(self);
    const at::Tensor *inputs[] = {&scratch, &self, &target, &scratch};
    const at::Tensor *outputs[] = {&loss, &total};
    const uint32_t mode = reduction == 1 ? 2 : reduction == 2 ? 4 : 5;
    run(inputs, 4, outputs, 2, mode, ignore_index.expect_int(),
        static_cast<uint32_t>(batch), static_cast<uint32_t>(classes));
    return {loss, total};
}

at::Tensor nll_loss_backward(const at::Tensor &grad_output, const at::Tensor &self,
                             const at::Tensor &target,
                             const c10::optional<at::Tensor> &weight, int64_t reduction,
                             c10::SymInt ignore_index, const at::Tensor &total_weight) {
    validate_float(grad_output, "grad_output");
    validate_logits(self);
    validate_labels(target, self.size(0));
    validate_float(total_weight, "total_weight");
    TORCH_CHECK(total_weight.dim() == 0 &&
                    (!weight.has_value() || !weight->defined() || weight->numel() == 0),
                "Vulkan nll_loss backward requires scalar total_weight and no weights");
    TORCH_CHECK(reduction >= 0 && reduction <= 2,
                "Vulkan nll_loss backward supports reduction none, mean, or sum");
    TORCH_CHECK(grad_output.dim() == (reduction == 0 ? 1 : 0) &&
                    (reduction != 0 || grad_output.size(0) == self.size(0)),
                "Vulkan nll_loss backward grad_output shape does not match reduction");
    TORCH_CHECK(ignore_index.expect_int() >= std::numeric_limits<int32_t>::min() &&
                    ignore_index.expect_int() <= std::numeric_limits<int32_t>::max(),
                "Vulkan nll_loss ignore_index does not fit in int32");
    auto output = at::empty_like(self);
    const at::Tensor *inputs[] = {&grad_output, &self, &target, &total_weight};
    const at::Tensor *outputs[] = {&output};
    const uint32_t mode = reduction == 0 ? 7 : reduction == 1 ? 3 : 6;
    run(inputs, 4, outputs, 1, mode, ignore_index.expect_int(),
        static_cast<uint32_t>(self.size(0)), static_cast<uint32_t>(self.size(1)));
    return output;
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("nll_loss_forward", &pytorch_vulkan::nll_loss_forward);
    m.impl("nll_loss_backward", &pytorch_vulkan::nll_loss_backward);
}
