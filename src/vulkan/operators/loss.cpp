#include "loss.h"

#include "capability.h"
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
void validate_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan mse_loss ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan mse_loss ", name, " requires a strided float32 tensor");
    // Validate readable device geometry before any materialization/dispatch.
    inspect_vulkan_tensor_layout(tensor, name);
}

uint32_t validate_common(const at::Tensor &input, const at::Tensor &target,
                         int64_t reduction, const char *name) {
    TORCH_CHECK(reduction >= 0 && reduction <= 2, "Vulkan mse_loss ", name,
                " supports reductions none, mean, and sum");
    validate_tensor(input, "input");
    validate_tensor(target, "target");
    TORCH_CHECK(input.sizes().equals(target.sizes()), "Vulkan mse_loss ", name,
                " requires matching input and target shapes");
    TORCH_CHECK(input.numel() >= 0 && static_cast<uint64_t>(input.numel()) <=
                                          std::numeric_limits<uint32_t>::max(),
                "Vulkan mse_loss ", name, " exceeds dispatch limits");
    return static_cast<uint32_t>(input.numel());
}

at::Tensor dispatch_loss(const at::Tensor &input_view,
                         const at::Tensor &target_view, int64_t reduction) {
    const uint32_t elements =
        validate_common(input_view, target_view, reduction, "forward");
    const auto input = input_view.contiguous();
    const auto target = target_view.contiguous();
    auto output =
        reduction == 0 ? at::empty_like(input) : at::empty({}, input.options());
    if (elements == 0) {
        if (reduction != 0) {
            auto layout = inspect_vulkan_tensor_layout(output, "mse_loss output");
            const auto &data = output.storage().data_ptr();
            validate_allocation(data, layout.allocation_bytes, "mse_loss output");
            allocation_platform(data).fill_buffer_sync(
                allocation_buffer(data).buffer(), layout.byte_offset, layout.byte_range,
                reduction == 1 ? 0x7fc00000U : 0U);
        }
        return output;
    }
    const auto input_layout = inspect_vulkan_tensor_layout(input, "mse_loss input");
    const auto target_layout = inspect_vulkan_tensor_layout(target, "mse_loss target");
    const auto output_layout = inspect_vulkan_tensor_layout(output, "mse_loss output");
    const auto &input_data = input.storage().data_ptr();
    const auto &target_data = target.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    validate_allocation(input_data, input_layout.allocation_bytes, "mse_loss input");
    validate_allocation(target_data, target_layout.allocation_bytes, "mse_loss target");
    validate_allocation(output_data, output_layout.allocation_bytes, "mse_loss output");
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(target_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan mse_loss requires one Vulkan platform");
    platform.compute().mse_loss(&allocation_buffer(input_data), input_layout,
                                &allocation_buffer(target_data), target_layout,
                                &allocation_buffer(input_data), input_layout,
                                &allocation_buffer(output_data), output_layout,
                                elements, static_cast<uint32_t>(reduction), false);
    return output;
}
} // namespace

at::Tensor mse_loss(const at::Tensor &input, const at::Tensor &target,
                    int64_t reduction) {
    return dispatch_loss(input, target, reduction);
}

at::Tensor mse_loss_backward(const at::Tensor &grad_view,
                             const at::Tensor &input_view,
                             const at::Tensor &target_view, int64_t reduction) {
    const uint32_t elements =
        validate_common(input_view, target_view, reduction, "backward");
    validate_tensor(grad_view, "grad_output");
    TORCH_CHECK(grad_view.dim() <= input_view.dim(),
                 "Vulkan mse_loss backward has an invalid grad_output shape");
    for (int64_t i = 0; i < grad_view.dim(); ++i) {
        const auto input_dim = input_view.dim() - grad_view.dim() + i;
        TORCH_CHECK(grad_view.size(i) == 1 ||
                        grad_view.size(i) == input_view.size(input_dim),
                    "Vulkan mse_loss backward has an invalid grad_output shape");
    }
    if (grad_view.numel() == 1) {
        const auto layout =
            inspect_vulkan_tensor_layout(grad_view, "mse_loss scalar grad");
        // Descriptors start at the allocation base. Zero is the dense sentinel;
        // scalar addressing stores the allocation element offset plus one.
        TORCH_CHECK(layout.storage_offset >= 0 &&
                        static_cast<uint64_t>(layout.storage_offset) <
                            std::numeric_limits<uint32_t>::max() &&
                        layout.element_bytes == sizeof(float) &&
                        layout.byte_offset ==
                            static_cast<uint64_t>(layout.storage_offset) * sizeof(float) &&
                        layout.byte_offset <= layout.allocation_bytes &&
                        sizeof(float) <= layout.allocation_bytes - layout.byte_offset,
                    "Vulkan mse_loss scalar grad exceeds addressing bounds");
        validate_allocation(grad_view.storage().data_ptr(), layout.allocation_bytes,
                            "mse_loss scalar grad");
    }
    // A one-element upstream is readable directly without an expanded copy.
    // Stock second reverse also passes full-shaped p for mean/sum: retain every
    // logical value there, independently of numerical reduction normalization.
    const auto input = input_view.contiguous();
    const auto target = target_view.contiguous();
    const auto grad_output = grad_view.numel() == 1
                                 ? grad_view
                                 : grad_view.expand(input_view.sizes()).contiguous();
    auto result = at::empty_like(input);
    if (elements == 0)
        return result;
    const auto input_layout =
        inspect_vulkan_tensor_layout(input, "mse_loss backward input");
    const auto target_layout =
        inspect_vulkan_tensor_layout(target, "mse_loss backward target");
    const auto grad_layout =
        inspect_vulkan_tensor_layout(grad_output, "mse_loss backward grad");
    const auto result_layout =
        inspect_vulkan_tensor_layout(result, "mse_loss backward result");
    const auto &input_data = input.storage().data_ptr();
    const auto &target_data = target.storage().data_ptr();
    const auto &grad_data = grad_output.storage().data_ptr();
    const auto &result_data = result.storage().data_ptr();
    validate_allocation(input_data, input_layout.allocation_bytes,
                        "mse_loss backward input");
    validate_allocation(target_data, target_layout.allocation_bytes,
                        "mse_loss backward target");
    validate_allocation(grad_data, grad_layout.allocation_bytes,
                        "mse_loss backward grad");
    validate_allocation(result_data, result_layout.allocation_bytes,
                        "mse_loss backward result");
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(target_data) &&
                    &platform == &allocation_platform(grad_data) &&
                    &platform == &allocation_platform(result_data),
                "Vulkan mse_loss backward requires one Vulkan platform");
    platform.compute().mse_loss(&allocation_buffer(input_data), input_layout,
                                &allocation_buffer(target_data), target_layout,
                                &allocation_buffer(grad_data), grad_layout,
                                &allocation_buffer(result_data), result_layout,
                                elements, static_cast<uint32_t>(reduction), true);
    return result;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mse_loss", &pytorch_vulkan::mse_loss);
    m.impl("mse_loss_backward", &pytorch_vulkan::mse_loss_backward);
}
