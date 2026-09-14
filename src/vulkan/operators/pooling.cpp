#include "pooling.h"
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
VkDeviceSize bytes(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.numel() >= 0 &&
                    static_cast<uint64_t>(tensor.numel()) <=
                        std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan pooling ", name, " byte count overflows");
    const uint64_t value = static_cast<uint64_t>(tensor.numel()) * sizeof(float);
    TORCH_CHECK(value <= std::numeric_limits<VkDeviceSize>::max(), "Vulkan pooling ",
                name, " byte count exceeds range");
    return static_cast<VkDeviceSize>(value);
}
void validate(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan pooling ", name, " requires vk:0");
    TORCH_CHECK(tensor.dim() == 4, "Vulkan pooling ", name,
                " requires rank 4 NCHW input");
    TORCH_CHECK(tensor.numel() > 0, "Vulkan pooling ", name,
                " requires a nonempty tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan pooling ", name, " requires a strided float32 tensor");
    TORCH_CHECK(tensor.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(2) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(3) <= std::numeric_limits<uint32_t>::max(),
                "Vulkan pooling ", name, " dimensions exceed Vulkan limits");
}
void validate_output_size(at::IntArrayRef output_size) {
    TORCH_CHECK(output_size.size() == 2 && output_size[0] == 1 && output_size[1] == 1,
                "Vulkan pooling supports only output size (1, 1)");
}
at::Tensor dispatch(const at::Tensor &input, const at::Tensor &grad,
                    uint32_t operation) {
    validate(input, "input");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "pooling input");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan pooling rejects overlapping input layouts");
    VulkanTensorLayout grad_layout = input_layout;
    if (operation == 1) {
        validate(grad, "grad_output");
        grad_layout = inspect_vulkan_tensor_layout(grad, "pooling grad_output");
        TORCH_CHECK(grad_layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan pooling rejects overlapping grad_output layouts");
        TORCH_CHECK(grad.sizes().equals({input.size(0), input.size(1), 1, 1}),
                    "Vulkan pooling grad_output has unsupported shape");
    }
    at::Tensor output = operation == 0 ? at::empty({input.size(0), input.size(1), 1, 1},
                                                   input.options())
                                       : at::empty(input.sizes(), input.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "pooling output");
    const auto &input_data = input.storage().data_ptr();
    const auto &grad_data = grad.defined() ? grad.storage().data_ptr() : input_data;
    const auto &output_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(grad_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan pooling requires Vulkan allocations on one device");
    validate_allocation(input_data, bytes(input, "input"), "pooling input");
    if (operation == 1)
        validate_allocation(grad_data, bytes(grad, "grad_output"),
                            "pooling grad_output");
    validate_allocation(output_data, bytes(output, "output"), "pooling output");
    platform.compute().pooling(
        allocation_buffer(operation == 0 ? input_data : grad_data).buffer(),
        allocation_buffer(output_data).buffer(), operation == 0 ? input_layout : grad_layout,
        output_layout, static_cast<uint32_t>(input.size(0)),
        static_cast<uint32_t>(input.size(1)), static_cast<uint32_t>(input.size(2)),
        static_cast<uint32_t>(input.size(3)), operation);
    return output;
}
} // namespace
at::Tensor adaptive_avg_pool2d(const at::Tensor &input, at::IntArrayRef output_size) {
    validate_output_size(output_size);
    return dispatch(input, at::Tensor(), 0);
}
at::Tensor adaptive_avg_pool2d_backward(const at::Tensor &grad_output,
                                        const at::Tensor &input) {
    return dispatch(input, grad_output, 1);
}
std::tuple<at::Tensor, at::Tensor>
max_pool2d_with_indices(const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                        at::IntArrayRef, at::IntArrayRef, bool) {
    TORCH_CHECK(false, "Vulkan pooling is not declared in the implemented model slice");
}
} // namespace pytorch_vulkan
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_adaptive_avg_pool2d", &pytorch_vulkan::adaptive_avg_pool2d);
    m.impl("_adaptive_avg_pool2d_backward",
           &pytorch_vulkan::adaptive_avg_pool2d_backward);
    m.impl("max_pool2d_with_indices", &pytorch_vulkan::max_pool2d_with_indices);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("_adaptive_avg_pool2d", &pytorch_vulkan::autograd_adaptive_avg_pool2d);
}
