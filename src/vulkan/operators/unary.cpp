#include "unary.h"

#include "binary.h"
#include "out.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1;
}

void validate_input(const at::Tensor &input, const char *operation_name) {
    TORCH_CHECK(is_vulkan_device(input.device()), "Vulkan ", operation_name,
                " requires a Vulkan tensor");
    TORCH_CHECK(input.device().index() == 0, "Vulkan ", operation_name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided, "Vulkan ", operation_name,
                " requires a strided tensor");
    TORCH_CHECK(input.is_contiguous(), "Vulkan ", operation_name,
                " requires a contiguous tensor");
    TORCH_CHECK(input.storage_offset() == 0, "Vulkan ", operation_name,
                " does not support tensors with non-zero storage_offset()");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Vulkan ", operation_name,
                " supports only float32 tensors");
    TORCH_CHECK(input.dim() != 0, "Vulkan ", operation_name,
                " does not support zero-dimensional tensor operands");
}

std::size_t checked_bytes(const at::Tensor &input, const char *operation_name) {
    const int64_t elements = input.numel();
    TORCH_CHECK(elements >= 0, "Vulkan ", operation_name,
                " has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / sizeof(float),
                "Vulkan ", operation_name, " byte count does not fit size_t");
    const auto bytes = count * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(), "Vulkan ",
                operation_name, " byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

at::Tensor dispatch_unary(const at::Tensor &input,
                          pytorch_vulkan::PointwiseOperation operation,
                          const char *operation_name) {
    validate_input(input, operation_name);
    at::Tensor output =
        at::empty(input.sizes(), input.options().device(input.device()));
    const std::size_t bytes = checked_bytes(input, operation_name);
    if (bytes == 0) {
        return output;
    }

    const at::DataPtr &input_data = input.storage().data_ptr();
    const at::DataPtr &output_data = output.storage().data_ptr();
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    pytorch_vulkan::validate_allocation(input_data, size, "input");
    pytorch_vulkan::validate_allocation(output_data, size, "output");
    const VulkanPlatform &input_platform =
        pytorch_vulkan::allocation_platform(input_data);
    const VulkanPlatform &output_platform =
        pytorch_vulkan::allocation_platform(output_data);
    TORCH_CHECK(&input_platform == &output_platform &&
                    input_platform.device() == output_platform.device(),
                "Vulkan ", operation_name,
                " requires input and output to use the same Vulkan platform/device");

    VulkanBuffer &input_buffer = pytorch_vulkan::allocation_buffer(input_data);
    VulkanBuffer &output_buffer = pytorch_vulkan::allocation_buffer(output_data);
    input_platform.compute().unary(input_buffer.buffer(), output_buffer.buffer(), size,
                                   static_cast<uint32_t>(operation));
    return output;
}

} // namespace

namespace pytorch_vulkan {

at::Tensor neg_tensor(const at::Tensor &input) {
    return dispatch_unary(input, PointwiseOperation::Neg, "neg");
}

at::Tensor abs_tensor(const at::Tensor &input) {
    return dispatch_unary(input, PointwiseOperation::Abs, "abs");
}

at::Tensor relu_tensor(const at::Tensor &input) {
    return dispatch_unary(input, PointwiseOperation::Relu, "relu");
}

at::Tensor &neg_out(const at::Tensor &input, at::Tensor &out) {
    return pytorch_vulkan::dispatch_unary_out(input, out, PointwiseOperation::Neg, "neg");
}

at::Tensor &abs_out(const at::Tensor &input, at::Tensor &out) {
    return pytorch_vulkan::dispatch_unary_out(input, out, PointwiseOperation::Abs, "abs");
}

at::Tensor &relu_out(const at::Tensor &input, at::Tensor &out) {
    return pytorch_vulkan::dispatch_unary_out(input, out, PointwiseOperation::Relu, "relu");
}

at::Tensor &reject_neg_inplace(at::Tensor &self) {
    TORCH_CHECK(false, "Vulkan neg in-place variants are unsupported");
    return self;
}

at::Tensor &reject_abs_inplace(at::Tensor &self) {
    TORCH_CHECK(false, "Vulkan abs in-place variants are unsupported");
    return self;
}

at::Tensor &reject_relu_inplace(at::Tensor &self) {
    TORCH_CHECK(false, "Vulkan relu in-place variants are unsupported");
    return self;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("neg", &pytorch_vulkan::neg_tensor);
    m.impl("abs", &pytorch_vulkan::abs_tensor);
    m.impl("relu", &pytorch_vulkan::relu_tensor);
    m.impl("neg.out", &pytorch_vulkan::neg_out);
    m.impl("abs.out", &pytorch_vulkan::abs_out);
    m.impl("relu.out", &pytorch_vulkan::relu_out);
    m.impl("neg_", &pytorch_vulkan::reject_neg_inplace);
    m.impl("abs_", &pytorch_vulkan::reject_abs_inplace);
    m.impl("relu_", &pytorch_vulkan::reject_relu_inplace);
}
