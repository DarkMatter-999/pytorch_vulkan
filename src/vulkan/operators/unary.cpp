#include "unary.h"

#include "autograd.h"
#include "binary.h"
#include "capability.h"
#include "out.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"
#include "vulkan_layout.h"
#include "formatter_double.h"
#include "fake_tensor.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/neg.h>
#include <torch/library.h>

#include <limits>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1;
}

pytorch_vulkan::VulkanTensorLayout validate_input(
    const at::Tensor &input, pytorch_vulkan::PointwiseOperation operation,
    const char *operation_name) {
    TORCH_CHECK(is_vulkan_device(input.device()), "Vulkan ", operation_name,
                " requires a Vulkan tensor");
    TORCH_CHECK(input.device().index() == 0, "Vulkan ", operation_name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided, "Vulkan ", operation_name,
                " requires a strided tensor");
    pytorch_vulkan::validate_unary_dtype(input.scalar_type(), operation, operation_name);
    auto layout = pytorch_vulkan::inspect_vulkan_tensor_layout(input, operation_name);
    TORCH_CHECK(layout.rank <= 8, "Vulkan ", operation_name, " supports ranks up to 8");
    TORCH_CHECK(layout.numel <= std::numeric_limits<uint32_t>::max(), "Vulkan ", operation_name,
                " element count exceeds the supported range");
    return layout;
}

std::size_t checked_bytes(const at::Tensor &input, const char *operation_name) {
    const int64_t elements = input.numel();
    TORCH_CHECK(elements >= 0, "Vulkan ", operation_name,
                " has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    const std::size_t element_bytes =
        pytorch_vulkan::vulkan_storage_bytes(input.scalar_type());
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / element_bytes,
                "Vulkan ", operation_name, " byte count does not fit size_t");
    const auto bytes = count * element_bytes;
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(), "Vulkan ",
                operation_name, " byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

at::Tensor dispatch_unary(const at::Tensor &input,
                          pytorch_vulkan::PointwiseOperation operation,
                          const char *operation_name) {
    if (pytorch_vulkan::is_fake_tensor(input)) {
        const auto dispatch = c10::DispatchKeySet(c10::DispatchKey::Meta);
        switch (operation) {
        case pytorch_vulkan::PointwiseOperation::Neg:
            return at::_ops::neg::redispatch(dispatch, input);
        case pytorch_vulkan::PointwiseOperation::Abs:
            return at::_ops::abs::redispatch(dispatch, input);
        case pytorch_vulkan::PointwiseOperation::Relu:
            return at::_ops::relu::redispatch(dispatch, input);
        default:
            TORCH_CHECK(false, "Vulkan compiler does not support fake unary operation");
        }
    }
    const auto input_layout = validate_input(input, operation, operation_name);
    at::Tensor output =
        at::empty(input.sizes(), input.options().device(input.device()));
    const auto output_layout = validate_input(output, operation, operation_name);
    if (input_layout.numel == 0) {
        return output;
    }

    const at::DataPtr &input_data = input.storage().data_ptr();
    const at::DataPtr &output_data = output.storage().data_ptr();
    pytorch_vulkan::validate_allocation(input_data, input_layout.allocation_bytes, "input");
    pytorch_vulkan::validate_allocation(output_data, output_layout.allocation_bytes, "output");
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
    input_platform.compute().unary(input_buffer.buffer(), input_layout, output_buffer.buffer(),
                                   output_layout, static_cast<uint32_t>(operation));
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

at::Tensor ceil_tensor(const at::Tensor &input) {
    if (input.scalar_type() == at::kDouble)
        return formatter_double_ceil(input);
    return dispatch_unary(input, PointwiseOperation::Ceil, "ceil");
}

at::Tensor abs_backward_tensor(const at::Tensor &input, const at::Tensor &grad) {
    return at::mul(dispatch_unary(input, PointwiseOperation::AbsBackward, "abs backward"), grad);
}

at::Tensor relu_backward_tensor(const at::Tensor &output, const at::Tensor &grad) {
    return at::mul(dispatch_unary(output, PointwiseOperation::ReluBackward, "relu backward"), grad);
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

at::Tensor &sqrt_out(const at::Tensor &input, at::Tensor &out) {
    return pytorch_vulkan::dispatch_unary_out(input, out, PointwiseOperation::Sqrt, "sqrt");
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
    m.impl("ceil", &pytorch_vulkan::ceil_tensor);
    m.impl("neg.out", &pytorch_vulkan::neg_out);
    m.impl("abs.out", &pytorch_vulkan::abs_out);
    m.impl("relu.out", &pytorch_vulkan::relu_out);
    m.impl("sqrt.out", &pytorch_vulkan::sqrt_out);
    m.impl("neg_", &pytorch_vulkan::reject_neg_inplace);
    m.impl("abs_", &pytorch_vulkan::reject_abs_inplace);
    m.impl("relu_", &pytorch_vulkan::reject_relu_inplace);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("neg", &pytorch_vulkan::autograd_neg);
    m.impl("abs", &pytorch_vulkan::autograd_abs);
    m.impl("relu", &pytorch_vulkan::autograd_relu);
}
