#include "autograd.h"
#include "binary.h"
#include "capability.h"
#include "out.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <string>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1;
}

std::size_t checked_bytes(const at::Tensor &tensor) {
    const int64_t elements = tensor.numel();
    TORCH_CHECK(elements >= 0, "Vulkan add has a negative element count");
    const auto count = static_cast<uint64_t>(elements);
    const std::size_t element_bytes =
        pytorch_vulkan::vulkan_storage_bytes(tensor.scalar_type());
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / element_bytes,
                "Vulkan add byte count does not fit size_t");
    const auto bytes = count * element_bytes;
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan add byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

pytorch_vulkan::VulkanTensorLayout
validate_tensor(const at::Tensor &tensor, pytorch_vulkan::PointwiseOperation operation,
                const char *operation_name) {
    TORCH_CHECK(is_vulkan_device(tensor.device()), "Vulkan ", operation_name,
                " requires a Vulkan tensor");
    TORCH_CHECK(tensor.device().index() == 0, "Vulkan ", operation_name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided, "Vulkan ", operation_name,
                " requires a strided tensor");
    pytorch_vulkan::validate_scalar_dtype(tensor.scalar_type(), operation,
                                          operation_name);
    auto layout = pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, operation_name);
    TORCH_CHECK(layout.rank <= 8, "Vulkan ", operation_name, " supports ranks up to 8");
    TORCH_CHECK(layout.numel <= std::numeric_limits<uint32_t>::max(), "Vulkan ",
                operation_name, " element count exceeds the supported range");
    return layout;
}

pytorch_vulkan::VulkanTensorLayout validate_binary_layout(const at::Tensor &tensor,
                                                          const char *operation_name) {
    TORCH_CHECK(is_vulkan_device(tensor.device()), "Vulkan ", operation_name,
                " requires a Vulkan tensor");
    TORCH_CHECK(tensor.device().index() == 0, "Vulkan ", operation_name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided, "Vulkan ", operation_name,
                " requires a strided tensor");
    auto layout = pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, operation_name);
    TORCH_CHECK(layout.rank <= 8, "Vulkan ", operation_name, " supports ranks up to 8");
    TORCH_CHECK(layout.numel <= std::numeric_limits<uint32_t>::max(), "Vulkan ",
                operation_name, " element count exceeds the supported range");
    return layout;
}

void validate(const at::Tensor &lhs, const at::Tensor &rhs, const at::Scalar &alpha,
              pytorch_vulkan::PointwiseOperation operation,
              const char *operation_name) {
    TORCH_CHECK(!(lhs.device().is_cpu() && rhs.device().is_cpu()), "Vulkan ",
                operation_name, " requires a Vulkan operand");
    TORCH_CHECK(!(lhs.device().is_cpu() && lhs.dim() == 0) &&
                    !(rhs.device().is_cpu() && rhs.dim() == 0),
                "Vulkan ", operation_name, " does not support a scalar operand");
    TORCH_CHECK(is_vulkan_device(lhs.device()) && is_vulkan_device(rhs.device()),
                "Vulkan ", operation_name,
                " requires operands on the same device; got ", lhs.device(), " and ",
                rhs.device());
    TORCH_CHECK(lhs.device() == rhs.device(), "Vulkan ", operation_name,
                " requires operands on the same device; got ", lhs.device(), " and ",
                rhs.device());
    TORCH_CHECK(lhs.device().index() == 0 && rhs.device().index() == 0, "Vulkan ",
                operation_name, " supports only Vulkan device index 0");
    TORCH_CHECK(lhs.layout() == at::kStrided && rhs.layout() == at::kStrided, "Vulkan ",
                operation_name, " requires strided tensors");
    pytorch_vulkan::validate_binary_dtypes(lhs.scalar_type(), rhs.scalar_type(),
                                           operation, operation_name);
    TORCH_CHECK(lhs.sizes().equals(rhs.sizes()), "Vulkan ", operation_name,
                " requires equal tensor sizes; broadcasting is unsupported");
    (void)validate_binary_layout(lhs, operation_name);
    (void)validate_binary_layout(rhs, operation_name);
}

float scalar_to_float(const at::Scalar &scalar, const char *operation_name) {
    TORCH_CHECK(!scalar.isComplex(), "Vulkan ", operation_name,
                " requires a real numeric scalar");
    try {
        const double value = scalar.toDouble();
        TORCH_CHECK(std::isfinite(value), "Vulkan ", operation_name,
                    " does not support non-finite scalar values");
        const float result = static_cast<float>(value);
        TORCH_CHECK(std::isfinite(result), "Vulkan ", operation_name,
                    " scalar is outside float32 range");
        TORCH_CHECK(value == 0.0 || result != 0.0, "Vulkan ", operation_name,
                    " scalar underflows to float32 zero");
        return result;
    } catch (const c10::Error &) {
        throw;
    } catch (const std::exception &error) {
        TORCH_CHECK(false, "Vulkan ", operation_name,
                    " could not convert scalar: ", error.what());
    }
}

at::Tensor dispatch_tensor_scalar(const at::Tensor &tensor, float scalar,
                                  pytorch_vulkan::PointwiseOperation operation,
                                  bool scalar_left, const char *operation_name) {
    const auto tensor_layout = validate_tensor(tensor, operation, operation_name);
    at::Tensor output =
        at::empty(tensor.sizes(), tensor.options().device(tensor.device()));
    const auto output_layout = validate_tensor(output, operation, operation_name);
    if (tensor_layout.numel == 0) {
        return output;
    }

    const at::DataPtr &tensor_data = tensor.storage().data_ptr();
    const at::DataPtr &output_data = output.storage().data_ptr();
    pytorch_vulkan::validate_allocation(tensor_data, tensor_layout.allocation_bytes,
                                        "tensor");
    pytorch_vulkan::validate_allocation(output_data, output_layout.allocation_bytes,
                                        "output");
    VulkanBuffer &tensor_buffer = pytorch_vulkan::allocation_buffer(tensor_data);
    VulkanBuffer &output_buffer = pytorch_vulkan::allocation_buffer(output_data);
    const VulkanPlatform &tensor_platform =
        pytorch_vulkan::allocation_platform(tensor_data);
    const VulkanPlatform &output_platform =
        pytorch_vulkan::allocation_platform(output_data);
    TORCH_CHECK(&tensor_platform == &output_platform &&
                    tensor_platform.device() == output_platform.device(),
                "Vulkan ", operation_name,
                " requires all tensors to use the same Vulkan platform/device");

    const uint32_t op = static_cast<uint32_t>(operation);
    if (scalar_left) {
        tensor_platform.compute().scalar_tensor(scalar, tensor_buffer.buffer(),
                                                tensor_layout, output_buffer.buffer(),
                                                output_layout, op);
    } else {
        tensor_platform.compute().tensor_scalar(tensor_buffer.buffer(), tensor_layout,
                                                output_buffer.buffer(), output_layout,
                                                scalar, op);
    }
    return output;
}

} // namespace

namespace pytorch_vulkan {

at::Tensor pointwise_tensor_operands(const at::Tensor &lhs, const at::Tensor &rhs,
                                     const at::Scalar &alpha,
                                     PointwiseOperation operation,
                                     const char *operation_name) {
    const bool lhs_wrapped_number = lhs.device().is_cpu() && lhs.dim() == 0 &&
                                    lhs.unsafeGetTensorImpl()->is_wrapped_number();
    const bool rhs_wrapped_number = rhs.device().is_cpu() && rhs.dim() == 0 &&
                                    rhs.unsafeGetTensorImpl()->is_wrapped_number();
    if (!lhs_wrapped_number && !rhs_wrapped_number) {
        validate(lhs, rhs, alpha, operation, operation_name);
        const bool uses_bool = pointwise_uses_bool(lhs.scalar_type(), operation);
        validate_pointwise_device_capability(
            uses_bool,
            !uses_bool ||
                allocation_platform(lhs.storage().data_ptr()).supports_bool_pointwise(),
            operation_name);
        at::Tensor output = at::empty(lhs.sizes(), lhs.options().device(lhs.device()));
        const auto lhs_layout =
            pytorch_vulkan::inspect_vulkan_tensor_layout(lhs, operation_name);
        const auto rhs_layout =
            pytorch_vulkan::inspect_vulkan_tensor_layout(rhs, operation_name);
        const auto output_layout =
            pytorch_vulkan::inspect_vulkan_tensor_layout(output, operation_name);
        if (lhs_layout.numel == 0) {
            return output;
        }
        const at::DataPtr &lhs_data = lhs.storage().data_ptr();
        const at::DataPtr &rhs_data = rhs.storage().data_ptr();
        const at::DataPtr &output_data = output.storage().data_ptr();
        validate_allocation(lhs_data, lhs_layout.allocation_bytes, "lhs");
        validate_allocation(rhs_data, rhs_layout.allocation_bytes, "rhs");
        validate_allocation(output_data, output_layout.allocation_bytes, "output");
        VulkanBuffer &lhs_buffer = allocation_buffer(lhs_data);
        VulkanBuffer &rhs_buffer = allocation_buffer(rhs_data);
        VulkanBuffer &output_buffer = allocation_buffer(output_data);
        const VulkanPlatform &lhs_platform = allocation_platform(lhs_data);
        const VulkanPlatform &rhs_platform = allocation_platform(rhs_data);
        const VulkanPlatform &output_platform = allocation_platform(output_data);
        TORCH_CHECK(&lhs_platform == &rhs_platform &&
                        &lhs_platform == &output_platform &&
                        lhs_platform.device() == rhs_platform.device() &&
                        lhs_platform.device() == output_platform.device(),
                    "Vulkan ", operation_name,
                    " requires all tensors to use the same Vulkan platform/device");
        lhs_platform.compute().tensor_tensor(
            lhs_buffer.buffer(), lhs_layout, rhs_buffer.buffer(), rhs_layout,
            output_buffer.buffer(), output_layout, static_cast<uint32_t>(operation),
            pointwise_uses_bool(lhs.scalar_type(), operation),
            scalar_to_float(alpha, operation_name));
        return output;
    }
    TORCH_CHECK(operation == PointwiseOperation::Mul || alpha.toDouble() == 1.0,
                "Vulkan ", operation_name, " supports only alpha == 1");
    TORCH_CHECK(lhs_wrapped_number != rhs_wrapped_number, "Vulkan ", operation_name,
                " requires one Python numeric scalar and one Vulkan tensor; scalar "
                "operand is unsupported");
    if (lhs_wrapped_number) {
        return pointwise_tensor_scalar(rhs, lhs.item(), operation, true,
                                       operation_name);
    }
    return pointwise_tensor_scalar(lhs, rhs.item(), operation, false, operation_name);
}

at::Tensor add_tensor(const at::Tensor &lhs, const at::Tensor &rhs,
                      const at::Scalar &alpha) {
    return pointwise_tensor_operands(lhs, rhs, alpha, PointwiseOperation::Add, "add");
}

at::Tensor div_tensor(const at::Tensor &lhs, const at::Tensor &rhs) {
    TORCH_CHECK(rhs.device().is_cpu() && rhs.dim() == 0,
                "Vulkan div supports only a Python scalar denominator");
    return pointwise_tensor_scalar(lhs, rhs.item(), PointwiseOperation::Div, false,
                                   "div");
}

at::Tensor pointwise_tensor_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                                   PointwiseOperation operation, bool scalar_left,
                                   const char *operation_name) {
    return dispatch_tensor_scalar(tensor, scalar_to_float(scalar, operation_name),
                                  operation, scalar_left, operation_name);
}

at::Tensor add_scalar(const at::Tensor &tensor, const at::Scalar &scalar,
                      const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan add supports only alpha == 1");
    return pointwise_tensor_scalar(tensor, scalar, PointwiseOperation::Add, false,
                                   "add");
}

at::Tensor &add_out(const at::Tensor &lhs, const at::Tensor &rhs,
                    const at::Scalar &alpha, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_tensor_out(
        lhs, rhs, alpha, out, pytorch_vulkan::PointwiseOperation::Add, "add");
}

at::Tensor &add_scalar_out(const at::Tensor &tensor, const at::Scalar &scalar,
                           const at::Scalar &alpha, at::Tensor &out) {
    return pytorch_vulkan::dispatch_tensor_scalar_out(
        tensor, scalar, alpha, out, pytorch_vulkan::PointwiseOperation::Add, "add");
}

at::Tensor &add_inplace_tensor(at::Tensor &self, const at::Tensor &other,
                               const at::Scalar &alpha) {
    return pytorch_vulkan::dispatch_tensor_tensor_alias(
        self, other, alpha, pytorch_vulkan::PointwiseOperation::Add, "add_");
}

at::Tensor &add_inplace_scalar(at::Tensor &self, const at::Scalar &other,
                               const at::Scalar &alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan add_ scalar supports only alpha == 1");
    return pytorch_vulkan::dispatch_tensor_scalar_alias(
        self, other, pytorch_vulkan::PointwiseOperation::Add, "add_");
}

} // namespace pytorch_vulkan

namespace pytorch_vulkan {

at::Tensor &lerp_scalar_out(const at::Tensor &self, const at::Tensor &end,
                            const at::Scalar &weight, at::Tensor &out) {
    return dispatch_tensor_tensor_out(self, end, weight, out, PointwiseOperation::Lerp,
                                      "lerp");
}

at::Tensor &lerp_scalar_inplace(at::Tensor &self, const at::Tensor &end,
                                const at::Scalar &weight) {
    return lerp_scalar_out(self, end, weight, self);
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", &pytorch_vulkan::add_tensor);
    m.impl("add.Scalar", &pytorch_vulkan::add_scalar);
    m.impl("add.out", &pytorch_vulkan::add_out);
    m.impl("add.Scalar_out", &pytorch_vulkan::add_scalar_out);
    m.impl("add_.Tensor", &pytorch_vulkan::add_inplace_tensor);
    m.impl("add_.Scalar", &pytorch_vulkan::add_inplace_scalar);
    m.impl("zero_", &pytorch_vulkan::dispatch_zero);
    m.impl("fill_.Scalar", &pytorch_vulkan::dispatch_fill);
    m.impl("lerp.Scalar_out", &pytorch_vulkan::lerp_scalar_out);
    m.impl("lerp_.Scalar", &pytorch_vulkan::lerp_scalar_inplace);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("add.Tensor", &pytorch_vulkan::autograd_add_tensor);
    m.impl("add.Scalar", &pytorch_vulkan::autograd_add_scalar);
}
