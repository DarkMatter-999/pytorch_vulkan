#include "formatter_double.h"

#include "binary.h"
#include "unary.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace pytorch_vulkan {
namespace {

enum : uint32_t { kAbs, kMin, kMax, kDiv, kNe, kGt, kCeil, kLt };

void validate(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.device().index() == 0,
                "Vulkan formatter Double ", name, " requires vk:0 input");
    TORCH_CHECK(input.scalar_type() == at::kDouble, "Vulkan formatter Double ", name,
                " requires float64 input");
    TORCH_CHECK(input.layout() == at::kStrided && input.is_contiguous() &&
                    input.storage_offset() == 0,
                "Vulkan formatter Double ", name,
                " requires a contiguous zero-offset strided input");
    TORCH_CHECK(pytorch_vulkan::formatter_double_supported(),
                "Vulkan formatter Double support is unavailable");
}

uint32_t checked_count(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.numel() > 0, "Vulkan formatter Double ", name,
                " does not support empty input");
    TORCH_CHECK(
        static_cast<uint64_t>(input.numel()) <= std::numeric_limits<uint32_t>::max(),
        "Vulkan formatter Double ", name, " element count exceeds uint32 range");
    return static_cast<uint32_t>(input.numel());
}

VkDeviceSize checked_bytes(uint64_t count, const char *name) {
    TORCH_CHECK(count <= std::numeric_limits<VkDeviceSize>::max() / sizeof(double),
                "Vulkan formatter Double ", name, " byte range overflows");
    return static_cast<VkDeviceSize>(count * sizeof(double));
}

at::Tensor dispatch(const at::Tensor &input, uint32_t operation, const char *name,
                    bool bool_output = false, double scalar = 0.0,
                    uint32_t output_numel = 0, bool reduction_output = false,
                    const at::Tensor *rhs_tensor = nullptr) {
    validate(input, name);
    const uint32_t count = checked_count(input, name);
    at::Tensor output =
        at::empty(reduction_output ? at::IntArrayRef{} : input.sizes(),
                  input.options().dtype(bool_output ? at::kBool : at::kDouble));
    if (output_numel == 0)
        output_numel = count;
    const VkDeviceSize input_bytes = checked_bytes(count, name);
    const VkDeviceSize output_bytes = bool_output ? static_cast<VkDeviceSize>(count)
                                                  : checked_bytes(output_numel, name);
    const auto &input_data = input.storage().data_ptr();
    const auto &rhs_data = rhs_tensor ? rhs_tensor->storage().data_ptr() : input_data;
    const auto &output_data = output.storage().data_ptr();
    const VkDeviceSize rhs_bytes =
        rhs_tensor ? checked_bytes(static_cast<uint64_t>(rhs_tensor->numel()), name)
                   : input_bytes;
    validate_allocation(input_data, input_bytes, "formatter input");
    validate_allocation(rhs_data, rhs_bytes, "formatter rhs");
    validate_allocation(output_data, output_bytes, "formatter output");
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(output_data),
                "Vulkan formatter Double ", name, " requires one Vulkan platform");
    platform.compute().formatter_double(
        allocation_buffer(input_data).buffer(), allocation_buffer(rhs_data).buffer(),
        allocation_buffer(output_data).buffer(), input_bytes, rhs_bytes, output_bytes,
        count, operation, scalar, output_numel, bool_output);
    return output;
}

} // namespace

at::Tensor formatter_double_abs(const at::Tensor &input) {
    return dispatch(input, kAbs, "abs");
}
at::Tensor formatter_double_min(const at::Tensor &input) {
    return dispatch(input, kMin, "min", false, 0.0, 1, true);
}
at::Tensor formatter_double_max(const at::Tensor &input) {
    return dispatch(input, kMax, "max", false, 0.0, 1, true);
}
at::Tensor formatter_double_div(const at::Tensor &input, const at::Tensor &other) {
    if (input.scalar_type() == at::kFloat)
        return pytorch_vulkan::div_tensor(input, other);
    validate(input, "div.Tensor");
    validate(other, "div.Tensor");
    TORCH_CHECK(other.dim() == 0 && other.device() == input.device() &&
                    other.layout() == at::kStrided && other.is_contiguous() &&
                    other.storage_offset() == 0,
                "Vulkan formatter Double div.Tensor requires a scalar on vk:0");
    const auto &input_data = input.storage().data_ptr();
    const auto &other_data = other.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(other_data),
                "Vulkan formatter Double div.Tensor requires one Vulkan platform");
    return dispatch(input, kDiv, "div.Tensor", false, 0.0, 0, false, &other);
}
at::Tensor formatter_double_ne(const at::Tensor &input, const at::Tensor &other) {
    if (input.scalar_type() == at::kFloat && other.scalar_type() == at::kFloat) {
        TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                        input.device().index() == 0 && input.is_contiguous() &&
                        other.device() == input.device() && other.is_contiguous() &&
                        other.numel() == input.numel(),
                    "Vulkan ne.Tensor requires equal Vulkan shapes and devices");
        TORCH_CHECK(
            (input.storage_offset() == 0 || input.dim() == 0) &&
                (other.storage_offset() == 0 || other.dim() == 0),
            "Vulkan ne.Tensor does not support tensors with non-zero storage_offset()");
        at::Tensor output = at::empty(input.sizes(), input.options().dtype(at::kBool));
        const auto count = checked_count(input, "ne.Tensor");
        const auto bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
        const auto &lhs = input.storage().data_ptr();
        const auto &rhs = other.storage().data_ptr();
        const auto &out = output.storage().data_ptr();
        validate_allocation(lhs, bytes, "ne lhs");
        validate_allocation(rhs, bytes, "ne rhs");
        validate_allocation(out, count, "ne output");
        const auto &platform = allocation_platform(lhs);
        TORCH_CHECK(&platform == &allocation_platform(rhs) &&
                        &platform == &allocation_platform(out),
                    "Vulkan ne.Tensor requires one Vulkan platform");
        platform.compute().comparison_tensor(
            allocation_buffer(lhs).buffer(),
            inspect_vulkan_tensor_layout(input, "ne lhs"),
            allocation_buffer(rhs).buffer(),
            inspect_vulkan_tensor_layout(other, "ne rhs"),
            allocation_buffer(out).buffer(),
            inspect_vulkan_tensor_layout(output, "ne output"), 10);
        return output;
    }
    validate(input, "ne.Tensor");
    validate(other, "ne.Tensor");
    TORCH_CHECK(other.dim() == 0 && other.device() == input.device() &&
                    other.layout() == at::kStrided && other.is_contiguous() &&
                    other.storage_offset() == 0,
                "Vulkan formatter Double ne.Tensor requires a scalar on vk:0");
    return dispatch(input, kNe, "ne.Tensor", true, 0.0, 0, false, &other);
}
at::Tensor formatter_double_gt(const at::Tensor &input, const at::Scalar &other) {
    TORCH_CHECK(!other.isComplex(),
                "Vulkan formatter Double gt.Scalar requires a real scalar");
    return dispatch(input, kGt, "gt.Scalar", true, other.toDouble());
}
at::Tensor formatter_double_lt(const at::Tensor &input, const at::Scalar &other) {
    TORCH_CHECK(!other.isComplex(),
                "Vulkan formatter Double lt.Scalar requires a real scalar");
    return dispatch(input, kLt, "lt.Scalar", true, other.toDouble());
}
at::Tensor formatter_double_ceil(const at::Tensor &input) {
    if (input.scalar_type() == at::kFloat)
        return ceil_tensor(input);
    return dispatch(input, kCeil, "ceil");
}
at::Scalar formatter_double_local_scalar_dense(const at::Tensor &input) {
    TORCH_CHECK(
        (input.scalar_type() == at::kBool || input.scalar_type() == at::kFloat ||
         input.scalar_type() == at::kDouble) &&
            input.device().type() == c10::DeviceType::PrivateUse1 &&
            input.device().index() == 0 && input.dim() == 0 && input.is_contiguous(),
        "Vulkan formatter _local_scalar_dense requires a contiguous scalar");
    at::Tensor cpu = at::empty({}, input.options().device(c10::kCPU));
    formatter_presentation_copy(cpu, input);
    return cpu.item();
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("min", &pytorch_vulkan::formatter_double_min);
    m.impl("max", &pytorch_vulkan::formatter_double_max);
    m.impl("div.Tensor", &pytorch_vulkan::formatter_double_div);
    m.impl("ne.Tensor", &pytorch_vulkan::formatter_double_ne);
    m.impl("gt.Scalar", &pytorch_vulkan::formatter_double_gt);
    m.impl("lt.Scalar", &pytorch_vulkan::formatter_double_lt);
    m.impl("_local_scalar_dense", &pytorch_vulkan::formatter_double_local_scalar_dense);
}
