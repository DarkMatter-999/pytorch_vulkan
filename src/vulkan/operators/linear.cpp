#include "linear.h"

#include "capability.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <limits>

namespace pytorch_vulkan {
namespace {
VkDeviceSize checked_bytes(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.numel() >= 0, "Vulkan ", name, " has a negative element count");
    const uint64_t elements = static_cast<uint64_t>(tensor.numel());
    TORCH_CHECK(elements <= std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan ", name, " byte count overflows uint64_t");
    const uint64_t bytes = elements * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() &&
                    bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan ", name, " byte count exceeds supported size");
    return static_cast<VkDeviceSize>(bytes);
}
}

at::Tensor linear(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.device().index() == 0,
                "Vulkan linear requires Vulkan device index 0");
    TORCH_CHECK(weight.device() == input.device(), "Vulkan linear requires weight on the input device");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "Vulkan linear supports only float32");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && input.size(1) == weight.size(1),
                "Vulkan linear supports contiguous 2-D input and weight with matching features");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && input.storage_offset() == 0 && weight.storage_offset() == 0,
                "Vulkan linear requires contiguous tensors with zero storage offset");
    TORCH_CHECK(bias.has_value(), "Vulkan linear currently requires a bias tensor");
    const auto &b = *bias;
    TORCH_CHECK(b.device() == input.device() && b.scalar_type() == at::kFloat && b.dim() == 1 &&
                    b.size(0) == weight.size(0) && b.is_contiguous() && b.storage_offset() == 0,
                "Vulkan linear requires a contiguous float32 bias with out_features elements");
    at::Tensor output = at::empty({input.size(0), weight.size(0)}, input.options());
    const auto &in_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &bias_data = b.storage().data_ptr();
    const auto &out_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(in_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) && &platform == &allocation_platform(bias_data) &&
                    &platform == &allocation_platform(out_data), "Vulkan linear requires one Vulkan platform");
    const VkDeviceSize input_bytes = checked_bytes(input, "linear input");
    const VkDeviceSize weight_bytes = checked_bytes(weight, "linear weight");
    const VkDeviceSize bias_bytes = checked_bytes(b, "linear bias");
    const VkDeviceSize output_bytes = checked_bytes(output, "linear output");
    validate_allocation(in_data, input_bytes, "linear input");
    validate_allocation(weight_data, weight_bytes, "linear weight");
    validate_allocation(bias_data, bias_bytes, "linear bias");
    validate_allocation(out_data, output_bytes, "linear output");
    TORCH_CHECK(input.size(0) <= std::numeric_limits<uint32_t>::max() && input.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    weight.size(0) <= std::numeric_limits<uint32_t>::max(), "Vulkan linear dimensions exceed dispatch limits");
    platform.compute().linear(allocation_buffer(in_data).buffer(), allocation_buffer(weight_data).buffer(),
                              allocation_buffer(bias_data).buffer(), allocation_buffer(out_data).buffer(),
                              static_cast<uint32_t>(input.size(0)), static_cast<uint32_t>(input.size(1)),
                              static_cast<uint32_t>(weight.size(0)));
    return output;
}
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("linear", &pytorch_vulkan::linear);
}
