#include "comparison.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace pytorch_vulkan {
namespace {

void validate_input(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.device().index() == 0,
                "Vulkan ", name, " requires vk:0 input");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Vulkan ", name,
                " requires float32 input");
    const auto layout = inspect_vulkan_tensor_layout(input, name);
    TORCH_CHECK(layout.rank <= 8, "Vulkan ", name, " supports ranks up to 8");
    TORCH_CHECK(layout.numel <= std::numeric_limits<uint32_t>::max(), "Vulkan ", name,
                " element count exceeds the supported range");
}

void validate_output(const at::Tensor &input, at::Tensor &out, const char *name) {
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kBool,
                "Vulkan ", name, " requires a bool vk:0 output");
    TORCH_CHECK(out.layout() == at::kStrided && out.sizes().equals(input.sizes()),
                "Vulkan ", name, " requires a strided output with equal shape");
    const auto layout = inspect_vulkan_tensor_layout(out, name);
    TORCH_CHECK(layout.rank <= 8, "Vulkan ", name, " supports ranks up to 8");
    TORCH_CHECK(at::has_internal_overlap(out) == at::MemOverlap::No, "Vulkan ", name,
                " output has internal overlap");
    const auto overlap = at::get_overlap_status(input, out);
    TORCH_CHECK(overlap != at::MemOverlapStatus::Partial &&
                    overlap != at::MemOverlapStatus::TooHard,
                "Vulkan ", name, " output partially overlaps an input");
    if (overlap == at::MemOverlapStatus::Full)
        TORCH_CHECK(input.data_ptr() == out.data_ptr() &&
                        input.strides().equals(out.strides()) &&
                        input.storage_offset() == out.storage_offset(),
                    "Vulkan ", name, " permits only exact full-tensor output aliases");
    const auto &input_data = input.storage().data_ptr();
    const auto &out_data = out.storage().data_ptr();
    const auto input_layout = inspect_vulkan_tensor_layout(input, name);
    const auto output_layout = inspect_vulkan_tensor_layout(out, name);
    TORCH_CHECK(is_vulkan_allocation(input_data) && is_vulkan_allocation(out_data),
                "Vulkan ", name, " requires Vulkan allocation provenance");
    const auto &input_platform = allocation_platform(input_data);
    const auto &out_platform = allocation_platform(out_data);
    TORCH_CHECK(&input_platform == &out_platform &&
                    input_platform.supports_bool_pointwise(),
                "Vulkan ", name, " requires bool pointwise Vulkan capability");
}

VkDeviceSize checked_bytes(const at::Tensor &input, const char *name) {
    TORCH_CHECK(input.numel() >= 0 &&
                    static_cast<uint64_t>(input.numel()) <=
                        std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan ", name, " input byte count overflows");
    return static_cast<VkDeviceSize>(static_cast<uint64_t>(input.numel()) *
                                     sizeof(float));
}

at::Tensor &dispatch(const at::Tensor &input, at::Tensor &out, bool finite,
                     float scalar, const char *name) {
    validate_input(input, name);
    validate_output(input, out, name);
    const auto input_layout = inspect_vulkan_tensor_layout(input, name);
    const auto output_layout = inspect_vulkan_tensor_layout(out, name);
    const VkDeviceSize bytes = checked_bytes(input, name);
    if (bytes == 0)
        return out;
    const auto &input_data = input.storage().data_ptr();
    const auto &out_data = out.storage().data_ptr();
    validate_allocation(input_data, bytes, "comparison input");
    validate_allocation(out_data,
                        static_cast<VkDeviceSize>(input.numel() * sizeof(bool)),
                        "comparison output");
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(out_data), "Vulkan ", name,
                " requires one Vulkan platform");
    if (finite) {
        platform.compute().isfinite(allocation_buffer(input_data).buffer(),
                                    input_layout, allocation_buffer(out_data).buffer(),
                                    output_layout);
    } else {
        platform.compute().comparison_scalar(
            allocation_buffer(input_data).buffer(), input_layout,
            allocation_buffer(out_data).buffer(), output_layout, scalar);
    }
    return out;
}

} // namespace

at::Tensor &ne_scalar_out(const at::Tensor &self, const at::Scalar &other,
                          at::Tensor &out) {
    TORCH_CHECK(!other.isComplex(), "Vulkan ne requires a real scalar");
    const double value = other.toDouble();
    const float scalar = other.toFloat();
    TORCH_CHECK(!std::isnan(value) && !std::isnan(scalar) &&
                    (std::isinf(value) || value == 0.0 || scalar != 0.0),
                "Vulkan ne scalar must be representable in float32, got ", value);
    return dispatch(self, out, false, scalar, "ne.Scalar_out");
}

at::Tensor &isfinite_out(const at::Tensor &self, at::Tensor &out) {
    return dispatch(self, out, true, 0.0F, "isfinite.out");
}

at::Tensor &eq_tensor_out(const at::Tensor &self, const at::Tensor &other,
                          at::Tensor &out) {
    validate_input(self, "eq.Tensor_out");
    validate_input(other, "eq.Tensor_out");
    TORCH_CHECK(other.device() == self.device() && other.sizes().equals(self.sizes()),
                "Vulkan eq.Tensor_out requires equal devices and shapes");
    const auto lhs_layout = inspect_vulkan_tensor_layout(self, "eq.Tensor_out");
    const auto rhs_layout = inspect_vulkan_tensor_layout(other, "eq.Tensor_out");
    validate_output(self, out, "eq.Tensor_out");
    const auto output_layout = inspect_vulkan_tensor_layout(out, "eq.Tensor_out");
    TORCH_CHECK(at::has_internal_overlap(out) == at::MemOverlap::No,
                "Vulkan eq.Tensor_out output has internal overlap");
    const auto lhs_overlap = at::get_overlap_status(self, out);
    const auto rhs_overlap = at::get_overlap_status(other, out);
    TORCH_CHECK(lhs_overlap == at::MemOverlapStatus::No &&
                    rhs_overlap == at::MemOverlapStatus::No,
                "Vulkan eq.Tensor_out output aliases an input");
    const VkDeviceSize bytes = checked_bytes(self, "eq.Tensor_out");
    if (bytes == 0)
        return out;
    const auto &lhs = self.storage().data_ptr();
    const auto &rhs = other.storage().data_ptr();
    const auto &result = out.storage().data_ptr();
    validate_allocation(lhs, bytes, "comparison lhs");
    validate_allocation(rhs, bytes, "comparison rhs");
    validate_allocation(result, static_cast<VkDeviceSize>(self.numel() * sizeof(bool)),
                        "comparison output");
    const auto &platform = allocation_platform(lhs);
    TORCH_CHECK(&platform == &allocation_platform(rhs) &&
                    &platform == &allocation_platform(result),
                "Vulkan eq.Tensor_out requires one Vulkan platform");
    platform.compute().comparison_tensor(
        allocation_buffer(lhs).buffer(), lhs_layout, allocation_buffer(rhs).buffer(),
        rhs_layout, allocation_buffer(result).buffer(), output_layout);
    return out;
}

at::Tensor &bitwise_and_tensor_out(const at::Tensor &self, const at::Tensor &other,
                                   at::Tensor &out) {
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1 &&
            other.device() == self.device() && out.device() == self.device() &&
            self.scalar_type() == at::kBool && other.scalar_type() == at::kBool &&
            out.scalar_type() == at::kBool && self.sizes().equals(other.sizes()) &&
            self.sizes().equals(out.sizes()),
        "Vulkan bitwise_and.Tensor_out requires equal-shape bool "
        "vk:0 tensors");
    const auto &lhs = self.storage().data_ptr();
    const auto &rhs = other.storage().data_ptr();
    const auto &result = out.storage().data_ptr();
    const auto lhs_layout = inspect_vulkan_tensor_layout(self, "bitwise_and lhs");
    const auto rhs_layout = inspect_vulkan_tensor_layout(other, "bitwise_and rhs");
    const auto output_layout = inspect_vulkan_tensor_layout(out, "bitwise_and output");
    for (const auto *layout : {&lhs_layout, &rhs_layout, &output_layout}) {
        TORCH_CHECK(layout->rank <= 8,
                    "Vulkan bitwise_and.Tensor_out supports ranks up to 8");
        TORCH_CHECK(
            layout->numel <= std::numeric_limits<uint32_t>::max(),
            "Vulkan bitwise_and.Tensor_out element count exceeds the supported range");
    }
    TORCH_CHECK(at::has_internal_overlap(out) == at::MemOverlap::No,
                "Vulkan bitwise_and.Tensor_out output has internal overlap");
    for (const auto *input : {&self, &other}) {
        const auto overlap = at::get_overlap_status(*input, out);
        TORCH_CHECK(overlap != at::MemOverlapStatus::Partial &&
                        overlap != at::MemOverlapStatus::TooHard,
                    "Vulkan bitwise_and.Tensor_out output partially overlaps an input");
        if (overlap == at::MemOverlapStatus::Full)
            TORCH_CHECK(
                input->data_ptr() == out.data_ptr() &&
                    input->sizes().equals(out.sizes()) &&
                    input->strides().equals(out.strides()) &&
                    input->storage_offset() == out.storage_offset(),
                "Vulkan bitwise_and.Tensor_out permits only exact full-tensor aliases");
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(self.numel() * sizeof(bool));
    validate_allocation(lhs, bytes, "bitwise_and lhs");
    validate_allocation(rhs, bytes, "bitwise_and rhs");
    validate_allocation(result, bytes, "bitwise_and output");
    const auto &platform = allocation_platform(lhs);
    TORCH_CHECK(&platform == &allocation_platform(rhs) &&
                    &platform == &allocation_platform(result) &&
                    platform.supports_bool_pointwise(),
                "Vulkan bitwise_and.Tensor_out requires bool pointwise capability");
    platform.compute().tensor_tensor(
        allocation_buffer(lhs).buffer(), lhs_layout, allocation_buffer(rhs).buffer(),
        rhs_layout, allocation_buffer(result).buffer(), output_layout, 2, true);
    return out;
}

at::Tensor masked_select(const at::Tensor &self, const at::Tensor &mask) {
    validate_input(self, "masked_select");
    TORCH_CHECK(mask.device() == self.device(),
                "Vulkan masked_select requires mask and input on the same vk:0 device");
    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "Vulkan masked_select requires a bool mask");
    TORCH_CHECK(mask.layout() == at::kStrided && mask.is_contiguous(),
                "Vulkan masked_select requires a contiguous mask");
    TORCH_CHECK(
        mask.storage_offset() == 0,
        "Vulkan masked_select does not support non-zero storage_offset() on the mask");
    const auto value_layout =
        inspect_vulkan_tensor_layout(self, "masked_select values");
    const auto mask_layout = inspect_vulkan_tensor_layout(mask, "masked_select mask");
    TORCH_CHECK(value_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan masked_select values have internal overlap");
    TORCH_CHECK(std::all_of(value_layout.strides.begin(), value_layout.strides.end(),
                            [](int64_t stride) { return stride > 0; }),
                "Vulkan masked_select requires positive-stride value views");
    TORCH_CHECK(mask_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan masked_select mask has internal overlap");
    TORCH_CHECK(mask.sizes().equals(self.sizes()),
                "Vulkan masked_select requires equal shapes");
    const auto &self_data = self.storage().data_ptr();
    const auto &mask_data = mask.storage().data_ptr();
    TORCH_CHECK(is_vulkan_allocation(self_data) && is_vulkan_allocation(mask_data),
                "Vulkan masked_select requires Vulkan allocation provenance");
    const auto &platform = allocation_platform(self_data);
    TORCH_CHECK(&platform == &allocation_platform(mask_data) &&
                    platform.supports_bool_pointwise(),
                "Vulkan masked_select requires bool pointwise Vulkan capability");
    // The shader consumes a densely packed logical sequence.  Validation above
    // must remain side-effect free; only then may a supported value view be
    // materialized on Vulkan.
    const bool materializes_values =
        !self.is_contiguous() || self.storage_offset() != 0;
    at::Tensor values = materializes_values ? self.contiguous() : self;
    if (values.storage_offset() != 0)
        values = values.clone();
    const auto &input_data = values.storage().data_ptr();
    TORCH_CHECK(static_cast<uint64_t>(values.numel()) <=
                    std::numeric_limits<uint32_t>::max(),
                "Vulkan masked_select exceeds the supported element count");
    if (values.numel() == 0)
        return at::empty({0}, values.options());
    const uint32_t element_count = static_cast<uint32_t>(values.numel());
    const VkDeviceSize input_bytes =
        static_cast<VkDeviceSize>(element_count) * sizeof(float);
    validate_allocation(input_data, input_bytes, "masked_select input");
    validate_allocation(mask_data, element_count, "masked_select mask");

    VulkanBuffer counter(platform, sizeof(uint32_t),
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    TORCH_CHECK(
        (counter.memory_properties() & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        "Vulkan masked_select requires coherent host-visible counter memory");
    const uint32_t zero = 0;
    counter.write(&zero, sizeof(zero));
    platform.compute().masked_select_count(allocation_buffer(input_data).buffer(),
                                           allocation_buffer(mask_data).buffer(),
                                           counter.buffer(), element_count);
    uint32_t selected = 0;
    counter.read(&selected, sizeof(selected));
    TORCH_CHECK(selected <= element_count,
                "Vulkan masked_select counter exceeded input size");
    at::Tensor result = at::empty({static_cast<int64_t>(selected)}, values.options());
    if (selected == 0)
        return result;
    const auto &result_data = result.storage().data_ptr();
    validate_allocation(result_data,
                        static_cast<VkDeviceSize>(selected) * sizeof(float),
                        "masked_select output");
    counter.write(&zero, sizeof(zero));
    platform.compute().masked_select_compact(allocation_buffer(input_data).buffer(),
                                             allocation_buffer(mask_data).buffer(),
                                             allocation_buffer(result_data).buffer(),
                                             counter.buffer(), element_count, selected);
    return result;
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("ne.Scalar_out", &pytorch_vulkan::ne_scalar_out);
    m.impl("isfinite.out", &pytorch_vulkan::isfinite_out);
    m.impl("eq.Tensor_out", &pytorch_vulkan::eq_tensor_out);
    m.impl("bitwise_and.Tensor_out", &pytorch_vulkan::bitwise_and_tensor_out);
    m.impl("masked_select", &pytorch_vulkan::masked_select);
}
