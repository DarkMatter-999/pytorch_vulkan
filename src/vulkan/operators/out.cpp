#include "out.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_platform.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <limits>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1 ||
           device.type() == c10::DeviceType::Vulkan;
}

std::size_t checked_bytes(const at::Tensor &tensor, const char *name) {
    const int64_t elements = tensor.numel();
    TORCH_CHECK(elements >= 0, "Vulkan ", name, " has a negative element count");
    const uint64_t count = static_cast<uint64_t>(elements);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max() / sizeof(float),
                "Vulkan ", name, " byte count does not fit size_t");
    const uint64_t bytes = count * sizeof(float);
    TORCH_CHECK(bytes <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan ", name, " byte count does not fit VkDeviceSize");
    return static_cast<std::size_t>(bytes);
}

int64_t expected_numel(at::IntArrayRef sizes, const char *name) {
    int64_t result = 1;
    for (const int64_t size : sizes) {
        TORCH_CHECK(size >= 0 &&
                        (size == 0 || result <= std::numeric_limits<int64_t>::max() / size),
                    "Vulkan ", name, " output size is invalid");
        result *= size;
    }
    return result;
}

void validate_input(const at::Tensor &input, const char *name) {
    TORCH_CHECK(is_vulkan_device(input.device()), "Vulkan ", name,
                " requires a Vulkan tensor");
    TORCH_CHECK(input.device().index() == 0, "Vulkan ", name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(input.layout() == at::kStrided, "Vulkan ", name,
                " requires a strided tensor");
    TORCH_CHECK(input.is_contiguous(), "Vulkan ", name,
                " requires a contiguous tensor");
    TORCH_CHECK(input.storage_offset() == 0, "Vulkan ", name,
                " does not support tensors with non-zero storage_offset()");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Vulkan ", name,
                " supports only float32 tensors");
    TORCH_CHECK(input.dim() != 0, "Vulkan ", name,
                " does not support zero-dimensional tensor operands");
}

void validate_output_metadata(const at::Tensor &out, const char *name) {
    TORCH_CHECK(is_vulkan_device(out.device()), "Vulkan ", name,
                " requires a Vulkan output tensor");
    TORCH_CHECK(out.device().index() == 0, "Vulkan ", name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(out.layout() == at::kStrided, "Vulkan ", name,
                " requires a strided output tensor");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "Vulkan ", name,
                " supports only float32 output tensors");
    TORCH_CHECK(out.storage_offset() == 0, "Vulkan ", name,
                " does not support output tensors with non-zero storage_offset()");
}

void resize_output(at::Tensor &out, at::IntArrayRef sizes, const char *name) {
    const int64_t requested_numel = expected_numel(sizes, name);
    if (requested_numel == 0) {
        const at::DataPtr &data = out.storage().data_ptr();
        TORCH_CHECK(pytorch_vulkan::is_vulkan_allocation(data) &&
                        data.device() == out.device(),
                    "Vulkan ", name, " output has invalid Vulkan storage provenance");
        out.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
        return;
    }
    const c10::Device before_device = out.device();
    const at::DataPtr &before_data = out.storage().data_ptr();
    const auto before_deleter = before_data.get_deleter();
    TORCH_CHECK(pytorch_vulkan::is_vulkan_allocation(before_data) &&
                    before_data.device() == before_device,
                "Vulkan ", name, " output has invalid Vulkan storage provenance");
    if (requested_numel != 0 && out.numel() != requested_numel) {
        at::Tensor resized;
        try {
            resized = at::empty(sizes, out.options().device(out.device()));
            out.unsafeGetTensorImpl()->set_storage_keep_dtype(resized.storage());
        } catch (const std::exception &error) {
            TORCH_CHECK(false, "Vulkan ", name,
                        " does not support resizing the output: ", error.what());
        }
    }
    out.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
    const at::DataPtr &after_data = out.storage().data_ptr();
    TORCH_CHECK(out.device() == before_device &&
                    pytorch_vulkan::is_vulkan_allocation(after_data) &&
                    after_data.get_deleter() == before_deleter &&
                    is_vulkan_device(out.device()) && out.layout() == at::kStrided &&
                    out.scalar_type() == at::kFloat && out.storage_offset() == 0 &&
                    out.is_contiguous(),
                "Vulkan ", name,
                " cannot safely resize the Vulkan output without replacing it");
}

void validate_zero_allocation(const at::Tensor &tensor, const char *name) {
    const at::DataPtr &data = tensor.storage().data_ptr();
    TORCH_CHECK(pytorch_vulkan::is_vulkan_allocation(data) &&
                    data.device() == tensor.device(),
                "Vulkan ", name, " has invalid Vulkan storage provenance");
}

at::MemOverlapStatus classify_overlap(const at::Tensor &input,
                                      const at::Tensor &out,
                                      const char *name) {
    TORCH_CHECK(at::has_internal_overlap(out) == at::MemOverlap::No,
                "Vulkan ", name, " output has internal overlap");
    return at::get_overlap_status(input, out);
}

void validate_overlap_status(at::MemOverlapStatus overlap, const at::Tensor &input,
                             const at::Tensor &out, const char *name) {
    TORCH_CHECK(overlap != at::MemOverlapStatus::Partial &&
                    overlap != at::MemOverlapStatus::TooHard,
                "Vulkan ", name, " output partially overlaps an input");
    if (overlap == at::MemOverlapStatus::Full) {
        TORCH_CHECK(input.data_ptr() == out.data_ptr() &&
                        input.numel() == out.numel(),
                    "Vulkan ", name,
                    " permits only exact full-tensor output aliases");
    }
}

float scalar_to_float(const at::Scalar &scalar, const char *name) {
    TORCH_CHECK(!scalar.isComplex(), "Vulkan ", name,
                " requires a real numeric scalar");
    const double value = scalar.toDouble();
    TORCH_CHECK(std::isfinite(value), "Vulkan ", name,
                " does not support non-finite scalar values");
    const float result = static_cast<float>(value);
    TORCH_CHECK(std::isfinite(result), "Vulkan ", name,
                " scalar is outside float32 range");
    TORCH_CHECK(value == 0.0 || result != 0.0, "Vulkan ", name,
                " scalar underflows to float32 zero");
    return result;
}

const VulkanPlatform &validate_allocations(const at::Tensor *first,
                                           const at::Tensor *second,
                                           const at::Tensor &out,
                                           std::size_t bytes,
                                           const char *name) {
    const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
    const at::DataPtr &out_data = out.storage().data_ptr();
    pytorch_vulkan::validate_allocation(out_data, size, "output");
    const at::DataPtr &first_data = first->storage().data_ptr();
    pytorch_vulkan::validate_allocation(first_data, size, "input");
    const VulkanPlatform &platform = pytorch_vulkan::allocation_platform(first_data);
    TORCH_CHECK(platform.device() != VK_NULL_HANDLE, "Vulkan ", name,
                " requires an initialized Vulkan device");
    const VulkanPlatform &out_platform = pytorch_vulkan::allocation_platform(out_data);
    TORCH_CHECK(&platform == &out_platform && platform.device() == out_platform.device(),
                "Vulkan ", name, " requires all tensors to use the same Vulkan platform/device");
    if (second) {
        const at::DataPtr &second_data = second->storage().data_ptr();
        pytorch_vulkan::validate_allocation(second_data, size, "input");
        const VulkanPlatform &second_platform = pytorch_vulkan::allocation_platform(second_data);
        TORCH_CHECK(&platform == &second_platform && platform.device() == second_platform.device(),
                    "Vulkan ", name,
                    " requires all tensors to use the same Vulkan platform/device");
    }
    return platform;
}

} // namespace

namespace pytorch_vulkan {

at::Tensor &dispatch_unary_out(const at::Tensor &input, at::Tensor &out,
                               PointwiseOperation operation, const char *name) {
    validate_input(input, name);
    validate_output_metadata(out, name);
    const auto before_overlap = classify_overlap(input, out, name);
    validate_overlap_status(before_overlap, input, out, name);
    TORCH_CHECK(out.is_contiguous(), "Vulkan ", name,
                " requires a contiguous output tensor");
    resize_output(out, input.sizes(), name);
    validate_overlap_status(classify_overlap(input, out, name), input, out, name);
    const std::size_t bytes = checked_bytes(input, name);
    if (bytes == 0) {
        validate_zero_allocation(input, name);
        validate_zero_allocation(out, name);
        return out;
    }
    const VulkanPlatform &platform = validate_allocations(&input, nullptr, out, bytes, name);
    const auto input_buffer = allocation_buffer(input.storage().data_ptr()).buffer();
    const auto output_buffer = allocation_buffer(out.storage().data_ptr()).buffer();
    if (classify_overlap(input, out, name) == at::MemOverlapStatus::Full) {
        platform.compute().unary_alias(input_buffer, output_buffer,
                                       static_cast<VkDeviceSize>(bytes),
                                       static_cast<uint32_t>(operation));
    } else {
        platform.compute().unary(input_buffer, output_buffer,
                                 static_cast<VkDeviceSize>(bytes),
                                 static_cast<uint32_t>(operation));
    }
    return out;
}

at::Tensor &dispatch_tensor_tensor_out(const at::Tensor &lhs, const at::Tensor &rhs,
                                        const at::Scalar &alpha, at::Tensor &out,
                                        PointwiseOperation operation, const char *name) {
    const bool lhs_wrapped_number = lhs.device().is_cpu() && lhs.dim() == 0 &&
                                    lhs.unsafeGetTensorImpl()->is_wrapped_number();
    const bool rhs_wrapped_number = rhs.device().is_cpu() && rhs.dim() == 0 &&
                                    rhs.unsafeGetTensorImpl()->is_wrapped_number();
    if (lhs_wrapped_number || rhs_wrapped_number) {
        TORCH_CHECK(lhs_wrapped_number != rhs_wrapped_number, "Vulkan ", name,
                    " requires one Python numeric scalar and one Vulkan tensor; scalar operand is unsupported");
        if (lhs_wrapped_number) {
            return dispatch_tensor_scalar_out(rhs, lhs.item(), alpha, out, operation, name, true);
        }
        return dispatch_tensor_scalar_out(lhs, rhs.item(), alpha, out, operation, name);
    }
    validate_input(lhs, name);
    validate_input(rhs, name);
    TORCH_CHECK(lhs.device() == rhs.device(), "Vulkan ", name,
                " requires equal devices");
    TORCH_CHECK(lhs.sizes().equals(rhs.sizes()), "Vulkan ", name,
                " requires equal tensor sizes");
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan ", name, " supports only alpha == 1");
    validate_output_metadata(out, name);
    const auto lhs_before_overlap = classify_overlap(lhs, out, name);
    const auto rhs_before_overlap = classify_overlap(rhs, out, name);
    validate_overlap_status(lhs_before_overlap, lhs, out, name);
    validate_overlap_status(rhs_before_overlap, rhs, out, name);
    TORCH_CHECK(out.is_contiguous(), "Vulkan ", name,
                " requires a contiguous output tensor");
    resize_output(out, lhs.sizes(), name);
    validate_overlap_status(classify_overlap(lhs, out, name), lhs, out, name);
    validate_overlap_status(classify_overlap(rhs, out, name), rhs, out, name);
    const std::size_t bytes = checked_bytes(lhs, name);
    if (bytes == 0) {
        validate_zero_allocation(lhs, name);
        validate_zero_allocation(rhs, name);
        validate_zero_allocation(out, name);
        return out;
    }
    const VulkanPlatform &platform = validate_allocations(&lhs, &rhs, out, bytes, name);
    const auto lhs_buffer = allocation_buffer(lhs.storage().data_ptr()).buffer();
    const auto rhs_buffer = allocation_buffer(rhs.storage().data_ptr()).buffer();
    const auto output_buffer = allocation_buffer(out.storage().data_ptr()).buffer();
    if (lhs_before_overlap == at::MemOverlapStatus::Full ||
        rhs_before_overlap == at::MemOverlapStatus::Full) {
        platform.compute().tensor_tensor_alias(
            lhs_buffer, rhs_buffer, output_buffer, static_cast<VkDeviceSize>(bytes),
            static_cast<uint32_t>(operation));
    } else {
        platform.compute().tensor_tensor(
            lhs_buffer, rhs_buffer, output_buffer, static_cast<VkDeviceSize>(bytes),
            static_cast<uint32_t>(operation));
    }
    return out;
}

at::Tensor &dispatch_tensor_scalar_out(const at::Tensor &tensor, const at::Scalar &scalar,
                                        const at::Scalar &alpha, at::Tensor &out,
                                        PointwiseOperation operation, const char *name,
                                        bool scalar_left) {
    validate_input(tensor, name);
    TORCH_CHECK(alpha.toDouble() == 1.0, "Vulkan ", name, " supports only alpha == 1");
    const float value = scalar_to_float(scalar, name);
    validate_output_metadata(out, name);
    const auto before_overlap = classify_overlap(tensor, out, name);
    validate_overlap_status(before_overlap, tensor, out, name);
    TORCH_CHECK(out.is_contiguous(), "Vulkan ", name,
                " requires a contiguous output tensor");
    resize_output(out, tensor.sizes(), name);
    validate_overlap_status(classify_overlap(tensor, out, name), tensor, out, name);
    const std::size_t bytes = checked_bytes(tensor, name);
    if (bytes == 0) {
        validate_zero_allocation(tensor, name);
        validate_zero_allocation(out, name);
        return out;
    }
    const VulkanPlatform &platform = validate_allocations(&tensor, nullptr, out, bytes, name);
    const auto tensor_buffer = allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto output_buffer = allocation_buffer(out.storage().data_ptr()).buffer();
    if (classify_overlap(tensor, out, name) == at::MemOverlapStatus::Full) {
        if (scalar_left) {
            platform.compute().scalar_tensor_alias(
                value, tensor_buffer, output_buffer, static_cast<VkDeviceSize>(bytes),
                static_cast<uint32_t>(operation));
        } else {
            platform.compute().tensor_scalar_alias(
                tensor_buffer, output_buffer, static_cast<VkDeviceSize>(bytes), value,
                static_cast<uint32_t>(operation));
        }
    } else if (scalar_left) {
        platform.compute().scalar_tensor(
            value, tensor_buffer, output_buffer, static_cast<VkDeviceSize>(bytes),
            static_cast<uint32_t>(operation));
    } else {
        platform.compute().tensor_scalar(
            tensor_buffer, output_buffer, static_cast<VkDeviceSize>(bytes), value,
            static_cast<uint32_t>(operation));
    }
    return out;
}

} // namespace pytorch_vulkan
