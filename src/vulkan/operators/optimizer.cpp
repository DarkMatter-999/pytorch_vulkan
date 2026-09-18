#include "optimizer.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

namespace {

bool is_vulkan_device(const c10::Device &device) {
    return device.type() == c10::DeviceType::PrivateUse1 ||
           device.type() == c10::DeviceType::Vulkan;
}

float scalar_to_float(const at::Scalar &value, const char *name) {
    TORCH_CHECK(!value.isComplex(), "Vulkan ", name, " requires a real numeric scalar");
    const double double_value = value.toDouble();
    TORCH_CHECK(std::isfinite(double_value), "Vulkan ", name,
                " does not support non-finite scalar values");
    const float result = static_cast<float>(double_value);
    TORCH_CHECK(std::isfinite(result), "Vulkan ", name,
                " scalar is outside float32 range");
    TORCH_CHECK(double_value == 0.0 || result != 0.0, "Vulkan ", name,
                " scalar underflows to float32 zero");
    return result;
}

pytorch_vulkan::VulkanTensorLayout validate_tensor(const at::Tensor &tensor,
                                                   const char *name) {
    TORCH_CHECK(is_vulkan_device(tensor.device()), "Vulkan ", name,
                " requires a Vulkan tensor");
    TORCH_CHECK(tensor.device().index() == 0, "Vulkan ", name,
                " supports only Vulkan device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided && tensor.scalar_type() == at::kFloat,
                "Vulkan ", name, " requires a strided float32 tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Vulkan ", name,
                " requires a contiguous tensor");
    TORCH_CHECK(at::has_internal_overlap(tensor) == at::MemOverlap::No, "Vulkan ", name,
                " tensor has internal overlap");
    auto layout = pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, name);
    TORCH_CHECK(layout.rank <= 8 &&
                    layout.numel <= std::numeric_limits<uint32_t>::max(),
                "Vulkan ", name, " tensor layout exceeds supported range");
    return layout;
}

void validate_overlap(const at::Tensor &input, const at::Tensor &self,
                      const char *name) {
    const auto overlap = at::get_overlap_status(input, self);
    TORCH_CHECK(overlap != at::MemOverlapStatus::Partial &&
                    overlap != at::MemOverlapStatus::TooHard,
                "Vulkan ", name, " tensors partially overlap");
}

at::Tensor &dispatch_compound(at::Tensor &self, const at::Tensor &tensor1,
                              const at::Tensor &tensor2, const at::Scalar &value,
                              uint32_t operation, const char *name) {
    const auto self_layout = validate_tensor(self, name);
    const auto tensor1_layout = validate_tensor(tensor1, name);
    const auto tensor2_layout = validate_tensor(tensor2, name);
    TORCH_CHECK(self.device() == tensor1.device() &&
                    self.device() == tensor2.device() &&
                    self.sizes().equals(tensor1.sizes()) &&
                    self.sizes().equals(tensor2.sizes()),
                "Vulkan ", name, " requires matching Vulkan tensor operands");
    const float scalar = scalar_to_float(value, name);
    validate_overlap(tensor1, self, name);
    validate_overlap(tensor2, self, name);
    const auto &self_data = self.storage().data_ptr();
    const auto &tensor1_data = tensor1.storage().data_ptr();
    const auto &tensor2_data = tensor2.storage().data_ptr();
    pytorch_vulkan::validate_allocation(self_data, self_layout.allocation_bytes,
                                        "self");
    pytorch_vulkan::validate_allocation(tensor1_data, tensor1_layout.allocation_bytes,
                                        "tensor1");
    pytorch_vulkan::validate_allocation(tensor2_data, tensor2_layout.allocation_bytes,
                                        "tensor2");
    const auto &platform = pytorch_vulkan::allocation_platform(self_data);
    TORCH_CHECK(&platform == &pytorch_vulkan::allocation_platform(tensor1_data) &&
                    &platform == &pytorch_vulkan::allocation_platform(tensor2_data),
                "Vulkan ", name, " requires one Vulkan platform/device");
    if (self_layout.numel != 0) {
        platform.compute().compound_tensor_tensor_alias(
            pytorch_vulkan::allocation_buffer(self_data).buffer(), self_layout,
            pytorch_vulkan::allocation_buffer(tensor1_data).buffer(), tensor1_layout,
            pytorch_vulkan::allocation_buffer(tensor2_data).buffer(), tensor2_layout,
            pytorch_vulkan::allocation_buffer(self_data).buffer(), self_layout, scalar,
            operation);
    }
    return self;
}

} // namespace

namespace pytorch_vulkan {

at::Tensor &addcmul_inplace(at::Tensor &self, const at::Tensor &tensor1,
                            const at::Tensor &tensor2, const at::Scalar &value) {
    return dispatch_compound(self, tensor1, tensor2, value, 0, "addcmul_");
}

at::Tensor &addcdiv_inplace(at::Tensor &self, const at::Tensor &tensor1,
                            const at::Tensor &tensor2, const at::Scalar &value) {
    return dispatch_compound(self, tensor1, tensor2, value, 1, "addcdiv_");
}

} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("addcmul_", &pytorch_vulkan::addcmul_inplace);
    m.impl("addcdiv_", &pytorch_vulkan::addcdiv_inplace);
}
