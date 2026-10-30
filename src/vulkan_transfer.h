#pragma once

#include "vulkan_copy_route.h"

#include <ATen/ATen.h>

namespace pytorch_vulkan {

at::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,
                        bool non_blocking);

// Route for a validated, destination-shaped F32 copy. Resolves pinned provenance
// of the CPU operand; Vulkan tensors are classified from metadata only.
F32CopyRoute classify_f32_copy(const at::Tensor &destination, const at::Tensor &source,
                               bool non_blocking, bool async_execution,
                               bool training_scope);

// Executes the classified route; same return/exception contract as copy_tensor.
// Public non_blocking=True remains rejected by copy_tensor until it is wired here.
at::Tensor &async_copy_tensor(at::Tensor &destination, const at::Tensor &source,
                              bool non_blocking);

at::Tensor vulkan_contiguous_copy(const at::Tensor &source);

// Materialize a validated stack with one recorded/submitted transfer operation.
at::Tensor vulkan_stack_copy(at::TensorList tensors, int64_t dim);

// Internal final-value presentation path used by tensor formatting.
at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source);

} // namespace pytorch_vulkan
