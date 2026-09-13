#pragma once

#include <c10/core/ScalarType.h>

#include "binary.h"

namespace pytorch_vulkan {

// Validates the dtype boundary before allocation, transfer, or operator work.
void validate_vulkan_dtype(c10::ScalarType dtype, const char *operation);
void validate_unary_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                          const char *operation_name);
void validate_binary_dtypes(c10::ScalarType lhs, c10::ScalarType rhs,
                            PointwiseOperation operation,
                            const char *operation_name);
void validate_scalar_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                           const char *operation_name);
void validate_output_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                           const char *operation_name);
void validate_reduction_dtype(c10::ScalarType dtype, bool mean,
                              const char *operation_name);
void validate_index_input_dtype(c10::ScalarType dtype, const char *operation_name);
void validate_index_output_dtype(c10::ScalarType dtype, const char *operation_name);
void validate_operator_rank(int64_t rank, const char *operation_name);
bool pointwise_uses_bool(c10::ScalarType dtype, PointwiseOperation operation);
void validate_pointwise_device_capability(bool uses_bool, bool bool_supported,
                                          const char *operation_name);

} // namespace pytorch_vulkan
