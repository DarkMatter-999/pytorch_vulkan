#include "capability.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include "vulkan_platform.h"

namespace pytorch_vulkan {

bool formatter_double_supported() {
    return platform()->supports_formatter_double();
}

void validate_vulkan_dtype(c10::ScalarType dtype, const char *operation) {
    if (dtype == at::kHalf) {
        TORCH_CHECK(false, "Vulkan float16 support is deferred; ", operation,
                    " cannot use float16 until its storage, transfer, shader, promotion, "
                    "and autograd contracts are implemented");
    }
    if (dtype == at::kDouble) {
        TORCH_CHECK(formatter_double_supported(),
                    "Vulkan formatter Double support requires the shaderFloat64 device feature");
        return;
    }
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBool,
                "Vulkan ", operation,
                " supports only float32 and bool tensors, got ", dtype);
}

namespace {

bool bool_binary(PointwiseOperation operation) {
    return operation == PointwiseOperation::Add || operation == PointwiseOperation::Mul;
}

void validate_float_or_approved_bool(c10::ScalarType dtype, PointwiseOperation operation,
                                     const char *operation_name) {
    TORCH_CHECK(dtype == at::kFloat || (dtype == at::kBool && bool_binary(operation)),
                "Vulkan ", operation_name, " does not support dtype ", dtype);
}

} // namespace

void validate_unary_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                          const char *operation_name) {
    validate_vulkan_dtype(dtype, operation_name);
    TORCH_CHECK(dtype != at::kDouble, "Vulkan ", operation_name,
                " does not support Double dtype outside formatter operations");
    TORCH_CHECK(dtype == at::kFloat, "Vulkan ", operation_name,
                " supports only float32 tensors");
    (void)operation;
}

void validate_binary_dtypes(c10::ScalarType lhs, c10::ScalarType rhs,
                            PointwiseOperation operation,
                            const char *operation_name) {
    validate_vulkan_dtype(lhs, operation_name);
    validate_vulkan_dtype(rhs, operation_name);
    TORCH_CHECK(lhs == rhs, "Vulkan ", operation_name,
                " requires matching dtypes");
    validate_float_or_approved_bool(lhs, operation, operation_name);
}

void validate_scalar_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                           const char *operation_name) {
    validate_vulkan_dtype(dtype, operation_name);
    TORCH_CHECK(dtype == at::kFloat, "Vulkan ", operation_name,
                " scalar forms support only float32 tensors");
    (void)operation;
}

void validate_output_dtype(c10::ScalarType dtype, PointwiseOperation operation,
                           const char *operation_name) {
    validate_vulkan_dtype(dtype, operation_name);
    validate_float_or_approved_bool(dtype, operation, operation_name);
}

void validate_reduction_dtype(c10::ScalarType dtype, bool mean,
                              const char *operation_name) {
    validate_vulkan_dtype(dtype, operation_name);
    TORCH_CHECK(dtype == at::kFloat, "Vulkan ", operation_name,
                " supports only float32 tensors", mean ? " for mean" : "");
}

void validate_index_input_dtype(c10::ScalarType dtype, const char *operation_name) {
    validate_vulkan_dtype(dtype, operation_name);
    TORCH_CHECK(dtype == at::kFloat, "Vulkan ", operation_name,
                " supports only float32 input tensors");
}

void validate_index_output_dtype(c10::ScalarType dtype, const char *operation_name) {
    TORCH_CHECK(dtype == at::kLong, "Vulkan ", operation_name,
                " requires int64 output indices");
}

void validate_operator_rank(int64_t rank, const char *operation_name) {
    TORCH_CHECK(rank >= 0 && rank <= 8, "Vulkan ", operation_name,
                " supports ranks up to 8, got ", rank);
}

bool pointwise_uses_bool(c10::ScalarType dtype, PointwiseOperation operation) {
    return dtype == at::kBool && bool_binary(operation);
}

void validate_pointwise_device_capability(bool uses_bool, bool bool_supported,
                                          const char *operation_name) {
    TORCH_CHECK(!uses_bool || bool_supported, "Vulkan ", operation_name,
                " bool pointwise support requires Vulkan 1.2 8-bit storage and int8 shader features");
}

} // namespace pytorch_vulkan
