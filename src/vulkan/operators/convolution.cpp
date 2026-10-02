#include "convolution.h"
#include "autograd.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include <c10/util/Exception.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <torch/library.h>

namespace pytorch_vulkan {
namespace {

uint64_t checked_add(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        throw std::overflow_error("Vulkan convolution addition overflow");
    return lhs + rhs;
}
uint64_t checked_mul(uint64_t lhs, uint64_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error("Vulkan convolution multiplication overflow");
    return lhs * rhs;
}
int64_t checked_output_extent(int64_t extent, int64_t kernel, int64_t stride,
                              int64_t padding, int64_t dilation) {
    if (extent <= 0 || kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0)
        throw std::invalid_argument("Vulkan convolution geometry is invalid");
    constexpr uint64_t shader_max = std::numeric_limits<int32_t>::max();
    const uint64_t padded = checked_add(static_cast<uint64_t>(extent),
                                        checked_mul(2, static_cast<uint64_t>(padding)));
    const uint64_t span = checked_add(
        checked_mul(static_cast<uint64_t>(dilation), static_cast<uint64_t>(kernel - 1)), 1);
    if (padded > shader_max || span > padded ||
        static_cast<uint64_t>(stride) > shader_max ||
        static_cast<uint64_t>(padding) > shader_max ||
        static_cast<uint64_t>(dilation) > shader_max)
        throw std::invalid_argument("Vulkan convolution geometry exceeds shader range");
    return static_cast<int64_t>((padded - span) / static_cast<uint64_t>(stride) + 1);
}

int64_t checked_transposed_output_extent(int64_t extent, int64_t kernel,
                                         int64_t stride, int64_t padding,
                                         int64_t dilation,
                                         int64_t output_padding) {
    TORCH_CHECK(extent > 0 && kernel > 0 && stride > 0 && dilation > 0 &&
                    padding >= 0 && output_padding >= 0,
                "Vulkan transposed convolution geometry is invalid");
    TORCH_CHECK(output_padding < stride || output_padding < dilation,
                "Vulkan transposed convolution output_padding must be smaller than stride or dilation");
    const __int128 result = static_cast<__int128>(extent - 1) * stride -
        static_cast<__int128>(2) * padding +
        static_cast<__int128>(dilation) * (kernel - 1) + output_padding + 1;
    TORCH_CHECK(result > 0 &&
                    result <= std::numeric_limits<int32_t>::max(),
                "Vulkan transposed convolution output extent exceeds shader range");
    return static_cast<int64_t>(result);
}

enum class ConvolutionSourceRole : uint8_t { Input, Weight, GradOutput, Bias };

struct ConvolutionOutputSpec {
    uint32_t operation;
    uint32_t result_slot;
    std::array<ConvolutionSourceRole, 3> source_roles;
    uint32_t source_count;
    std::array<int64_t, 4> shape;
    uint32_t rank;
    uint32_t numel;
    VkDeviceSize bytes;
    uint32_t kernel_height, kernel_width;
    VulkanConvolutionGeometry shader_geometry;
};
struct ConvolutionPreflight {
    VulkanTensorLayout input_layout, weight_layout, bias_layout, grad_layout;
    std::array<ConvolutionOutputSpec, 3> outputs{};
    size_t output_count = 0;
    bool transposed = false;
    bool forward = false;
};

void validate_tensor(const at::Tensor &tensor, const char *name, int64_t expected_rank) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan convolution ", name, " requires vk:0");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "Vulkan convolution ", name, " requires float32");
    TORCH_CHECK(tensor.layout() == at::kStrided,
                "Vulkan convolution ", name, " requires strided layout");
    TORCH_CHECK(tensor.dim() == expected_rank, "Vulkan convolution ", name,
                " requires rank ", expected_rank);
    for (const auto extent : tensor.sizes())
        TORCH_CHECK(extent > 0, "Vulkan convolution ", name,
                    " rejects empty tensors and non-positive extents");
}

bool canonical_strides(const VulkanTensorLayout &layout) {
    uint64_t expected = 1;
    for (int64_t dim = layout.rank - 1; dim >= 0; --dim) {
        if (layout.sizes[dim] > 1 && layout.strides[dim] != expected)
            return false;
        expected = checked_mul(expected, static_cast<uint64_t>(layout.sizes[dim]));
    }
    return true;
}
void validate_read_layout(const at::Tensor &tensor, const VulkanTensorLayout &layout,
                          const char *name, bool rank_four = true,
                          bool allow_broadcast_dimensions = false) {
    TORCH_CHECK(layout.internal_overlap == VulkanOverlap::No ||
                    (allow_broadcast_dimensions &&
                     is_non_overlapping_except_broadcast_dims(layout)),
                "Vulkan convolution rejects overlapping ", name, " layouts");
    TORCH_CHECK(!rank_four || !tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
                    canonical_strides(layout),
                "Vulkan convolution rejects channels-last-only ", name, " layout");
}

uint32_t narrow_shader(uint64_t value, const char *name) {
    TORCH_CHECK(value <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "Vulkan convolution ", name,
                " exceeds signed 32-bit shader range");
    return static_cast<uint32_t>(value);
}
uint32_t output_numel(const std::array<int64_t, 4> &shape, uint32_t rank) {
    uint64_t count = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        TORCH_CHECK(shape[i] > 0, "Vulkan convolution rejects empty output extents");
        count = checked_mul(count, static_cast<uint64_t>(shape[i]));
    }
    TORCH_CHECK(count <= std::numeric_limits<uint32_t>::max(),
                "Vulkan convolution output numel exceeds UINT32_MAX");
    return static_cast<uint32_t>(count);
}

void validate_source_allocation(const at::Tensor &tensor,
                                const VulkanTensorLayout &layout,
                                const char *name) {
    const auto &data = tensor.storage().data_ptr();
    validate_allocation(data, layout.allocation_bytes, name);
    TORCH_CHECK(layout.byte_offset <= layout.allocation_bytes &&
                    layout.byte_range <= layout.allocation_bytes - layout.byte_offset,
                "Vulkan convolution ", name,
                " view storage-offset/stride range exceeds its allocation");
}

void validate_shader_read_domain(const VulkanTensorLayout &layout,
                                 const char *name) {
    TORCH_CHECK(layout.numel > 0 &&
                    static_cast<uint64_t>(layout.numel) <=
                        std::numeric_limits<uint32_t>::max(),
                "Vulkan convolution ", name,
                " logical numel exceeds UINT32 shader range");
    for (const auto extent : layout.sizes)
        TORCH_CHECK(extent > 0 && extent <= std::numeric_limits<int32_t>::max(),
                    "Vulkan convolution ", name,
                    " extent exceeds signed shader range");
}

ConvolutionPreflight preflight_convolution(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor *bias,
    const at::Tensor *grad_output, const ConvolutionGeometry &geometry,
    bool transposed, std::array<int64_t, 2> output_padding,
    std::array<bool, 3> requested_outputs, bool forward) {
    validate_tensor(input, "input", 4);
    validate_tensor(weight, "weight", 4);
    TORCH_CHECK(input.device() == weight.device(),
                "Vulkan convolution tensors require one device");
    TORCH_CHECK(geometry.groups >= 1,
                "Vulkan convolution requires groups >= 1, got ", geometry.groups);
    TORCH_CHECK(input.size(1) > 0 && weight.size(0) > 0 && weight.size(1) > 0,
                "Vulkan convolution rejects empty channel dimensions");
    int64_t output_channels;
    if (transposed) {
        TORCH_CHECK(input.size(1) == weight.size(0),
                    "Vulkan transposed convolution input channels do not match weight.size(0)");
        TORCH_CHECK(input.size(1) % geometry.groups == 0,
                    "Vulkan transposed convolution groups must divide input channels");
        const __int128 channels = static_cast<__int128>(weight.size(1)) * geometry.groups;
        TORCH_CHECK(channels > 0 &&
                        channels <= std::numeric_limits<int32_t>::max(),
                    "Vulkan transposed convolution output channels exceed shader range");
        output_channels = static_cast<int64_t>(channels);
    } else {
        TORCH_CHECK(input.size(1) % geometry.groups == 0 &&
                        weight.size(0) % geometry.groups == 0,
                    "Vulkan convolution requires groups to divide input and output channels");
        TORCH_CHECK(weight.size(1) == input.size(1) / geometry.groups,
                    "Vulkan convolution grouped input channels do not match weight size");
        output_channels = weight.size(0);
    }
    const auto input_layout = inspect_vulkan_tensor_layout(input, "convolution input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "convolution weight");
    validate_shader_read_domain(input_layout, "input");
    validate_shader_read_domain(weight_layout, "weight");
    validate_read_layout(input, input_layout, "input");
    validate_read_layout(weight, weight_layout, "weight");
    validate_source_allocation(input, input_layout, "convolution input");
    validate_source_allocation(weight, weight_layout, "convolution weight");

    VulkanTensorLayout bias_layout = input_layout;
    const bool has_bias = bias != nullptr && bias->defined();
    if (forward && has_bias) {
        validate_tensor(*bias, "bias", 1);
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == output_channels,
                    "Vulkan convolution bias must have output channel elements");
        TORCH_CHECK(bias->device() == input.device(),
                    "Vulkan convolution tensors require one device");
        bias_layout = inspect_vulkan_tensor_layout(*bias, "convolution bias");
        validate_shader_read_domain(bias_layout, "bias");
        TORCH_CHECK(bias_layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan convolution rejects overlapping bias layouts");
        validate_source_allocation(*bias, bias_layout, "convolution bias");
    }
    VulkanTensorLayout grad_layout = input_layout;
    if (!forward) {
        TORCH_CHECK(grad_output != nullptr, "Vulkan convolution requires grad_output");
        validate_tensor(*grad_output, "grad_output", 4);
        TORCH_CHECK(grad_output->device() == input.device(),
                    "Vulkan convolution tensors require one device");
        grad_layout = inspect_vulkan_tensor_layout(*grad_output,
                                                   "convolution grad_output");
        validate_shader_read_domain(grad_layout, "grad_output");
        validate_read_layout(*grad_output, grad_layout, "grad_output", true, true);
        validate_source_allocation(input, input_layout, "convolution input");
        validate_source_allocation(*grad_output, grad_layout,
                                   "convolution grad_output");
    }

    int64_t out_h, out_w;
    if (transposed) {
        out_h = checked_transposed_output_extent(
            input.size(2), weight.size(2), geometry.stride_height,
            geometry.padding_height, geometry.dilation_height, output_padding[0]);
        out_w = checked_transposed_output_extent(
            input.size(3), weight.size(3), geometry.stride_width,
            geometry.padding_width, geometry.dilation_width, output_padding[1]);
    } else {
        out_h = checked_output_extent(
            input.size(2), weight.size(2), geometry.stride_height,
            geometry.padding_height, geometry.dilation_height);
        out_w = checked_output_extent(
            input.size(3), weight.size(3), geometry.stride_width,
            geometry.padding_width, geometry.dilation_width);
    }
    TORCH_CHECK(geometry.groups <= std::numeric_limits<int32_t>::max(),
                "Vulkan convolution groups exceed shader range");
    VulkanConvolutionGeometry shader_geometry{
        narrow_shader(geometry.stride_height, "stride"),
        narrow_shader(geometry.stride_width, "stride"),
        narrow_shader(geometry.padding_height, "padding"),
        narrow_shader(geometry.padding_width, "padding"),
        narrow_shader(geometry.dilation_height, "dilation"),
        narrow_shader(geometry.dilation_width, "dilation"),
        narrow_shader(geometry.groups, "groups")};

    ConvolutionPreflight plan{input_layout, weight_layout, bias_layout, grad_layout};
    plan.transposed = transposed;
    plan.forward = forward;
    auto append_output = [&](uint32_t operation, uint32_t result_slot,
                             std::array<int64_t, 4> shape, uint32_t rank,
                             int64_t kh, int64_t kw) {
        const uint32_t count = output_numel(shape, rank);
        const uint64_t byte_count = checked_mul(count, sizeof(float));
        TORCH_CHECK(byte_count <= std::numeric_limits<VkDeviceSize>::max(),
                    "Vulkan convolution output bytes exceed VkDeviceSize");
        std::array<ConvolutionSourceRole, 3> roles;
        uint32_t source_count;
        if (forward) {
            roles = has_bias
                ? std::array<ConvolutionSourceRole, 3>{
                      ConvolutionSourceRole::Input, ConvolutionSourceRole::Weight,
                      ConvolutionSourceRole::Bias}
                : std::array<ConvolutionSourceRole, 3>{
                      ConvolutionSourceRole::Input, ConvolutionSourceRole::Weight,
                      ConvolutionSourceRole::Input};
            source_count = has_bias ? 3 : 2;
        } else if (transposed && operation == 0) {
            roles = {ConvolutionSourceRole::GradOutput, ConvolutionSourceRole::Weight,
                     ConvolutionSourceRole::GradOutput};
            source_count = 2;
        } else if (operation == 2) {
            roles = transposed
                ? std::array<ConvolutionSourceRole, 3>{
                      ConvolutionSourceRole::Input, ConvolutionSourceRole::GradOutput,
                      ConvolutionSourceRole::GradOutput}
                : std::array<ConvolutionSourceRole, 3>{
                      ConvolutionSourceRole::GradOutput, ConvolutionSourceRole::Input,
                      ConvolutionSourceRole::GradOutput};
            source_count = 2;
        } else if (operation == 3) {
            roles = {ConvolutionSourceRole::GradOutput, ConvolutionSourceRole::GradOutput,
                     ConvolutionSourceRole::GradOutput};
            source_count = 1;
        } else {
            roles = {ConvolutionSourceRole::GradOutput, ConvolutionSourceRole::Weight,
                     ConvolutionSourceRole::GradOutput};
            source_count = 2;
        }
        ConvolutionOutputSpec spec{operation, result_slot, roles, source_count, shape, rank, count,
                                   static_cast<VkDeviceSize>(byte_count),
                                   narrow_shader(kh, "kernel height"),
                                   narrow_shader(kw, "kernel width"), shader_geometry};
        uint32_t active_sources = 0;
        for (size_t role_index = 0; role_index < spec.source_roles.size(); ++role_index) {
            bool first_occurrence = true;
            for (size_t previous = 0; previous < role_index; ++previous)
                first_occurrence &= spec.source_roles[previous] !=
                                    spec.source_roles[role_index];
            active_sources += static_cast<uint32_t>(first_occurrence);
        }
        TORCH_CHECK(active_sources == spec.source_count,
                    "Vulkan convolution active source roles do not match source_count");
        auto role_tensor = [&](ConvolutionSourceRole role) -> const at::Tensor & {
            switch (role) {
            case ConvolutionSourceRole::Input: return input;
            case ConvolutionSourceRole::Weight: return weight;
            case ConvolutionSourceRole::GradOutput: return *grad_output;
            case ConvolutionSourceRole::Bias: return has_bias ? *bias : input;
            }
            TORCH_CHECK(false, "Vulkan convolution has an invalid source role");
        };
        auto role_layout = [&](ConvolutionSourceRole role) -> const VulkanTensorLayout & {
            switch (role) {
            case ConvolutionSourceRole::Input: return input_layout;
            case ConvolutionSourceRole::Weight: return weight_layout;
            case ConvolutionSourceRole::GradOutput: return grad_layout;
            case ConvolutionSourceRole::Bias: return has_bias ? bias_layout : input_layout;
            }
            TORCH_CHECK(false, "Vulkan convolution has an invalid source role");
        };
        const auto &read_input_tensor = role_tensor(roles[0]);
        const auto &read_weight_tensor = role_tensor(roles[1]);
        const auto &read_bias_tensor = role_tensor(roles[2]);
        const auto &read_input = role_layout(roles[0]);
        const auto &read_weight = role_layout(roles[1]);
        const auto &read_bias = role_layout(roles[2]);
        const auto &platform = allocation_platform(read_input_tensor.storage().data_ptr());
        TORCH_CHECK(&platform == &allocation_platform(read_weight_tensor.storage().data_ptr()) &&
                        &platform == &allocation_platform(read_bias_tensor.storage().data_ptr()),
                    "Vulkan convolution requires Vulkan allocations on one platform");
        validate_source_allocation(read_input_tensor, read_input, "convolution input");
        validate_source_allocation(read_weight_tensor, read_weight, "convolution weight");
        validate_source_allocation(read_bias_tensor, read_bias, "convolution bias");
        platform.compute().validate_convolution_dispatch(
            read_input.allocation_bytes, read_weight.allocation_bytes,
            read_bias.allocation_bytes, spec.bytes, spec.numel,
            operation == 2 || operation == 3);

        auto checked_axis = [](int64_t coordinate_max, int64_t kernel_max,
                               int64_t stride, int64_t padding, int64_t dilation,
                               uint32_t op) {
            const __int128 stride_term = static_cast<__int128>(coordinate_max) * stride;
            const __int128 kernel_term = static_cast<__int128>(kernel_max) * dilation;
            const __int128 minimum = std::numeric_limits<int32_t>::min();
            const __int128 maximum = std::numeric_limits<int32_t>::max();
            TORCH_CHECK((op == 1 || stride_term <= maximum) &&
                            kernel_term <= maximum &&
                            stride <= maximum && dilation <= maximum &&
                            padding >= 0 && padding <= maximum,
                        "Vulkan convolution coordinate product exceeds signed shader range");
            if (op == 0) {
                // GLSL evaluates (oh * stride + kh * dilation) before
                // subtracting padding, so the sum itself must be representable.
                const __int128 sum = stride_term + kernel_term;
                const __int128 result = sum - padding;
                TORCH_CHECK(sum >= minimum && sum <= maximum &&
                                result >= minimum && result <= maximum,
                            "Vulkan convolution coordinate expression exceeds signed shader range");
            } else if (op == 1) {
                // GLSL evaluates target + padding before subtracting kh*dilation.
                const __int128 sum = static_cast<__int128>(coordinate_max) + padding;
                const __int128 low = static_cast<__int128>(padding) - kernel_term;
                TORCH_CHECK(sum <= maximum && low >= minimum &&
                                low <= maximum,
                            "Vulkan convolution coordinate expression exceeds signed shader range");
            } else {
                // op2 evaluates (oh * stride - padding) + kh*dilation.
                const __int128 before_kernel = stride_term - padding;
                const __int128 result = before_kernel + kernel_term;
                TORCH_CHECK(before_kernel >= minimum && before_kernel <= maximum &&
                                result >= minimum && result <= maximum,
                            "Vulkan convolution coordinate expression exceeds signed shader range");
            }
        };
        if (operation == 0) {
            checked_axis(shape[2] - 1, kh - 1, geometry.stride_height,
                         geometry.padding_height, geometry.dilation_height, 0);
            checked_axis(shape[3] - 1, kw - 1, geometry.stride_width,
                         geometry.padding_width, geometry.dilation_width, 0);
        } else if (operation == 1) {
            checked_axis(shape[2] - 1, kh - 1, geometry.stride_height,
                         geometry.padding_height, geometry.dilation_height, 1);
            checked_axis(shape[3] - 1, kw - 1, geometry.stride_width,
                         geometry.padding_width, geometry.dilation_width, 1);
        } else if (operation == 2) {
            checked_axis(read_input.sizes[2] - 1, kh - 1, geometry.stride_height,
                         geometry.padding_height, geometry.dilation_height, 2);
            checked_axis(read_input.sizes[3] - 1, kw - 1, geometry.stride_width,
                         geometry.padding_width, geometry.dilation_width, 2);
            constexpr uint64_t max_reduction_count =
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 255;
            const uint64_t reduction = checked_mul(
                static_cast<uint64_t>(read_input.sizes[0]),
                checked_mul(static_cast<uint64_t>(read_input.sizes[2]),
                            static_cast<uint64_t>(read_input.sizes[3])));
            TORCH_CHECK(reduction <= max_reduction_count,
                        "Vulkan convolution op2 reader reduction exceeds shader index range");
        } else {
            constexpr uint64_t max_reduction_count =
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 255;
            const uint64_t reduction = checked_mul(
                static_cast<uint64_t>(grad_output->size(0)),
                checked_mul(static_cast<uint64_t>(grad_output->size(2)),
                            static_cast<uint64_t>(grad_output->size(3))));
            TORCH_CHECK(grad_layout.numel <= std::numeric_limits<uint32_t>::max() &&
                            reduction <= max_reduction_count,
                        "Vulkan convolution grad_output logical numel/reduction exceeds shader index range");
        }
        plan.outputs[plan.output_count++] = spec;
    };

    if (forward) {
        append_output(transposed ? 1u : 0u, 0,
                      {input.size(0), output_channels, out_h, out_w}, 4,
                      weight.size(2), weight.size(3));
    } else {
        const bool request_input = requested_outputs[0];
        const bool request_weight = requested_outputs[1];
        const bool request_bias = requested_outputs[2];
        if (request_input || request_weight) {
            TORCH_CHECK(grad_output->size(0) == input.size(0) &&
                            grad_output->size(1) == output_channels &&
                            grad_output->size(2) == out_h &&
                            grad_output->size(3) == out_w,
                        "Vulkan convolution backward grad_output shape expected (",
                        input.size(0), ", ", output_channels, ", ", out_h,
                        ", ", out_w, "), actual (", grad_output->size(0),
                        ", ", grad_output->size(1), ", ", grad_output->size(2),
                        ", ", grad_output->size(3), ")");
        } else if (request_bias) {
            TORCH_CHECK(grad_output->size(0) == input.size(0) &&
                            grad_output->size(1) == output_channels,
                        "Vulkan convolution backward dBias grad_output requires matching batch and output channels");
        }

        if (request_input)
            append_output(transposed ? 0u : 1u, 0,
                          {input.size(0), input.size(1), input.size(2), input.size(3)},
                          4, weight.size(2), weight.size(3));
        if (request_weight)
            append_output(2, 1,
                          {weight.size(0), weight.size(1), weight.size(2),
                           weight.size(3)}, 4, weight.size(2), weight.size(3));
        if (request_bias)
            append_output(3, 2, {output_channels, 0, 0, 0}, 1, 1, 1);
    }
    return plan;
}

at::Tensor allocate_convolution_output(const ConvolutionOutputSpec &spec,
                                       at::TensorOptions options) {
    return at::empty(at::IntArrayRef(spec.shape.data(), spec.rank), options);
}

void dispatch_convolution_output(const ConvolutionOutputSpec &spec,
                                 const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const at::Tensor *bias,
                                 const at::Tensor *grad_output,
                                 const at::Tensor &output,
                                 const ConvolutionPreflight &plan) {
    auto role_tensor = [&](ConvolutionSourceRole role) -> const at::Tensor & {
        switch (role) {
        case ConvolutionSourceRole::Input: return input;
        case ConvolutionSourceRole::Weight: return weight;
        case ConvolutionSourceRole::GradOutput: return *grad_output;
        case ConvolutionSourceRole::Bias:
            return bias != nullptr && bias->defined() ? *bias : input;
        }
        TORCH_CHECK(false, "Vulkan convolution dispatch has an invalid source role");
    };
    auto role_layout = [&](ConvolutionSourceRole role) -> const VulkanTensorLayout & {
        switch (role) {
        case ConvolutionSourceRole::Input: return plan.input_layout;
        case ConvolutionSourceRole::Weight: return plan.weight_layout;
        case ConvolutionSourceRole::GradOutput: return plan.grad_layout;
        case ConvolutionSourceRole::Bias:
            return bias != nullptr && bias->defined() ? plan.bias_layout
                                                       : plan.input_layout;
        }
        TORCH_CHECK(false, "Vulkan convolution dispatch has an invalid source role");
    };
    const auto &input_tensor = role_tensor(spec.source_roles[0]);
    const auto &weight_tensor = role_tensor(spec.source_roles[1]);
    const auto &bias_tensor = role_tensor(spec.source_roles[2]);
    const auto &input_layout = role_layout(spec.source_roles[0]);
    const auto &weight_layout = role_layout(spec.source_roles[1]);
    const auto &bias_layout = role_layout(spec.source_roles[2]);
    const bool has_bias = plan.forward && bias != nullptr && bias->defined();
    auto output_layout = inspect_vulkan_tensor_layout(output, "convolution output");
    TORCH_CHECK(output_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan convolution output must be writable and non-overlapping");
    const auto &input_data = input_tensor.storage().data_ptr();
    const auto &weight_data = weight_tensor.storage().data_ptr();
    const auto &bias_data = bias_tensor.storage().data_ptr();
    const auto &output_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(bias_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan convolution requires Vulkan allocations");
    validate_allocation(output_data, spec.bytes, "convolution output");
    platform.compute().convolution(
        allocation_buffer(input_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(bias_data).buffer(), allocation_buffer(output_data).buffer(),
        input_layout, weight_layout, bias_layout, output_layout, spec.operation,
        spec.kernel_height, spec.kernel_width, spec.shader_geometry,
        has_bias);
}

at::Tensor run_single(const at::Tensor &input, const at::Tensor &weight,
                      const at::Tensor *bias, const at::Tensor *grad_output,
                      const ConvolutionGeometry &geometry, bool transposed,
                      std::array<int64_t, 2> output_padding) {
    std::array<bool, 3> mask{true, true, true};
    auto plan = preflight_convolution(input, weight, bias, grad_output, geometry,
                                     transposed, output_padding, mask,
                                     true);
    const auto &spec = plan.outputs[0];
    at::Tensor output = allocate_convolution_output(spec, input.options());
    dispatch_convolution_output(spec, input, weight, bias, grad_output, output, plan);
    return output;
}

at::Tensor run_legacy_backward_operation(
    const at::Tensor &input_role, const at::Tensor &weight_role,
    const at::Tensor &bias_role, uint32_t operation,
    const ConvolutionGeometry &geometry, int64_t input_height = 0,
    int64_t input_width = 0, int64_t kernel_height = 0,
    int64_t kernel_width = 0) {
    validate_tensor(input_role, "backward read input", 4);
    validate_tensor(weight_role, "backward read weight", 4);
    validate_tensor(bias_role, "backward read bias", 4);
    TORCH_CHECK(geometry.groups > 0,
                "Vulkan convolution backward requires groups >= 1");
    TORCH_CHECK(input_role.device() == weight_role.device() &&
                    input_role.device() == bias_role.device(),
                "Vulkan convolution tensors require one device");
    const auto input_layout = inspect_vulkan_tensor_layout(input_role,
                                                           "convolution input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight_role,
                                                            "convolution weight");
    const auto bias_layout = inspect_vulkan_tensor_layout(bias_role,
                                                          "convolution bias");
    validate_read_layout(input_role, input_layout, "grad_output", true, true);
    validate_read_layout(weight_role, weight_layout,
                         operation == 3 ? "grad_output" :
                             (operation == 1 ? "weight" : "input"),
                         true, operation == 3);
    validate_read_layout(bias_role, bias_layout, "grad_output", true, true);
    std::array<int64_t, 4> shape{input_role.size(0), input_role.size(1), 1, 1};
    uint32_t rank = 1;
    int64_t kh = kernel_height;
    int64_t kw = kernel_width;
    if (operation == 1) {
        TORCH_CHECK(input_height > 0 && input_width > 0,
                    "Vulkan convolution backward input extent must be positive");
        kh = weight_role.size(2);
        kw = weight_role.size(3);
        const auto expected_h = checked_output_extent(
            input_height, kh, geometry.stride_height, geometry.padding_height,
            geometry.dilation_height);
        const auto expected_w = checked_output_extent(
            input_width, kw, geometry.stride_width, geometry.padding_width,
            geometry.dilation_width);
        TORCH_CHECK(input_role.size(2) == expected_h &&
                        input_role.size(3) == expected_w,
                    "Vulkan convolution backward grad_output spatial shape mismatch");
        const uint64_t input_channels =
            checked_mul(weight_role.size(1), geometry.groups);
        TORCH_CHECK(input_channels <=
                        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "Vulkan convolution backward input channels overflow");
        shape = {input_role.size(0), static_cast<int64_t>(input_channels),
                 input_height, input_width};
        rank = 4;
    } else if (operation == 2) {
        TORCH_CHECK(kh > 0 && kw > 0,
                    "Vulkan convolution backward kernel extent must be positive");
        const auto expected_h = checked_output_extent(
            weight_role.size(2), kh, geometry.stride_height,
            geometry.padding_height, geometry.dilation_height);
        const auto expected_w = checked_output_extent(
            weight_role.size(3), kw, geometry.stride_width,
            geometry.padding_width, geometry.dilation_width);
        TORCH_CHECK(input_role.size(2) == expected_h &&
                        input_role.size(3) == expected_w,
                    "Vulkan convolution backward grad_output spatial shape mismatch");
        shape = {input_role.size(1), weight_role.size(1) / geometry.groups,
                 kh, kw};
        rank = 4;
    } else if (operation == 3) {
        shape = {input_role.size(1), 0, 0, 0};
        kh = kw = 1;
    } else {
        throw std::invalid_argument("Vulkan convolution backward operation is invalid");
    }
    const uint32_t count = output_numel(shape, rank);
    ConvolutionOutputSpec spec{
        operation, 0,
        {ConvolutionSourceRole::GradOutput, ConvolutionSourceRole::Weight,
         ConvolutionSourceRole::GradOutput}, 2, shape, rank, count,
        static_cast<VkDeviceSize>(checked_mul(count, sizeof(float))),
        narrow_shader(kh, "kernel height"), narrow_shader(kw, "kernel width"),
        {narrow_shader(geometry.stride_height, "stride"),
         narrow_shader(geometry.stride_width, "stride"),
         narrow_shader(geometry.padding_height, "padding"),
         narrow_shader(geometry.padding_width, "padding"),
         narrow_shader(geometry.dilation_height, "dilation"),
         narrow_shader(geometry.dilation_width, "dilation"),
         narrow_shader(geometry.groups, "groups")}};
    const auto &input_data = input_role.storage().data_ptr();
    const auto &weight_data = weight_role.storage().data_ptr();
    const auto &bias_data = bias_role.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(bias_data),
                "Vulkan convolution requires allocations on one platform");
    validate_source_allocation(input_role, input_layout, "convolution input");
    validate_source_allocation(weight_role, weight_layout, "convolution weight");
    validate_source_allocation(bias_role, bias_layout, "convolution bias");
    platform.compute().validate_convolution_dispatch(
        input_layout.allocation_bytes, weight_layout.allocation_bytes,
        bias_layout.allocation_bytes, spec.bytes, spec.numel,
        operation == 2 || operation == 3);
    at::Tensor output = allocate_convolution_output(spec, input_role.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output,
                                                            "convolution output");
    const auto &output_data = output.storage().data_ptr();
    validate_allocation(output_data, spec.bytes, "convolution output");
    platform.compute().convolution(
        allocation_buffer(input_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(bias_data).buffer(), allocation_buffer(output_data).buffer(),
        input_layout, weight_layout, bias_layout, output_layout, operation,
        spec.kernel_height, spec.kernel_width, spec.shader_geometry, false);
    return output;
}

} // namespace

at::Tensor convolution(const at::Tensor &input, const at::Tensor &weight,
                       const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       bool transposed, at::IntArrayRef output_padding,
                       int64_t groups) {
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Vulkan convolution requires 2-D stride, padding and dilation");
    TORCH_CHECK(output_padding.size() == 2,
                "Vulkan convolution requires 2-D output_padding");
    TORCH_CHECK(transposed || output_padding.equals({0, 0}),
                "Vulkan convolution supports only zero output_padding for ordinary convolutions");
    const bool has_bias = bias.has_value() && bias->defined();
    const ConvolutionGeometry geometry{stride[0], stride[1], padding[0], padding[1],
                                       dilation[0], dilation[1], groups};
    const at::Tensor *defined_bias = has_bias ? &*bias : nullptr;
    return run_single(input, weight, defined_bias, nullptr, geometry, transposed,
                      {output_padding[0], output_padding[1]});
}

at::Tensor convolution_backward_input(const at::Tensor &grad, const at::Tensor &weight,
                                     const ConvolutionGeometry &geometry,
                                     int64_t input_height, int64_t input_width) {
    return run_legacy_backward_operation(grad, weight, grad, 1, geometry,
                                         input_height, input_width);
}
at::Tensor convolution_backward_weight(const at::Tensor &grad,
                                       const at::Tensor &input,
                                       const ConvolutionGeometry &geometry,
                                       int64_t kernel_height, int64_t kernel_width) {
    return run_legacy_backward_operation(grad, input, grad, 2, geometry,
                                         0, 0, kernel_height, kernel_width);
}
at::Tensor convolution_backward_bias(const at::Tensor &grad) {
    return run_legacy_backward_operation(grad, grad, grad, 3, {});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &weight,
    c10::OptionalArrayRef<int64_t> bias_sizes, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, std::array<bool, 3> output_mask) {
    (void)bias_sizes;
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Vulkan convolution backward requires 2-D stride, padding and dilation");
    TORCH_CHECK(output_padding.size() == 2,
                "Vulkan convolution backward requires 2-D output_padding");
    TORCH_CHECK(transposed || output_padding.equals({0, 0}),
                "Vulkan convolution backward supports only zero output_padding for ordinary convolutions");
    const ConvolutionGeometry geometry{stride[0], stride[1], padding[0], padding[1],
                                       dilation[0], dilation[1], groups};
    auto plan = preflight_convolution(input, weight, nullptr, &grad_output,
                                       geometry, transposed,
                                       {output_padding[0], output_padding[1]},
                                       output_mask, false);
    std::array<at::Tensor, 3> outputs;
    for (size_t i = 0; i < plan.output_count; ++i) {
        const auto &spec = plan.outputs[i];
        outputs[spec.result_slot] = allocate_convolution_output(spec, input.options());
    }
    for (size_t i = 0; i < plan.output_count; ++i)
    {
        const auto &spec = plan.outputs[i];
        dispatch_convolution_output(spec, input, weight, nullptr,
                                    &grad_output, outputs[spec.result_slot], plan);
    }
    return {outputs[0], outputs[1], outputs[2]};
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("convolution", &pytorch_vulkan::convolution);
    m.impl("convolution_overrideable", &pytorch_vulkan::convolution);
    m.impl("convolution_backward", &pytorch_vulkan::convolution_backward);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("convolution", &pytorch_vulkan::autograd_convolution);
    m.impl("convolution_overrideable", &pytorch_vulkan::autograd_convolution);
}
