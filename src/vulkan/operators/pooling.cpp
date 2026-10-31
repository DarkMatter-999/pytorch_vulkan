#include "pooling.h"
#include "autograd.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include <ATen/native/Pool.h>
#include <c10/util/Exception.h>
#include <limits>
#include <torch/library.h>
namespace pytorch_vulkan {
namespace {
VkDeviceSize bytes(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.numel() >= 0 &&
                    static_cast<uint64_t>(tensor.numel()) <=
                        std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan pooling ", name, " byte count overflows");
    const uint64_t value = static_cast<uint64_t>(tensor.numel()) * sizeof(float);
    TORCH_CHECK(value <= std::numeric_limits<VkDeviceSize>::max(), "Vulkan pooling ",
                name, " byte count exceeds range");
    return static_cast<VkDeviceSize>(value);
}
void validate(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan pooling ", name, " requires vk:0");
    TORCH_CHECK(tensor.dim() == 4, "Vulkan pooling ", name,
                " requires rank 4 NCHW input");
    TORCH_CHECK(tensor.numel() > 0, "Vulkan pooling ", name,
                " requires a nonempty tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.layout() == at::kStrided,
                "Vulkan pooling ", name, " requires a strided float32 tensor");
    TORCH_CHECK(tensor.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(2) <= std::numeric_limits<uint32_t>::max() &&
                    tensor.size(3) <= std::numeric_limits<uint32_t>::max(),
                "Vulkan pooling ", name, " dimensions exceed Vulkan limits");
}
void validate_output_size(at::IntArrayRef output_size) {
    TORCH_CHECK(output_size.size() == 2 && output_size[0] == 1 && output_size[1] == 1,
                "Vulkan pooling supports only output size (1, 1)");
}
at::Tensor dispatch(const at::Tensor &input, const at::Tensor &grad,
                    uint32_t operation) {
    validate(input, "input");
    const auto input_layout = inspect_vulkan_tensor_layout(input, "pooling input");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan pooling rejects overlapping input layouts");
    VulkanTensorLayout grad_layout = input_layout;
    if (operation == 1) {
        validate(grad, "grad_output");
        grad_layout = inspect_vulkan_tensor_layout(grad, "pooling grad_output");
        TORCH_CHECK(grad_layout.internal_overlap == VulkanOverlap::No,
                    "Vulkan pooling rejects overlapping grad_output layouts");
        TORCH_CHECK(grad.sizes().equals({input.size(0), input.size(1), 1, 1}),
                    "Vulkan pooling grad_output has unsupported shape");
    }
    return with_vulkan_operation(allocation_runtime(input.storage().data_ptr()), [&]() -> at::Tensor {
    at::Tensor output = operation == 0 ? at::empty({input.size(0), input.size(1), 1, 1},
                                                   input.options())
                                       : at::empty(input.sizes(), input.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "pooling output");
    const auto &input_data = input.storage().data_ptr();
    const auto &grad_data = grad.defined() ? grad.storage().data_ptr() : input_data;
    const auto &output_data = output.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(grad_data) &&
                    &platform == &allocation_platform(output_data),
                "Vulkan pooling requires Vulkan allocations on one device");
    validate_allocation(input_data, bytes(input, "input"), "pooling input");
    if (operation == 1)
        validate_allocation(grad_data, bytes(grad, "grad_output"),
                            "pooling grad_output");
    validate_allocation(output_data, bytes(output, "output"), "pooling output");
    platform.compute().pooling(
        allocation_buffer(operation == 0 ? input_data : grad_data).buffer(),
        allocation_buffer(output_data).buffer(),
        operation == 0 ? input_layout : grad_layout, output_layout,
        static_cast<uint32_t>(input.size(0)), static_cast<uint32_t>(input.size(1)),
        static_cast<uint32_t>(input.size(2)), static_cast<uint32_t>(input.size(3)),
        operation);
    return output;
    });
}
} // namespace
at::Tensor adaptive_avg_pool2d(const at::Tensor &input, at::IntArrayRef output_size) {
    validate_output_size(output_size);
    return dispatch(input, at::Tensor(), 0);
}
at::Tensor adaptive_avg_pool2d_backward(const at::Tensor &grad_output,
                                        const at::Tensor &input) {
    return dispatch(input, grad_output, 1);
}
namespace {
// ---- general 2-D pooling (operations 2-6 of pooling.comp) --------------------

constexpr uint64_t kPoolParameterLimit = 65535;

struct Pool2dArguments {
    int64_t kH, kW, dH, dW, padH, padW, dilH, dilW;
    bool ceil_mode;
};

void check_pool_tensor(const at::Tensor &tensor, const char *op, const char *name) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0) &&
                    tensor.scalar_type() == at::kFloat &&
                    tensor.layout() == at::kStrided,
                "Vulkan ", op, " ", name, " requires a strided float32 vk:0 tensor");
}

void check_pool_indices(const at::Tensor &tensor, const char *op) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0) &&
                    tensor.scalar_type() == at::kLong && tensor.layout() == at::kStrided,
                "Vulkan ", op, " indices requires a strided int64 vk:0 tensor");
}

// Mirrors the argument normalization in PyTorch's avg_pool2d / max_pool2d meta.
Pool2dArguments normalize_pool2d(const char *label, at::IntArrayRef kernel_size,
                                 at::IntArrayRef stride, at::IntArrayRef padding,
                                 at::IntArrayRef dilation, bool ceil_mode) {
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2, label,
                ": kernel_size must either be a single int, or a tuple of two ints");
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2, label,
                ": stride must either be omitted, a single int, or a tuple of two ints");
    TORCH_CHECK(padding.size() == 1 || padding.size() == 2, label,
                ": padding must either be a single int, or a tuple of two ints");
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2, label,
                ": dilation must be either a single int, or a tuple of two ints");
    Pool2dArguments a{};
    a.kH = kernel_size[0];
    a.kW = kernel_size.size() == 1 ? a.kH : kernel_size[1];
    a.dH = stride.empty() ? a.kH : stride[0];
    a.dW = stride.empty() ? a.kW : stride.size() == 1 ? a.dH : stride[1];
    a.padH = padding[0];
    a.padW = padding.size() == 1 ? a.padH : padding[1];
    a.dilH = dilation[0];
    a.dilW = dilation.size() == 1 ? a.dilH : dilation[1];
    a.ceil_mode = ceil_mode;
    return a;
}

struct Pool2dShape {
    int64_t batch, planes, height, width, out_height, out_width;
    bool batched;
    at::MemoryFormat memory_format;
};

Pool2dShape pool2d_shape(const char *op, const at::Tensor &input, const Pool2dArguments &a) {
    TORCH_CHECK(input.dim() == 3 || input.dim() == 4,
                "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for "
                "input, but got:",
                input.sizes());
    Pool2dShape shape{};
    shape.batched = input.dim() == 4;
    shape.batch = shape.batched ? input.size(0) : 1;
    shape.planes = input.size(-3);
    shape.height = input.size(-2);
    shape.width = input.size(-1);
    shape.out_height = at::native::pooling_output_shape<int64_t>(
        shape.height, a.kH, a.padH, a.dH, a.dilH, a.ceil_mode);
    shape.out_width = at::native::pooling_output_shape<int64_t>(
        shape.width, a.kW, a.padW, a.dW, a.dilW, a.ceil_mode);
    shape.memory_format = input.suggest_memory_format();
    (void)op;
    return shape;
}

void check_pool_limits(const char *op, const Pool2dArguments &a, const Pool2dShape &shape) {
    for (const int64_t value : {a.kH, a.kW, a.dH, a.dW, a.padH, a.padW, a.dilH, a.dilW,
                                shape.height, shape.width, shape.out_height, shape.out_width})
        TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kPoolParameterLimit,
                    "Vulkan ", op, " parameter ", value, " exceeds the shader limit of ",
                    kPoolParameterLimit);
    const uint64_t input_elements = static_cast<uint64_t>(shape.batch) * shape.planes *
                                    shape.height * shape.width;
    const uint64_t output_elements = static_cast<uint64_t>(shape.batch) * shape.planes *
                                     shape.out_height * shape.out_width;
    TORCH_CHECK(input_elements <= std::numeric_limits<uint32_t>::max() &&
                    output_elements <= std::numeric_limits<uint32_t>::max(),
                "Vulkan ", op, " element count exceeds the uint32_t shader limit");
}

std::vector<int64_t> pool_output_sizes(const Pool2dShape &shape) {
    std::vector<int64_t> sizes;
    if (shape.batched)
        sizes.push_back(shape.batch);
    sizes.insert(sizes.end(), {shape.planes, shape.out_height, shape.out_width});
    return sizes;
}

at::Tensor pool_empty(at::IntArrayRef sizes, const at::TensorOptions &options,
                      const Pool2dShape &shape, const at::Tensor *like = nullptr) {
    (void)like;
    if (shape.batched)
        return at::empty(sizes, options, shape.memory_format);
    return at::empty(sizes, options);
}

// The shader indexes N,C,H,W through metadata; present 3-D tensors as 4-D.
VulkanTensorLayout pool_layout(const at::Tensor &tensor, const char *op, bool batched,
                               bool readable) {
    VulkanTensorLayout layout = inspect_vulkan_tensor_layout(tensor, op);
    TORCH_CHECK(layout.byte_range > 0 &&
                    layout.byte_offset + layout.byte_range <= layout.allocation_bytes,
                "Vulkan ", op, " tensor layout exceeds its allocation");
    TORCH_CHECK(readable || layout.internal_overlap == VulkanOverlap::No, "Vulkan ", op,
                " rejects overlapping output layouts");
    if (!batched) {
        layout.rank = 4;
        layout.sizes.insert(layout.sizes.begin(), 1);
        layout.strides.insert(layout.strides.begin(),
                              layout.sizes[1] * layout.strides[0]);
    }
    return layout;
}

VulkanPooling2dGeometry pool_geometry(const Pool2dArguments &a, const Pool2dShape &s,
                                      bool count_include_pad,
                                      const c10::optional<int64_t> &divisor) {
    VulkanPooling2dGeometry g{};
    g.batch = static_cast<uint32_t>(s.batch);
    g.channels = static_cast<uint32_t>(s.planes);
    g.height = static_cast<uint32_t>(s.height);
    g.width = static_cast<uint32_t>(s.width);
    g.out_height = static_cast<uint32_t>(s.out_height);
    g.out_width = static_cast<uint32_t>(s.out_width);
    g.kernel_h = static_cast<uint32_t>(a.kH);
    g.kernel_w = static_cast<uint32_t>(a.kW);
    g.stride_h = static_cast<uint32_t>(a.dH);
    g.stride_w = static_cast<uint32_t>(a.dW);
    g.pad_h = static_cast<uint32_t>(a.padH);
    g.pad_w = static_cast<uint32_t>(a.padW);
    g.dilation_h = static_cast<uint32_t>(a.dilH);
    g.dilation_w = static_cast<uint32_t>(a.dilW);
    g.count_include_pad = count_include_pad;
    g.has_divisor = divisor.has_value();
    g.divisor = divisor.has_value() ? static_cast<uint32_t>(*divisor) : 0U;
    return g;
}

VulkanRuntimeLifecycle *pool_runtime(const at::Tensor &tensor) {
    return tensor.numel() > 0 ? allocation_runtime(tensor.storage().data_ptr()) : nullptr;
}

const VkBuffer pool_buffer(const at::Tensor &tensor) {
    return allocation_buffer(tensor.storage().data_ptr()).buffer();
}

const VulkanPlatform &pool_platform(const at::Tensor &a, const at::Tensor &b) {
    const auto &platform = allocation_platform(a.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(b.storage().data_ptr()),
                "Vulkan pooling requires Vulkan allocations on one device");
    return platform;
}

void check_divisor(const c10::optional<int64_t> &divisor) {
    TORCH_CHECK(!divisor.has_value() || *divisor != 0, "divisor must be not zero");
    TORCH_CHECK(!divisor.has_value() ||
                    (*divisor > 0 && static_cast<uint64_t>(*divisor) <=
                                         std::numeric_limits<uint32_t>::max()),
                "Vulkan avg_pool2d divisor_override must be a positive 32-bit value");
}
} // namespace

at::Tensor avg_pool2d(const at::Tensor &input, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                      bool count_include_pad, c10::optional<int64_t> divisor_override) {
    check_pool_tensor(input, "avg_pool2d", "input");
    const auto a = normalize_pool2d("avg_pool2d", kernel_size, stride, padding, {1}, ceil_mode);
    check_divisor(divisor_override);
    const auto s = pool2d_shape("avg_pool2d", input, a);
    at::native::pool2d_shape_check(input, a.kH, a.kW, a.dH, a.dW, a.padH, a.padW, 1, 1,
                                   s.planes, s.height, s.width, s.out_height, s.out_width,
                                   s.memory_format);
    check_pool_limits("avg_pool2d", a, s);
    const auto sizes = pool_output_sizes(s);
    if (input.numel() == 0 || s.batch == 0)
        return pool_empty(sizes, input.options(), s);
    return with_vulkan_operation(pool_runtime(input), [&]() -> at::Tensor {
    at::Tensor output = pool_empty(sizes, input.options(), s);
    const auto input_layout = pool_layout(input, "avg_pool2d", s.batched, true);
    const auto output_layout = pool_layout(output, "avg_pool2d", s.batched, false);
    const auto &platform = pool_platform(input, output);
    const uint32_t invocations = static_cast<uint32_t>(output.numel());
    platform.compute().pooling2d(2, pool_buffer(input), pool_buffer(output), VK_NULL_HANDLE,
                                 VK_NULL_HANDLE, input_layout, output_layout, nullptr, 0,
                                 pool_geometry(a, s, count_include_pad, divisor_override),
                                 invocations);
    return output;
    });
}

at::Tensor avg_pool2d_backward(const at::Tensor &grad_output, const at::Tensor &input,
                               at::IntArrayRef kernel_size, at::IntArrayRef stride,
                               at::IntArrayRef padding, bool ceil_mode,
                               bool count_include_pad,
                               c10::optional<int64_t> divisor_override) {
    check_pool_tensor(grad_output, "avg_pool2d_backward", "grad_output");
    check_pool_tensor(input, "avg_pool2d_backward", "input");
    const auto a = normalize_pool2d("avg_pool2d", kernel_size, stride, padding, {1}, ceil_mode);
    check_divisor(divisor_override);
    const auto s = pool2d_shape("avg_pool2d_backward", input, a);
    at::native::avg_pool2d_backward_shape_check(
        input, grad_output, s.batch, a.kH, a.kW, a.dH, a.dW, a.padH, a.padW, s.planes,
        s.height, s.width, s.out_height, s.out_width, s.memory_format);
    check_pool_limits("avg_pool2d_backward", a, s);
    const auto make_grad_input = [&] {
        return s.batched ? at::empty(input.sizes(), input.options(), s.memory_format)
                         : at::empty(input.sizes(), input.options());
    };
    if (input.numel() == 0 || s.batch == 0)
        return make_grad_input();
    return with_vulkan_operation(pool_runtime(input), [&]() -> at::Tensor {
    at::Tensor grad_input = make_grad_input();
    const auto grad_layout = pool_layout(grad_output, "avg_pool2d_backward", s.batched, true);
    const auto input_layout = pool_layout(grad_input, "avg_pool2d_backward", s.batched, false);
    const auto &platform = pool_platform(grad_output, grad_input);
    platform.compute().pooling2d(3, pool_buffer(grad_output), pool_buffer(grad_input),
                                 VK_NULL_HANDLE, VK_NULL_HANDLE, grad_layout, input_layout,
                                 nullptr, 0,
                                 pool_geometry(a, s, count_include_pad, divisor_override),
                                 static_cast<uint32_t>(grad_input.numel()));
    return grad_input;
    });
}

std::tuple<at::Tensor, at::Tensor>
max_pool2d_with_indices(const at::Tensor &input, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode) {
    check_pool_tensor(input, "max_pool2d_with_indices", "input");
    const auto a = normalize_pool2d("max_pool2d", kernel_size, stride, padding, dilation, ceil_mode);
    const auto s = pool2d_shape("max_pool2d_with_indices", input, a);
    at::native::pool2d_shape_check(input, a.kH, a.kW, a.dH, a.dW, a.padH, a.padW, a.dilH,
                                   a.dilW, s.planes, s.height, s.width, s.out_height,
                                   s.out_width, s.memory_format);
    check_pool_limits("max_pool2d_with_indices", a, s);
    TORCH_CHECK(static_cast<uint64_t>(s.height) * static_cast<uint64_t>(s.width) <=
                    std::numeric_limits<int32_t>::max(),
                "Vulkan max_pool2d_with_indices plane size exceeds the int32 index limit");
    const auto sizes = pool_output_sizes(s);
    const auto make_indices = [&] {
        VulkanIndexOutputAllocationGuard guard;
        return pool_empty(sizes, input.options().dtype(at::kLong), s);
    };
    if (input.numel() == 0 || s.batch == 0)
        return {pool_empty(sizes, input.options(), s), make_indices()};
    return with_vulkan_operation(pool_runtime(input), [&]() -> std::tuple<at::Tensor, at::Tensor> {
    at::Tensor output = pool_empty(sizes, input.options(), s);
    at::Tensor indices = make_indices();
    const auto input_layout = pool_layout(input, "max_pool2d_with_indices", s.batched, true);
    const auto output_layout = pool_layout(output, "max_pool2d_with_indices", s.batched, false);
    const auto indices_layout = pool_layout(indices, "max_pool2d_with_indices", s.batched, false);
    const auto &platform = pool_platform(input, output);
    platform.compute().pooling2d(4, pool_buffer(input), pool_buffer(output),
                                 pool_buffer(indices), VK_NULL_HANDLE, input_layout,
                                 output_layout, &indices_layout, 0,
                                 pool_geometry(a, s, true, c10::nullopt),
                                 static_cast<uint32_t>(output.numel()));
    return {output, indices};
    });
}

at::Tensor max_pool2d_with_indices_backward(
    const at::Tensor &grad_output, const at::Tensor &input, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode, const at::Tensor &indices) {
    const char *op = "max_pool2d_with_indices_backward";
    check_pool_tensor(grad_output, op, "grad_output");
    check_pool_tensor(input, op, "input");
    check_pool_indices(indices, op);
    const auto a = normalize_pool2d("max_pool2d", kernel_size, stride, padding, dilation, ceil_mode);
    const auto s = pool2d_shape(op, input, a);
    at::native::max_pool2d_backward_shape_check(
        input, grad_output, indices, a.kH, a.kW, a.dH, a.dW, a.padH, a.padW, a.dilH, a.dilW,
        s.planes, s.height, s.width, s.out_height, s.out_width, s.memory_format);
    check_pool_limits(op, a, s);
    TORCH_CHECK(static_cast<uint64_t>(s.height) * static_cast<uint64_t>(s.width) <=
                    std::numeric_limits<int32_t>::max(),
                "Vulkan max_pool2d_with_indices_backward plane size exceeds the int32 index limit");
    const auto make_grad_input = [&] {
        return s.batched ? at::empty(input.sizes(), input.options(), s.memory_format)
                         : at::empty(input.sizes(), input.options());
    };
    if (input.numel() == 0 || s.batch == 0)
        return make_grad_input();
    return with_vulkan_operation(pool_runtime(input), [&]() -> at::Tensor {
    at::Tensor grad_input = make_grad_input();
    // One float per (n, c) plane: written by the validation pass, read by the
    // gather pass, so it needs no zero fill.
    at::Tensor flags = at::empty({s.batch * s.planes}, input.options());
    const auto grad_layout = pool_layout(grad_output, op, s.batched, true);
    const auto input_layout = pool_layout(grad_input, op, s.batched, false);
    const auto indices_layout = pool_layout(indices, op, s.batched, true);
    const auto flags_layout = inspect_vulkan_tensor_layout(flags, op);
    const auto &platform = pool_platform(grad_output, grad_input);
    const auto geometry = pool_geometry(a, s, true, c10::nullopt);
    platform.compute().pooling2d(6, pool_buffer(grad_output), pool_buffer(grad_input),
                                 pool_buffer(indices), pool_buffer(flags), grad_layout,
                                 input_layout, &indices_layout, flags_layout.allocation_bytes,
                                 geometry, static_cast<uint32_t>(s.batch * s.planes));
    platform.compute().pooling2d(5, pool_buffer(grad_output), pool_buffer(grad_input),
                                 pool_buffer(indices), pool_buffer(flags), grad_layout,
                                 input_layout, &indices_layout, flags_layout.allocation_bytes,
                                 geometry, static_cast<uint32_t>(grad_input.numel()));
    return grad_input;
    });
}
} // namespace pytorch_vulkan
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_adaptive_avg_pool2d", &pytorch_vulkan::adaptive_avg_pool2d);
    m.impl("_adaptive_avg_pool2d_backward",
           &pytorch_vulkan::adaptive_avg_pool2d_backward);
    m.impl("avg_pool2d", &pytorch_vulkan::avg_pool2d);
    m.impl("avg_pool2d_backward", &pytorch_vulkan::avg_pool2d_backward);
    m.impl("max_pool2d_with_indices", &pytorch_vulkan::max_pool2d_with_indices);
    m.impl("max_pool2d_with_indices_backward",
           &pytorch_vulkan::max_pool2d_with_indices_backward);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("_adaptive_avg_pool2d", &pytorch_vulkan::autograd_adaptive_avg_pool2d);
}
