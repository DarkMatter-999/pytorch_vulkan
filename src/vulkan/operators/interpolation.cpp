#include "interpolation.h"
#include "autograd.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include <ATen/native/UpSample.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <cmath>
#include <limits>
#include <torch/library.h>

// 2-D interpolation leaves (nearest, nearest-exact, bilinear) for interpolation.comp.
// Size / scale-factor resolution, the .vec lowering and autograd stay with PyTorch;
// these are the six dispatcher leaves it reaches.  Scales are computed here with
// ATen's own inline helpers so the hint / double-rounding rules are reproduced by
// construction; the shader only runs correctly rounded float operations.
namespace pytorch_vulkan {
namespace {
enum class Mode : uint32_t { Nearest = 0, Exact = 1, Bilinear = 2 };

constexpr int64_t kMaxSpatial = int64_t{1} << 23;
// float(out) * scale beyond this is outside ATen's defined float->int64 cast range.
constexpr double kMaxSourceCoordinate = 4.0e18;

struct Plan {
    int64_t batch, channels, in_h, in_w, out_h, out_w;
    float scale_h, scale_w;
};

void check_tensor(const at::Tensor &tensor, const char *op) {
    TORCH_CHECK(tensor.device() == c10::Device(c10::DeviceType::PrivateUse1, 0) &&
                    tensor.scalar_type() == at::kFloat &&
                    tensor.layout() == at::kStrided,
                "Vulkan ", op, " requires a strided float32 vk:0 tensor");
}

// The scale the shader uses for one dimension, or 0 when the shader never reads it
// (ATen ignores the hint for these size relations).  Rejects scales whose source
// coordinate leaves the range where ATen's float->int64 cast is defined.
float resolve_scale(Mode mode, const char *op, int64_t in, int64_t out, bool align_corners,
                    const c10::optional<double> &hint) {
    bool used = true;
    bool hint_matters = true;
    if (mode == Mode::Nearest)
        used = out != in && out != 2 * in;
    else if (mode == Mode::Bilinear) {
        used = out != in;
        hint_matters = !align_corners;
    }
    if (!used)
        return 0.0F;
    if (hint_matters && hint.has_value() && *hint > 0.0)
        TORCH_CHECK(1.0 / *hint <= static_cast<double>(std::numeric_limits<float>::max()),
                    "Vulkan ", op, " scale hint produces an unsupported source scale");
    const float scale = mode == Mode::Bilinear
        ? at::native::area_pixel_compute_scale<float>(in, out, align_corners, hint)
        : at::native::compute_scales_value<float>(hint, in, out);
    TORCH_CHECK(std::isfinite(scale) && scale >= 0.0F &&
                    static_cast<double>(out) * static_cast<double>(scale) < kMaxSourceCoordinate,
                "Vulkan ", op, " scale hint produces an unsupported source scale");
    return scale;
}

void check_limits(const char *op, const Plan &plan) {
    TORCH_CHECK(plan.in_h <= kMaxSpatial && plan.in_w <= kMaxSpatial &&
                    plan.out_h <= kMaxSpatial && plan.out_w <= kMaxSpatial,
                "Vulkan ", op, " spatial sizes must not exceed ", kMaxSpatial);
}

void check_element_limit(const char *op, int64_t batch, int64_t channels, int64_t height,
                         int64_t width) {
    const uint64_t planes = static_cast<uint64_t>(batch) * static_cast<uint64_t>(channels);
    const uint64_t plane = static_cast<uint64_t>(height) * static_cast<uint64_t>(width);
    TORCH_CHECK(plane == 0 || planes <= std::numeric_limits<uint32_t>::max() / plane,
                "Vulkan ", op, " element count exceeds the uint32_t shader limit");
}

VulkanTensorLayout layout_of(const at::Tensor &tensor, const char *op, bool readable) {
    VulkanTensorLayout layout = inspect_vulkan_tensor_layout(tensor, op);
    TORCH_CHECK(layout.byte_range > 0 &&
                    layout.byte_offset + layout.byte_range <= layout.allocation_bytes,
                "Vulkan ", op, " tensor layout exceeds its allocation");
    TORCH_CHECK(readable || layout.internal_overlap == VulkanOverlap::No, "Vulkan ", op,
                " rejects overlapping output layouts");
    return layout;
}

VulkanRuntimeLifecycle *runtime_of(const at::Tensor &tensor) {
    return tensor.numel() > 0 ? allocation_runtime(tensor.storage().data_ptr()) : nullptr;
}

VkBuffer buffer_of(const at::Tensor &tensor) {
    return allocation_buffer(tensor.storage().data_ptr()).buffer();
}

const VulkanPlatform &platform_of(const at::Tensor &a, const at::Tensor &b, const char *op) {
    const auto &platform = allocation_platform(a.storage().data_ptr());
    TORCH_CHECK(&platform == &allocation_platform(b.storage().data_ptr()), "Vulkan ", op,
                " requires Vulkan allocations on one device");
    return platform;
}

VulkanInterpolation2dGeometry geometry_of(const Plan &plan, bool align_corners) {
    VulkanInterpolation2dGeometry g{};
    g.batch = static_cast<uint32_t>(plan.batch);
    g.channels = static_cast<uint32_t>(plan.channels);
    g.in_height = static_cast<uint32_t>(plan.in_h);
    g.in_width = static_cast<uint32_t>(plan.in_w);
    g.out_height = static_cast<uint32_t>(plan.out_h);
    g.out_width = static_cast<uint32_t>(plan.out_w);
    g.scale_h = plan.scale_h;
    g.scale_w = plan.scale_w;
    g.align_corners = align_corners;
    return g;
}

at::Tensor forward(Mode mode, const char *op, const at::Tensor &input,
                   at::IntArrayRef output_size, bool align_corners,
                   const c10::optional<double> &scales_h,
                   const c10::optional<double> &scales_w) {
    check_tensor(input, op);
    // ATen's own validation first, so messages and ordering match CPU.
    const auto full = at::native::upsample_2d_common_check(input.sizes(), output_size);
    TORCH_CHECK(input.numel() != 0 ||
                    c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
                "Non-empty 4D data tensor expected but got a tensor with sizes ",
                input.sizes());
    Plan plan{full[0], full[1], input.size(2), input.size(3), full[2], full[3], 0.0F, 0.0F};
    check_limits(op, plan);
    plan.scale_h = resolve_scale(mode, op, plan.in_h, plan.out_h, align_corners, scales_h);
    plan.scale_w = resolve_scale(mode, op, plan.in_w, plan.out_w, align_corners, scales_w);
    check_element_limit(op, plan.batch, plan.channels, plan.out_h, plan.out_w);
    check_element_limit(op, plan.batch, plan.channels, plan.in_h, plan.in_w);
    const std::vector<int64_t> sizes{plan.batch, plan.channels, plan.out_h, plan.out_w};
    const auto format = input.suggest_memory_format();
    if (input.numel() == 0 || plan.batch == 0)
        return at::empty(sizes, input.options(), format);
    return with_vulkan_operation(runtime_of(input), [&]() -> at::Tensor {
        at::Tensor output = at::empty(sizes, input.options(), format);
        const auto input_layout = layout_of(input, op, true);
        const auto output_layout = layout_of(output, op, false);
        platform_of(input, output, op).compute().interpolate2d(
            static_cast<uint32_t>(mode), buffer_of(input), buffer_of(output), input_layout,
            output_layout, geometry_of(plan, align_corners),
            static_cast<uint32_t>(output.numel()));
        return output;
    });
}

at::Tensor backward(Mode mode, const char *op, const at::Tensor &grad_output,
                    at::IntArrayRef output_size, at::IntArrayRef input_size,
                    bool align_corners, const c10::optional<double> &scales_h,
                    const c10::optional<double> &scales_w) {
    check_tensor(grad_output, op);
    const auto full = at::native::upsample_2d_common_check(input_size, output_size);
    for (int64_t i = 0; i < 4; ++i)
        TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(i) == full[i],
                    "Expected grad_output to have the same shape as output;",
                    " output.size(", i, ") = ", full[i], " but got grad_output.size(", i,
                    ") = ", grad_output.dim() == 4 ? grad_output.size(i) : -1);
    Plan plan{input_size[0], input_size[1], input_size[2], input_size[3], full[2], full[3],
              0.0F, 0.0F};
    check_limits(op, plan);
    plan.scale_h = resolve_scale(mode, op, plan.in_h, plan.out_h, align_corners, scales_h);
    plan.scale_w = resolve_scale(mode, op, plan.in_w, plan.out_w, align_corners, scales_w);
    check_element_limit(op, plan.batch, plan.channels, plan.out_h, plan.out_w);
    check_element_limit(op, plan.batch, plan.channels, plan.in_h, plan.in_w);
    const std::vector<int64_t> sizes{plan.batch, plan.channels, plan.in_h, plan.in_w};
    const auto format = grad_output.suggest_memory_format();
    if (grad_output.numel() == 0 || plan.batch == 0 || plan.channels == 0)
        return at::empty(sizes, grad_output.options(), format);
    return with_vulkan_operation(runtime_of(grad_output), [&]() -> at::Tensor {
        at::Tensor grad_input = at::empty(sizes, grad_output.options(), format);
        const auto grad_layout = layout_of(grad_output, op, true);
        const auto input_layout = layout_of(grad_input, op, false);
        platform_of(grad_output, grad_input, op).compute().interpolate2d(
            3U + static_cast<uint32_t>(mode), buffer_of(grad_output), buffer_of(grad_input),
            grad_layout, input_layout, geometry_of(plan, align_corners),
            static_cast<uint32_t>(grad_input.numel()));
        return grad_input;
    });
}
} // namespace

at::Tensor upsample_nearest2d(const at::Tensor &input, at::IntArrayRef output_size,
                              c10::optional<double> scales_h, c10::optional<double> scales_w) {
    return forward(Mode::Nearest, "upsample_nearest2d", input, output_size, false, scales_h,
                   scales_w);
}
at::Tensor upsample_nearest2d_backward(const at::Tensor &grad_output,
                                       at::IntArrayRef output_size, at::IntArrayRef input_size,
                                       c10::optional<double> scales_h,
                                       c10::optional<double> scales_w) {
    return backward(Mode::Nearest, "upsample_nearest2d_backward", grad_output, output_size,
                    input_size, false, scales_h, scales_w);
}
at::Tensor upsample_nearest_exact2d(const at::Tensor &input, at::IntArrayRef output_size,
                                    c10::optional<double> scales_h,
                                    c10::optional<double> scales_w) {
    return forward(Mode::Exact, "_upsample_nearest_exact2d", input, output_size, false,
                   scales_h, scales_w);
}
at::Tensor upsample_nearest_exact2d_backward(const at::Tensor &grad_output,
                                             at::IntArrayRef output_size,
                                             at::IntArrayRef input_size,
                                             c10::optional<double> scales_h,
                                             c10::optional<double> scales_w) {
    return backward(Mode::Exact, "_upsample_nearest_exact2d_backward", grad_output,
                    output_size, input_size, false, scales_h, scales_w);
}
at::Tensor upsample_bilinear2d(const at::Tensor &input, at::IntArrayRef output_size,
                               bool align_corners, c10::optional<double> scales_h,
                               c10::optional<double> scales_w) {
    return forward(Mode::Bilinear, "upsample_bilinear2d", input, output_size, align_corners,
                   scales_h, scales_w);
}
at::Tensor upsample_bilinear2d_backward(const at::Tensor &grad_output,
                                        at::IntArrayRef output_size,
                                        at::IntArrayRef input_size, bool align_corners,
                                        c10::optional<double> scales_h,
                                        c10::optional<double> scales_w) {
    return backward(Mode::Bilinear, "upsample_bilinear2d_backward", grad_output, output_size,
                    input_size, align_corners, scales_h, scales_w);
}

// Generated autograd (UpsampleNearest2DBackward0 ...) attaches on its own; no
// Autograd registrations or derivative wrappers are added here.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("upsample_nearest2d", &pytorch_vulkan::upsample_nearest2d);
    m.impl("upsample_nearest2d_backward", &pytorch_vulkan::upsample_nearest2d_backward);
    m.impl("_upsample_nearest_exact2d", &pytorch_vulkan::upsample_nearest_exact2d);
    m.impl("_upsample_nearest_exact2d_backward",
           &pytorch_vulkan::upsample_nearest_exact2d_backward);
    m.impl("upsample_bilinear2d", &pytorch_vulkan::upsample_bilinear2d);
    m.impl("upsample_bilinear2d_backward", &pytorch_vulkan::upsample_bilinear2d_backward);
}
} // namespace pytorch_vulkan
