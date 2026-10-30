#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace pytorch_vulkan {

enum class F32CopyRoute { Blocking, PinnedHostToDevice, PageableHostToDevice,
                          PinnedDeviceToHost, DeviceToDevice };

constexpr uint64_t kAsyncStagingMaxBytes = 64ull << 20;

namespace detail {

inline bool copy_route_is_vulkan(const at::Tensor &tensor) {
    return tensor.device().type() == c10::DeviceType::PrivateUse1 ||
           tensor.device().type() == c10::DeviceType::Vulkan;
}

// D1: both post-broadcast operands contiguous, or identical sizes+strides and
// dense non-overlapping (one storage-extent region, e.g. preserve-format E2).
inline bool copy_route_layout_eligible(const at::Tensor &destination,
                                       const at::Tensor &source) {
    if (destination.is_contiguous() && source.is_contiguous())
        return true;
    return destination.sizes() == source.sizes() &&
           destination.strides() == source.strides() &&
           destination.is_non_overlapping_and_dense() &&
           source.is_non_overlapping_and_dense();
}

} // namespace detail

// Pure metadata classification (D1-D7). `host_pinned` is the pinned-provenance
// result for the CPU operand; callers that cannot resolve it pass false.
inline F32CopyRoute classify_f32_copy_resolved(const at::Tensor &destination,
                                               const at::Tensor &source,
                                               bool non_blocking,
                                               bool async_execution,
                                               bool training_scope,
                                               bool host_pinned) {
    if (destination.scalar_type() != at::kFloat || source.scalar_type() != at::kFloat ||
        destination.layout() != at::kStrided || source.layout() != at::kStrided ||
        destination.numel() == 0 || source.numel() != destination.numel() ||
        !detail::copy_route_layout_eligible(destination, source))
        return F32CopyRoute::Blocking;
    const bool destination_vulkan = detail::copy_route_is_vulkan(destination);
    const bool source_vulkan = detail::copy_route_is_vulkan(source);
    // M1 (ratified): host routes are Blocking under an active training scope,
    // symmetric with sync execution mode. D7 (an accepted non_blocking completes
    // before return) is satisfied by completion-before-return; the engine's
    // training-wait path remains only as a defensive backstop. D2D classification
    // is unchanged: async mode and not training, otherwise Blocking.
    if (!async_execution || training_scope)
        return F32CopyRoute::Blocking;
    if (destination_vulkan && source_vulkan)
        return F32CopyRoute::DeviceToDevice;
    if (!non_blocking)
        return F32CopyRoute::Blocking;
    if (source.device().is_cpu() && destination_vulkan) {
        if (host_pinned)
            return F32CopyRoute::PinnedHostToDevice;
        const uint64_t bytes = static_cast<uint64_t>(source.numel()) * sizeof(float);
        return bytes <= kAsyncStagingMaxBytes ? F32CopyRoute::PageableHostToDevice
                                              : F32CopyRoute::Blocking;
    }
    if (source_vulkan && destination.device().is_cpu())
        return host_pinned ? F32CopyRoute::PinnedDeviceToHost : F32CopyRoute::Blocking;
    return F32CopyRoute::Blocking;
}

} // namespace pytorch_vulkan
