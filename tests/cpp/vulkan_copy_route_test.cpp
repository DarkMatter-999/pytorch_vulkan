// CPU-only table test for the pure F32 copy classifier. Vulkan-device operands are
// metadata-only from_blob views over CPU memory; nothing here touches a GPU.
#include "vulkan_copy_route.h"

#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using pytorch_vulkan::F32CopyRoute;
using pytorch_vulkan::kAsyncStagingMaxBytes;

namespace {

int failures = 0;

// from_blob on a PrivateUse1 device asks the hooks registry; no device is touched.
struct StubHooks : at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex) const override { return false; }
    at::Device getDeviceFromPtr(void *) const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
};
StubHooks stub_hooks;

void expect(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

const char *name(F32CopyRoute route) {
    switch (route) {
    case F32CopyRoute::Blocking: return "Blocking";
    case F32CopyRoute::PinnedHostToDevice: return "PinnedHostToDevice";
    case F32CopyRoute::PageableHostToDevice: return "PageableHostToDevice";
    case F32CopyRoute::PinnedDeviceToHost: return "PinnedDeviceToHost";
    case F32CopyRoute::DeviceToDevice: return "DeviceToDevice";
    }
    return "?";
}

// Tensor with the given geometry whose device tag is `vulkan` or CPU. The Vulkan
// variant aliases CPU backing memory with a no-op deleter (metadata only).
struct Operand {
    at::Tensor backing;
    at::Tensor tensor;
};

Operand make(const at::Tensor &cpu, bool vulkan) {
    if (!vulkan) return {cpu, cpu};
    auto options = at::TensorOptions().dtype(cpu.dtype()).device(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto view = at::from_blob(cpu.data_ptr(), cpu.sizes(), cpu.strides(), [](void *) {}, options);
    return {cpu, view};
}

// Layout oracle: .superpowers/sdd/2026-10-20-vulkan-f32-copy-layout/receipts/
// async-t0-cpu-probe-03.json (result.stdout "layouts"). `eligible` is the D1
// verdict for a preserve-format clone destination and the layout as source.
struct LayoutCase {
    const char *label;
    at::Tensor tensor;
    bool eligible;
};

std::vector<LayoutCase> layouts() {
    const auto f = at::TensorOptions().dtype(at::kFloat);
    std::vector<LayoutCase> cases;
    cases.push_back({"contiguous", at::zeros({2, 3, 4}, f), true});
    cases.push_back({"offset_contiguous_slice", at::zeros({40}, f).slice(0, 8, 28).view({5, 4}), true});
    cases.push_back({"transposed", at::zeros({3, 4}, f).t(), true});
    cases.push_back({"channels_last", at::zeros({2, 3, 4, 5}, f.memory_format(at::MemoryFormat::ChannelsLast)), true});
    cases.push_back({"channels_last_3d", at::zeros({2, 3, 4, 5, 6}, f.memory_format(at::MemoryFormat::ChannelsLast3d)), true});
    cases.push_back({"stride0_expand", at::zeros({1}, f).expand({3, 4}), false});
    cases.push_back({"stride_hole", at::zeros({2, 3}, f).slice(1, 0, 2), false});
    cases.push_back({"zero_numel", at::zeros({0, 3}, f), false});
    return cases;
}

enum Dir { H2D, D2H, D2D };
enum Mode { Async, Sync, Training };

// Literal expected routes for an ELIGIBLE layout pair (D1). Hard-coded per cell, not
// derived: H2D/D2H non_blocking only in async mode outside training (M1 ratified:
// host routes are Blocking under training, symmetric with sync); D2D any flag in
// async mode outside training; every other cell is Blocking.
struct Cell { Dir dir; int non_blocking; Mode mode; int pinned; F32CopyRoute expected; };
const Cell kEligibleTable[] = {
    {H2D, 0, Async, 0, F32CopyRoute::Blocking},
    {H2D, 0, Async, 1, F32CopyRoute::Blocking},
    {H2D, 0, Sync, 0, F32CopyRoute::Blocking},
    {H2D, 0, Sync, 1, F32CopyRoute::Blocking},
    {H2D, 0, Training, 0, F32CopyRoute::Blocking},
    {H2D, 0, Training, 1, F32CopyRoute::Blocking},
    {H2D, 1, Async, 0, F32CopyRoute::PageableHostToDevice},
    {H2D, 1, Async, 1, F32CopyRoute::PinnedHostToDevice},
    {H2D, 1, Sync, 0, F32CopyRoute::Blocking},
    {H2D, 1, Sync, 1, F32CopyRoute::Blocking},
    {H2D, 1, Training, 0, F32CopyRoute::Blocking},
    {H2D, 1, Training, 1, F32CopyRoute::Blocking},
    {D2H, 0, Async, 0, F32CopyRoute::Blocking},
    {D2H, 0, Async, 1, F32CopyRoute::Blocking},
    {D2H, 0, Sync, 0, F32CopyRoute::Blocking},
    {D2H, 0, Sync, 1, F32CopyRoute::Blocking},
    {D2H, 0, Training, 0, F32CopyRoute::Blocking},
    {D2H, 0, Training, 1, F32CopyRoute::Blocking},
    {D2H, 1, Async, 0, F32CopyRoute::Blocking},
    {D2H, 1, Async, 1, F32CopyRoute::PinnedDeviceToHost},
    {D2H, 1, Sync, 0, F32CopyRoute::Blocking},
    {D2H, 1, Sync, 1, F32CopyRoute::Blocking},
    {D2H, 1, Training, 0, F32CopyRoute::Blocking},
    {D2H, 1, Training, 1, F32CopyRoute::Blocking},
    {D2D, 0, Async, 0, F32CopyRoute::DeviceToDevice},
    {D2D, 0, Async, 1, F32CopyRoute::DeviceToDevice},
    {D2D, 0, Sync, 0, F32CopyRoute::Blocking},
    {D2D, 0, Sync, 1, F32CopyRoute::Blocking},
    {D2D, 0, Training, 0, F32CopyRoute::Blocking},
    {D2D, 0, Training, 1, F32CopyRoute::Blocking},
    {D2D, 1, Async, 0, F32CopyRoute::DeviceToDevice},
    {D2D, 1, Async, 1, F32CopyRoute::DeviceToDevice},
    {D2D, 1, Sync, 0, F32CopyRoute::Blocking},
    {D2D, 1, Sync, 1, F32CopyRoute::Blocking},
    {D2D, 1, Training, 0, F32CopyRoute::Blocking},
    {D2D, 1, Training, 1, F32CopyRoute::Blocking},
};

void check_table() {
    // Layout verdicts come from the Task 0 oracle: receipts/async-t0-cpu-probe-03.json
    // (result.stdout "layouts"). Note async-t0-cpu-probe-01.json is a corrupt 19-byte file.
    for (const auto &layout : layouts()) {
        for (const auto &cell : kEligibleTable) {
            const bool dst_vk = cell.dir != D2H, src_vk = cell.dir != H2D;
            auto dst = make(layout.tensor.clone(at::MemoryFormat::Preserve), dst_vk);
            auto src = make(layout.tensor, src_vk);
            // Ineligible layouts (stride0_expand, stride_hole, zero_numel) are Blocking in every cell.
            const F32CopyRoute expected = layout.eligible ? cell.expected : F32CopyRoute::Blocking;
            const auto actual = pytorch_vulkan::classify_f32_copy_resolved(
                dst.tensor, src.tensor, cell.non_blocking, cell.mode != Sync,
                cell.mode == Training, cell.pinned);
            expect(actual == expected,
                   std::string(layout.label) + " dir=" + std::to_string(cell.dir) +
                       " nb=" + std::to_string(cell.non_blocking) + " mode=" + std::to_string(cell.mode) +
                       " pinned=" + std::to_string(cell.pinned) + " expected " + name(expected) +
                       " got " + name(actual));
        }
    }
}

void check_mixed_and_dtype() {
    const auto f = at::TensorOptions().dtype(at::kFloat);
    auto contiguous = make(at::zeros({3, 4}, f), true);
    auto transposed_cpu = make(at::zeros({4, 3}, f).t(), false);
    auto hole_cpu = make(at::zeros({3, 6}, f).slice(1, 0, 4), false);
    const auto route = [](const Operand &d, const Operand &s) {
        return pytorch_vulkan::classify_f32_copy_resolved(d.tensor, s.tensor, true, true, false, true);
    };
    expect(route(contiguous, transposed_cpu) == F32CopyRoute::Blocking, "contiguous<-transposed must degrade");
    expect(route(contiguous, hole_cpu) == F32CopyRoute::Blocking, "contiguous<-hole must degrade");
    expect(route(make(at::zeros({3, 4}, f), true), make(at::zeros({3, 4}, f), false)) ==
               F32CopyRoute::PinnedHostToDevice, "contiguous pinned H2D baseline");
    // Mixed layouts in every direction degrade (oracle: transposed != contiguous strides).
    for (const Dir dir : {H2D, D2H, D2D}) {
        auto d = make(at::zeros({3, 4}, f), dir != D2H);
        auto t = make(at::zeros({4, 3}, f).t(), dir != H2D);
        expect(route(d, t) == F32CopyRoute::Blocking, "mixed layout must degrade for dir " + std::to_string(dir));
    }
    // Mixed F32/F64 operands are Blocking in every direction and orientation.
    const auto d64 = at::TensorOptions().dtype(at::kDouble);
    for (const Dir dir : {H2D, D2H, D2D}) {
        const bool dst_vk = dir != D2H, src_vk = dir != H2D;
        expect(route(make(at::zeros({4}, f), dst_vk), make(at::zeros({4}, d64), src_vk)) ==
                   F32CopyRoute::Blocking, "F32<-F64 must be Blocking");
        expect(route(make(at::zeros({4}, d64), dst_vk), make(at::zeros({4}, f), src_vk)) ==
                   F32CopyRoute::Blocking, "F64<-F32 must be Blocking");
    }
    for (const auto dtype : {at::kDouble, at::kLong, at::kBool}) {
        auto d = make(at::zeros({4}, at::TensorOptions().dtype(dtype)), true);
        auto s = make(at::zeros({4}, at::TensorOptions().dtype(dtype)), false);
        expect(route(d, s) == F32CopyRoute::Blocking, "non-F32 must be Blocking");
    }
}

void check_staging_threshold() {
    const int64_t limit = static_cast<int64_t>(kAsyncStagingMaxBytes / sizeof(float));
    const auto f = at::TensorOptions().dtype(at::kFloat);
    // Metadata-only views over a one-float sentinel; classification never reads data.
    const auto cpu_view = [&](int64_t n) {
        static float sentinel = 0.0f;
        return at::from_blob(&sentinel, {n}, {1}, [](void *) {}, f);
    };
    const auto vk_view = [&](int64_t n) {
        static float sentinel = 0.0f;
        return at::from_blob(&sentinel, {n}, {1}, [](void *) {},
                             f.device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    };
    expect(pytorch_vulkan::classify_f32_copy_resolved(vk_view(limit), cpu_view(limit), true, true, false, false) ==
               F32CopyRoute::PageableHostToDevice, "64 MiB pageable H2D is eligible");
    expect(pytorch_vulkan::classify_f32_copy_resolved(vk_view(limit + 1), cpu_view(limit + 1), true, true, false, false) ==
               F32CopyRoute::Blocking, "64 MiB + 4 pageable H2D must degrade");
    expect(pytorch_vulkan::classify_f32_copy_resolved(vk_view(limit + 1), cpu_view(limit + 1), true, true, false, true) ==
               F32CopyRoute::PinnedHostToDevice, "oversized pinned H2D stays pinned");
}

} // namespace

int main() {
    try {
        at::RegisterPrivateUse1HooksInterface(&stub_hooks);
        check_table();
        check_mixed_and_dtype();
        check_staging_threshold();
    } catch (const std::exception &error) {
        std::cerr << "exception: " << error.what() << "\n";
        return 1;
    }
    if (failures) {
        std::cerr << failures << " classification failures\n";
        return 1;
    }
    std::cout << "classify_f32_copy table passed\n";
    return 0;
}
