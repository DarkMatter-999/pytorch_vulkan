#include "vulkan_pinned_allocator.h"

#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/EmptyTensor.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstdint>

#ifdef PYTORCH_VULKAN_TESTING
#include <atomic>
namespace pytorch_vulkan::testing {
namespace {
std::atomic<uint64_t> pinned_kernel_calls{0};
}
void reset_pinned_memory_kernel_calls_for_testing() noexcept {
    pinned_kernel_calls.store(0, std::memory_order_relaxed);
}
uint64_t pinned_memory_kernel_calls_for_testing() noexcept {
    return pinned_kernel_calls.load(std::memory_order_relaxed);
}
} // namespace pytorch_vulkan::testing
#endif

namespace {

void record_pinned_kernel_call() noexcept {
#ifdef PYTORCH_VULKAN_TESTING
    pytorch_vulkan::testing::pinned_kernel_calls.fetch_add(1, std::memory_order_relaxed);
#endif
}

bool is_pinned_vulkan(const at::Tensor &self,
                      c10::optional<c10::Device> device) {
    record_pinned_kernel_call();
    (void)device;
    if (!self.device().is_cpu())
        return false;
    return self.storage().data_ptr().get() != nullptr &&
           is_vulkan_pinned_storage(self.storage());
}

at::Tensor pin_memory_vulkan(const at::Tensor &self,
                             c10::optional<c10::Device> device) {
    record_pinned_kernel_call();
    TORCH_CHECK(!device.has_value() ||
                    (device->type() == c10::DeviceType::PrivateUse1 &&
                     device->index() <= 0),
                "pin_memory device must be vk");

    auto *allocator = vulkan_pinned_host_allocator_instance();
    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(self.sizes(), self.strides(),
                                         self.dtype().itemsize()),
        allocator, /*resizable=*/false);
    auto result = at::cpu::empty({0}, self.options())
                      .set_(storage, 0, self.sizes(), self.strides());
    result.copy_(self);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("is_pinned", TORCH_FN(is_pinned_vulkan));
    m.impl("_pin_memory", TORCH_FN(pin_memory_vulkan));
}
