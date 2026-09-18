#include "vulkan_allocator.h"
#include "vulkan_device_guard.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_bool_storage_and_round_trip() {
    auto source = at::empty({3}, at::TensorOptions().dtype(at::kBool));
    auto *source_data = source.data_ptr<bool>();
    source_data[0] = true;
    source_data[1] = false;
    source_data[2] = true;
    auto device_tensor = at::empty({3}, source.options().device(kDevice));
    expect(device_tensor.nbytes() == source.numel() * sizeof(bool),
           "bool Vulkan storage does not use one byte per element");
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(device_tensor, "bool layout test");
    expect(layout.byte_offset == 0 && layout.byte_range == 3 * sizeof(bool),
           "bool Vulkan layout has the wrong contiguous byte range");
    expect(layout.allocation_bytes >= layout.byte_offset + layout.byte_range,
           "bool Vulkan layout exceeds its allocation");

    auto offset_base = at::empty({5}, source.options().device(kDevice));
    auto offset = offset_base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> offset_sizes{3};
    const std::vector<int64_t> offset_strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    const auto offset_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(offset, "bool offset layout test");
    expect(offset_layout.byte_offset == sizeof(bool) &&
               offset_layout.byte_range == 3 * sizeof(bool) &&
               offset_layout.allocation_bytes >=
                   offset_layout.byte_offset + offset_layout.byte_range,
           "bool Vulkan offset layout has the wrong byte range");
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(result, device_tensor, false);
    expect(result.equal(source), "bool Vulkan round trip changed data");
}

void test_empty_bool_storage() {
    auto tensor =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kBool).device(kDevice));
    expect(tensor.nbytes() == 0 && tensor.numel() == 0,
           "empty bool Vulkan tensor has nonzero storage");
}

} // namespace

int main() {
    try {
        (void)vulkan_allocator_instance();
        try {
            (void)pytorch_vulkan::platform();
        } catch (const VulkanUnavailable &) {
            return 77;
        }
        test_bool_storage_and_round_trip();
        test_empty_bool_storage();
        std::cout << "Vulkan bool dtype tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
