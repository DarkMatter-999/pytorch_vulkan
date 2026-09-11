#include "vulkan_allocator.h"
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

void expect_error(const std::function<void()> &operation, const char *text) {
    try {
        operation();
    } catch (const c10::Error &error) {
        expect(std::string(error.what()).find(text) != std::string::npos,
               "unexpected Vulkan transfer error");
        return;
    }
    throw std::runtime_error("Vulkan transfer accepted invalid input");
}

void test_copy_round_trip() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto result = at::empty({4}, source.options());
    pytorch_vulkan::copy_tensor(result, device_tensor, false);
    expect(result.equal(source), "Vulkan tensor round trip changed data");
}

void test_validation_boundaries() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, source, true); },
                 "non_blocking=True");
    expect_error([&] { pytorch_vulkan::copy_tensor(source, source, false); },
                 "exactly one CPU and one Vulkan device");
    auto wrong_type = at::ones({4}, at::TensorOptions().dtype(at::kDouble));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, wrong_type, false); },
                 "only float32");
    auto wrong_size = at::ones({3}, at::TensorOptions().dtype(at::kFloat));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, wrong_size, false); },
                 "matching sizes");
}

void test_zero_tensor_copy_is_noop() {
    auto source = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
    auto device_tensor = at::empty({0}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto result = at::empty({0}, source.options());
    pytorch_vulkan::copy_tensor(result, device_tensor, false);
    expect(device_tensor.device() == kDevice, "empty Vulkan tensor has wrong device");
    expect(device_tensor.numel() == 0 && device_tensor.sizes() == source.sizes(),
           "empty Vulkan tensor has wrong metadata");
    expect(result.numel() == 0 && result.scalar_type() == at::kFloat,
           "empty Vulkan round trip has wrong metadata");

    for (int index = 0; index < 32; ++index) {
        auto repeated = at::empty({0}, source.options().device(kDevice));
        expect(repeated.numel() == 0, "repeated empty Vulkan tensor is non-empty");
    }
}

void test_zero_allocator_payload() {
    auto data = vulkan_allocator_instance()->allocate(0);
    expect(data.device() == kDevice, "empty allocation has wrong device");
    expect(data.get() == data.get_context(), "empty allocation context mismatch");
    expect(data.get() != nullptr, "empty allocation payload is null");
}

void test_foreign_payload_rejected() {
    at::DataPtr foreign(nullptr, nullptr, nullptr, kDevice);
    expect_error([&] { (void)pytorch_vulkan::allocation_buffer(foreign); },
                 "not a Vulkan allocation");
}

} // namespace

int main() {
    try {
        (void)pytorch_vulkan::platform();
        test_copy_round_trip();
        test_validation_boundaries();
        test_zero_tensor_copy_is_noop();
        test_zero_allocator_payload();
        test_foreign_payload_rejected();
        std::cout << "Vulkan tensor transfer tests passed\n";
        return 0;
    } catch (const VulkanUnavailable &error) {
        std::cerr << error.what() << '\n';
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
