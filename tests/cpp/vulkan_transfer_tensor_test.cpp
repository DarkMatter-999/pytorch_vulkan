#include "vulkan_allocator.h"
#include "vulkan/operators/add.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>

#include <functional>
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

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

void test_copy_returns_without_pending_transfer_resources() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();

    pytorch_vulkan::copy_tensor(device_tensor, source, false);

    expect(platform->pending_transfer_count() == 0,
           "synchronous tensor copy left pending transfer resources");
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

void test_nonzero_storage_offset_is_rejected() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    auto destination_base =
        at::empty({5}, source.options().device(kDevice));
    auto destination = destination_base;
    destination.unsafeGetTensorImpl()->set_storage_offset(1);
    destination.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(destination.is_contiguous() && destination.storage_offset() != 0,
           "destination test tensor is not a contiguous offset view");
    expect_error([&] { pytorch_vulkan::copy_tensor(destination, source, false); },
                 "storage_offset");

    auto source_base = at::empty({5}, source.options().device(kDevice));
    auto source_view = source_base;
    source_view.unsafeGetTensorImpl()->set_storage_offset(1);
    source_view.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    auto result = at::empty({4}, source.options());
    expect(source_view.is_contiguous() && source_view.storage_offset() != 0,
           "source test tensor is not a contiguous offset view");
    expect_error([&] { pytorch_vulkan::copy_tensor(result, source_view, false); },
                 "storage_offset");
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

void test_repeated_add_dispatch_and_retained_output() {
    auto lhs = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto rhs = at::tensor({10.0F, 20.0F, 30.0F, 40.0F});
    auto retained = pytorch_vulkan::add_tensor(
        at::empty_like(lhs, lhs.options().device(kDevice)),
        at::empty_like(rhs, rhs.options().device(kDevice)), 1.0F);

    pytorch_vulkan::copy_tensor(retained, lhs, false);
    auto device_rhs = at::empty_like(rhs, rhs.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_rhs, rhs, false);
    retained = pytorch_vulkan::add_tensor(retained, device_rhs, 1.0F);

    for (int iteration = 0; iteration < 16; ++iteration) {
        auto device_lhs = at::empty_like(lhs, lhs.options().device(kDevice));
        auto device_addend = at::empty_like(rhs, rhs.options().device(kDevice));
        pytorch_vulkan::copy_tensor(device_lhs, lhs, false);
        pytorch_vulkan::copy_tensor(device_addend, rhs, false);
        auto output = pytorch_vulkan::add_tensor(device_lhs, device_addend, 1.0F);
        auto result = at::empty_like(lhs);
        pytorch_vulkan::copy_tensor(result, output, false);
        expect(result.equal(at::tensor({11.0F, 22.0F, 33.0F, 44.0F})),
               "repeated Vulkan add changed data");
    }

    lhs = at::Tensor();
    rhs = at::Tensor();
    device_rhs = at::Tensor();
    auto retained_result = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
    pytorch_vulkan::copy_tensor(retained_result, retained, false);
    expect(retained_result.equal(at::tensor({11.0F, 22.0F, 33.0F, 44.0F})),
           "retained Vulkan add output changed after input release");
}

void test_zero_element_add_does_not_dispatch() {
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto lhs = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto rhs = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = pytorch_vulkan::add_tensor(lhs, rhs, 1.0F);
    expect(output.numel() == 0, "zero-element Vulkan add returned non-empty output");
    expect(platform->compute_dispatch_count() == before,
           "zero-element Vulkan add submitted a compute dispatch");
}

void test_concurrent_add_dispatches_are_serialized() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto lhs = at::empty({4}, source.options().device(kDevice));
    auto rhs = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, source, false);
    pytorch_vulkan::copy_tensor(rhs, source, false);

    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    constexpr int kThreads = 8;
    constexpr int kOperationsPerThread = 8;
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int thread = 0; thread < kThreads; ++thread) {
        workers.emplace_back([&] {
            try {
                for (int operation = 0; operation < kOperationsPerThread; ++operation) {
                    auto output = pytorch_vulkan::add_tensor(lhs, rhs, 1.0F);
                    auto result = at::empty_like(source);
                    pytorch_vulkan::copy_tensor(result, output, false);
                    expect(result.equal(at::tensor({2.0F, 4.0F, 6.0F, 8.0F})),
                           "concurrent Vulkan add changed data");
                }
            } catch (...) {
                failed.store(true);
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    expect(!failed.load(), "concurrent Vulkan add failed");
    expect(platform->compute_dispatch_count() - before ==
               static_cast<std::size_t>(kThreads * kOperationsPerThread),
           "concurrent Vulkan add lost a dispatch count");
}

void test_add_invalid_input_cleans_up() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto base = at::empty({5}, source.options().device(kDevice));
    auto offset = base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    auto rhs = at::empty({4}, source.options().device(kDevice));
    expect_error([&] { (void)pytorch_vulkan::add_tensor(offset, rhs, 1.0F); },
                 "storage_offset");
    expect(pytorch_vulkan::platform()->pending_transfer_count() == 0,
           "invalid Vulkan add left pending transfer resources");
}

void test_platform_destruction_is_nothrow() {
    static_assert(std::is_nothrow_destructible<VulkanPlatform>::value,
                  "Vulkan cleanup must not throw during ownership quarantine");
}

} // namespace

int main() {
    try {
        (void)pytorch_vulkan::platform();
        test_copy_round_trip();
        test_copy_returns_without_pending_transfer_resources();
        test_validation_boundaries();
        test_nonzero_storage_offset_is_rejected();
        test_zero_tensor_copy_is_noop();
        test_zero_allocator_payload();
        test_foreign_payload_rejected();
        test_repeated_add_dispatch_and_retained_output();
        test_zero_element_add_does_not_dispatch();
        test_concurrent_add_dispatches_are_serialized();
        test_add_invalid_input_cleans_up();
        test_platform_destruction_is_nothrow();
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
