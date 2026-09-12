#include "vulkan_allocator.h"
#include "vulkan/operators/abs.h"
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

bool is_missing_view_dispatch(const std::string &message) {
    return message.find("Could not run 'aten::view'") != std::string::npos;
}

void expect_view_rejected(const std::function<void()> &operation, const char *text) {
    try {
        operation();
    } catch (const c10::Error &error) {
        const std::string message(error.what());
        if (is_missing_view_dispatch(message)) {
            return;
        }
        expect(message.find(text) != std::string::npos,
               "Vulkan view rejected input for an unexpected reason");
        return;
    }
    throw std::runtime_error("Vulkan view accepted invalid input");
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

void test_tensor_tensor_sub_and_mul_dispatch() {
    auto lhs_source = at::tensor({1.0F, -2.0F, 3.0F});
    auto rhs_source = at::tensor({4.0F, 5.0F, -6.0F});
    auto lhs = at::empty_like(lhs_source, lhs_source.options().device(kDevice));
    auto rhs = at::empty_like(rhs_source, rhs_source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::copy_tensor(rhs, rhs_source, false);

    auto sub = pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, 1.0F, pytorch_vulkan::PointwiseOperation::Sub, "sub");
    auto mul = pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, 1.0F, pytorch_vulkan::PointwiseOperation::Mul, "mul");
    auto sub_result = at::empty_like(lhs_source);
    auto mul_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(sub_result, sub, false);
    pytorch_vulkan::copy_tensor(mul_result, mul, false);

    expect(sub_result.equal(at::tensor({-3.0F, -7.0F, 9.0F})),
           "Vulkan tensor/tensor sub changed data");
    expect(mul_result.equal(at::tensor({4.0F, -10.0F, -18.0F})),
           "Vulkan tensor/tensor mul changed data");
    auto lhs_result = at::empty_like(lhs_source);
    auto rhs_result = at::empty_like(rhs_source);
    pytorch_vulkan::copy_tensor(lhs_result, lhs, false);
    pytorch_vulkan::copy_tensor(rhs_result, rhs, false);
    expect(lhs_result.equal(lhs_source) && rhs_result.equal(rhs_source),
           "Vulkan tensor/tensor arithmetic changed inputs");
    expect(sub.data_ptr() != lhs.data_ptr() && sub.data_ptr() != rhs.data_ptr() &&
               mul.data_ptr() != lhs.data_ptr() && mul.data_ptr() != rhs.data_ptr() &&
               sub.data_ptr() != mul.data_ptr(),
           "Vulkan tensor/tensor arithmetic aliased storage");
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

void test_scalar_add_dispatch_modes_and_lifecycle() {
    auto source = at::tensor({1.0F, -2.0F, 3.5F});
    auto tensor = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(tensor, source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();

    auto tensor_scalar = at::add(tensor, at::Scalar(2.5F));
    auto tensor_scalar_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(tensor_scalar_result, tensor_scalar, false);

    expect(tensor_scalar_result.equal(at::tensor({3.5F, 0.5F, 6.0F})),
           "tensor/scalar Vulkan add changed data");
    expect(platform->compute_dispatch_count() - before == 1,
           "scalar Vulkan add lost a dispatch count");
    expect(platform->pending_transfer_count() == 0,
           "scalar Vulkan add left pending transfer resources");
}

void test_zero_element_scalar_add_does_not_dispatch() {
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto tensor = at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = at::add(tensor, at::Scalar(1.25F));
    expect(output.numel() == 0 && output.sizes() == tensor.sizes(),
           "zero-element scalar Vulkan add returned wrong metadata");
    expect(platform->compute_dispatch_count() == before,
           "zero-element scalar Vulkan add submitted a compute dispatch");
    expect(platform->pending_transfer_count() == 0,
           "zero-element scalar Vulkan add left pending transfer resources");
}

void test_scalar_pointwise_offset_is_rejected() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto base = at::empty({5}, source.options().device(kDevice));
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "scalar offset test tensor is not a contiguous offset view");

    const auto expect_offset_or_unwired = [](const std::function<void()> &operation,
                                             const char *unwired_error) {
        try {
            operation();
        } catch (const c10::Error &error) {
            const std::string message(error.what());
            if (message.find("storage_offset") != std::string::npos) {
                return;
            }
            expect(message.find(unwired_error) != std::string::npos,
                   "scalar offset rejected for an unexpected reason");
            return;
        }
        throw std::runtime_error("scalar offset was accepted");
    };

    expect_offset_or_unwired(
        [&] { (void)at::add(offset, at::Scalar(1.0F)); },
        "Vulkan add does not support a scalar operand");
    expect_offset_or_unwired(
        [&] { (void)at::sub(offset, at::Scalar(1.0F)); },
        "Could not run 'aten::sub.out'");
    expect_offset_or_unwired(
        [&] { (void)at::mul(offset, at::Scalar(1.0F)); },
        "Could not run 'aten::mul.out'");
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

at::Tensor formatter_view(const at::Tensor &tensor) {
    const std::vector<c10::SymInt> shape{c10::SymInt(tensor.numel())};
    return at::_ops::view::call(tensor, shape);
}

void test_formatter_view_aliases_storage_and_keeps_lifetime() {
    auto source = at::tensor({1.0F, -2.5F, 3.25F, 0.0F});
    auto device_tensor = at::empty({2, 2}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source.reshape({2, 2}), false);

    auto view = formatter_view(device_tensor);
    expect(view.storage().data_ptr().get() == device_tensor.storage().data_ptr().get(),
           "Vulkan formatter view did not alias storage");
    expect(view.dim() == 1 && view.sizes().size() == 1 && view.size(0) == 4 &&
               view.numel() == 4,
           "Vulkan formatter view has wrong one-dimensional metadata");
    expect(view.stride(0) == 1 && view.is_contiguous() && view.storage_offset() == 0,
           "Vulkan formatter view has wrong contiguous metadata");
    expect(view.scalar_type() == at::kFloat && view.device() == kDevice,
           "Vulkan formatter view has wrong dtype or device");
    auto before_release = at::empty_like(source);
    pytorch_vulkan::copy_tensor(before_release, view, false);
    expect(before_release.equal(source), "Vulkan formatter view changed values");

    device_tensor = at::Tensor();
    auto after_release = at::empty_like(source);
    pytorch_vulkan::copy_tensor(after_release, view, false);
    expect(after_release.equal(source),
           "Vulkan formatter view did not retain source allocation lifetime");
}

void test_formatter_view_rejects_unsupported_inputs() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto non_contiguous = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(non_contiguous, source, false);
    const std::vector<int64_t> non_contiguous_sizes{2, 2};
    const std::vector<int64_t> non_contiguous_strides{1, 2};
    non_contiguous.unsafeGetTensorImpl()->set_sizes_and_strides(
        non_contiguous_sizes, non_contiguous_strides);
    expect_view_rejected([&] { (void)formatter_view(non_contiguous); }, "contiguous");

    auto offset_base = at::empty({5}, source.options().device(kDevice));
    auto offset = offset_base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> offset_sizes{3};
    const std::vector<int64_t> offset_strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    expect_view_rejected([&] { (void)formatter_view(offset); }, "storage_offset");

    auto wrong_dtype = at::empty({4}, at::TensorOptions().dtype(at::kDouble).device(kDevice));
    expect_view_rejected([&] { (void)formatter_view(wrong_dtype); }, "float32");
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    const std::vector<c10::SymInt> non_formatter_shape{c10::SymInt(2), c10::SymInt(2)};
    expect_view_rejected(
        [&] { (void)at::_ops::view::call(device_tensor, non_formatter_shape); },
        "one-dimensional");
}

void test_abs_out_dispatch_identity_and_rejection() {
    auto source = at::tensor({1.0F, -2.5F, 0.0F, 3.25F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    auto output = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    const auto output_sizes = output.sizes().vec();
    auto &returned = pytorch_vulkan::abs_out(input, output);
    expect(&returned == &output, "Vulkan abs.out changed output identity");
    expect(output.sizes().vec() == output_sizes && output.stride(0) == 1 &&
               output.storage_offset() == 0 && output.is_contiguous(),
           "Vulkan abs.out changed output metadata");
    expect(platform->compute_dispatch_count() - before == 1,
           "Vulkan abs.out lost a dispatch count");
    auto result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(result, output, false);
    expect(result.equal(at::tensor({1.0F, 2.5F, 0.0F, 3.25F})),
           "Vulkan abs.out produced incorrect values");

    auto retained_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(retained_result, output, false);
    expect(retained_result.equal(at::tensor({1.0F, 2.5F, 0.0F, 3.25F})),
           "Vulkan abs.out output became invalid after input release");

    auto zero_input = at::empty({0, 2}, source.options().device(kDevice));
    auto zero_output = at::empty({0, 2}, source.options().device(kDevice));
    const std::size_t zero_before = platform->compute_dispatch_count();
    pytorch_vulkan::abs_out(zero_input, zero_output);
    expect(platform->compute_dispatch_count() == zero_before,
           "zero-element Vulkan abs.out submitted a compute dispatch");

    auto offset_base = at::empty({5}, source.options().device(kDevice));
    auto offset = offset_base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> offset_sizes{4};
    const std::vector<int64_t> offset_strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    expect_error([&] { pytorch_vulkan::abs_out(offset, output); }, "storage_offset");
    auto non_contiguous = at::empty({2, 2}, source.options().device(kDevice));
    const std::vector<int64_t> non_contiguous_sizes{2, 2};
    const std::vector<int64_t> non_contiguous_strides{1, 2};
    non_contiguous.unsafeGetTensorImpl()->set_sizes_and_strides(
        non_contiguous_sizes, non_contiguous_strides);
    expect_error([&] { pytorch_vulkan::abs_out(input, non_contiguous); }, "contiguous");
    auto wrong_size = at::empty({3}, source.options().device(kDevice));
    expect_error([&] { pytorch_vulkan::abs_out(input, wrong_size); }, "matching sizes");
    auto cpu_output = at::empty_like(source);
    expect_error([&] { pytorch_vulkan::abs_out(input, cpu_output); }, "on Vulkan");
    expect(platform->pending_transfer_count() == 0,
           "Vulkan abs.out left pending transfer resources");
}

void test_vk1_construction_is_rejected_at_supported_device_boundary() {
    expect_error(
        [] {
            (void)at::empty(
                {4}, at::TensorOptions().dtype(at::kFloat).device(
                         c10::Device(c10::DeviceType::PrivateUse1, 1)));
        },
        "only device index 0");
}

void test_cpu_view_and_print_inputs_remain_supported() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    const std::vector<c10::SymInt> shape{c10::SymInt(2), c10::SymInt(2)};
    auto view = at::_ops::view::call(source, shape);
    expect(view.device().is_cpu() && view.equal(source.reshape({2, 2})),
           "CPU view behavior regressed");
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
        test_tensor_tensor_sub_and_mul_dispatch();
        test_zero_element_add_does_not_dispatch();
        test_scalar_pointwise_offset_is_rejected();
        test_scalar_add_dispatch_modes_and_lifecycle();
        test_zero_element_scalar_add_does_not_dispatch();
        test_concurrent_add_dispatches_are_serialized();
        test_add_invalid_input_cleans_up();
        test_platform_destruction_is_nothrow();
        bool view_positive_is_unimplemented = false;
        try {
            test_formatter_view_aliases_storage_and_keeps_lifetime();
        } catch (const c10::Error &error) {
            if (!is_missing_view_dispatch(error.what())) {
                throw;
            }
            view_positive_is_unimplemented = true;
            std::cerr << "Expected red state: aten::view is not registered for Vulkan\n";
        }
        test_formatter_view_rejects_unsupported_inputs();
        test_vk1_construction_is_rejected_at_supported_device_boundary();
        test_cpu_view_and_print_inputs_remain_supported();
        test_abs_out_dispatch_identity_and_rejection();
        if (view_positive_is_unimplemented) {
            throw std::runtime_error("formatter view positive contract is unimplemented");
        }
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
