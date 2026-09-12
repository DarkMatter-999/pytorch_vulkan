#include "vulkan_allocator.h"
#include "vulkan_compute.h"
#include "vulkan_buffer.h"
#include "vulkan/operators/binary.h"
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

void test_repeated_unary_dispatch_and_input_readability() {
    auto source = at::tensor({-3.5F, 0.0F, 2.25F, -1.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto expected = at::neg(source);

    for (int iteration = 0; iteration < 16; ++iteration) {
        auto output = at::neg(input);
        expect(output.data_ptr() != input.data_ptr(),
               "Vulkan unary output aliased its input");
        auto result = at::empty_like(source);
        pytorch_vulkan::copy_tensor(result, output, false);
        expect(result.equal(expected), "repeated Vulkan unary changed data");
    }

    auto input_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(input_result, input, false);
    expect(input_result.equal(source), "Vulkan unary changed its input");
}

void test_unary_nonzero_storage_offset_is_rejected() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto base = at::empty({5}, source.options().device(kDevice));
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "unary offset test tensor is not a contiguous offset view");

    expect_error([&] { (void)at::neg(offset); }, "storage_offset");
}

void test_shared_out_dispatch_helpers() {
    auto source = at::tensor({-3.0F, 2.0F, 4.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);

    auto unary_out = at::empty({1}, source.options().device(kDevice));
    auto *unary_impl = unary_out.unsafeGetTensorImpl();
    auto &unary_result = pytorch_vulkan::dispatch_unary_out(
        input, unary_out, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(&unary_result == &unary_out && unary_out.unsafeGetTensorImpl() == unary_impl,
           "unary out did not preserve tensor identity");
    auto unary_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(unary_cpu, unary_out, false);
    expect(unary_cpu.equal(at::tensor({3.0F, -2.0F, -4.0F})),
           "unary out helper changed values");

    auto scalar_out = at::empty({1}, source.options().device(kDevice));
    auto &scalar_result = pytorch_vulkan::dispatch_tensor_scalar_out(
        input, at::Scalar(1.5F), at::Scalar(1.0F), scalar_out,
        pytorch_vulkan::PointwiseOperation::Add, "add");
    expect(&scalar_result == &scalar_out && scalar_out.sizes() == source.sizes(),
           "tensor scalar out helper did not resize in place");
    auto scalar_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(scalar_cpu, scalar_out, false);
    expect(scalar_cpu.equal(at::tensor({-1.5F, 3.5F, 5.5F})),
           "tensor scalar out helper changed values");

    auto rhs = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(rhs, at::tensor({1.0F, 3.0F, 5.0F}), false);
    auto binary_out = at::empty({1}, source.options().device(kDevice));
    auto &binary_result = pytorch_vulkan::dispatch_tensor_tensor_out(
        input, rhs, at::Scalar(1.0F), binary_out,
        pytorch_vulkan::PointwiseOperation::Sub, "sub");
    expect(&binary_result == &binary_out && binary_out.sizes() == source.sizes(),
           "tensor tensor out helper did not preserve identity or resize");
    auto binary_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(binary_cpu, binary_out, false);
    expect(binary_cpu.equal(at::tensor({-4.0F, -1.0F, -1.0F})),
           "tensor tensor out helper changed values");
}

void test_shared_out_empty_path_does_not_dispatch() {
    auto input = at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto out = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    pytorch_vulkan::dispatch_unary_out(
        input, out, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(out.sizes() == input.sizes() && platform->compute_dispatch_count() == before,
           "empty unary out path dispatched or kept the wrong shape");
}

void test_shared_out_rejects_offset_and_noncontiguous_output() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);

    auto offset = at::empty({4}, source.options().device(kDevice));
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> offset_sizes{3};
    const std::vector<int64_t> offset_strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    expect_error([&] {
        pytorch_vulkan::dispatch_unary_out(
            input, offset, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    }, "storage_offset");

    auto partial = offset;
    expect_error([&] {
        pytorch_vulkan::dispatch_tensor_tensor_out(
            input, input, at::Scalar(1.0F), partial,
            pytorch_vulkan::PointwiseOperation::Add, "add");
    }, "storage_offset");

    auto noncontiguous = at::empty_strided({3, 2}, {1, 3},
                                           source.options().device(kDevice));
    auto matrix_input = at::empty({3, 2}, source.options().device(kDevice));
    expect_error([&] {
        pytorch_vulkan::dispatch_unary_out(
            matrix_input, noncontiguous,
            pytorch_vulkan::PointwiseOperation::Neg, "neg");
    }, "contiguous");

    auto alias = input;
    auto &alias_result = pytorch_vulkan::dispatch_unary_out(
        input, alias, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(&alias_result == &alias &&
               alias.unsafeGetTensorImpl() == input.unsafeGetTensorImpl(),
           "exact full-tensor output alias was not permitted");
}

void test_shared_out_rejects_partial_and_internal_overlap() {
    auto values = at::tensor({1.0F, 2.0F, 3.0F});
    auto input = at::empty_like(values, values.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, values, false);

    auto storage = at::empty({5}, values.options().device(kDevice));
    auto partial_input = storage.detach();
    partial_input.unsafeGetTensorImpl()->set_sizes_and_strides(
        std::vector<int64_t>{3}, std::vector<int64_t>{1});
    auto rhs = at::empty_like(values, values.options().device(kDevice));
    auto partial_output = storage;
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] {
        pytorch_vulkan::dispatch_tensor_tensor_out(
            partial_input, rhs, at::Scalar(1.0F), partial_output,
            pytorch_vulkan::PointwiseOperation::Add, "add");
    }, "partially overlaps");
    expect(platform->compute_dispatch_count() == before,
           "partial overlap rejection submitted a compute dispatch");

    auto internal_output = at::empty_strided({3}, {0},
                                             values.options().device(kDevice));
    expect_error([&] {
        pytorch_vulkan::dispatch_tensor_tensor_out(
            input, rhs, at::Scalar(1.0F), internal_output,
            pytorch_vulkan::PointwiseOperation::Add, "add");
    }, "internal overlap");
    expect(platform->compute_dispatch_count() == before,
           "internal overlap rejection submitted a compute dispatch");
}

void test_inplace_rejections_preserve_inputs_and_dispatch_count() {
    auto values = at::tensor({1.0F, -2.0F, 3.0F});
    auto input = at::empty_like(values, values.options().device(kDevice));
    auto other = at::empty_like(values, values.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, values, false);
    pytorch_vulkan::copy_tensor(other, values, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] { input.add_(other); }, "in-place");
    expect(platform->compute_dispatch_count() == before,
           "in-place add rejection submitted a compute dispatch");
    auto result = at::empty_like(values);
    pytorch_vulkan::copy_tensor(result, input, false);
    expect(result.equal(values), "in-place add rejection mutated its input");

    const auto expect_unchanged = [&](const std::function<void()> &operation,
                                      const char *message) {
        pytorch_vulkan::copy_tensor(input, values, false);
        const std::size_t dispatches = platform->compute_dispatch_count();
        expect_error(operation, "in-place");
        expect(platform->compute_dispatch_count() == dispatches, message);
        pytorch_vulkan::copy_tensor(result, input, false);
        expect(result.equal(values), "in-place rejection mutated its input");
    };
    expect_unchanged([&] { input.add_(1.0F); },
                     "in-place scalar add rejection submitted a compute dispatch");
    expect_unchanged([&] { input.sub_(other); },
                     "in-place sub rejection submitted a compute dispatch");
    expect_unchanged([&] { input.sub_(1.0F); },
                     "in-place scalar sub rejection submitted a compute dispatch");
    expect_unchanged([&] { input.mul_(other); },
                     "in-place mul rejection submitted a compute dispatch");
    expect_unchanged([&] { input.mul_(1.0F); },
                     "in-place scalar mul rejection submitted a compute dispatch");
    expect_unchanged([&] { input.neg_(); },
                     "in-place neg rejection submitted a compute dispatch");
    expect_unchanged([&] { input.abs_(); },
                     "in-place abs rejection submitted a compute dispatch");
    expect_unchanged([&] { input.relu_(); },
                     "in-place relu rejection submitted a compute dispatch");
}

void test_shared_out_exact_aliases_preserve_values_and_lifecycles() {
    auto lhs_source = at::tensor({1.0F, 2.0F, 3.0F});
    auto rhs_source = at::tensor({10.0F, 20.0F, 30.0F});
    auto lhs = at::empty_like(lhs_source, lhs_source.options().device(kDevice));
    auto rhs = at::empty_like(rhs_source, rhs_source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::copy_tensor(rhs, rhs_source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();

    pytorch_vulkan::dispatch_tensor_tensor_out(
        lhs, rhs, at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Add, "add");
    auto lhs_result = at::empty_like(lhs_source);
    auto rhs_after_lhs = at::empty_like(rhs_source);
    pytorch_vulkan::copy_tensor(lhs_result, lhs, false);
    pytorch_vulkan::copy_tensor(rhs_after_lhs, rhs, false);
    expect(lhs_result.equal(at::tensor({11.0F, 22.0F, 33.0F})) &&
               rhs_after_lhs.equal(rhs_source),
           "binary out=lhs alias produced wrong values or changed rhs");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_tensor_out(
        lhs, rhs, at::Scalar(1.0F), rhs,
        pytorch_vulkan::PointwiseOperation::Sub, "sub");
    auto rhs_result = at::empty_like(rhs_source);
    auto lhs_after_rhs = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(rhs_result, rhs, false);
    pytorch_vulkan::copy_tensor(lhs_after_rhs, lhs, false);
    expect(rhs_result.equal(at::tensor({-9.0F, -18.0F, -27.0F})) &&
               lhs_after_rhs.equal(lhs_source),
           "binary out=rhs alias produced wrong values or changed lhs");

    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Mul, "mul");
    auto scalar_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({2.0F, 4.0F, 6.0F})),
           "tensor/scalar exact alias produced wrong values");
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Add, "add", true);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({3.0F, 4.0F, 5.0F})),
           "scalar-left add exact alias produced wrong values");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Sub, "rsub", true);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({1.0F, 0.0F, -1.0F})),
           "scalar-left rsub exact alias produced wrong values");
    expect(platform->compute_dispatch_count() - before == 5,
           "exact alias dispatches were not counted");
    expect(platform->pending_transfer_count() == 0,
           "exact alias dispatch left pending transfer resources");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    platform->compute().unary_alias(
        pytorch_vulkan::allocation_buffer(lhs.storage().data_ptr()).buffer(),
        pytorch_vulkan::allocation_buffer(lhs.storage().data_ptr()).buffer(),
        static_cast<VkDeviceSize>(lhs.numel() * sizeof(float)),
        static_cast<uint32_t>(pytorch_vulkan::PointwiseOperation::Neg));
    auto direct_alias_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(direct_alias_result, lhs, false);
    expect(direct_alias_result.equal(at::tensor({-1.0F, -2.0F, -3.0F})),
           "direct unary exact alias dispatch produced wrong values");
}

void test_shared_out_device_index_is_rejected_by_native_allocator() {
    expect_error([&] {
        (void)at::empty({3}, at::TensorOptions().dtype(at::kFloat).device(
            c10::Device(c10::DeviceType::PrivateUse1, 1)));
    }, "device index 0");
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
        test_tensor_tensor_sub_and_mul_dispatch();
        test_zero_element_add_does_not_dispatch();
        test_scalar_pointwise_offset_is_rejected();
        test_scalar_add_dispatch_modes_and_lifecycle();
        test_zero_element_scalar_add_does_not_dispatch();
        test_concurrent_add_dispatches_are_serialized();
        test_add_invalid_input_cleans_up();
        test_repeated_unary_dispatch_and_input_readability();
        test_unary_nonzero_storage_offset_is_rejected();
        test_shared_out_dispatch_helpers();
        test_shared_out_empty_path_does_not_dispatch();
        test_shared_out_rejects_offset_and_noncontiguous_output();
        test_shared_out_rejects_partial_and_internal_overlap();
        test_inplace_rejections_preserve_inputs_and_dispatch_count();
        test_shared_out_exact_aliases_preserve_values_and_lifecycles();
        test_shared_out_device_index_is_rejected_by_native_allocator();
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
