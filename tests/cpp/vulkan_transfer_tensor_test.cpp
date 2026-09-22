#include "vulkan/operators/binary.h"
#include "vulkan/operators/comparison.h"
#include "vulkan/operators/convolution.h"
#include "vulkan/operators/pooling.h"
#include "vulkan/operators/reduction.h"
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace pytorch_vulkan {
at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source);
}

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
    } catch (const std::exception &error) {
        expect(std::string(error.what()).find(text) != std::string::npos,
               "unexpected Vulkan transfer error");
        return;
    }
    throw std::runtime_error("Vulkan transfer accepted invalid input");
}

void test_rnn_limit_validation_uses_supplied_limits() {
    constexpr uint32_t batch = 4;
    constexpr uint32_t sequence = 3;
    constexpr uint32_t input_dimension = 8;
    constexpr uint32_t hidden_dimension = 16;
    const VkDeviceSize input_bytes = batch * sequence * input_dimension * sizeof(float);
    const VkDeviceSize weight_bytes = input_dimension * hidden_dimension * sizeof(float);
    const VkDeviceSize recurrent_bytes = hidden_dimension * hidden_dimension * sizeof(float);
    const VkDeviceSize bias_bytes = hidden_dimension * sizeof(float);
    const VkDeviceSize output_bytes = batch * sequence * hidden_dimension * sizeof(float);
    const VkDeviceSize largest_range = std::max({input_bytes, weight_bytes, recurrent_bytes,
                                                  bias_bytes, output_bytes});
    const VkDeviceSize backward_range = std::max(
        {largest_range, batch * weight_bytes, batch * recurrent_bytes, batch * bias_bytes});

    VulkanCompute::validate_rnn_sequence_limits(
        batch, sequence, input_dimension, hidden_dimension, largest_range, batch);
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_limits(
                batch, sequence, input_dimension, hidden_dimension, output_bytes - 1, batch);
        },
        "maxStorageBufferRange");
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_limits(
                batch, sequence, input_dimension, hidden_dimension, largest_range, batch - 1);
        },
        "batch must be between 1 and 16");

    VulkanCompute::validate_rnn_sequence_backward_limits(
        batch, sequence, input_dimension, hidden_dimension, backward_range, batch);
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_backward_limits(
                1, 1, 512, 256, std::numeric_limits<VkDeviceSize>::max(), 1);
        },
        "maxComputeWorkGroupCount[0]");
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

void test_contiguous_reshape_copy_is_bulk() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_source = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_source, source, false);
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();

    auto result = pytorch_vulkan::vulkan_contiguous_copy(device_source);

    expect(platform->vulkan_copy_count() == 1,
           "contiguous reshape transfer was not counted once");
    expect(platform->copy_command_count() == 1,
           "contiguous reshape did not record one Vulkan copy command");
    auto cpu_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(cpu_result, result, false);
    expect(cpu_result.equal(source), "contiguous reshape bulk copy changed data");
}

void test_execution_counters_reset_and_read_stably() {
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();
    expect(platform->compute_dispatch_count() == 0,
           "reset did not clear compute dispatch count");
    expect(platform->explicit_transfer_count() == 0,
           "reset did not clear explicit transfer count");

    auto source = at::tensor({1.0F, -2.0F, 3.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto transfer_after_first = platform->explicit_transfer_count();
    expect(transfer_after_first == 1,
           "CPU-to-Vulkan copy did not increment explicit transfer count");
    auto first = at::neg(input);
    const auto dispatch_after_first = platform->compute_dispatch_count();
    expect(dispatch_after_first > 0,
           "unary operation did not increment dispatch count");

    auto second = at::neg(input);
    expect(platform->compute_dispatch_count() > dispatch_after_first,
           "second execution did not increment dispatch count");
    expect(platform->explicit_transfer_count() == transfer_after_first,
           "Vulkan-only execution changed explicit transfer count");
    auto result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(result, second, false);
    expect(platform->explicit_transfer_count() == transfer_after_first + 1,
           "Vulkan-to-CPU copy did not increment explicit transfer count");
    (void)first;
}

void test_vulkan_to_vulkan_copy_is_counted_once() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto view = at::as_strided(input, {2}, {1}, 1);
    auto output = at::empty({2}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();
    pytorch_vulkan::copy_tensor(output, view, false);
    expect(platform->vulkan_copy_count() == 1,
           "generic Vulkan-to-Vulkan copy was not counted exactly once");
    expect(platform->explicit_transfer_count() == 0,
           "Vulkan-to-Vulkan copy was counted as an explicit transfer");
    auto result = at::empty({2}, source.options());
    pytorch_vulkan::copy_tensor(result, output, false);
    expect(platform->vulkan_copy_count() == 1,
           "Vulkan-to-CPU presentation changed Vulkan copy count");
}

void test_formatter_presentation_copy_reads_exact_range_and_waits() {
    auto source = at::tensor({10.0F, 20.0F, 30.0F, 40.0F, 50.0F});
    auto device_tensor = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto pending = at::add(device_tensor, at::Scalar(1.0F));
    auto view = at::as_strided(pending, {2}, {1}, 2);
    auto result = at::empty({2}, source.options());
    const auto platform = pytorch_vulkan::platform();

    pytorch_vulkan::formatter_presentation_copy(result, view);

    expect(result.equal(at::tensor({31.0F, 41.0F})),
           "formatter presentation copy read the wrong value range");
    expect(platform->pending_transfer_count() == 0,
           "formatter presentation copy left pending transfer resources");
    auto full_result = at::empty({5}, source.options());
    pytorch_vulkan::copy_tensor(full_result, pending, false);
    expect(full_result.equal(at::tensor({11.0F, 21.0F, 31.0F, 41.0F, 51.0F})),
           "formatter presentation copy invalidated the source buffer");
}

void test_formatter_presentation_copy_preserves_double_and_rejects_general_readback() {
    const auto platform = pytorch_vulkan::platform();
    if (!platform->supports_formatter_double())
        return;

    auto source = at::tensor({1.25F, -2.5F, 7.0F});
    auto input = at::empty({3}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto double_tensor = input.to(at::kDouble);
    auto result = at::empty({2}, at::TensorOptions().dtype(at::kDouble));
    auto view = at::as_strided(double_tensor, {2}, {1}, 1);
    pytorch_vulkan::formatter_presentation_copy(result, view);
    expect(
        result.equal(at::tensor({-2.5, 7.0}, at::TensorOptions().dtype(at::kDouble))),
        "formatter presentation copy changed Double values");
    expect_error(
        [&] {
            auto generic_view_result =
                at::empty({2}, at::TensorOptions().dtype(at::kDouble));
            pytorch_vulkan::copy_tensor(generic_view_result, view, false);
        },
        "readback to CPU is unsupported");
    expect_error(
        [&] {
            auto generic = at::empty({3}, at::TensorOptions().dtype(at::kDouble));
            pytorch_vulkan::copy_tensor(generic, double_tensor, false);
        },
        "readback to CPU is unsupported");
    expect(platform->pending_transfer_count() == 0,
           "Double formatter presentation copy left pending resources");
}

void test_formatter_presentation_copy_rejects_malformed_sources() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto input = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto result = at::empty({2}, source.options());

    auto noncontiguous = at::as_strided(input, {2, 2}, {1, 2});
    auto strided_result = at::empty({4}, source.options());
    pytorch_vulkan::formatter_presentation_copy(strided_result, noncontiguous);
    expect(strided_result.equal(at::ones({4}, source.options())),
           "strided presentation copy changed values");

    auto out_of_range = input;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(3);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{2},
                                                              std::vector<int64_t>{1});
    expect_error(
        [&] { pytorch_vulkan::formatter_presentation_copy(result, out_of_range); },
        "undersized");
    expect(pytorch_vulkan::platform()->pending_transfer_count() == 0,
           "malformed presentation source left pending resources");
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

void test_formatter_double_storage_and_conversion() {
    const auto platform = pytorch_vulkan::platform();
    auto source = at::tensor({1.25F, -2.5F, 0.0F, 7.0F});
    if (!platform->supports_formatter_double()) {
        expect_error(
            [&] {
                (void)at::empty({4},
                                at::TensorOptions().dtype(at::kDouble).device(kDevice));
            },
            "shaderFloat64");
        return;
    }

    auto double_tensor =
        at::empty({4}, at::TensorOptions().dtype(at::kDouble).device(kDevice));
    expect(double_tensor.nbytes() == 4 * sizeof(double),
           "Double Vulkan storage does not use native sizeof(double)");
    auto input = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto converted = input.to(at::kDouble);
    expect(converted.device() == kDevice && converted.scalar_type() == at::kDouble &&
               converted.is_contiguous() && converted.storage_offset() == 0,
           "F32-to-Double conversion did not preserve Vulkan formatter layout");
}

void test_compute_range_rejects_before_descriptor_setup() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().linear(buffer, buffer, buffer, buffer, layout, layout,
                                       layout, layout,
                                       std::numeric_limits<uint32_t>::max(), 1, 1);
        },
        "invalid range");
    expect(platform->compute_dispatch_count() == before,
           "maxStorageBufferRange rejection submitted a compute dispatch");
}

void test_linear_output_count_overflow_rejects_before_dispatch() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().linear(buffer, buffer, buffer, buffer, layout, layout,
                                       layout, layout,
                                       std::numeric_limits<uint32_t>::max(), 1, 2);
        },
        "linear output count overflow");
    expect(platform->compute_dispatch_count() == before,
           "linear output count overflow submitted a compute dispatch");
}

void test_convolution_rejects_undersized_allocation_before_dispatch() {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(kDevice);
    auto input = at::empty({2, 1, 8, 8}, options);
    auto weight = at::empty({4, 1, 3, 3}, options);
    auto bias = at::empty({4}, options);
    input.storage().set_data_ptr(vulkan_allocator_instance()->allocate(sizeof(float)));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            (void)pytorch_vulkan::convolution(input, weight, bias, {1, 1}, {1, 1},
                                              {1, 1}, false, {0, 0}, 1);
        },
        "outside its Vulkan allocation");
    expect(platform->compute_dispatch_count() == before,
           "undersized Conv2d allocation submitted a compute dispatch");
}

void test_pooling_rejects_undersized_allocation_before_dispatch() {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(kDevice);
    auto input = at::empty({2, 4, 3, 5}, options);
    input.storage().set_data_ptr(vulkan_allocator_instance()->allocate(sizeof(float)));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] { (void)pytorch_vulkan::adaptive_avg_pool2d(input, {1, 1}); },
                 "outside its Vulkan allocation");
    expect(platform->compute_dispatch_count() == before,
           "undersized pooling allocation submitted a compute dispatch");
}

void test_pooling_rejects_dimension_product_overflow_before_dispatch() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().pooling(buffer, buffer, layout, layout,
                                        std::numeric_limits<uint32_t>::max(), 2, 2, 2);
        },
        "truncation");
    expect(platform->compute_dispatch_count() == before,
           "overflowing pooling metadata submitted a compute dispatch");
}

void test_nonzero_storage_offset_is_supported() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    auto destination_base = at::empty({5}, source.options().device(kDevice));
    auto destination = destination_base;
    destination.unsafeGetTensorImpl()->set_storage_offset(1);
    destination.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(destination.is_contiguous() && destination.storage_offset() != 0,
           "destination test tensor is not a contiguous offset view");
    pytorch_vulkan::copy_tensor(destination, source, false);
    auto destination_result = at::empty({4}, source.options());
    pytorch_vulkan::copy_tensor(destination_result, destination, false);
    expect(destination_result.equal(source), "offset destination copy changed values");

    auto source_base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(source_base, at::ones({5}, source.options()), false);
    auto source_view = source_base;
    source_view.unsafeGetTensorImpl()->set_storage_offset(1);
    source_view.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    auto result = at::empty({4}, source.options());
    expect(source_view.is_contiguous() && source_view.storage_offset() != 0,
           "source test tensor is not a contiguous offset view");
    pytorch_vulkan::copy_tensor(result, source_view, false);
    expect(result.equal(at::ones({4}, source.options())),
           "formatter presentation copy rejected a valid source range");
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

void test_strided_copy_reads_and_writes_logical_indices() {
    auto source = at::arange(12, at::TensorOptions().dtype(at::kFloat))
                      .reshape({3, 4})
                      .transpose(0, 1)
                      .narrow(1, 1, 2);
    auto destination =
        at::empty_strided({4, 2}, {1, 4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(destination, source, false);
    auto result = at::empty({4, 2}, source.options());
    pytorch_vulkan::copy_tensor(result, destination, false);
    expect(result.equal(source), "strided Vulkan copy changed logical values");
}

void test_strided_copy_rejects_unsafe_overlap() {
    auto source = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto destination =
        at::empty_strided({2, 2}, {0, 1}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] { pytorch_vulkan::copy_tensor(destination, source, false); },
                 "internal overlap");
    expect(platform->compute_dispatch_count() == before,
           "overlapping transfer submitted compute work");

    auto storage = at::empty({5}, source.options().device(kDevice));
    auto source_view = at::as_strided(storage, {3}, {1}, 0);
    auto destination_view = at::as_strided(storage, {3}, {1}, 1);
    expect_error(
        [&] { pytorch_vulkan::copy_tensor(destination_view, source_view, false); },
        "partially overlap");
}

void test_identical_vulkan_copy_is_a_noop() {
    auto tensor = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->pending_transfer_count();
    pytorch_vulkan::copy_tensor(tensor, tensor, false);
    expect(platform->pending_transfer_count() == before,
           "identical Vulkan copy left pending transfer resources");
}

void test_strided_copy_zero_elements_is_noop() {
    auto source =
        at::empty_strided({0, 3}, {3, 1}, at::TensorOptions().dtype(at::kFloat));
    auto destination =
        at::empty_strided({0, 3}, {1, 1}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(destination, source, false);
    expect(destination.numel() == 0, "strided zero-element copy changed size");
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

void test_vulkan_layout_inspection() {
    const std::vector<int64_t> offset_sizes{4};
    const std::vector<int64_t> offset_strides{1};
    auto tensor =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "layout test");
    expect(layout.storage_offset == 0 && layout.numel == 6,
           "contiguous Vulkan layout has wrong element metadata");
    expect(layout.byte_offset == 0 && layout.byte_range == 6 * sizeof(float),
           "contiguous Vulkan layout has wrong byte range");
    expect(layout.allocation_bytes >= layout.byte_range,
           "contiguous Vulkan layout has too-small allocation");

    auto empty = at::empty({0, 3}, tensor.options());
    const auto empty_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(empty, "layout test");
    expect(empty_layout.numel == 0 && empty_layout.byte_range == 0 &&
               empty_layout.allocation_bytes == 0,
           "empty Vulkan layout requires a buffer or has a nonzero byte range");
    auto foreign_empty = empty;
    foreign_empty.storage().set_data_ptr(
        at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(foreign_empty,
                                                               "layout test");
        },
        "invalid allocation payload");

    auto offset_base = at::empty({5}, tensor.options());
    auto offset = offset_base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    const auto offset_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(offset, "layout test");
    expect(offset_layout.byte_offset == sizeof(float) &&
               offset_layout.byte_range == 4 * sizeof(float),
           "offset Vulkan layout has wrong byte range");

    auto foreign = tensor;
    foreign.storage().set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(foreign, "layout test");
        },
        "invalid allocation payload");

    auto out_of_range = offset_base;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(2);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes,
                                                              offset_strides);
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(out_of_range,
                                                               "layout test");
        },
        "outside its Vulkan allocation");

    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(offset_base, {2}, {-1}, 0,
                                                             "layout test");
        },
        "negative stride");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                offset_base, {2}, {std::numeric_limits<int64_t>::max()}, 1,
                "layout test");
        },
        "overflow");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                offset_base, {1}, {0}, std::numeric_limits<int64_t>::max(),
                "layout test");
        },
        "overflow");
}

void test_vulkan_layout_descriptor_and_index_mapping() {
    auto tensor =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto transposed = at::as_strided(tensor, {3, 2}, {1, 3}, 0);
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(transposed, "descriptor test");
    expect(layout.rank == 2 && layout.sizes == std::vector<int64_t>({3, 2}) &&
               layout.strides == std::vector<int64_t>({1, 3}) &&
               layout.element_bytes == sizeof(float) &&
               layout.internal_overlap == pytorch_vulkan::VulkanOverlap::No,
           "Vulkan layout descriptor lost shape, stride, or overlap metadata");
    expect(pytorch_vulkan::vulkan_storage_offset(layout, {0, 0}) == 0 &&
               pytorch_vulkan::vulkan_storage_offset(layout, {2, 1}) == 5 &&
               pytorch_vulkan::vulkan_storage_offset(layout, 5) == 5,
           "Vulkan layout descriptor mapped logical indices incorrectly");

    const auto broadcast = at::as_strided(tensor, {2, 3}, {0, 1}, 0);
    const auto broadcast_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(broadcast, "descriptor test");
    expect(broadcast_layout.internal_overlap == pytorch_vulkan::VulkanOverlap::Yes,
           "zero-stride Vulkan view was not classified as internally overlapping");
    expect(pytorch_vulkan::vulkan_storage_offset(broadcast_layout, 5) == 2,
           "zero-stride Vulkan view mapped its linear index incorrectly");

    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, {3, 0}); },
                 "coordinate");
    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, 6); },
                 "linear index");

    auto overlapping = at::empty_strided({3, 3}, {2, 4}, tensor.options());
    const auto overlapping_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(overlapping, "descriptor test");
    expect(overlapping_layout.internal_overlap != pytorch_vulkan::VulkanOverlap::No,
           "non-unit-stride repeated addresses were classified as non-overlapping");
}

void test_empty_reductions_stay_on_vulkan() {
    auto input =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto sum = pytorch_vulkan::sum_tensor(input, 0, false, c10::nullopt);
    auto mean = pytorch_vulkan::mean_tensor(input, 0, false, c10::nullopt);
    expect(sum.device() == kDevice && mean.device() == kDevice,
           "empty reductions materialized their result on CPU");
    auto sum_cpu = at::empty_like(sum, sum.options().device(c10::kCPU));
    auto mean_cpu = at::empty_like(mean, mean.options().device(c10::kCPU));
    pytorch_vulkan::copy_tensor(sum_cpu, sum, false);
    pytorch_vulkan::copy_tensor(mean_cpu, mean, false);
    expect(sum_cpu.equal(at::zeros_like(sum_cpu)) &&
               at::isnan(mean_cpu).all().item<bool>(),
           "empty Vulkan reductions produced incorrect identity values");
}

void test_empty_reductions_validate_input_layout() {
    auto input = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto foreign = input;
    foreign.storage().set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] { (void)pytorch_vulkan::sum_tensor(foreign, 0, false, c10::nullopt); }, "");

    auto out_of_range = input;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(1);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{0},
                                                              std::vector<int64_t>{1});
    expect_error(
        [&] {
            (void)pytorch_vulkan::mean_tensor(out_of_range, 0, false, c10::nullopt);
        },
        "");
}

void test_vulkan_layout_address_rejects_storage_offset_overflow() {
    const pytorch_vulkan::VulkanTensorLayout layout{
        1,
        {2},
        {1},
        static_cast<int>(at::kBool),
        1,
        std::numeric_limits<int64_t>::max(),
        2,
        static_cast<VkDeviceSize>(std::numeric_limits<int64_t>::max()),
        2,
        std::numeric_limits<VkDeviceSize>::max(),
        pytorch_vulkan::VulkanOverlap::No};
    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, {1}); },
                 "address exceeds int64 range");
}

void test_vulkan_layout_rejects_shader_address_overflow() {
    auto input = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const int64_t max_uint32 = std::numeric_limits<uint32_t>::max();

    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(input, {2}, {max_uint32},
                                                             1, "shader address");
        },
        "shader address");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                input, {2, 2},
                {static_cast<int64_t>(uint32_t{0x80000000}),
                 static_cast<int64_t>(uint32_t{0x80000000})},
                0, "shader address");
        },
        "shader address");
}

void test_empty_vulkan_view_checks_allocation_boundary() {
    auto input = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto valid = pytorch_vulkan::inspect_vulkan_view_layout(input, {0}, {1}, 4,
                                                                  "empty boundary");
    expect(valid.numel == 0 && valid.byte_range == 0 &&
               valid.byte_offset == valid.allocation_bytes,
           "empty Vulkan view at allocation boundary was rejected");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(input, {0}, {1}, 5,
                                                             "empty boundary");
        },
        "empty boundary");
}

void test_empty_vulkan_view_does_not_use_shader_address_limit() {
    auto input = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                input, {0}, {1},
                static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1,
                "empty address");
        },
        "outside its Vulkan allocation");
}

void test_metadata_only_views() {
    auto source = at::tensor({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    auto input = at::empty({2, 3}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source.reshape({2, 3}), false);
    const auto source_sizes = input.sizes().vec();
    const auto source_strides = input.strides().vec();
    const auto platform = pytorch_vulkan::platform();

    const auto expect_view = [&](const at::Tensor &view, const char *message) {
        expect(view.data_ptr() == input.data_ptr() && view.is_contiguous(), message);
        expect(view.sizes().equals({6}) && view.strides().equals({1}),
               "Vulkan metadata-only view has wrong metadata");
        auto result = at::empty({6}, source.options());
        pytorch_vulkan::copy_tensor(result, view, false);
        expect(result.equal(source), "Vulkan metadata-only view changed values");
        expect(input.sizes().equals(source_sizes) &&
                   input.strides().equals(source_strides),
               "Vulkan metadata-only view changed its source metadata");
    };

    const std::size_t before = platform->compute_dispatch_count();
    expect_view(at::as_strided(input, {6}, {1}),
                "Vulkan as_strided did not preserve contiguous storage identity");
    expect_view(input.view({6}),
                "Vulkan view did not preserve contiguous storage identity");
    expect_view(at::_reshape_alias(input, {6}, {1}),
                "Vulkan _reshape_alias did not preserve contiguous storage identity");
    expect(platform->compute_dispatch_count() == before,
           "metadata-only view submitted compute work");

    const auto expect_metadata = [&](const at::Tensor &view, at::IntArrayRef sizes,
                                     at::IntArrayRef strides, int64_t offset) {
        expect(view.storage().data_ptr().get() == input.storage().data_ptr().get() &&
                   view.sizes().equals(sizes) && view.strides().equals(strides) &&
                   view.storage_offset() == offset,
               "Vulkan as_strided did not preserve general metadata");
    };
    expect_metadata(at::as_strided(input, {2}, {2}), {2}, {2}, 0);
    expect_metadata(at::as_strided(input, {2, 3}, {1, 2}), {2, 3}, {1, 2}, 0);
    expect_metadata(at::as_strided(input, {2, 3}, {0, 1}), {2, 3}, {0, 1}, 0);
    expect_metadata(at::as_strided(input, {2}, {1}, 1), {2}, {1}, 1);

    const auto expect_rejected = [&](const std::function<void()> &operation,
                                     const char *message) {
        const std::size_t dispatches = platform->compute_dispatch_count();
        (void)message;
        expect_error(operation, "");
        expect(platform->compute_dispatch_count() == dispatches,
               "invalid metadata-only view submitted compute work");
    };
    expect_rejected([&] { (void)at::as_strided(input, {6}, {1}, 1); },
                    "storage offset");
    expect_rejected([&] { (void)input.view({5}); }, "invalid for input");
    expect_rejected([&] { (void)at::as_strided(input, {2, 3}, {1, 3}); },
                    "Vulkan as_strided");
    expect_rejected([&] { (void)at::as_strided(input, {2, 3}, {-1, 1}); },
                    "Vulkan as_strided");
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
    auto tensor =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = at::add(tensor, at::Scalar(1.25F));
    expect(output.numel() == 0 && output.sizes() == tensor.sizes(),
           "zero-element scalar Vulkan add returned wrong metadata");
    expect(platform->compute_dispatch_count() == before,
           "zero-element scalar Vulkan add submitted a compute dispatch");
    expect(platform->pending_transfer_count() == 0,
           "zero-element scalar Vulkan add left pending transfer resources");
}

void test_formatter_comparison_dispatch_is_counted() {
    auto source = at::tensor({-1.0F, 0.0F, 1.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto result = at::empty_like(input, input.options().dtype(at::kBool));
    pytorch_vulkan::isfinite_out(input, result);
    expect(result.sizes() == input.sizes(),
           "formatter comparison returned wrong metadata");
    expect(platform->compute_dispatch_count() - before == 1,
           "formatter comparison did not submit exactly one dispatch");
}

void test_scalar_pointwise_offset_is_supported() {
    auto source = at::tensor({2.0F, 3.0F, 4.0F, 5.0F});
    auto base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(base, at::tensor({0.0F, 2.0F, 3.0F, 4.0F, 5.0F}),
                                false);
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "scalar offset test tensor is not a contiguous offset view");

    for (const auto &[result, expected] :
         std::vector<std::pair<at::Tensor, at::Tensor>>{
             {at::add(offset, at::Scalar(1.0F)), at::tensor({3.0F, 4.0F, 5.0F, 6.0F})},
             {at::sub(offset, at::Scalar(1.0F)), at::tensor({1.0F, 2.0F, 3.0F, 4.0F})},
             {at::mul(offset, at::Scalar(2.0F)),
              at::tensor({4.0F, 6.0F, 8.0F, 10.0F})}}) {
        auto cpu_result = at::empty_like(expected);
        pytorch_vulkan::copy_tensor(cpu_result, result, false);
        expect(cpu_result.equal(expected), "scalar offset pointwise changed values");
    }
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
    const std::vector<int64_t> rank_nine(9, 1);
    auto lhs =
        at::empty(rank_nine, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto rhs = at::empty_like(lhs);
    expect_error([&] { (void)pytorch_vulkan::add_tensor(lhs, rhs, 1.0F); },
                 "ranks up to 8");
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

void test_unary_nonzero_storage_offset_is_supported() {
    auto source = at::tensor({2.0F, 3.0F, 4.0F, 5.0F});
    auto base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(base, at::tensor({0.0F, 2.0F, 3.0F, 4.0F, 5.0F}),
                                false);
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "unary offset test tensor is not a contiguous offset view");

    auto result = at::neg(offset);
    auto cpu_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(cpu_result, result, false);
    expect(cpu_result.equal(at::tensor({-2.0F, -3.0F, -4.0F, -5.0F})),
           "unary offset dispatch changed values");
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
    auto input =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto out = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    pytorch_vulkan::dispatch_unary_out(input, out,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");
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
    pytorch_vulkan::dispatch_unary_out(input, offset,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");

    auto noncontiguous =
        at::empty_strided({3, 2}, {1, 3}, source.options().device(kDevice));
    auto matrix_input = at::empty({3, 2}, source.options().device(kDevice));
    pytorch_vulkan::dispatch_unary_out(matrix_input, noncontiguous,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");

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
    partial_input.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{3},
                                                               std::vector<int64_t>{1});
    auto rhs = at::empty_like(values, values.options().device(kDevice));
    auto partial_output = storage;
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            pytorch_vulkan::dispatch_tensor_tensor_out(
                partial_input, rhs, at::Scalar(1.0F), partial_output,
                pytorch_vulkan::PointwiseOperation::Add, "add");
        },
        "partially overlaps");
    expect(platform->compute_dispatch_count() == before,
           "partial overlap rejection submitted a compute dispatch");

    auto internal_output =
        at::empty_strided({3}, {0}, values.options().device(kDevice));
    expect_error(
        [&] {
            pytorch_vulkan::dispatch_tensor_tensor_out(
                input, rhs, at::Scalar(1.0F), internal_output,
                pytorch_vulkan::PointwiseOperation::Add, "add");
        },
        "internal overlap");
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

    pytorch_vulkan::dispatch_tensor_tensor_out(lhs, rhs, at::Scalar(1.0F), lhs,
                                               pytorch_vulkan::PointwiseOperation::Add,
                                               "add");
    auto lhs_result = at::empty_like(lhs_source);
    auto rhs_after_lhs = at::empty_like(rhs_source);
    pytorch_vulkan::copy_tensor(lhs_result, lhs, false);
    pytorch_vulkan::copy_tensor(rhs_after_lhs, rhs, false);
    expect(lhs_result.equal(at::tensor({11.0F, 22.0F, 33.0F})) &&
               rhs_after_lhs.equal(rhs_source),
           "binary out=lhs alias produced wrong values or changed rhs");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_tensor_out(lhs, rhs, at::Scalar(1.0F), rhs,
                                               pytorch_vulkan::PointwiseOperation::Sub,
                                               "sub");
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
        pytorch_vulkan::inspect_vulkan_tensor_layout(lhs, "direct unary input"),
        pytorch_vulkan::allocation_buffer(lhs.storage().data_ptr()).buffer(),
        pytorch_vulkan::inspect_vulkan_tensor_layout(lhs, "direct unary output"),
        static_cast<uint32_t>(pytorch_vulkan::PointwiseOperation::Neg));
    auto direct_alias_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(direct_alias_result, lhs, false);
    expect(direct_alias_result.equal(at::tensor({-1.0F, -2.0F, -3.0F})),
           "direct unary exact alias dispatch produced wrong values");
}

void test_shared_out_device_index_is_rejected_by_native_allocator() {
    expect_error(
        [&] {
            (void)at::empty({3},
                            at::TensorOptions()
                                .dtype(at::kFloat)
                                .device(c10::Device(c10::DeviceType::PrivateUse1, 1)));
        },
        "device index 0");
}

void test_platform_destruction_is_nothrow() {
    static_assert(std::is_nothrow_destructible<VulkanPlatform>::value,
                  "Vulkan cleanup must not throw during ownership quarantine");
}

void test_reduction_indexing_reject_malformed_metadata() {
    auto input =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = at::empty({2, 1}, input.options());
    const auto input_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(input, "test input");
    const auto output_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(output, "test output");
    auto input_buffer =
        pytorch_vulkan::allocation_buffer(input.storage().data_ptr()).buffer();
    auto output_buffer =
        pytorch_vulkan::allocation_buffer(output.storage().data_ptr()).buffer();
    auto platform = pytorch_vulkan::platform();

    auto missing_sizes = input_layout;
    missing_sizes.sizes.pop_back();
    expect_error(
        [&] {
            platform->compute().reduction(input_buffer, missing_sizes, output_buffer,
                                          output_layout, 2U, 3U, 2U, false);
        },
        "metadata lengths");

    auto rank_nine = input_layout;
    rank_nine.rank = 9;
    rank_nine.sizes.resize(9, 1);
    rank_nine.strides.resize(9, 1);
    expect_error(
        [&] {
            platform->compute().argmax(input_buffer, rank_nine, output_buffer,
                                       output_layout, 0U, 1U, 1U);
        },
        "ranks up to 8");

    auto missing_output = output_layout;
    missing_output.strides.pop_back();
    expect_error(
        [&] {
            platform->compute().broadcast(input_buffer, input_layout, output_buffer,
                                          missing_output, 2U, 1.0F);
        },
        "output metadata lengths");
}

} // namespace

int main() {
    try {
        (void)pytorch_vulkan::platform();
        test_copy_round_trip();
        test_copy_returns_without_pending_transfer_resources();
        test_contiguous_reshape_copy_is_bulk();
        test_execution_counters_reset_and_read_stably();
        test_vulkan_to_vulkan_copy_is_counted_once();
        test_formatter_presentation_copy_reads_exact_range_and_waits();
        test_formatter_presentation_copy_preserves_double_and_rejects_general_readback();
        test_formatter_presentation_copy_rejects_malformed_sources();
        test_validation_boundaries();
        test_formatter_double_storage_and_conversion();
        test_compute_range_rejects_before_descriptor_setup();
        test_linear_output_count_overflow_rejects_before_dispatch();
        test_convolution_rejects_undersized_allocation_before_dispatch();
        test_pooling_rejects_undersized_allocation_before_dispatch();
        test_pooling_rejects_dimension_product_overflow_before_dispatch();
        test_nonzero_storage_offset_is_supported();
        test_zero_tensor_copy_is_noop();
        test_strided_copy_reads_and_writes_logical_indices();
        test_strided_copy_rejects_unsafe_overlap();
        test_identical_vulkan_copy_is_a_noop();
        test_strided_copy_zero_elements_is_noop();
        test_zero_allocator_payload();
        test_foreign_payload_rejected();
        test_vulkan_layout_inspection();
        test_vulkan_layout_descriptor_and_index_mapping();
        test_empty_reductions_stay_on_vulkan();
        test_empty_reductions_validate_input_layout();
        test_vulkan_layout_address_rejects_storage_offset_overflow();
        test_vulkan_layout_rejects_shader_address_overflow();
        test_empty_vulkan_view_checks_allocation_boundary();
        test_empty_vulkan_view_does_not_use_shader_address_limit();
        test_metadata_only_views();
        test_repeated_add_dispatch_and_retained_output();
        test_tensor_tensor_sub_and_mul_dispatch();
        test_zero_element_add_does_not_dispatch();
        test_scalar_pointwise_offset_is_supported();
        test_scalar_add_dispatch_modes_and_lifecycle();
        test_zero_element_scalar_add_does_not_dispatch();
        test_formatter_comparison_dispatch_is_counted();
        test_concurrent_add_dispatches_are_serialized();
        test_add_invalid_input_cleans_up();
        test_repeated_unary_dispatch_and_input_readability();
        test_unary_nonzero_storage_offset_is_supported();
        test_shared_out_dispatch_helpers();
        test_shared_out_empty_path_does_not_dispatch();
        test_shared_out_rejects_offset_and_noncontiguous_output();
        test_shared_out_rejects_partial_and_internal_overlap();
        test_inplace_rejections_preserve_inputs_and_dispatch_count();
        test_shared_out_exact_aliases_preserve_values_and_lifecycles();
        test_shared_out_device_index_is_rejected_by_native_allocator();
        test_platform_destruction_is_nothrow();
        test_reduction_indexing_reject_malformed_metadata();
        test_rnn_limit_validation_uses_supplied_limits();
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
