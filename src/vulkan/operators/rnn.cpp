#include "rnn.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_execution.h"

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <array>
#include <limits>
#include <memory>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>

namespace pytorch_vulkan {
namespace {
void validate_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device().index() == 0,
                "Vulkan RNN ", name, " requires PrivateUse1 device index 0");
    TORCH_CHECK(tensor.layout() == at::kStrided && tensor.scalar_type() == at::kFloat,
                "Vulkan RNN ", name, " requires a strided float32 tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Vulkan RNN ", name,
                " requires a contiguous tensor");
}

uint32_t dimension(int64_t value, const char *name) {
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<uint32_t>::max(),
                "Vulkan RNN ", name, " is outside the supported positive uint32 range");
    return static_cast<uint32_t>(value);
}

VkDeviceSize bytes(const at::Tensor &tensor, const char *name) {
    const uint64_t elements = static_cast<uint64_t>(tensor.numel());
    TORCH_CHECK(elements <= std::numeric_limits<uint64_t>::max() / sizeof(float),
                "Vulkan RNN ", name, " byte count overflows");
    const uint64_t result = elements * sizeof(float);
    TORCH_CHECK(result <= std::numeric_limits<VkDeviceSize>::max(),
                "Vulkan RNN ", name, " byte count exceeds Vulkan range");
    return static_cast<VkDeviceSize>(result);
}
} // namespace

at::Tensor rnn_sequence_forward(const at::Tensor &input, const at::Tensor &weight,
                                const at::Tensor &recurrent_weight, const at::Tensor &bias) {
    validate_tensor(input, "input");
    validate_tensor(weight, "weight");
    validate_tensor(recurrent_weight, "recurrent_weight");
    validate_tensor(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "Vulkan RNN input requires rank 3 [batch, sequence, input]");
    TORCH_CHECK(weight.dim() == 2 && recurrent_weight.dim() == 2 && bias.dim() == 1,
                "Vulkan RNN weights must be rank 2 and bias rank 1");
    TORCH_CHECK(input.device() == weight.device() && input.device() == recurrent_weight.device() &&
                    input.device() == bias.device(),
                "Vulkan RNN tensors must share one device");

    const uint32_t batch = dimension(input.size(0), "batch");
    const uint32_t sequence = dimension(input.size(1), "sequence");
    const uint32_t input_dimension = dimension(input.size(2), "input dimension");
    const uint32_t hidden_dimension = dimension(weight.size(0), "hidden dimension");
    TORCH_CHECK(hidden_dimension <= 256, "Vulkan RNN hidden dimension exceeds workgroup size");
    TORCH_CHECK(weight.size(1) == input.size(2),
                "Vulkan RNN input weight dimensions do not match input");
    TORCH_CHECK(recurrent_weight.sizes().equals({weight.size(0), weight.size(0)}),
                "Vulkan RNN recurrent weight must be [hidden, hidden]");
    TORCH_CHECK(bias.size(0) == weight.size(0),
                "Vulkan RNN bias must have hidden dimension elements");

    const auto input_layout = inspect_vulkan_tensor_layout(input, "RNN input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "RNN weight");
    const auto recurrent_layout =
        inspect_vulkan_tensor_layout(recurrent_weight, "RNN recurrent weight");
    const auto bias_layout = inspect_vulkan_tensor_layout(bias, "RNN bias");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No &&
                    weight_layout.internal_overlap == VulkanOverlap::No &&
                    recurrent_layout.internal_overlap == VulkanOverlap::No &&
                    bias_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan RNN rejects overlapping input layouts");

    const auto &input_data = input.storage().data_ptr();
    const auto &weight_data = weight.storage().data_ptr();
    const auto &recurrent_data = recurrent_weight.storage().data_ptr();
    const auto &bias_data = bias.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    TORCH_CHECK(&platform == &allocation_platform(weight_data) &&
                    &platform == &allocation_platform(recurrent_data) &&
                    &platform == &allocation_platform(bias_data),
                "Vulkan RNN requires Vulkan allocations");
    validate_allocation(input_data, bytes(input, "input"), "RNN input");
    validate_allocation(weight_data, bytes(weight, "weight"), "RNN weight");
    validate_allocation(recurrent_data, bytes(recurrent_weight, "recurrent weight"),
                        "RNN recurrent weight");
    validate_allocation(bias_data, bytes(bias, "bias"), "RNN bias");

    // All rejection checks precede output allocation so malformed raw calls have
    // no Vulkan allocation or dispatch side effects.
    platform.compute().validate_rnn_sequence(batch, sequence, input_dimension,
                                              hidden_dimension);
    at::Tensor output = at::empty({input.size(0), input.size(1), weight.size(0)},
                                  input.options());
    const auto output_layout = inspect_vulkan_tensor_layout(output, "RNN output");
    const auto &output_data = output.storage().data_ptr();
    TORCH_CHECK(&platform == &allocation_platform(output_data),
                "Vulkan RNN output requires the input Vulkan platform");
    validate_allocation(output_data, bytes(output, "output"), "RNN output");
    platform.compute().rnn_sequence(
        allocation_buffer(input_data).buffer(), allocation_buffer(weight_data).buffer(),
        allocation_buffer(recurrent_data).buffer(), allocation_buffer(bias_data).buffer(),
        allocation_buffer(output_data).buffer(), batch, sequence, input_dimension,
        hidden_dimension);
    (void)input_layout;
    (void)weight_layout;
    (void)recurrent_layout;
    (void)bias_layout;
    (void)output_layout;
    return output;
}

std::array<at::Tensor, 4> rnn_sequence_backward_impl(
    const at::Tensor &gradient_output, const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &recurrent_weight, const at::Tensor &bias, const at::Tensor &saved_output,
    bool need_input, bool need_weight, bool need_recurrent_weight, bool need_bias) {
    validate_tensor(gradient_output, "gradient output");
    validate_tensor(saved_output, "saved output");
    validate_tensor(input, "input");
    validate_tensor(weight, "weight");
    validate_tensor(recurrent_weight, "recurrent_weight");
    validate_tensor(bias, "bias");
    TORCH_CHECK(input.dim() == 3 && gradient_output.dim() == 3 && saved_output.dim() == 3,
                "Vulkan RNN backward requires rank 3 tensors");
    TORCH_CHECK(weight.dim() == 2 && recurrent_weight.dim() == 2 && bias.dim() == 1,
                "Vulkan RNN backward weights must be rank 2 and bias rank 1");
    TORCH_CHECK(input.device() == weight.device() && input.device() == recurrent_weight.device() &&
                    input.device() == bias.device() && input.device() == gradient_output.device() &&
                    input.device() == saved_output.device(),
                "Vulkan RNN backward tensors must share one device");
    TORCH_CHECK(gradient_output.sizes().equals(saved_output.sizes()) &&
                    saved_output.sizes().equals({input.size(0), input.size(1), weight.size(0)}),
                "Vulkan RNN backward gradient shape does not match the saved output");
    TORCH_CHECK(weight.size(1) == input.size(2) &&
                    recurrent_weight.sizes().equals({weight.size(0), weight.size(0)}) &&
                    bias.size(0) == weight.size(0),
                "Vulkan RNN backward parameter dimensions do not match input");
    TORCH_CHECK(weight.size(0) <= 256, "Vulkan RNN hidden dimension exceeds workgroup size");

    const uint32_t batch = dimension(input.size(0), "batch");
    const uint32_t sequence = dimension(input.size(1), "sequence");
    const uint32_t input_dimension = dimension(input.size(2), "input dimension");
    const uint32_t hidden_dimension = dimension(weight.size(0), "hidden dimension");
    const auto &input_data = input.storage().data_ptr();
    const auto &platform = allocation_platform(input_data);
    const at::Tensor outputs[] = {gradient_output, saved_output};
    for (const auto &tensor : outputs) {
        TORCH_CHECK(&platform == &allocation_platform(tensor.storage().data_ptr()),
                    "Vulkan RNN backward requires one Vulkan platform");
    }
    const at::Tensor parameters[] = {weight, recurrent_weight, bias};
    for (const auto &tensor : parameters) {
        TORCH_CHECK(&platform == &allocation_platform(tensor.storage().data_ptr()),
                    "Vulkan RNN backward requires one Vulkan platform");
    }
    const auto input_layout = inspect_vulkan_tensor_layout(input, "RNN backward input");
    const auto weight_layout = inspect_vulkan_tensor_layout(weight, "RNN backward weight");
    const auto recurrent_layout =
        inspect_vulkan_tensor_layout(recurrent_weight, "RNN backward recurrent weight");
    const auto bias_layout = inspect_vulkan_tensor_layout(bias, "RNN backward bias");
    const auto grad_output_layout =
        inspect_vulkan_tensor_layout(gradient_output, "RNN backward gradient output");
    const auto saved_layout = inspect_vulkan_tensor_layout(saved_output, "RNN backward saved output");
    TORCH_CHECK(input_layout.internal_overlap == VulkanOverlap::No &&
                    weight_layout.internal_overlap == VulkanOverlap::No &&
                    recurrent_layout.internal_overlap == VulkanOverlap::No &&
                    bias_layout.internal_overlap == VulkanOverlap::No &&
                    grad_output_layout.internal_overlap == VulkanOverlap::No &&
                    saved_layout.internal_overlap == VulkanOverlap::No,
                "Vulkan RNN backward rejects overlapping layouts");
    validate_allocation(input_data, bytes(input, "backward input"), "RNN backward input");
    validate_allocation(weight.storage().data_ptr(), bytes(weight, "backward weight"),
                        "RNN backward weight");
    validate_allocation(recurrent_weight.storage().data_ptr(),
                        bytes(recurrent_weight, "backward recurrent weight"),
                        "RNN backward recurrent weight");
    validate_allocation(bias.storage().data_ptr(), bytes(bias, "backward bias"),
                        "RNN backward bias");
    validate_allocation(gradient_output.storage().data_ptr(),
                        bytes(gradient_output, "backward gradient output"),
                        "RNN backward gradient output");
    validate_allocation(saved_output.storage().data_ptr(),
                        bytes(saved_output, "backward saved output"),
                        "RNN backward saved output");
    platform.compute().validate_rnn_sequence_backward(batch, sequence, input_dimension,
                                                      hidden_dimension);

    TORCH_CHECK(need_input || need_weight || need_recurrent_weight || need_bias,
                "Vulkan RNN backward received no requested gradients");

    at::Tensor gradient_input = need_input ? at::empty_like(input) : at::Tensor();
    at::Tensor gradient_weight = need_weight ? at::empty_like(weight) : at::Tensor();
    at::Tensor gradient_recurrent = need_recurrent_weight ? at::empty_like(recurrent_weight)
                                                           : at::Tensor();
    at::Tensor gradient_bias = need_bias ? at::empty_like(bias) : at::Tensor();

    const VkDeviceSize partial_weight_bytes = static_cast<VkDeviceSize>(batch) *
                                              bytes(weight, "partial weight");
    const VkDeviceSize partial_recurrent_bytes = static_cast<VkDeviceSize>(batch) *
                                                 bytes(recurrent_weight, "partial recurrent weight");
    const VkDeviceSize partial_bias_bytes = static_cast<VkDeviceSize>(batch) *
                                            bytes(bias, "partial bias");
    auto partial_weight = std::make_shared<VulkanBuffer>(platform, partial_weight_bytes);
    auto partial_recurrent = std::make_shared<VulkanBuffer>(platform, partial_recurrent_bytes);
    auto partial_bias = std::make_shared<VulkanBuffer>(platform, partial_bias_bytes);

    // The shader ABI always has output buffers for all gradient families. Keep
    // unused families device-resident without allocating user-visible tensors.
    std::array<std::shared_ptr<VulkanBuffer>, 4> scratch_gradients;
    const auto gradient_buffer = [&](const at::Tensor &gradient,
                                     const at::Tensor &reference,
                                     std::shared_ptr<VulkanBuffer> &scratch) {
        if (gradient.defined())
            return allocation_buffer(gradient.storage().data_ptr()).buffer();
        scratch = std::make_shared<VulkanBuffer>(platform, bytes(reference, "scratch gradient"));
        return scratch->buffer();
    };
    const VkBuffer gradient_input_buffer =
        gradient_buffer(gradient_input, input, scratch_gradients[0]);
    const VkBuffer gradient_weight_buffer =
        gradient_buffer(gradient_weight, weight, scratch_gradients[1]);
    const VkBuffer gradient_recurrent_buffer =
        gradient_buffer(gradient_recurrent, recurrent_weight, scratch_gradients[2]);
    const VkBuffer gradient_bias_buffer =
        gradient_buffer(gradient_bias, bias, scratch_gradients[3]);
    platform.compute().rnn_sequence_backward(
        allocation_buffer(input_data).buffer(), allocation_buffer(weight.storage().data_ptr()).buffer(),
        allocation_buffer(recurrent_weight.storage().data_ptr()).buffer(),
        allocation_buffer(bias.storage().data_ptr()).buffer(),
        allocation_buffer(saved_output.storage().data_ptr()).buffer(),
        allocation_buffer(gradient_output.storage().data_ptr()).buffer(),
        gradient_input_buffer,
        partial_weight->buffer(), partial_recurrent->buffer(), partial_bias->buffer(),
        gradient_weight_buffer, gradient_recurrent_buffer, gradient_bias_buffer,
        batch, sequence, input_dimension, hidden_dimension);
    if (platform.compute().training_step_active()) {
        platform.execution_context().defer_destruction(
            [partial_weight, partial_recurrent, partial_bias, scratch_gradients] {});
    }
    return {need_input ? gradient_input : at::Tensor(),
            need_weight ? gradient_weight : at::Tensor(),
            need_recurrent_weight ? gradient_recurrent : at::Tensor(),
            need_bias ? gradient_bias : at::Tensor()};
}

class RnnSequenceAutograd final : public torch::autograd::Function<RnnSequenceAutograd> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor &input, const at::Tensor &weight,
                              const at::Tensor &recurrent_weight, const at::Tensor &bias) {
        at::AutoDispatchBelowAutograd guard;
        auto output = rnn_sequence_forward(input, weight, recurrent_weight, bias);
        ctx->save_for_backward({input, weight, recurrent_weight, bias, output});
        return output;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list gradients) {
        at::AutoDispatchBelowAutograd guard;
        TORCH_CHECK(!c10::GradMode::is_enabled(),
                    "Vulkan RNN sequence does not support higher-order gradients");
        if (!gradients[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        auto gradients_out = rnn_sequence_backward_impl(
            gradients[0], saved[0], saved[1], saved[2], saved[3], saved[4],
            saved[0].requires_grad(), saved[1].requires_grad(),
            saved[2].requires_grad(), saved[3].requires_grad());
        return {gradients_out[0], gradients_out[1], gradients_out[2], gradients_out[3]};
    }
};

at::Tensor rnn_sequence(const at::Tensor &input, const at::Tensor &weight,
                        const at::Tensor &recurrent_weight, const at::Tensor &bias) {
    return RnnSequenceAutograd::apply(input, weight, recurrent_weight, bias);
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_FRAGMENT(pytorch_vulkan, m) {
    m.def("rnn_sequence(Tensor input, Tensor weight, Tensor recurrent_weight, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(pytorch_vulkan, PrivateUse1, m) {
    m.impl("rnn_sequence", &pytorch_vulkan::rnn_sequence_forward);
}

TORCH_LIBRARY_IMPL(pytorch_vulkan, AutogradPrivateUse1, m) {
    m.impl("rnn_sequence", &pytorch_vulkan::rnn_sequence);
}
