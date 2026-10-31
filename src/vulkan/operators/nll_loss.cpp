#include "nll_loss.h"

#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"

#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstring>
#include <limits>
#include <sstream>

// aten::nll_loss_forward / nll_loss_backward leaves for nll_loss.comp.  The input is
// already log-probabilities (cross_entropy_loss is composite: _log_softmax then NLL);
// the kernel is a gather and a fixed-order reduction.  Targets are validated exactly
// at the call (CPU raises IndexError), see validate_targets().
namespace pytorch_vulkan {
namespace {
enum Operation : uint32_t {
    kInit = 0,
    kValidate = 1,
    kCapture = 2,
    kForwardNone = 3,
    kForwardReduce = 4,
    kBackward = 5
};

// The reduction is one workgroup; at most 2048 tile sums of 2048 terms fit two levels.
constexpr int64_t kMaxReduceElements = int64_t{1} << 22;

struct NllLossParams {
    uint32_t operation, n, c, reduction, has_weight, ignore_lo, ignore_hi;
    uint32_t target_offset, input_offset, weight_offset, grad_offset, total_offset, count;
};
static_assert(sizeof(NllLossParams) == kNllLossPushConstantBytes,
              "nll_loss push-constant ABI size mismatch");

const c10::Device kVulkan(c10::DeviceType::PrivateUse1, 0);

void check_float(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.device() == kVulkan && tensor.scalar_type() == at::kFloat &&
                    tensor.layout() == at::kStrided,
                "Vulkan nll_loss ", name, " requires a strided float32 vk:0 tensor");
}

bool has_weight(const c10::optional<at::Tensor> &weight) {
    return weight.has_value() && weight->defined();
}

// Mirrors the argument checks of ATen's nll_loss_forward / backward meta functions, in
// the same order and wording where ATen exposes them.
void check_arguments(const at::Tensor &self, const at::Tensor &target,
                     const c10::optional<at::Tensor> &weight, int64_t reduction) {
    check_float(self, "input");
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2,
                "Vulkan nll_loss supports 1-D or 2-D input, got ", self.dim(),
                " dimensions");
    TORCH_CHECK(target.device() == kVulkan && target.layout() == at::kStrided,
                "Vulkan nll_loss target requires a strided vk:0 tensor");
    TORCH_CHECK(target.scalar_type() == at::kLong,
                "Vulkan nll_loss target requires int64 (uint8 and other integer "
                "targets are not supported), got ", target.scalar_type());
    const bool shape_ok = self.dim() == 1
                              ? target.dim() == 0
                              : target.dim() == 1 && target.size(0) == self.size(0);
    TORCH_CHECK(shape_ok, "size mismatch (got input: ", self.sizes(),
                ", target: ", target.sizes(), ")");
    // An empty label tensor owns no allocation, so it carries no provenance to check.
    TORCH_CHECK(target.numel() == 0 || is_validated_label_allocation(target.storage().data_ptr()),
                "Vulkan nll_loss labels have no validated CPU provenance");
    const int64_t classes = self.size(self.dim() - 1);
    if (has_weight(weight)) {
        check_float(*weight, "weight");
        TORCH_CHECK(weight->dim() == 1 && weight->size(0) == classes,
                    "weight tensor should be defined either for all ", classes,
                    " classes or no classes but got weight tensor of shape: ",
                    weight->sizes());
    }
    TORCH_CHECK(classes >= 1, "Vulkan nll_loss requires at least one class");
    TORCH_CHECK(reduction >= 0 && reduction <= 2,
                "Vulkan nll_loss supports reduction none, mean, or sum");
    const int64_t rows = self.dim() == 1 ? 1 : self.size(0);
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() &&
                    rows * classes <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan nll_loss element count exceeds the supported range");
    TORCH_CHECK(reduction == 0 || rows <= kMaxReduceElements,
                "Vulkan nll_loss reductions support at most ", kMaxReduceElements,
                " rows, got ", rows);
    TORCH_CHECK(target.is_contiguous(), "Vulkan nll_loss target requires a contiguous tensor");
}

uint32_t checked_offset(const at::Tensor &tensor) {
    TORCH_CHECK(tensor.storage_offset() >= 0 &&
                    tensor.storage_offset() <= std::numeric_limits<int32_t>::max(),
                "Vulkan nll_loss tensor offset exceeds the supported range");
    return static_cast<uint32_t>(tensor.storage_offset());
}

at::Tensor dense(const at::Tensor &tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

struct Slots {
    const at::Tensor *inputs[5];
    const at::Tensor *outputs[3];
};

void dispatch(const Slots &slots, NllLossParams params, uint32_t invocations) {
    const auto &platform = allocation_platform(slots.inputs[0]->storage().data_ptr());
    VulkanTensorLayout input_layouts[5], output_layouts[3];
    const VulkanTensorLayout *input_layout_ptrs[5], *output_layout_ptrs[3];
    const VulkanBuffer *input_buffers[5], *output_buffers[3];
    for (int i = 0; i < 5; ++i) {
        input_layouts[i] = inspect_vulkan_tensor_layout(*slots.inputs[i], "nll_loss input");
        input_layout_ptrs[i] = &input_layouts[i];
        input_buffers[i] = &allocation_buffer(slots.inputs[i]->storage().data_ptr());
    }
    for (int i = 0; i < 3; ++i) {
        output_layouts[i] = inspect_vulkan_tensor_layout(*slots.outputs[i], "nll_loss output");
        output_layout_ptrs[i] = &output_layouts[i];
        output_buffers[i] = &allocation_buffer(slots.outputs[i]->storage().data_ptr());
    }
    platform.compute().compute_multi_output(input_buffers, input_layout_ptrs,
                                            output_buffers, output_layout_ptrs, 5, 3,
                                            invocations, &params, sizeof(params), true);
}

NllLossParams base_params(Operation operation, int64_t rows, int64_t classes,
                          int64_t reduction, int64_t ignore_index, bool weighted) {
    NllLossParams params{};
    params.operation = operation;
    params.n = static_cast<uint32_t>(rows);
    params.c = static_cast<uint32_t>(classes);
    params.reduction = static_cast<uint32_t>(reduction);
    params.has_weight = weighted ? 1U : 0U;
    const uint64_t bits = static_cast<uint64_t>(ignore_index);
    params.ignore_lo = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
    params.ignore_hi = static_cast<uint32_t>(bits >> 32);
    return params;
}

// Exact target validation (decision: CPU raises IndexError, so does Vulkan).  Three tiny
// dispatches fill a 4-word status buffer: `init`, `validate` (one atomicMax on the
// encoding 0xFFFFFFFF - index, so the result is the lowest invalid index independent
// of scheduling), and `capture` (one invocation reads the int64 value at that index).
// The host then waits for the queue and reads 16 bytes through an internal
// host-visible read (not a user-visible transfer).  A recording training scope cannot
// synchronize, so validation there is refused rather than skipped.
void validate_targets(const at::Tensor &target, const at::Tensor &filler,
                      const at::Tensor &out0, const at::Tensor &out1, int64_t rows,
                      int64_t classes, int64_t ignore_index) {
    const auto &platform = allocation_platform(target.storage().data_ptr());
    TORCH_CHECK(!platform.compute().training_step_active(),
                "Vulkan nll_loss target validation requires synchronization, which a "
                "recording training scope cannot perform");
    at::Tensor status = at::empty({4}, filler.options());
    const Slots slots{{&target, &filler, &filler, &filler, &filler},
                      {&out0, &out1, &status}};
    NllLossParams params = base_params(kInit, rows, classes, 0, ignore_index, false);
    params.target_offset = checked_offset(target);
    params.count = 4;
    dispatch(slots, params, 4);
    params.operation = kValidate;
    params.count = static_cast<uint32_t>(rows);
    dispatch(slots, params, static_cast<uint32_t>(rows));
    params.operation = kCapture;
    params.count = 1;
    dispatch(slots, params, 1);
    uint32_t words[4] = {0, 0, 0, 0};
    VulkanBuffer &buffer = allocation_buffer(status.storage().data_ptr());
    if ((buffer.memory_properties() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        auto lock = platform.acquire_host_transfer_lock();
        buffer.read(words, sizeof(words));
    } else {
        // Same staged path the public transfer uses, but internal: it is validation
        // plumbing, not a user-visible device-to-host copy.
        std::unique_lock<std::mutex> transfer_lock(platform.transfer_mutex());
        VulkanBuffer &staging = platform.staging_buffer(sizeof(words));
        platform.copy_buffer_sync(buffer.buffer(), staging.buffer(), sizeof(words), 0, 0);
        staging.read(words, sizeof(words));
    }
    if (words[0] != 0) {
        const uint64_t bits = (static_cast<uint64_t>(words[3]) << 32) | words[2];
        TORCH_CHECK_INDEX(false, "Target ", static_cast<int64_t>(bits),
                          " is out of bounds.");
    }
}
} // namespace

std::tuple<at::Tensor, at::Tensor>
nll_loss_forward(const at::Tensor &self, const at::Tensor &target,
                 const c10::optional<at::Tensor> &weight, int64_t reduction,
                 c10::SymInt ignore_index_symint) {
    check_arguments(self, target, weight, reduction);
    const int64_t ignore_index = ignore_index_symint.expect_int();
    const int64_t rows = self.dim() == 1 ? 1 : self.size(0);
    const int64_t classes = self.size(self.dim() - 1);
    const bool weighted = has_weight(weight);
    const std::vector<int64_t> loss_shape =
        reduction == 0 && self.dim() == 2 ? std::vector<int64_t>{rows} : std::vector<int64_t>{};
    if (rows == 0) {
        at::Tensor loss = reduction == 0
                              ? at::empty({0}, self.options())
                              : at::full({}, reduction == 1
                                                 ? std::numeric_limits<float>::quiet_NaN()
                                                 : 0.0F, self.options());
        return {loss, at::zeros({}, self.options())};
    }
    return with_vulkan_operation(allocation_runtime(self.storage().data_ptr()), [&]()
                                     -> std::tuple<at::Tensor, at::Tensor> {
        const at::Tensor input = dense(self);
        const at::Tensor weights = weighted ? dense(*weight) : input;
        at::Tensor loss = at::empty(loss_shape, self.options());
        // total_weight is 0 for reduction none: the kernel never writes it there.
        at::Tensor total = reduction == 0 ? at::zeros({}, self.options())
                                          : at::empty({}, self.options());
        validate_targets(target, input, loss, total, rows, classes, ignore_index);
        at::Tensor status = at::empty({4}, self.options());
        const Slots slots{{&target, &input, &weights, &input, &input},
                          {&loss, &total, &status}};
        NllLossParams params = base_params(reduction == 0 ? kForwardNone : kForwardReduce,
                                           rows, classes, reduction, ignore_index, weighted);
        params.target_offset = checked_offset(target);
        params.input_offset = checked_offset(input);
        params.weight_offset = weighted ? checked_offset(weights) : 0U;
        const uint32_t invocations = reduction == 0 ? static_cast<uint32_t>(rows) : 256U;
        params.count = invocations;
        dispatch(slots, params, invocations);
        return {loss, total};
    });
}

at::Tensor nll_loss_backward(const at::Tensor &grad_output, const at::Tensor &self,
                             const at::Tensor &target,
                             const c10::optional<at::Tensor> &weight, int64_t reduction,
                             c10::SymInt ignore_index_symint,
                             const at::Tensor &total_weight) {
    check_arguments(self, target, weight, reduction);
    check_float(grad_output, "grad_output");
    check_float(total_weight, "total_weight");
    TORCH_CHECK(total_weight.numel() == 1,
                "Vulkan nll_loss total_weight must be a single-element tensor");
    const int64_t ignore_index = ignore_index_symint.expect_int();
    const int64_t rows = self.dim() == 1 ? 1 : self.size(0);
    const int64_t classes = self.size(self.dim() - 1);
    const bool weighted = has_weight(weight);
    if (reduction == 0) {
        TORCH_CHECK(grad_output.numel() == rows && grad_output.dim() == (self.dim() == 1 ? 0 : 1),
                    "Vulkan nll_loss backward grad_output shape does not match reduction none");
    } else {
        TORCH_CHECK(grad_output.numel() == 1,
                    "Expected a single element grad_output tensor, but got: ",
                    grad_output.sizes());
    }
    if (rows == 0)
        return at::empty(self.sizes(), self.options());
    return with_vulkan_operation(allocation_runtime(self.storage().data_ptr()), [&]() -> at::Tensor {
        at::Tensor grad_input = at::empty(self.sizes(), self.options());
        const at::Tensor grad = dense(grad_output);
        const at::Tensor total = dense(total_weight);
        const at::Tensor weights = weighted ? dense(*weight) : grad;
        at::Tensor dummy = at::empty({1}, self.options());
        validate_targets(target, grad, grad_input, dummy, rows, classes, ignore_index);
        at::Tensor status = at::empty({4}, self.options());
        const Slots slots{{&target, &grad, &weights, &grad, &total},
                          {&grad_input, &dummy, &status}};
        NllLossParams params =
            base_params(kBackward, rows, classes, reduction, ignore_index, weighted);
        params.target_offset = checked_offset(target);
        params.weight_offset = weighted ? checked_offset(weights) : 0U;
        params.grad_offset = checked_offset(grad);
        params.total_offset = checked_offset(total);
        const uint32_t invocations = static_cast<uint32_t>(rows * classes);
        params.count = invocations;
        dispatch(slots, params, invocations);
        return grad_input;
    });
}
} // namespace pytorch_vulkan

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("nll_loss_forward", &pytorch_vulkan::nll_loss_forward);
    m.impl("nll_loss_backward", &pytorch_vulkan::nll_loss_backward);
}
