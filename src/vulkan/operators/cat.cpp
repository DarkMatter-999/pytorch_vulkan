#include "cat.h"

#include <algorithm>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include "vulkan_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_execution.h"
#include <limits>
#include <stdexcept>

namespace pytorch_vulkan {
uint32_t validate_cat_dispatch_limits(uint64_t input_numel, uint64_t output_numel,
                                      VkDeviceSize input_bytes, VkDeviceSize output_bytes,
                                      VkDeviceSize metadata_bytes,
                                      CatDispatchLimits limits) {
  constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  if (input_numel > max_u32 || output_numel > max_u32)
    throw std::invalid_argument("cat element count exceeds uint32 shader indexing");
  if ((input_numel > 0 && input_bytes < sizeof(float)) ||
      output_numel > output_bytes / sizeof(float))
    throw std::invalid_argument("cat allocation is smaller than its reachable/output range");
  if (input_bytes > limits.max_storage_buffer_range ||
      output_bytes > limits.max_storage_buffer_range ||
      metadata_bytes > limits.max_storage_buffer_range)
    throw std::invalid_argument("cat descriptor range exceeds device limit");
  if (input_numel == 0) return 0;
  if (limits.max_compute_workgroup_count_x == 0)
    throw std::invalid_argument("cat device has zero X workgroup limit");
  const uint64_t needed = (input_numel + 255) / 256;
  const uint64_t max_groups = max_u32 / 256;
  return static_cast<uint32_t>(std::min({needed, max_groups,
      static_cast<uint64_t>(limits.max_compute_workgroup_count_x)}));
}

uint32_t checked_cat_max_element_offset(uint64_t offset,
    const std::array<uint32_t,4>& sizes, const std::array<uint32_t,4>& strides,
    uint32_t rank) {
  if (rank == 0 || rank > 4 || offset > UINT32_MAX)
    throw std::invalid_argument("cat rank or storage offset is unsupported");
  uint64_t maximum = offset;
  for (uint32_t d = 0; d < rank; ++d) {
    if (sizes[d] == 0) throw std::invalid_argument("cat dispatched extent is zero");
    const uint64_t term = static_cast<uint64_t>(sizes[d] - 1) * strides[d];
    if (term > UINT32_MAX || maximum > UINT32_MAX - term)
      throw std::invalid_argument("cat source element offset overflows uint32");
    maximum += term;
  }
  return static_cast<uint32_t>(maximum);
}

CatPlan make_cat_plan(const at::TensorList &inputs, int64_t dim) {
  TORCH_CHECK(!inputs.empty(), "torch.cat(): expected a non-empty list of Tensors");
  for (const auto &input : inputs)
    TORCH_CHECK(input.dim() > 0, "zero-dimensional tensor cannot be concatenated");
  std::size_t first = inputs.size();
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!at::native::cat_should_skip_tensor(inputs[i])) { first = i; break; }
  c10::MemoryFormat format = c10::MemoryFormat::Contiguous;
  bool have_format = false;
  for (const auto &input : inputs) {
    const auto suggested = input.suggest_memory_format();
    if (!have_format) { format = suggested; have_format = true; }
    else if (suggested != format) format = c10::MemoryFormat::Contiguous;
    if (suggested == c10::MemoryFormat::Contiguous) format = c10::MemoryFormat::Contiguous;
  }
  if (first == inputs.size()) return {dim, {0}, {1}, format, {}, true};
  const auto rank = inputs[first].dim();
  TORCH_CHECK(rank > 0 && rank <= 4, "Vulkan cat supports ranks 1 through 4");
  const auto axis = at::maybe_wrap_dim(dim, rank);
  std::vector<int64_t> sizes(inputs[first].sizes().begin(), inputs[first].sizes().end());
  sizes[axis] = 0;
  std::vector<CatRegion> regions;
  uint64_t begin = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &t = inputs[i];
    if (at::native::cat_should_skip_tensor(t)) continue;
    TORCH_CHECK(t.dim() == rank, "Tensors must have same number of dimensions");
    for (int64_t d = 0; d < rank; ++d)
      if (d != axis) TORCH_CHECK(t.size(d) == sizes[d], "Sizes of tensors must match except in dimension ", axis);
    TORCH_CHECK(t.size(axis) >= 0 && begin <= static_cast<uint64_t>(INT64_MAX - t.size(axis)), "cat concatenated extent overflow");
    if (t.numel() != 0) regions.push_back({i, begin, static_cast<uint64_t>(t.size(axis))});
    begin += static_cast<uint64_t>(t.size(axis));
  }
  sizes[axis] = static_cast<int64_t>(begin);
  uint64_t output_numel = 1;
  for (const int64_t extent : sizes) {
    TORCH_CHECK(extent >= 0 &&
                    (extent == 0 || output_numel <= static_cast<uint64_t>(INT64_MAX) /
                                                       static_cast<uint64_t>(extent)),
                "Vulkan cat output element count overflows int64");
    output_numel *= static_cast<uint64_t>(extent);
  }
  TORCH_CHECK(output_numel <= UINT32_MAX, "Vulkan cat output exceeds uint32 shader indexing");
  std::vector<int64_t> strides(rank, 1);
  if (format == c10::MemoryFormat::ChannelsLast && rank == 4) {
    strides[1] = 1;
    strides[3] = sizes[1];
    TORCH_CHECK(sizes[3] == 0 || strides[3] <= INT64_MAX / sizes[3], "cat output stride overflow");
    strides[2] = sizes[3] * strides[3];
    TORCH_CHECK(sizes[2] == 0 || strides[2] <= INT64_MAX / sizes[2], "cat output stride overflow");
    strides[0] = sizes[2] * strides[2];
  } else {
    for (int64_t d = rank - 2; d >= 0; --d) {
      TORCH_CHECK(sizes[d + 1] == 0 || strides[d + 1] <= INT64_MAX / sizes[d + 1], "cat output stride overflow");
      strides[d] = strides[d + 1] * sizes[d + 1];
    }
  }
  return {axis, std::move(sizes), std::move(strides), format, std::move(regions), false};
}

at::Tensor cat(const at::TensorList &inputs, int64_t dim) {
  TORCH_CHECK(!inputs.empty(), "torch.cat(): expected a non-empty list of Tensors");
  const auto plan = make_cat_plan(inputs, dim);
  const auto first_it = std::find_if(inputs.begin(), inputs.end(), [](const at::Tensor &t) {
    return !at::native::cat_should_skip_tensor(t);
  });
  const auto &first = first_it == inputs.end() ? inputs[0] : *first_it;
  for (const auto &t : inputs) {
    TORCH_CHECK(t.device() == first.device() && t.device().type() == c10::DeviceType::PrivateUse1 && t.device().index() == 0,
                "Vulkan cat requires all inputs on vk:0");
    TORCH_CHECK(t.scalar_type() == at::kFloat, "Vulkan cat requires float32 inputs");
    TORCH_CHECK(t.dim() > 0 && t.dim() <= 4, "Vulkan cat supports ranks 1 through 4");
    if (!at::native::cat_should_skip_tensor(t)) {
      const auto layout = inspect_vulkan_tensor_layout(t, "cat input");
      TORCH_CHECK(layout.internal_overlap == VulkanOverlap::No ||
                      is_non_overlapping_except_broadcast_dims(layout),
                  "Vulkan cat input layout overlaps");
    }
  }
  if (!plan.all_skipped) {
    uint64_t output_numel = 1;
    for (const auto extent : plan.sizes) output_numel *= static_cast<uint64_t>(extent);
    if (output_numel != 0) {
      TORCH_CHECK(!plan.regions.empty(), "positive Vulkan cat output has no gathered input region");
      const auto &platform_source = inputs[plan.regions.front().input_index];
      auto &platform = allocation_platform(platform_source.storage().data_ptr());
      const auto bytes = output_numel * sizeof(float);
      TORCH_CHECK(bytes / sizeof(float) == output_numel,
                  "Vulkan cat output storage size overflows");
      const auto max_range = platform.compute().cat_max_storage_buffer_range();
      TORCH_CHECK(bytes <= max_range,
                  "Vulkan cat output descriptor range exceeds device limit");
      for (const auto &region : plan.regions) {
        const auto &source = inputs[region.input_index];
        const auto layout = inspect_vulkan_tensor_layout(source, "cat input");
        const auto &source_data = source.storage().data_ptr();
        TORCH_CHECK(&allocation_platform(source_data) == &platform,
                    "Vulkan cat inputs require one allocation platform");
        validate_allocation(source_data, layout.allocation_bytes, "cat input");
        TORCH_CHECK(layout.numel >= 0 && static_cast<uint64_t>(layout.numel) <= UINT32_MAX,
                    "Vulkan cat input exceeds uint32 indexing");
        platform.compute().validate_cat_gather(layout.allocation_bytes, bytes,
                                                static_cast<uint32_t>(layout.numel));
        std::array<uint32_t, 4> source_sizes{}, source_strides{};
        for (uint32_t d = 0; d < static_cast<uint32_t>(layout.rank); ++d) {
          TORCH_CHECK(layout.sizes[d] > 0 && static_cast<uint64_t>(layout.sizes[d]) <= UINT32_MAX &&
                          layout.strides[d] >= 0 && static_cast<uint64_t>(layout.strides[d]) <= UINT32_MAX,
                      "Vulkan cat input dimensions or strides exceed shader metadata");
          source_sizes[d] = static_cast<uint32_t>(layout.sizes[d]);
          source_strides[d] = static_cast<uint32_t>(layout.strides[d]);
        }
        TORCH_CHECK(layout.storage_offset >= 0, "Vulkan cat storage offset must be nonnegative");
        const auto maximum = checked_cat_max_element_offset(
            static_cast<uint64_t>(layout.storage_offset), source_sizes, source_strides,
            static_cast<uint32_t>(layout.rank));
        TORCH_CHECK(static_cast<uint64_t>(maximum) * sizeof(float) + sizeof(float) <=
                        layout.allocation_bytes,
                    "Vulkan cat source address exceeds allocation");
      }
    }
  }
  auto output = at::empty(plan.sizes, first.options().memory_format(plan.memory_format));
  if (plan.all_skipped || output.numel() == 0) return output;
  TORCH_CHECK(static_cast<uint64_t>(output.numel()) <= UINT32_MAX,
              "Vulkan cat output exceeds uint32 shader indexing");
  const auto output_layout = inspect_vulkan_tensor_layout(output, "cat output");
  TORCH_CHECK(output_layout.strides == plan.strides,
              "Vulkan cat output allocation does not match planned memory format");
  const auto &out_data = output.storage().data_ptr();
  const auto &platform = allocation_platform(out_data);
  validate_allocation(out_data, output_layout.allocation_bytes, "cat output");
  struct Pending { at::Tensor tensor; VulkanTensorLayout layout; VulkanBuffer *buffer; CatGatherMetadata meta; };
  std::vector<Pending> pending;
  pending.reserve(plan.regions.size());
  for (const auto &region : plan.regions) {
    const auto &input = inputs[region.input_index];
    const auto layout = inspect_vulkan_tensor_layout(input, "cat input");
    TORCH_CHECK(layout.rank >= 1 && layout.rank <= 4, "Vulkan cat supports ranks 1 through 4");
    TORCH_CHECK(layout.internal_overlap == VulkanOverlap::No || is_non_overlapping_except_broadcast_dims(layout), "Vulkan cat input layout overlaps");
    TORCH_CHECK(layout.numel >= 0 && static_cast<uint64_t>(layout.numel) <= UINT32_MAX, "Vulkan cat input exceeds uint32 indexing");
    const auto &data = input.storage().data_ptr();
    TORCH_CHECK(&allocation_platform(data) == &platform, "Vulkan cat inputs require one allocation platform");
    auto &buffer = allocation_buffer(data);
    const VkDeviceSize allocation_bytes = buffer.size();
    validate_allocation(data, allocation_bytes, "cat input");
    std::array<uint32_t,4> sz{}, st{}, os{}, ost{};
    CatGatherMetadata meta{};
    meta[kRank] = static_cast<uint32_t>(layout.rank); meta[kAxis] = static_cast<uint32_t>(plan.dim);
    meta[kAxisBegin] = static_cast<uint32_t>(region.axis_begin); meta[kInputNumel] = static_cast<uint32_t>(layout.numel);
    uint64_t output_offset = 0;
    for (uint32_t d = 0; d < static_cast<uint32_t>(layout.rank); ++d) {
      TORCH_CHECK(layout.sizes[d] > 0 && static_cast<uint64_t>(layout.sizes[d]) <= UINT32_MAX, "Vulkan cat dispatched dimensions must be positive uint32 values");
      TORCH_CHECK(layout.strides[d] >= 0 && static_cast<uint64_t>(layout.strides[d]) <= UINT32_MAX, "Vulkan cat strides must be uint32 values");
      sz[d] = static_cast<uint32_t>(layout.sizes[d]); st[d] = static_cast<uint32_t>(layout.strides[d]);
      os[d] = static_cast<uint32_t>(output_layout.sizes[d]); ost[d] = static_cast<uint32_t>(output_layout.strides[d]);
      meta[kInputSizes+d] = sz[d]; meta[kInputStrides+d] = st[d];
      meta[kOutputSizes+d] = os[d]; meta[kOutputStrides+d] = ost[d];
    }
    TORCH_CHECK(layout.storage_offset >= 0, "Vulkan cat storage offset must be nonnegative");
    meta[kInputStorageOffset] = static_cast<uint32_t>(layout.storage_offset);
    const auto max_src = checked_cat_max_element_offset(layout.storage_offset, sz, st, layout.rank);
    TORCH_CHECK(static_cast<uint64_t>(max_src) * sizeof(float) + sizeof(float) <= allocation_bytes, "Vulkan cat source address exceeds allocation");
    output_offset = region.axis_begin;
    uint64_t max_dst = 0;
    for (uint32_t d=0; d<static_cast<uint32_t>(layout.rank); ++d) {
      const uint64_t coord = d == static_cast<uint32_t>(plan.dim) ? output_offset + sz[d]-1 : sz[d]-1;
      TORCH_CHECK(coord == 0 || ost[d] <= (UINT32_MAX-max_dst)/coord, "Vulkan cat output address overflows uint32");
      max_dst += coord * ost[d];
    }
    TORCH_CHECK(max_dst < static_cast<uint64_t>(output.numel()), "Vulkan cat output address exceeds result");
    pending.push_back({input, layout, &buffer, meta});
  }
  for (const auto &entry : pending)
    platform.compute().validate_cat_gather(entry.layout.allocation_bytes,
                                           output_layout.allocation_bytes,
                                           static_cast<uint32_t>(entry.layout.numel));
  for (const auto &entry : pending) {
    platform.compute().cat_gather(entry.buffer->buffer(), entry.layout.allocation_bytes,
                                  allocation_buffer(out_data).buffer(), output_layout.allocation_bytes,
                                  entry.meta, static_cast<uint32_t>(entry.layout.numel));
    platform.execution_context().retain_until_completion([input=entry.tensor, output] {});
  }
  return output;
}
}

namespace {
at::Tensor cat_dispatch(const c10::IListRef<at::Tensor> &inputs, int64_t dim) {
  std::vector<at::Tensor> values;
  values.reserve(inputs.size());
  for (const auto &input : inputs) values.push_back(input);
  return pytorch_vulkan::cat(at::TensorList(values), dim);
}
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("cat", TORCH_FN(cat_dispatch)); }
