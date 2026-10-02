#pragma once

#include <ATen/ATen.h>
#include <c10/core/MemoryFormat.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace pytorch_vulkan {
struct CatRegion {
  std::size_t input_index;
  uint64_t axis_begin, axis_extent;
};
struct CatPlan {
  int64_t dim;
  std::vector<int64_t> sizes, strides;
  c10::MemoryFormat memory_format;
  std::vector<CatRegion> regions;
  bool all_skipped;
};
struct CatDispatchLimits {
  VkDeviceSize max_storage_buffer_range;
  uint32_t max_compute_workgroup_count_x;
};
using CatGatherMetadata = std::array<uint32_t, 21>;
constexpr std::size_t kRank = 0, kAxis = 1, kAxisBegin = 2, kInputNumel = 3;
constexpr std::size_t kInputSizes = 4, kInputStrides = 8;
constexpr std::size_t kInputStorageOffset = 12, kOutputSizes = 13;
constexpr std::size_t kOutputStrides = 17;
static_assert(CatGatherMetadata{}.size() == 21);
static_assert(sizeof(CatGatherMetadata) == 84);
static_assert(kRank == 0 && kAxis == 1 && kAxisBegin == 2 && kInputNumel == 3);
static_assert(kInputSizes == 4 && kInputStrides == 8);
static_assert(kInputStorageOffset == 12 && kOutputSizes == 13 && kOutputStrides == 17);

CatPlan make_cat_plan(const at::TensorList&, int64_t);
uint32_t validate_cat_dispatch_limits(uint64_t, uint64_t, VkDeviceSize,
                                      VkDeviceSize, VkDeviceSize,
                                      CatDispatchLimits);
uint32_t checked_cat_max_element_offset(uint64_t,
    const std::array<uint32_t,4>&, const std::array<uint32_t,4>&, uint32_t);
at::Tensor cat(const at::TensorList&, int64_t);
}
