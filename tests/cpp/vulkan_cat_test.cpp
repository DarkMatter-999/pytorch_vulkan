#include "vulkan/operators/cat.h"
#include <cstdint>
#include <stdexcept>
#define CAT_CHECK(condition) do { if (!(condition)) throw std::runtime_error("cat check failed: " #condition); } while (false)
int main() {
  using pytorch_vulkan::CatDispatchLimits;
  using pytorch_vulkan::validate_cat_dispatch_limits;
  using pytorch_vulkan::checked_cat_max_element_offset;
  const CatDispatchLimits limits{4096, 8};
  CAT_CHECK(validate_cat_dispatch_limits(257, 600, 1028, 2400, 84, limits) == 2);
  auto throws = [](auto fn) { try { fn(); } catch (const std::invalid_argument&) { return true; } return false; };
  CAT_CHECK(validate_cat_dispatch_limits(257, 600, 1028, 2400, 84, {4096, 1}) == 1);
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(1, 1, 4097, 4, 84, limits); }));
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(1, 1, 4, 4097, 84, limits); }));
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(1, 1, 4, 4, 4097, limits); }));
  CAT_CHECK(validate_cat_dispatch_limits(12, 12, 12, 48, 84, limits) == 1);
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(257, 600, 1028, 2399, 84, limits); }));
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(
      1, UINT64_C(0x100000000), 4, UINT64_C(17179869184), 84,
      {UINT64_C(1) << 40, 8}); }));
  // The wrapper's allocation preflight must reject representable uint32-sized
  // output ranges that exceed maxStorageBufferRange before at::empty.
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(1, 1025, 4, 4100, 84, {4096, 8}); }));
  CAT_CHECK(throws([&] { checked_cat_max_element_offset(0, {UINT32_MAX, 2, 1, 1}, {2, 1, 1, 1}, 2); }));
  CAT_CHECK(throws([&] { validate_cat_dispatch_limits(1, 1, 4, 4, 84, {4096, 0}); }));
  CAT_CHECK(checked_cat_max_element_offset(7, {2, 3, 1, 1}, {5, 1, 0, 0}, 2) == 14);
  constexpr uint64_t n = UINT64_C(0xfffffffe);
  const uint64_t step = (UINT32_MAX / 256) * 256ULL;
  for (uint64_t start : {UINT64_C(0), UINT64_C(1), step - 1}) {
    uint64_t ordinal = start;
    unsigned visits = 0;
    while (ordinal < n) {
      ++visits;
      if (n - 1 - ordinal < step) break;
      ordinal += step;
    }
    CAT_CHECK(visits == (start < n - step ? 2u : 1u));
  }
}
