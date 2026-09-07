#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace dif::runtime {

// Generic opt-in approximation mechanism. A frontend chooses a contiguous
// residual region and owns its numerical admission/invalidation policy.
enum class ResidualCacheAction { Exact, Refresh, Reuse };
struct ResidualCacheRegion {
  std::uint32_t front_first_operation{}, middle_first_operation{}, back_first_operation{};
  std::uint32_t before_front_tensor{}, after_front_tensor{}, after_middle_tensor{};
  std::uint64_t sequence{}, hidden{}, group_size{32};
  std::vector<std::int32_t> probe_rows, audio_probe_rows;
  bool operator==(const ResidualCacheRegion &) const = default;
};
struct ResidualCacheCallbacks {
  // If false, no probes are gathered; decide is still called with empty spans
  // so the frontend can account for this exact evaluation.
  std::function<bool()> needs_probes;
  std::function<ResidualCacheAction(std::span<const float>, std::span<const float>,
                                  std::span<const float>, std::span<const float>)> decide;
  // Called only after a refresh kernel was successfully enqueued.
  std::function<void()> refreshed;
};
struct ResidualCacheDeviceLayout {
  std::uint64_t elements{}, groups{}, main_probe_elements{}, audio_probe_elements{};
  std::uint64_t indices{}, before_probe{}, after_probe{};
  std::uint64_t front_codes{}, front_scales{}, residual_codes{}, residual_scales{};
  std::uint64_t required_bytes{};
};

// No device/context allocation or compilation here. The shared runtime admits
// this full worst-case scratch size, suballocates it from its existing arena,
// and appends the source to its existing generated module before hashing it.
ResidualCacheDeviceLayout residual_cache_device_layout(const ResidualCacheRegion &region);
std::string_view residual_cache_cuda_source();

} // namespace dif::runtime
