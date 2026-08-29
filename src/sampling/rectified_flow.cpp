#include "dif/sampling/rectified_flow.hpp"

#include "dif/support/error.hpp"

#include <cmath>

namespace dif::sampling {

ShiftedSigmaSchedule make_exponential_shifted_schedule(std::uint32_t points,
                                                       float shift) {
  if (points < 2U)
    fail("shifted sigma schedule requires at least two points");
  if (!(shift > 0.0F) || !std::isfinite(shift))
    fail("shifted sigma schedule requires a finite positive shift");

  ShiftedSigmaSchedule schedule;
  schedule.sigmas.reserve(points);
  const auto denominator = static_cast<float>(points - 1U);
  for (std::uint32_t index = 0; index < points; ++index) {
    const auto base = index == 0U
                          ? 1.0F
                          : index + 1U == points
                                ? 0.0F
                                : 1.0F - static_cast<float>(index) / denominator;
    const auto sigma =
        shift * base / (1.0F + (shift - 1.0F) * base);
    if (schedule.sigmas.empty() || schedule.sigmas.back() != sigma)
      schedule.sigmas.push_back(sigma);
  }
  if (schedule.sigmas.size() < 2U || schedule.sigmas.back() != 0.0F)
    fail("shifted sigma schedule collapsed below two points or lost terminal zero");

  schedule.timesteps.reserve(schedule.sigmas.size() - 1U);
  for (std::size_t index = 0; index + 1U < schedule.sigmas.size(); ++index)
    schedule.timesteps.push_back(1.0F - schedule.sigmas[index]);
  return schedule;
}

} // namespace dif::sampling
