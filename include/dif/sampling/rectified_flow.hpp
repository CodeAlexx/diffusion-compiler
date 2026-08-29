#pragma once

#include <cstdint>
#include <vector>

namespace dif::sampling {

struct ShiftedSigmaSchedule {
  std::vector<float> sigmas;
  std::vector<float> timesteps;
};

// Builds a float32 grid from 1 to 0 (terminal zero included), applies
// shift*sigma/(1+(shift-1)*sigma), collapses consecutive float32 duplicates,
// and exposes t=1-sigma for every model evaluation.
ShiftedSigmaSchedule make_exponential_shifted_schedule(std::uint32_t points,
                                                       float shift);

} // namespace dif::sampling
