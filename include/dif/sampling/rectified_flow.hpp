#pragma once

#include <cstdint>
#include <span>
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

// Source-faithful MiniMax-H3 data-ward Euler update. The denoised estimate
// deliberately recovers sigma from the rounded timestep while the blend ratio
// uses the sigma grid, and every arithmetic boundary is rounded to F32.
void h3_euler_step_in_place(std::span<float> sample,
                            std::span<const float> model_output,
                            std::uint64_t row_width,
                            std::uint64_t first_generated_row,
                            float timestep, float sigma, float sigma_next);

} // namespace dif::sampling
