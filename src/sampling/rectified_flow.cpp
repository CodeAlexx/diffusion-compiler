#include "dif/sampling/rectified_flow.hpp"

#include "dif/support/error.hpp"

#include <cmath>
#include <limits>

namespace dif::sampling {

ShiftedSigmaSchedule make_exponential_shifted_schedule(std::uint32_t points,
                                                       float shift) {
  if (points < 2U)
    fail("shifted sigma schedule requires at least two points");
  if (!(shift > 0.0F) || !std::isfinite(shift))
    fail("shifted sigma schedule requires a finite positive shift");

  ShiftedSigmaSchedule schedule;
  schedule.sigmas.reserve(points);
  const auto step = -1.0F / static_cast<float>(points - 1U);
  const auto halfway = points / 2U;
  for (std::uint32_t index = 0; index < points; ++index) {
    // torch.linspace fills a float32 CPU grid from both ends using an FMA.
    const auto base = index < halfway
                          ? std::fma(step, static_cast<float>(index), 1.0F)
                          : std::fma(-step,
                                     static_cast<float>(points - index - 1U),
                                     0.0F);
    // These barriers preserve the source's individual float32 tensor-op
    // boundaries and prevent a release build from contracting the denominator.
    volatile float numerator = shift * base;
    volatile float denominator_term = (shift - 1.0F) * base;
    volatile float denominator = 1.0F + denominator_term;
    const auto sigma = numerator / denominator;
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

void h3_euler_step_in_place(std::span<float> sample,
                            std::span<const float> model_output,
                            std::uint64_t row_width,
                            std::uint64_t first_generated_row,
                            float timestep, float sigma, float sigma_next) {
  if (row_width == 0U || sample.size() != model_output.size() ||
      sample.size() % row_width != 0U ||
      first_generated_row > sample.size() / row_width ||
      !(sigma > 0.0F) || sigma_next < 0.0F || sigma_next > sigma ||
      !std::isfinite(timestep))
    fail("invalid H3 scheduler step");
  volatile float sigma_from_timestep = 1.0F - timestep;
  volatile float ratio = sigma_next / sigma;
  volatile float one_minus_ratio = 1.0F - ratio;
  const auto first = first_generated_row * row_width;
  for (std::uint64_t index = first; index < sample.size(); ++index) {
    volatile float scaled_velocity =
        sigma_from_timestep * model_output[static_cast<std::size_t>(index)];
    volatile float denoised =
        sample[static_cast<std::size_t>(index)] + scaled_velocity;
    volatile float kept = ratio * sample[static_cast<std::size_t>(index)];
    volatile float moved = one_minus_ratio * denoised;
    volatile float updated = kept + moved;
    sample[static_cast<std::size_t>(index)] = updated;
  }
}

} // namespace dif::sampling
