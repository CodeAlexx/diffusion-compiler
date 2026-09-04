#pragma once

// The reference sampler's discrete DDPM sigma schedule, the one SDXL uses.
//
// Mirrors comfy/model_sampling.py (ModelSamplingDiscrete, EPS) and
// comfy/samplers.py (normal_scheduler). Every float32 rounding boundary the
// reference crosses is reproduced: the beta/alpha chain runs in double and
// is rounded to float once when the tables are stored, the log table is the
// double logarithm rounded to float, and the interpolation, the argmin and
// the timestep grid then run entirely in float.
//
// sigma_at() is deliberately the exponential of log_sigma_at(): the
// reference's float32 exp comes from its vector math library, which is
// within one ulp of correctly rounded but not reproducible from libm, so
// callers that need bit-exactness gate the interpolant instead.

#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dif::sampling {

struct DiscreteSigmaTable {
  // Ascending in t; t = 0 is the least noisy row. Both float32, as the
  // reference registers them.
  std::vector<float> sigmas;
  std::vector<float> log_sigmas;

  // The scaled-linear beta schedule: betas = linspace(sqrt(a), sqrt(b), T)^2
  // in double, alphas_cumprod = cumprod(1 - betas), and
  // sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod). SDXL's defaults.
  static DiscreteSigmaTable linear(double linear_start = 0.00085,
                                   double linear_end = 0.012,
                                   std::uint32_t timesteps = 1000) {
    if (timesteps < 2U || !(linear_start > 0.0) ||
        !(linear_end > linear_start))
      fail("discrete sigma table requires T>=2 and 0 < start < end");
    DiscreteSigmaTable table;
    table.sigmas.resize(timesteps);
    table.log_sigmas.resize(timesteps);
    const double root_start = std::sqrt(linear_start);
    const double root_end = std::sqrt(linear_end);
    const double span = root_end - root_start;
    const auto last = static_cast<double>(timesteps - 1U);
    double cumulative = 1.0;
    for (std::uint32_t step = 0U; step < timesteps; ++step) {
      const double root =
          root_start + span * (static_cast<double>(step) / last);
      const double beta = root * root;
      cumulative *= 1.0 - beta;
      const double sigma = std::sqrt((1.0 - cumulative) / cumulative);
      table.sigmas[step] = static_cast<float>(sigma);
      table.log_sigmas[step] = static_cast<float>(std::log(sigma));
    }
    return table;
  }

  std::uint32_t size() const {
    return static_cast<std::uint32_t>(sigmas.size());
  }
  float sigma_min() const { return sigmas.front(); }
  float sigma_max() const { return sigmas.back(); }

  // ModelSamplingDiscrete.sigma without its final exponential: clamp the
  // timestep into the table and interpolate the LOG sigmas linearly.
  float log_sigma_at(float timestep) const {
    if (sigmas.empty())
      fail("sigma table is empty");
    const auto last = static_cast<float>(size() - 1U);
    const float clamped = std::clamp(timestep, 0.0F, last);
    const float floor_value = std::floor(clamped);
    const auto low = static_cast<std::uint32_t>(floor_value);
    const auto high = static_cast<std::uint32_t>(std::ceil(clamped));
    const float weight = clamped - floor_value;
    return (1.0F - weight) * log_sigmas[low] + weight * log_sigmas[high];
  }

  // ModelSamplingDiscrete.sigma.
  float sigma_at(float timestep) const {
    return std::exp(log_sigma_at(timestep));
  }

  // ModelSamplingDiscrete.timestep: the row whose log sigma is nearest.
  // torch.argmin keeps the FIRST minimum on a tie.
  std::uint32_t timestep_for(float sigma) const {
    if (sigmas.empty())
      fail("sigma table is empty");
    if (!(sigma > 0.0F))
      fail("timestep lookup requires a positive sigma");
    const float target = std::log(sigma);
    std::uint32_t best = 0U;
    float best_distance = std::abs(target - log_sigmas[0]);
    for (std::uint32_t index = 1U; index < size(); ++index) {
      const float distance = std::abs(target - log_sigmas[index]);
      if (distance < best_distance) {
        best_distance = distance;
        best = index;
      }
    }
    return best;
  }
};

// torch.linspace for float32. The CPU kernel counts up from the start over
// the first half and down from the end over the second, so both endpoints
// are exact and the halves round symmetrically, and its vectorized path
// FUSES each multiply-add. Reproducing the fusion matters: without it the
// grid drifts by an ulp at thirty steps and more, which then moves the
// interpolated sigma. Verified against torch 2.12 for 2..100 steps.
inline std::vector<float> torch_linspace(float start, float end,
                                         std::uint32_t steps) {
  std::vector<float> values(steps);
  if (steps == 0U)
    return values;
  if (steps == 1U) {
    values[0] = start;
    return values;
  }
  const std::uint32_t halfway = steps / 2U;
  const float step = (end - start) / static_cast<float>(steps - 1U);
  for (std::uint32_t index = 0U; index < steps; ++index)
    values[index] =
        index < halfway
            ? std::fma(step, static_cast<float>(index), start)
            : std::fma(-step, static_cast<float>(steps - 1U - index), end);
  return values;
}

// samplers.py normal_scheduler: the fractional timestep grid, walked
// linearly from the noisiest row to the least noisy one. The reference
// takes one extra step instead of appending a zero when the least noisy row
// already is zero, which this schedule's table never is.
inline std::vector<float>
normal_schedule_timesteps(const DiscreteSigmaTable &table,
                          std::uint32_t steps) {
  if (steps == 0U)
    fail("normal schedule requires at least one step");
  const auto start = table.timestep_for(table.sigma_max());
  const auto end = table.timestep_for(table.sigma_min());
  auto count = steps;
  if (std::abs(table.sigma_at(static_cast<float>(end))) <= 1.0e-5F)
    count += 1U;
  return torch_linspace(static_cast<float>(start), static_cast<float>(end),
                        count);
}

// The sampler's sigma list: one sigma per step plus the terminal zero.
inline std::vector<float> normal_schedule(const DiscreteSigmaTable &table,
                                          std::uint32_t steps) {
  const auto grid = normal_schedule_timesteps(table, steps);
  const auto end = table.timestep_for(table.sigma_min());
  const bool append_zero =
      !(std::abs(table.sigma_at(static_cast<float>(end))) <= 1.0e-5F);
  std::vector<float> sigmas;
  sigmas.reserve(grid.size() + 1U);
  for (const auto timestep : grid)
    sigmas.push_back(table.sigma_at(timestep));
  if (append_zero)
    sigmas.push_back(0.0F);
  return sigmas;
}

// The integer-valued timestep the denoiser receives for each non-terminal
// sigma (model_base.BaseModel._apply_model).
inline std::vector<float> unet_timesteps(const DiscreteSigmaTable &table,
                                         const std::vector<float> &sigmas) {
  if (sigmas.size() < 2U)
    fail("a sampling schedule needs a terminal sigma");
  std::vector<float> timesteps;
  timesteps.reserve(sigmas.size() - 1U);
  for (std::size_t index = 0U; index + 1U < sigmas.size(); ++index)
    timesteps.push_back(
        static_cast<float>(table.timestep_for(sigmas[index])));
  return timesteps;
}

// EPS.calculate_input divides the latent by this.
inline float eps_input_divisor(float sigma) {
  return std::sqrt(sigma * sigma + 1.0F);
}

inline float eps_input_scale(float sigma) {
  return 1.0F / eps_input_divisor(sigma);
}

// EPS.calculate_denoised: x0 = x - sigma * eps.
inline float eps_denoised(float x, float epsilon, float sigma) {
  return x - epsilon * sigma;
}

// KSAMPLER.max_denoise: whether the first sigma is the table's maximum, in
// which case the initial noise is scaled by sqrt(1 + sigma^2) rather than
// sigma (EPS.noise_scaling).
inline bool max_denoise(const DiscreteSigmaTable &table,
                        const std::vector<float> &sigmas) {
  if (sigmas.empty())
    fail("max_denoise needs a schedule");
  const float maximum = table.sigma_max();
  const float first = sigmas.front();
  return std::abs(first - maximum) <=
             1.0e-5F * std::max(std::abs(first), std::abs(maximum)) ||
         first > maximum;
}

inline float initial_noise_scale(float first_sigma, bool denoise_fully) {
  return denoise_fully ? eps_input_divisor(first_sigma) : first_sigma;
}

} // namespace dif::sampling
