#include "dif/sampling/rectified_flow.hpp"

#include "dif/support/error.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace dif::sampling {

namespace {

float shifted_sigma(float base, float shift) {
  volatile float numerator = shift * base;
  volatile float denominator_term = (shift - 1.0F) * base;
  volatile float denominator = 1.0F + denominator_term;
  return numerator / denominator;
}

void validate_audio_geometry(std::span<const float> values,
                             std::uint64_t row_width,
                             std::uint64_t first_generated_row) {
  if (row_width == 0U || values.size() % row_width != 0U ||
      first_generated_row > values.size() / row_width)
    fail("invalid H3 AV audio carry geometry");
}

} // namespace

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
    const auto sigma = shifted_sigma(base, shift);
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

H3AVSigmaSchedule make_h3_simple_av_schedule(std::uint32_t evaluations) {
  if (evaluations == 0U || evaluations > 1000U)
    fail("H3 simple schedule evaluations must be in [1,1000]");

  H3AVSigmaSchedule schedule;
  schedule.video_sigmas.reserve(static_cast<std::size_t>(evaluations) + 1U);
  schedule.audio_sigmas.reserve(static_cast<std::size_t>(evaluations) + 1U);
  const auto stride = 1000.0 / static_cast<double>(evaluations);
  for (std::uint32_t step = 0U; step < evaluations; ++step) {
    const auto offset = static_cast<std::uint32_t>(
        static_cast<double>(step) * stride);
    const auto table_index = 999U - offset;
    volatile float base =
        static_cast<float>(table_index + 1U) / 1000.0F;
    const auto video_sigma = shifted_sigma(base, 12.0F);

    // MiniMax's time_shift_sigma(video, 12, 3), with every tensor operation
    // rounded to float32 in the same order as the Python source.
    volatile float inverse_denominator_term = video_sigma * (1.0F - 12.0F);
    volatile float inverse_denominator = 12.0F + inverse_denominator_term;
    volatile float unshifted = video_sigma / inverse_denominator;
    const auto audio_sigma = shifted_sigma(unshifted, 3.0F);
    schedule.video_sigmas.push_back(video_sigma);
    schedule.audio_sigmas.push_back(audio_sigma);
  }
  schedule.video_sigmas.push_back(0.0F);
  schedule.audio_sigmas.push_back(0.0F);
  return schedule;
}

double flux2_empirical_mu(std::uint64_t image_sequence_length,
                          std::uint32_t steps) {
  if (steps == 0U)
    fail("FLUX.2 schedule requires at least one step");
  const auto sequence = static_cast<double>(image_sequence_length);
  if (image_sequence_length > 4300U) {
    volatile double scaled = 0.00016927 * sequence;
    volatile double result = scaled + 0.45666666;
    return result;
  }
  volatile double m200_scaled = 0.00016927 * sequence;
  volatile double m200 = m200_scaled + 0.45666666;
  volatile double m10_scaled = 8.73809524e-05 * sequence;
  volatile double m10 = m10_scaled + 1.89833333;
  volatile double difference = m200 - m10;
  volatile double slope = difference / 190.0;
  volatile double intercept_term = 200.0 * slope;
  volatile double intercept = m200 - intercept_term;
  volatile double step_term = static_cast<double>(steps) * slope;
  volatile double result = step_term + intercept;
  return result;
}

std::vector<float> make_flux2_klein_schedule(
    std::uint32_t steps, std::uint64_t image_sequence_length) {
  if (steps == 0U)
    fail("FLUX.2 schedule requires at least one step");
  const auto points = steps + 1U;
  if (points == 0U)
    fail("FLUX.2 schedule point count overflow");
  const auto mu = flux2_empirical_mu(image_sequence_length, steps);
  const auto exponential = static_cast<float>(std::exp(mu));
  std::vector<float> result;
  result.reserve(points);
  const auto step = -1.0F / static_cast<float>(steps);
  const auto halfway = points / 2U;
  for (std::uint32_t index = 0; index < points; ++index) {
    const auto base = index < halfway
                          ? std::fma(step, static_cast<float>(index), 1.0F)
                          : std::fma(-step,
                                     static_cast<float>(points - index - 1U),
                                     0.0F);
    volatile float reciprocal = 1.0F / base;
    volatile float odds = reciprocal - 1.0F;
    volatile float denominator = exponential + odds;
    // torch's CPU true_divide path evaluates this scalar/tensor form as a
    // float reciprocal followed by a float multiply.
    volatile float denominator_reciprocal = 1.0F / denominator;
    volatile float shifted = exponential * denominator_reciprocal;
    result.push_back(static_cast<float>(shifted));
  }
  if (result.front() != 1.0F || result.back() != 0.0F)
    fail("FLUX.2 schedule lost an endpoint");
  return result;
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

void h3_res_multistep_step_in_place(
    std::span<float> sample, std::span<const float> model_output,
    std::uint64_t row_width, std::uint64_t first_generated_row,
    float timestep, float sigma, float sigma_next,
    H3ResMultistepState &state) {
  if (row_width == 0U || sample.size() != model_output.size() ||
      sample.size() % row_width != 0U ||
      first_generated_row > sample.size() / row_width ||
      !(sigma > 0.0F) || sigma_next < 0.0F || sigma_next > sigma ||
      !std::isfinite(timestep))
    fail("invalid H3 RES multistep scheduler step");

  const auto first = first_generated_row * row_width;
  const auto generated_values = sample.size() - first;
  if (state.has_previous &&
      state.previous_denoised.size() != generated_values)
    fail("H3 RES multistep state geometry changed between evaluations");

  // The H3 velocity wrapper observed by the sampler is
  // denoised = sample + sigma * velocity.  Preserve the individual F32
  // boundaries used by the source model wrapper before mutating `sample`.
  std::vector<float> current_denoised(generated_values);
  volatile float sigma_from_timestep = 1.0F - timestep;
  for (std::size_t index = 0U; index < generated_values; ++index) {
    volatile float scaled_velocity =
        sigma_from_timestep * model_output[first + index];
    volatile float denoised = sample[first + index] + scaled_velocity;
    current_denoised[index] = denoised;
  }

  if (sigma_next == 0.0F || !state.has_previous) {
    // ComfyUI's first and terminal RES updates fall back to Euler.  Use the
    // already gated H3 expression so its float32 rounding remains unchanged.
    h3_euler_step_in_place(sample, model_output, row_width,
                           first_generated_row, timestep, sigma, sigma_next);
  } else {
    if (!(state.previous_sigma_down > 0.0F) ||
        !std::isfinite(state.previous_sigma_down))
      fail("H3 RES multistep previous sigma is invalid");

    if (!(state.previous_sigma > 0.0F) ||
        !std::isfinite(state.previous_sigma))
      fail("H3 RES multistep previous released sigma is invalid");

    // sample_res_multistep, eta=0, cfg_pp=false. Its scalar tensors are F32;
    // volatile barriers below retain those source-visible operation
    // boundaries instead of permitting contraction in a release build.
    volatile float t = -std::log(sigma);
    volatile float t_old = -std::log(state.previous_sigma_down);
    volatile float t_next = -std::log(sigma_next);
    volatile float t_prev = -std::log(state.previous_sigma);
    volatile float h = t_next - t;
    volatile float c2 = (t_prev - t_old) / h;
    volatile float negative_h = -h;
    volatile float phi1 = std::expm1(negative_h) / negative_h;
    volatile float phi2 = (phi1 - 1.0F) / negative_h;
    volatile float b1 = phi1 - phi2 / c2;
    volatile float b2 = phi2 / c2;
    if (std::isnan(b1))
      b1 = 0.0F;
    if (std::isnan(b2))
      b2 = 0.0F;
    if (!std::isfinite(h) || !std::isfinite(b1) || !std::isfinite(b2))
      fail("H3 RES multistep coefficients are nonfinite");
    volatile float decay = std::exp(negative_h);
    for (std::size_t index = 0U; index < generated_values; ++index) {
      volatile float current_term = b1 * current_denoised[index];
      volatile float previous_term = b2 * state.previous_denoised[index];
      volatile float combined = current_term + previous_term;
      volatile float history = h * combined;
      volatile float retained = decay * sample[first + index];
      volatile float updated = retained + history;
      sample[first + index] = updated;
    }
  }

  state.previous_denoised = std::move(current_denoised);
  state.previous_sigma = sigma;
  state.previous_sigma_down = sigma_next;
  state.has_previous = true;
}

void h3_av_audio_carry_to_model_input(
    std::span<float> model_input, std::span<const float> carry,
    std::uint64_t row_width, std::uint64_t first_generated_row,
    float video_sigma, float audio_sigma) {
  validate_audio_geometry(carry, row_width, first_generated_row);
  if (model_input.size() != carry.size() || !(video_sigma > 0.0F) ||
      audio_sigma < 0.0F || audio_sigma > video_sigma)
    fail("invalid H3 AV audio carry conversion");
  volatile float ratio = audio_sigma / video_sigma;
  const auto first = first_generated_row * row_width;
  for (std::size_t index = 0U; index < carry.size(); ++index) {
    if (index < first) {
      model_input[index] = carry[index];
    } else {
      volatile float physical = carry[index] * ratio;
      model_input[index] = physical;
    }
  }
}

void h3_res_multistep_av_audio_step_in_place(
    std::span<float> carry, std::span<const float> model_input,
    std::span<const float> physical_velocity, std::uint64_t row_width,
    std::uint64_t first_generated_row, float video_sigma, float audio_sigma,
    float video_sigma_next, float audio_scale,
    H3ResMultistepState &state) {
  validate_audio_geometry(carry, row_width, first_generated_row);
  if (model_input.size() != carry.size() ||
      physical_velocity.size() != carry.size() || !(video_sigma > 0.0F) ||
      audio_sigma < 0.0F || audio_sigma > video_sigma ||
      video_sigma_next < 0.0F || video_sigma_next > video_sigma ||
      !(audio_scale > 0.0F) || !std::isfinite(audio_scale))
    fail("invalid H3 AV RES multistep scheduler step");

  std::vector<float> carry_velocity(carry.size(), 0.0F);
  volatile float scale_minus_one = audio_scale - 1.0F;
  volatile float velocity_factor_term = scale_minus_one * audio_sigma;
  volatile float velocity_factor = 1.0F + velocity_factor_term;
  const auto first = first_generated_row * row_width;
  for (std::size_t index = first; index < carry.size(); ++index) {
    volatile float source_term = scale_minus_one * model_input[index];
    volatile float velocity_term = velocity_factor * physical_velocity[index];
    volatile float effective_velocity = source_term + velocity_term;
    carry_velocity[index] = effective_velocity;
  }
  volatile float video_timestep = 1.0F - video_sigma;
  h3_res_multistep_step_in_place(
      carry, carry_velocity, row_width, first_generated_row, video_timestep,
      video_sigma, video_sigma_next, state);
}

void h3_av_audio_carry_to_physical_in_place(
    std::span<float> carry, std::uint64_t row_width,
    std::uint64_t first_generated_row, float video_sigma, float audio_sigma,
    float audio_scale) {
  validate_audio_geometry(carry, row_width, first_generated_row);
  if (video_sigma < 0.0F || audio_sigma < 0.0F ||
      !(audio_scale > 0.0F) || !std::isfinite(audio_scale) ||
      ((video_sigma == 0.0F) != (audio_sigma == 0.0F)) ||
      (video_sigma > 0.0F && audio_sigma > video_sigma))
    fail("invalid H3 AV audio output conversion");
  volatile float ratio = video_sigma == 0.0F
                             ? 1.0F / audio_scale
                             : audio_sigma / video_sigma;
  const auto first = first_generated_row * row_width;
  for (std::size_t index = first; index < carry.size(); ++index) {
    volatile float physical = carry[index] * ratio;
    carry[index] = physical;
  }
}

} // namespace dif::sampling
