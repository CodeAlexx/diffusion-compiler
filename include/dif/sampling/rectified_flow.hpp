#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace dif::sampling {

struct ShiftedSigmaSchedule {
  std::vector<float> sigmas;
  std::vector<float> timesteps;
};

struct H3AVSigmaSchedule {
  std::vector<float> video_sigmas;
  std::vector<float> audio_sigmas;
};

// Builds a float32 grid from 1 to 0 (terminal zero included), applies
// shift*sigma/(1+(shift-1)*sigma), collapses consecutive float32 duplicates,
// and exposes t=1-sigma for every model evaluation.
ShiftedSigmaSchedule make_exponential_shifted_schedule(std::uint32_t points,
                                                       float shift);

// Reproduces ComfyUI's `simple` scheduler for MiniMax H3.  The released video
// sigmas are selected from ModelSamplingDiscreteFlow's 1000-entry float32
// table; audio sigmas are then mapped from those released video sigmas exactly
// as MiniMaxH3Model.forward does, rather than independently resampling a second
// table.
H3AVSigmaSchedule make_h3_simple_av_schedule(std::uint32_t evaluations);

// Source-faithful MiniMax-H3 data-ward Euler update. The denoised estimate
// deliberately recovers sigma from the rounded timestep while the blend ratio
// uses the sigma grid, and every arithmetic boundary is rounded to F32.
void h3_euler_step_in_place(std::span<float> sample,
                            std::span<const float> model_output,
                            std::uint64_t row_width,
                            std::uint64_t first_generated_row,
                            float timestep, float sigma, float sigma_next);

// State carried by ComfyUI's deterministic res_multistep sampler (eta=0).
// The stored denoised tensor covers generated rows only; keyframe-condition
// rows remain clamped by the caller's row boundary.
struct H3ResMultistepState {
  std::vector<float> previous_denoised;
  float previous_sigma{};
  float previous_sigma_down{};
  bool has_previous{};
};

// Source-faithful second-order RES multistep update used by the frozen H3
// ComfyUI comparator.  The first and terminal updates are Euler, matching
// sample_res_multistep(..., eta=0); interior updates carry the prior denoised
// prediction and sigma-down value in `state`.
void h3_res_multistep_step_in_place(
    std::span<float> sample, std::span<const float> model_output,
    std::uint64_t row_width, std::uint64_t first_generated_row,
    float timestep, float sigma, float sigma_next,
    H3ResMultistepState &state);

// ComfyUI packs H3 video and audio into one sampler tensor.  The audio slice is
// a carry variable on the video sigma schedule, while the transformer consumes
// the physical audio latent on its own shifted schedule.  These helpers expose
// that boundary without making the transformer or the generic RES integrator
// aware of nested tensors.
void h3_av_audio_carry_to_model_input(
    std::span<float> model_input, std::span<const float> carry,
    std::uint64_t row_width, std::uint64_t first_generated_row,
    float video_sigma, float audio_sigma);

void h3_res_multistep_av_audio_step_in_place(
    std::span<float> carry, std::span<const float> model_input,
    std::span<const float> physical_velocity, std::uint64_t row_width,
    std::uint64_t first_generated_row, float video_sigma, float audio_sigma,
    float video_sigma_next, float audio_scale,
    H3ResMultistepState &state);

void h3_av_audio_carry_to_physical_in_place(
    std::span<float> carry, std::uint64_t row_width,
    std::uint64_t first_generated_row, float video_sigma, float audio_sigma,
    float audio_scale);

} // namespace dif::sampling
