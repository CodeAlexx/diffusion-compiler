#pragma once

#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <filesystem>

namespace dif::frontend {

std::uint64_t h3_motion_context_steps(std::uint64_t context_frames);
std::uint64_t h3_motion_context_pixel_frames(std::uint64_t latent_steps);
std::uint64_t h3_motion_context_audio_latents(std::uint64_t pixel_frames);
std::uint64_t h3_motion_context_endpoint_steps(
    std::uint64_t requested_pixel_frames, std::uint64_t available_latent_steps);

struct H3MotionContext {
  runtime::Tensor video_rows; // normalized F32 [context_steps * rows/frame,96]
  runtime::Tensor audio_rows; // normalized F32 [2 * context_audio_latents,32]
  std::uint64_t source_video_latent_frames{};
  std::uint64_t source_audio_latents{};
  double source_audio_overhang{};
  std::uint64_t context_frames{};
};

// Reads the compatible Mojo schema-1 compact SafeTensors artifact, or a full
// normalized video_state_rows/audio_state_rows handoff without condition
// prefixes. Only selected tail bytes are copied; no device allocation occurs.
H3MotionContext load_h3_motion_context(
    const std::filesystem::path &path, std::uint64_t width,
    std::uint64_t height, std::uint64_t context_frames,
    std::uint64_t patch_height = 2U, std::uint64_t patch_width = 2U);

struct H3MotionState {
  runtime::Tensor video_rows;
  runtime::Tensor audio_rows;
};

// Uses supplied noise to preserve the caller's established RNG draw order.
// Video condition: 0.999F * clean + (1.0F - 0.999F) * noise; audio is exact.
// Prefixes stay fixed under the existing H3 scheduler condition-row masks.
H3MotionState prepare_h3_motion_context_state(
    const H3MotionContext &context,
    const runtime::Tensor &condition_video_noise,
    const runtime::Tensor &target_video_noise,
    const runtime::Tensor &target_audio_noise);

// Strip condition prefixes and save the largest admitted tail ending nearest
// the delivered endpoint. Returns the saved overlap (5,22,39). Source tensors
// remain normalized row-space F32. Never decode/re-encode the preceding clip.
std::uint64_t save_h3_motion_context_tail(
    const std::filesystem::path &path, const runtime::Tensor &video_rows,
    const runtime::Tensor &audio_rows, std::uint64_t num_latent_frames,
    std::uint64_t num_audio_latents, std::uint64_t width, std::uint64_t height,
    std::uint64_t delivered_endpoint_frames,
    std::uint64_t condition_video_rows = 0U,
    std::uint64_t condition_audio_rows = 0U,
    std::uint64_t patch_height = 2U, std::uint64_t patch_width = 2U);

struct H3MotionTrim {
  std::uint64_t trim_start_frames{};
  std::uint64_t trim_start_audio_samples{};
  std::uint64_t output_frames{};
  std::uint64_t output_audio_samples{};
  std::uint64_t continuation_endpoint_frames{};
};

// Native 24 FPS / 32 kHz delivery; output_frames excludes the overlap.
// Zero context_frames is a cap-only delivery with no continuation overlap.
H3MotionTrim make_h3_motion_trim(std::uint64_t context_frames,
                                std::uint64_t output_frames);

} // namespace dif::frontend
