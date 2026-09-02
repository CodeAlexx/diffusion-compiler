#pragma once

#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <filesystem>

namespace dif::frontend {

// Released H3 video packing: [1, 24, T, H, W] ->
// [T * (H / 2) * (W / 2), 96], with (C, patch-y, patch-x) columns.
runtime::Tensor pack_h3_video_latent(const runtime::Tensor &video_latent,
                                     std::uint64_t patch_height = 2U,
                                     std::uint64_t patch_width = 2U);

// Released H3 audio packing: [1, 32, 2, T] -> [2 * T, 32], with all
// channel-zero time rows followed by all channel-one time rows.
runtime::Tensor pack_h3_audio_latent(const runtime::Tensor &audio_latent);

// Inverse of the released H3 channel-slowest (C, pt, ph, pw) row packing.
runtime::Tensor unpack_h3_video_rows(const runtime::Tensor &video_rows,
                                     std::uint64_t condition_rows,
                                     std::uint64_t latent_frames,
                                     std::uint64_t latent_height,
                                     std::uint64_t latent_width,
                                     std::uint64_t patch_height = 2U,
                                     std::uint64_t patch_width = 2U);

// Inverse of the released stereo-major audio packing. The returned tensor is
// [2, 32, T]; when requested, latent normalization is inverted per channel.
runtime::Tensor unpack_h3_audio_rows(const runtime::Tensor &audio_rows,
                                     std::uint64_t condition_rows,
                                     std::uint64_t num_audio_latents,
                                     bool denormalize);

// Process-boundary handoff consumed directly by the retained Serenity H3
// decode-only runtime. Both tensors remain normalized F32 row-space states.
void write_h3_latent_handoff(const std::filesystem::path &path,
                             const runtime::Tensor &video_rows,
                             const runtime::Tensor &audio_rows);

} // namespace dif::frontend
