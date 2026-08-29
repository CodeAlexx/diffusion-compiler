#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace dif::frontend {

enum class H3KeyframeAnchor { First, Last };

// Padless [text | keyframe conditions | target audio | target video] layout
// consumed by the H3 denoiser. Maps use IndexedUpdateRows inverse-map
// semantics: -1 preserves the destination row, otherwise the value selects a
// source row from the named modality.
struct H3PackedLayout {
  std::uint64_t sequence_length{};
  std::vector<float> position_ids;
  std::vector<std::int32_t> token_tags;
  std::vector<std::int32_t> text_indices;
  std::vector<std::int32_t> video_indices;
  std::vector<std::int32_t> audio_indices;
  std::vector<std::int32_t> text_map;
  std::vector<std::int32_t> video_map;
  std::vector<std::int32_t> audio_map;
  std::uint64_t num_condition_video_rows{};
  std::uint64_t num_condition_audio_rows{};
};

H3PackedLayout make_h3_t2va_layout(
    std::span<const std::int32_t> text_token_tags,
    std::uint64_t num_latent_frames, std::uint64_t latent_height,
    std::uint64_t latent_width, std::uint64_t num_audio_latents,
    std::uint64_t patch_t, std::uint64_t patch_h, std::uint64_t patch_w,
    std::span<const H3KeyframeAnchor> keyframe_anchors = {});

struct H3RowTimestepPlan {
  std::vector<float> timesteps;
  std::vector<std::int32_t> timestep_indices;
  std::vector<std::int32_t> adaln_indices;
};

H3RowTimestepPlan make_h3_row_timestep_plan(
    const H3PackedLayout &layout, float video_timestep,
    float audio_timestep, float condition_video_timestep,
    float condition_audio_timestep);

} // namespace dif::frontend
