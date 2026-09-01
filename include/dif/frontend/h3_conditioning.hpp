#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace dif::frontend {

enum class H3KeyframeAnchor { First, Last };

enum class H3ReferenceKind { Image, Video, Audio };

// Encoded reference geometry consumed by the Ref2VA packed-layout frontend.
// Media decoding and VAE encoding establish these dimensions before packing;
// the shared denoiser runtime only sees the resulting modality rows.
struct H3ReferenceGeometry {
  H3ReferenceKind kind{};
  std::uint64_t num_latent_frames{};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t num_audio_latents{};
};

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

// Padless [text | ordered reference blocks | target audio | target video]
// layout from the creator Ref2VA path. A video reference packs its optional
// soundtrack immediately before its video rows and both share one rotary-time
// origin. Each reference retains its own encoded spatial geometry.
H3PackedLayout make_h3_ref2va_layout(
    std::span<const std::int32_t> text_token_tags,
    std::span<const H3ReferenceGeometry> references,
    std::uint64_t num_latent_frames, std::uint64_t latent_height,
    std::uint64_t latent_width, std::uint64_t num_audio_latents,
    std::uint64_t patch_t, std::uint64_t patch_h, std::uint64_t patch_w);

struct H3RowTimestepPlan {
  std::vector<float> timesteps;
  std::vector<std::int32_t> timestep_indices;
  std::vector<std::int32_t> adaln_indices;
};

// Fixed-width schedule table consumed by a prepared H3 executable. Each
// evaluation stores the creator's sorted distinct row-timestep values and
// repeats the final value only to fill the graph's fixed number of slots.
// `make_h3_row_timestep_plan` indices remain valid because padding never
// changes the sorted prefix.
struct H3ScheduleTimestepTable {
  std::uint64_t evaluations{};
  std::uint64_t tables{};
  std::vector<float> timesteps;
};

H3ScheduleTimestepTable make_h3_schedule_timestep_table(
    std::span<const float> video_sigmas,
    std::span<const float> audio_sigmas, bool has_condition_video,
    bool has_condition_audio, float condition_video_floor = 0.999F,
    float condition_audio_timestep = 1.0F);

H3RowTimestepPlan make_h3_row_timestep_plan(
    const H3PackedLayout &layout, float video_timestep,
    float audio_timestep, float condition_video_timestep,
    float condition_audio_timestep);

} // namespace dif::frontend
