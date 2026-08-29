#include "dif/frontend/h3_conditioning.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace dif::frontend {
namespace {

constexpr std::int32_t kVideoTag = 0;
constexpr std::int32_t kAudioTag = 2;
constexpr std::uint64_t kAudioChannels = 2U;
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr std::array<std::uint64_t, 5> kFramesPerLatent = {1U, 4U, 4U,
                                                            4U, 4U};

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left)
    fail(std::string("H3 packed layout overflows ") + label);
  return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    fail(std::string("H3 packed layout overflows ") + label);
  return left + right;
}

std::vector<double> spatial_grid(std::uint64_t dim, std::uint64_t patch,
                                 double square_root_area) {
  const auto count = dim / patch;
  const auto ratio = static_cast<double>(dim) / square_root_area;
  const auto left = (1.0 - ratio) / 2.0;
  const auto step = ratio / static_cast<double>(count);
  std::vector<double> grid(count);
  for (std::uint64_t index = 0; index < count; ++index) {
    // NumPy's endpoint=False grid is start + arange(count) * step.
    volatile double offset = static_cast<double>(index) * step;
    volatile double coordinate = left + offset;
    grid[index] = coordinate * 32.0;
  }
  return grid;
}

std::vector<double> temporal_grid(std::uint64_t frames, double origin) {
  std::vector<double> grid(frames);
  auto current = origin;
  for (std::uint64_t frame = 0; frame < frames; ++frame) {
    grid[frame] = current;
    current += kFrameRescale * static_cast<double>(
                                   kFramesPerLatent[frame %
                                                    kFramesPerLatent.size()]);
  }
  return grid;
}

double temporal_span(std::uint64_t frames) {
  std::vector<double> spans(frames);
  for (std::uint64_t frame = 0; frame < frames; ++frame)
    spans[frame] =
        kFrameRescale * static_cast<double>(
                            kFramesPerLatent[frame % kFramesPerLatent.size()]);
  // Pairwise reduction follows the source NumPy summation contract closely
  // enough to preserve the subsequent F32 position boundary at native sizes.
  while (spans.size() > 1U) {
    std::vector<double> reduced;
    reduced.reserve((spans.size() + 1U) / 2U);
    std::size_t index = 0U;
    for (; index + 1U < spans.size(); index += 2U)
      reduced.push_back(spans[index] + spans[index + 1U]);
    if (index < spans.size())
      reduced.push_back(spans[index]);
    spans = std::move(reduced);
  }
  return spans.empty() ? 0.0 : spans.front();
}

void set_position(std::vector<float> &positions, std::uint64_t row,
                  double time, double height, double width) {
  positions[row * 3U] = static_cast<float>(time);
  positions[row * 3U + 1U] = static_cast<float>(height);
  positions[row * 3U + 2U] = static_cast<float>(width);
}

} // namespace

H3PackedLayout make_h3_t2va_layout(
    std::span<const std::int32_t> text_token_tags,
    std::uint64_t num_latent_frames, std::uint64_t latent_height,
    std::uint64_t latent_width, std::uint64_t num_audio_latents,
    std::uint64_t patch_t, std::uint64_t patch_h, std::uint64_t patch_w,
    std::span<const H3KeyframeAnchor> keyframe_anchors) {
  if (text_token_tags.empty() || num_latent_frames == 0U ||
      latent_height == 0U || latent_width == 0U ||
      num_audio_latents == 0U || patch_t != 1U || patch_h == 0U ||
      patch_w == 0U || latent_height % patch_h != 0U ||
      latent_width % patch_w != 0U)
    fail("invalid H3 t2va/fl2va packed-layout geometry");
  for (const auto tag : text_token_tags) {
    if (tag != 0 && tag != 1)
      fail("H3 text token tags must be video (0) or text (1)");
  }
  const auto rows_per_frame = checked_multiply(
      latent_height / patch_h, latent_width / patch_w, "rows per frame");
  const auto text_rows = static_cast<std::uint64_t>(text_token_tags.size());
  const auto condition_rows = checked_multiply(
      static_cast<std::uint64_t>(keyframe_anchors.size()), rows_per_frame,
      "conditioning rows");
  const auto audio_rows =
      checked_multiply(num_audio_latents, kAudioChannels, "audio rows");
  const auto video_target_rows = checked_multiply(
      num_latent_frames, rows_per_frame, "target video rows");
  const auto sequence = checked_add(
      checked_add(checked_add(text_rows, condition_rows, "sequence"),
                  audio_rows, "sequence"),
      video_target_rows, "sequence");
  if (sequence > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int32_t>::max()))
    fail("H3 packed layout exceeds I32 index range");

  const auto condition_start = text_rows;
  const auto audio_start = condition_start + condition_rows;
  const auto video_start = audio_start + audio_rows;
  const auto latent_area =
      checked_multiply(latent_height, latent_width, "latent area");
  const auto square_root_area = std::sqrt(static_cast<double>(latent_area));
  const auto height_grid =
      spatial_grid(latent_height, patch_h, square_root_area);
  const auto width_grid = spatial_grid(latent_width, patch_w, square_root_area);

  H3PackedLayout layout;
  layout.sequence_length = sequence;
  layout.position_ids.assign(sequence * 3U, 0.0F);
  layout.token_tags.resize(sequence);
  layout.text_map.assign(sequence, -1);
  layout.video_map.assign(sequence, -1);
  layout.audio_map.assign(sequence, -1);
  layout.num_condition_video_rows = condition_rows;
  layout.num_condition_audio_rows = 0U;

  layout.text_indices.reserve(text_rows);
  for (std::uint64_t row = 0; row < text_rows; ++row) {
    layout.text_indices.push_back(static_cast<std::int32_t>(row));
    layout.text_map[row] = static_cast<std::int32_t>(row);
    layout.token_tags[row] = text_token_tags[row];
    set_position(layout.position_ids, row, static_cast<double>(row), 0.0,
                 0.0);
  }

  layout.video_indices.reserve(condition_rows + video_target_rows);
  for (std::size_t condition = 0; condition < keyframe_anchors.size();
       ++condition) {
    const auto time = keyframe_anchors[condition] == H3KeyframeAnchor::First
                          ? static_cast<double>(text_rows)
                          : static_cast<double>(text_rows) +
                                temporal_span(num_latent_frames) -
                                kFrameRescale;
    for (std::uint64_t spatial = 0; spatial < rows_per_frame; ++spatial) {
      const auto row = condition_start + condition * rows_per_frame + spatial;
      const auto y = spatial / width_grid.size();
      const auto x = spatial % width_grid.size();
      layout.video_map[row] =
          static_cast<std::int32_t>(layout.video_indices.size());
      layout.video_indices.push_back(static_cast<std::int32_t>(row));
      layout.token_tags[row] = kVideoTag;
      set_position(layout.position_ids, row, time, height_grid[y],
                   width_grid[x]);
    }
  }

  layout.audio_indices.reserve(audio_rows);
  for (std::uint64_t channel = 0; channel < kAudioChannels; ++channel) {
    for (std::uint64_t latent = 0; latent < num_audio_latents; ++latent) {
      const auto row = audio_start + channel * num_audio_latents + latent;
      layout.audio_map[row] =
          static_cast<std::int32_t>(layout.audio_indices.size());
      layout.audio_indices.push_back(static_cast<std::int32_t>(row));
      layout.token_tags[row] = kAudioTag;
      set_position(layout.position_ids, row,
                   static_cast<double>(text_rows + latent), 0.0,
                   channel == 0U ? width_grid.front() : width_grid.back());
    }
  }

  const auto times = temporal_grid(num_latent_frames,
                                   static_cast<double>(text_rows));
  for (std::uint64_t frame = 0; frame < num_latent_frames; ++frame) {
    for (std::uint64_t spatial = 0; spatial < rows_per_frame; ++spatial) {
      const auto row = video_start + frame * rows_per_frame + spatial;
      const auto y = spatial / width_grid.size();
      const auto x = spatial % width_grid.size();
      layout.video_map[row] =
          static_cast<std::int32_t>(layout.video_indices.size());
      layout.video_indices.push_back(static_cast<std::int32_t>(row));
      layout.token_tags[row] = kVideoTag;
      set_position(layout.position_ids, row, times[frame], height_grid[y],
                   width_grid[x]);
    }
  }
  return layout;
}

H3RowTimestepPlan make_h3_row_timestep_plan(
    const H3PackedLayout &layout, float video_timestep,
    float audio_timestep, float condition_video_timestep,
    float condition_audio_timestep) {
  if (layout.token_tags.size() != layout.sequence_length)
    fail("H3 row-timestep layout is inconsistent");
  for (const auto timestep : {video_timestep, audio_timestep,
                              condition_video_timestep,
                              condition_audio_timestep}) {
    if (!std::isfinite(timestep))
      fail("H3 row timesteps must be finite");
  }
  std::vector<float> row_timesteps(layout.sequence_length, video_timestep);
  for (std::uint64_t index = 0; index < layout.num_condition_video_rows;
       ++index)
    row_timesteps[static_cast<std::size_t>(layout.video_indices[index])] =
        condition_video_timestep;
  for (std::uint64_t index = 0; index < layout.num_condition_audio_rows;
       ++index)
    row_timesteps[static_cast<std::size_t>(layout.audio_indices[index])] =
        condition_audio_timestep;
  for (std::uint64_t index = layout.num_condition_audio_rows;
       index < layout.audio_indices.size(); ++index)
    row_timesteps[static_cast<std::size_t>(layout.audio_indices[index])] =
        audio_timestep;

  H3RowTimestepPlan plan;
  plan.timesteps = row_timesteps;
  std::sort(plan.timesteps.begin(), plan.timesteps.end());
  plan.timesteps.erase(
      std::unique(plan.timesteps.begin(), plan.timesteps.end()),
      plan.timesteps.end());
  plan.timestep_indices.resize(layout.sequence_length);
  plan.adaln_indices.resize(layout.sequence_length);
  for (std::uint64_t row = 0; row < layout.sequence_length; ++row) {
    const auto found = std::lower_bound(plan.timesteps.begin(),
                                        plan.timesteps.end(),
                                        row_timesteps[row]);
    const auto timestep_index =
        static_cast<std::int32_t>(found - plan.timesteps.begin());
    plan.timestep_indices[row] = timestep_index;
    plan.adaln_indices[row] =
        timestep_index * 3 + std::max(layout.token_tags[row], 0);
  }
  return plan;
}

} // namespace dif::frontend
