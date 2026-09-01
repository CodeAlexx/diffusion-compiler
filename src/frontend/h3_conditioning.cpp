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

double sequential_temporal_span(std::uint64_t frames) {
  // packing_ref2va.py deliberately uses Python's sequential sum here, while
  // packing.py uses NumPy's pairwise reduction for FL2VA keyframe placement.
  // The two differ in the last F64 ulp at longer reference lengths.
  double span = 0.0;
  for (std::uint64_t frame = 0; frame < frames; ++frame)
    span += kFrameRescale * static_cast<double>(
                                kFramesPerLatent[frame %
                                                 kFramesPerLatent.size()]);
  return span;
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

H3PackedLayout make_h3_ref2va_layout(
    std::span<const std::int32_t> text_token_tags,
    std::span<const H3ReferenceGeometry> references,
    std::uint64_t num_latent_frames, std::uint64_t latent_height,
    std::uint64_t latent_width, std::uint64_t num_audio_latents,
    std::uint64_t patch_t, std::uint64_t patch_h, std::uint64_t patch_w) {
  if (text_token_tags.empty() || references.empty() ||
      num_latent_frames == 0U || latent_height == 0U ||
      latent_width == 0U || num_audio_latents == 0U || patch_t != 1U ||
      patch_h == 0U || patch_w == 0U || latent_height % patch_h != 0U ||
      latent_width % patch_w != 0U)
    fail("invalid H3 Ref2VA packed-layout geometry");
  for (const auto tag : text_token_tags) {
    if (tag != 0 && tag != 1)
      fail("H3 text token tags must be video (0) or text (1)");
  }

  const auto reference_video_rows = [&](const H3ReferenceGeometry &reference) {
    if (reference.kind == H3ReferenceKind::Audio)
      return std::uint64_t{0U};
    const auto spatial = checked_multiply(
        reference.latent_height / patch_h,
        reference.latent_width / patch_w, "reference rows per frame");
    return checked_multiply(reference.num_latent_frames, spatial,
                            "reference video rows");
  };
  const auto reference_audio_rows = [&](const H3ReferenceGeometry &reference) {
    return checked_multiply(reference.num_audio_latents, kAudioChannels,
                            "reference audio rows");
  };

  std::uint64_t num_condition_video_rows = 0U;
  std::uint64_t num_condition_audio_rows = 0U;
  for (const auto &reference : references) {
    if (reference.kind == H3ReferenceKind::Image) {
      if (reference.num_latent_frames != 1U ||
          reference.latent_height == 0U || reference.latent_width == 0U ||
          reference.num_audio_latents != 0U ||
          reference.latent_height % patch_h != 0U ||
          reference.latent_width % patch_w != 0U)
        fail("invalid H3 Ref2VA image-reference geometry");
    } else if (reference.kind == H3ReferenceKind::Video) {
      if (reference.num_latent_frames == 0U ||
          reference.latent_height == 0U || reference.latent_width == 0U ||
          reference.latent_height % patch_h != 0U ||
          reference.latent_width % patch_w != 0U)
        fail("invalid H3 Ref2VA video-reference geometry");
    } else if (reference.kind == H3ReferenceKind::Audio) {
      if (reference.num_latent_frames != 0U ||
          reference.latent_height != 0U || reference.latent_width != 0U ||
          reference.num_audio_latents == 0U)
        fail("invalid H3 Ref2VA audio-reference geometry");
    }
    num_condition_video_rows = checked_add(
        num_condition_video_rows, reference_video_rows(reference),
        "condition video rows");
    num_condition_audio_rows = checked_add(
        num_condition_audio_rows, reference_audio_rows(reference),
        "condition audio rows");
  }

  const auto target_rows_per_frame = checked_multiply(
      latent_height / patch_h, latent_width / patch_w,
      "target rows per frame");
  const auto target_video_rows = checked_multiply(
      num_latent_frames, target_rows_per_frame, "target video rows");
  const auto target_audio_rows = checked_multiply(
      num_audio_latents, kAudioChannels, "target audio rows");
  const auto text_rows = static_cast<std::uint64_t>(text_token_tags.size());
  auto sequence = checked_add(text_rows, num_condition_video_rows,
                              "Ref2VA sequence");
  sequence = checked_add(sequence, num_condition_audio_rows,
                         "Ref2VA sequence");
  sequence = checked_add(sequence, target_audio_rows, "Ref2VA sequence");
  sequence = checked_add(sequence, target_video_rows, "Ref2VA sequence");
  if (sequence > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int32_t>::max()))
    fail("H3 Ref2VA packed layout exceeds I32 index range");

  const auto target_area =
      checked_multiply(latent_height, latent_width, "target latent area");
  const auto target_sqrt_area =
      std::sqrt(static_cast<double>(target_area));
  const auto target_height_grid =
      spatial_grid(latent_height, patch_h, target_sqrt_area);
  const auto target_width_grid =
      spatial_grid(latent_width, patch_w, target_sqrt_area);

  H3PackedLayout layout;
  layout.sequence_length = sequence;
  layout.position_ids.assign(sequence * 3U, 0.0F);
  layout.token_tags.resize(sequence);
  layout.text_map.assign(sequence, -1);
  layout.video_map.assign(sequence, -1);
  layout.audio_map.assign(sequence, -1);
  layout.num_condition_video_rows = num_condition_video_rows;
  layout.num_condition_audio_rows = num_condition_audio_rows;
  layout.text_indices.reserve(text_rows);
  layout.video_indices.reserve(num_condition_video_rows + target_video_rows);
  layout.audio_indices.reserve(num_condition_audio_rows + target_audio_rows);

  for (std::uint64_t row = 0U; row < text_rows; ++row) {
    layout.text_indices.push_back(static_cast<std::int32_t>(row));
    layout.text_map[row] = static_cast<std::int32_t>(row);
    layout.token_tags[row] = text_token_tags[row];
    set_position(layout.position_ids, row, static_cast<double>(row), 0.0,
                 0.0);
  }

  auto add_audio_block = [&](std::uint64_t &cursor,
                             std::uint64_t audio_latents, double origin,
                             const std::vector<double> &width_grid) {
    for (std::uint64_t channel = 0U; channel < kAudioChannels; ++channel) {
      for (std::uint64_t latent = 0U; latent < audio_latents; ++latent) {
        const auto row = cursor++;
        layout.audio_map[row] =
            static_cast<std::int32_t>(layout.audio_indices.size());
        layout.audio_indices.push_back(static_cast<std::int32_t>(row));
        layout.token_tags[row] = kAudioTag;
        set_position(layout.position_ids, row,
                     origin + static_cast<double>(latent), 0.0,
                     channel == 0U ? width_grid.front() : width_grid.back());
      }
    }
  };

  auto add_video_block = [&](std::uint64_t &cursor,
                             const H3ReferenceGeometry &geometry,
                             double origin) {
    const auto area = checked_multiply(geometry.latent_height,
                                       geometry.latent_width,
                                       "reference latent area");
    const auto sqrt_area = std::sqrt(static_cast<double>(area));
    const auto height_grid =
        spatial_grid(geometry.latent_height, patch_h, sqrt_area);
    const auto width_grid =
        spatial_grid(geometry.latent_width, patch_w, sqrt_area);
    const auto times = temporal_grid(geometry.num_latent_frames, origin);
    const auto rows_per_frame = checked_multiply(
        height_grid.size(), width_grid.size(), "reference rows per frame");
    for (std::uint64_t frame = 0U; frame < geometry.num_latent_frames;
         ++frame) {
      for (std::uint64_t spatial = 0U; spatial < rows_per_frame; ++spatial) {
        const auto row = cursor++;
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
    return width_grid;
  };

  std::uint64_t cursor = text_rows;
  double rotary_time = static_cast<double>(text_rows);
  for (const auto &reference : references) {
    if (reference.kind == H3ReferenceKind::Image) {
      add_video_block(cursor, reference, rotary_time);
      rotary_time += 1.0;
    } else if (reference.kind == H3ReferenceKind::Audio) {
      add_audio_block(cursor, reference.num_audio_latents, rotary_time,
                      target_width_grid);
      rotary_time += static_cast<double>(reference.num_audio_latents);
    } else {
      const auto area = checked_multiply(reference.latent_height,
                                         reference.latent_width,
                                         "reference latent area");
      const auto reference_width_grid = spatial_grid(
          reference.latent_width, patch_w,
          std::sqrt(static_cast<double>(area)));
      add_audio_block(cursor, reference.num_audio_latents, rotary_time,
                      reference_width_grid);
      add_video_block(cursor, reference, rotary_time);
      rotary_time += std::max(
          static_cast<double>(reference.num_audio_latents),
          sequential_temporal_span(reference.num_latent_frames));
    }
  }

  add_audio_block(cursor, num_audio_latents, rotary_time,
                  target_width_grid);
  H3ReferenceGeometry target{H3ReferenceKind::Video, num_latent_frames,
                             latent_height, latent_width, 0U};
  add_video_block(cursor, target, rotary_time);
  if (cursor != sequence)
    fail("H3 Ref2VA packed layout cursor disagrees with sequence length");
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

H3ScheduleTimestepTable make_h3_schedule_timestep_table(
    std::span<const float> video_sigmas,
    std::span<const float> audio_sigmas, bool has_condition_video,
    bool has_condition_audio, float condition_video_floor,
    float condition_audio_timestep) {
  if (video_sigmas.size() < 2U || audio_sigmas.size() != video_sigmas.size())
    fail("H3 schedule timestep table requires matching sigma schedules");
  if (video_sigmas.back() != 0.0F || audio_sigmas.back() != 0.0F)
    fail("H3 sigma schedules must end at zero");
  for (const auto sigma : video_sigmas) {
    if (!std::isfinite(sigma) || sigma < 0.0F || sigma > 1.0F)
      fail("H3 video sigmas must be finite and within [0,1]");
  }
  for (const auto sigma : audio_sigmas) {
    if (!std::isfinite(sigma) || sigma < 0.0F || sigma > 1.0F)
      fail("H3 audio sigmas must be finite and within [0,1]");
  }
  for (const auto timestep : {condition_video_floor,
                              condition_audio_timestep}) {
    if (!std::isfinite(timestep) || timestep < 0.0F || timestep > 1.0F)
      fail("H3 condition timesteps must be finite and within [0,1]");
  }

  H3ScheduleTimestepTable table;
  table.evaluations = video_sigmas.size() - 1U;
  table.tables = 2U + static_cast<std::uint64_t>(has_condition_video) +
                 static_cast<std::uint64_t>(has_condition_audio);
  table.timesteps.reserve(table.evaluations * table.tables);
  for (std::size_t evaluation = 0U; evaluation < table.evaluations;
       ++evaluation) {
    const auto video_timestep = 1.0F - video_sigmas[evaluation];
    std::vector<float> local = {
        video_timestep, 1.0F - audio_sigmas[evaluation]};
    if (has_condition_video)
      local.push_back(std::max(video_timestep, condition_video_floor));
    if (has_condition_audio)
      local.push_back(condition_audio_timestep);
    std::sort(local.begin(), local.end());
    local.erase(std::unique(local.begin(), local.end()), local.end());
    while (local.size() < table.tables)
      local.push_back(local.back());
    table.timesteps.insert(table.timesteps.end(), local.begin(), local.end());
  }
  return table;
}

} // namespace dif::frontend
