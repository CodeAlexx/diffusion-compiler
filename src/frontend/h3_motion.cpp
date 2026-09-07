#include "dif/frontend/h3_motion.hpp"

#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace dif::frontend {
namespace {

std::uint64_t product(std::uint64_t a, std::uint64_t b) {
  if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
    fail("H3 motion geometry overflows");
  return a * b;
}

std::uint64_t sum(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    fail("H3 motion geometry overflows");
  return a + b;
}

std::uint64_t thirds(std::uint64_t numerator) {
  return numerator / 3U + (numerator % 3U == 2U ? 1U : 0U);
}

std::uint64_t rows_per_frame(std::uint64_t width, std::uint64_t height,
                              std::uint64_t ph, std::uint64_t pw) {
  if (!width || !height || !ph || !pw || width % 16U || height % 16U ||
      (width / 16U) % pw || (height / 16U) % ph)
    fail("H3 motion context requires matching native spatial/patch geometry");
  return product(height / 16U / ph, width / 16U / pw);
}

void check_rows(const runtime::Tensor &tensor, std::uint64_t columns) {
  tensor.validate();
  if (tensor.dtype != ir::DType::F32 || tensor.dims.size() != 2U ||
      tensor.dims[0] == 0U || tensor.dims[1] != columns)
    fail("H3 motion context requires F32 rows with matching channel width");
}

runtime::Tensor new_rows(std::uint64_t rows, std::uint64_t columns) {
  runtime::Tensor result{ir::DType::F32, {rows, columns}, {}};
  const auto bytes = product(product(rows, columns), sizeof(float));
  if (bytes > std::numeric_limits<std::size_t>::max())
    fail("H3 motion tensor exceeds address space");
  result.bytes.resize(static_cast<std::size_t>(bytes));
  return result;
}

void copy_rows(runtime::Tensor &to, std::uint64_t dest,
               const runtime::Tensor &from, std::uint64_t start,
               std::uint64_t rows) {
  if (to.dims[1] != from.dims[1] || start > from.dims[0] ||
      rows > from.dims[0] - start || dest > to.dims[0] ||
      rows > to.dims[0] - dest)
    fail("H3 motion row slice exceeds tensor bounds");
  const auto bytes_per_row = product(from.dims[1], sizeof(float));
  std::memcpy(to.mutable_data() + product(dest, bytes_per_row),
              from.data() + product(start, bytes_per_row),
              product(rows, bytes_per_row));
}

void require_finite(const runtime::Tensor &tensor) {
  for (const auto value : tensor.f32())
    if (!std::isfinite(value))
      fail("H3 motion context contains nonfinite latent values");
}

std::uint64_t metadata_integer(float value) {
  if (!std::isfinite(value) || value < 1.0F || value > 16777216.0F ||
      std::floor(value) != value)
    fail("H3 motion context metadata contains an invalid integer");
  return static_cast<std::uint64_t>(value);
}

double audio_overhang(std::uint64_t video_steps, std::uint64_t audio_steps) {
  if (video_steps < 2U || (video_steps - 2U) % 5U)
    fail("H3 motion context source endpoint is not phase-zero aligned");
  const auto frames = h3_motion_context_pixel_frames(video_steps);
  const auto residual = static_cast<double>(audio_steps) -
                         (5.0 / 3.0) * static_cast<double>(frames);
  if (!std::isfinite(residual) || residual < -0.5 || residual > 0.5)
    fail("H3 motion source audio has an invalid rounding residual");
  return residual < 0.0 && residual > -1e-9 ? 0.0 : residual;
}

} // namespace

std::uint64_t h3_motion_context_steps(std::uint64_t context_frames) {
  switch (context_frames) {
  case 5U: return 2U;
  case 22U: return 7U;
  case 39U: return 12U;
  default: fail("H3 motion context must be 5, 22, or 39 frames");
  }
}

std::uint64_t h3_motion_context_pixel_frames(std::uint64_t latent_steps) {
  const auto remainder = latent_steps % 5U;
  return sum(product(latent_steps / 5U, 17U),
             remainder ? 1U + (remainder - 1U) * 4U : 0U);
}

std::uint64_t h3_motion_context_audio_latents(std::uint64_t pixel_frames) {
  return thirds(product(pixel_frames, 5U));
}

std::uint64_t h3_motion_context_endpoint_steps(
    std::uint64_t requested_pixel_frames, std::uint64_t available_latent_steps) {
  if (!requested_pixel_frames || available_latent_steps < 2U ||
      (available_latent_steps - 2U) % 5U)
    fail("H3 continuation endpoint must be positive and phase-zero aligned");
  const auto groups = (available_latent_steps - 2U) / 5U;
  auto best = std::min(groups, requested_pixel_frames > 5U
                                   ? (requested_pixel_frames - 5U) / 17U : 0U);
  const auto lower = sum(5U, product(best, 17U));
  if (best < groups && requested_pixel_frames >= lower &&
      sum(lower, 17U) - requested_pixel_frames < requested_pixel_frames - lower)
    ++best;
  return sum(2U, product(best, 5U));
}

H3MotionContext load_h3_motion_context(
    const std::filesystem::path &path, std::uint64_t width,
    std::uint64_t height, std::uint64_t context_frames,
    std::uint64_t patch_height, std::uint64_t patch_width) {
  const auto steps = h3_motion_context_steps(context_frames);
  const auto audio_steps = h3_motion_context_audio_latents(context_frames);
  const auto rpf = rows_per_frame(width, height, patch_height, patch_width);
  const auto columns = product(24U, product(patch_height, patch_width));
  const auto file = weights::read_safetensors(path);
  const bool compact = file.find("video_context_rows") != nullptr;
  if (compact != (file.find("audio_context_rows") != nullptr))
    fail("H3 compact motion context requires both video and audio tails");
  const auto video = weights::map_safetensor(
      file, compact ? "video_context_rows" : "video_state_rows");
  const auto audio = weights::map_safetensor(
      file, compact ? "audio_context_rows" : "audio_state_rows");
  check_rows(video, columns);
  check_rows(audio, 32U);
  if (video.dims[0] % rpf || audio.dims[0] % 2U)
    fail("H3 motion context rows do not align with video/stereo geometry");
  const auto stored_steps = video.dims[0] / rpf;
  const auto stored_audio = audio.dims[0] / 2U;
  H3MotionContext result;
  result.context_frames = context_frames;
  result.source_video_latent_frames = stored_steps;
  result.source_audio_latents = stored_audio;
  if (compact) {
    const auto meta = weights::map_safetensor(file, "motion_context_meta");
    meta.validate();
    if (meta.dtype != ir::DType::F32 || meta.dims != std::vector<std::uint64_t>{10U})
      fail("H3 motion context metadata must be F32 [10]");
    const auto m = meta.f32();
    if (m[0] != 1.0F || metadata_integer(m[1]) != width ||
        metadata_integer(m[2]) != height || metadata_integer(m[6]) != stored_steps ||
        metadata_integer(m[7]) != stored_audio || metadata_integer(m[8]) != 24U ||
        metadata_integer(m[9]) != rpf ||
        h3_motion_context_steps(metadata_integer(m[5])) != stored_steps ||
        h3_motion_context_audio_latents(metadata_integer(m[5])) != stored_audio)
      fail("H3 motion context metadata does not match its source resolution/header");
    result.source_video_latent_frames = metadata_integer(m[3]);
    result.source_audio_latents = metadata_integer(m[4]);
  }
  if (stored_steps < steps || stored_audio < audio_steps ||
      (stored_steps - steps) % 5U || result.source_video_latent_frames < stored_steps ||
      result.source_audio_latents < stored_audio)
    fail("H3 motion context has no phase-aligned tail of the requested length");
  result.source_audio_overhang = audio_overhang(
      result.source_video_latent_frames, result.source_audio_latents);
  result.video_rows = new_rows(product(steps, rpf), columns);
  result.audio_rows = new_rows(product(audio_steps, 2U), 32U);
  copy_rows(result.video_rows, 0U, video, product(stored_steps - steps, rpf),
            result.video_rows.dims[0]);
  copy_rows(result.audio_rows, 0U, audio, stored_audio - audio_steps, audio_steps);
  copy_rows(result.audio_rows, audio_steps, audio,
            2U * stored_audio - audio_steps, audio_steps);
  require_finite(result.video_rows);
  require_finite(result.audio_rows);
  return result;
}

H3MotionState prepare_h3_motion_context_state(
    const H3MotionContext &context, const runtime::Tensor &condition_video_noise,
    const runtime::Tensor &target_video_noise,
    const runtime::Tensor &target_audio_noise) {
  const auto context_steps = h3_motion_context_steps(context.context_frames);
  check_rows(context.video_rows, 96U);
  check_rows(context.audio_rows, 32U);
  check_rows(condition_video_noise, 96U);
  check_rows(target_video_noise, 96U);
  check_rows(target_audio_noise, 32U);
  if (condition_video_noise.dims != context.video_rows.dims ||
      context.video_rows.dims[0] % context_steps ||
      context.audio_rows.dims[0] !=
          2U * h3_motion_context_audio_latents(context.context_frames))
    fail("H3 motion context/noise geometry mismatch");
  H3MotionState result{
      new_rows(sum(context.video_rows.dims[0], target_video_noise.dims[0]), 96U),
      new_rows(sum(context.audio_rows.dims[0], target_audio_noise.dims[0]), 32U)};
  auto out = result.video_rows.f32();
  const auto clean = context.video_rows.f32();
  const auto noise = condition_video_noise.f32();
  for (std::size_t index = 0U; index < clean.size(); ++index) {
    // Match the motion extension's literal coefficients and distinct F32
    // multiply/add boundaries (minimax_h3_t2va.mojo:2863-2868). The generic
    // creator augmentation's (1 - 0.999F) is a different F32 value.
    volatile float scaled_clean = 0.999F * clean[index];
    volatile float scaled_noise = 0.001F * noise[index];
    out[index] = scaled_clean + scaled_noise;
  }
  copy_rows(result.video_rows, context.video_rows.dims[0], target_video_noise,
            0U, target_video_noise.dims[0]);
  copy_rows(result.audio_rows, 0U, context.audio_rows, 0U, context.audio_rows.dims[0]);
  copy_rows(result.audio_rows, context.audio_rows.dims[0], target_audio_noise,
            0U, target_audio_noise.dims[0]);
  require_finite(result.video_rows);
  require_finite(result.audio_rows);
  return result;
}

std::uint64_t save_h3_motion_context_tail(
    const std::filesystem::path &path, const runtime::Tensor &video_rows,
    const runtime::Tensor &audio_rows, std::uint64_t num_latent_frames,
    std::uint64_t num_audio_latents, std::uint64_t width, std::uint64_t height,
    std::uint64_t delivered_endpoint_frames, std::uint64_t condition_video_rows,
    std::uint64_t condition_audio_rows, std::uint64_t patch_height,
    std::uint64_t patch_width) {
  const auto rpf = rows_per_frame(width, height, patch_height, patch_width);
  const auto columns = product(24U, product(patch_height, patch_width));
  check_rows(video_rows, columns);
  check_rows(audio_rows, 32U);
  if (video_rows.dims[0] != sum(condition_video_rows, product(num_latent_frames, rpf)) ||
      audio_rows.dims[0] != sum(condition_audio_rows, product(2U, num_audio_latents)))
    fail("H3 motion source rows disagree with target and condition geometry");
  const auto endpoint = h3_motion_context_endpoint_steps(
      delivered_endpoint_frames, num_latent_frames);
  const auto endpoint_audio = h3_motion_context_audio_latents(
      h3_motion_context_pixel_frames(endpoint));
  if (endpoint_audio > num_audio_latents)
    fail("H3 continuation endpoint exceeds source audio");
  const auto frames = endpoint >= 12U ? 39U : endpoint >= 7U ? 22U : 5U;
  const auto steps = h3_motion_context_steps(frames);
  const auto audio_steps = h3_motion_context_audio_latents(frames);
  auto video = new_rows(product(steps, rpf), columns);
  auto audio = new_rows(product(audio_steps, 2U), 32U);
  copy_rows(video, 0U, video_rows,
            sum(condition_video_rows, product(endpoint - steps, rpf)), video.dims[0]);
  copy_rows(audio, 0U, audio_rows,
            sum(condition_audio_rows, endpoint_audio - audio_steps), audio_steps);
  copy_rows(audio, audio_steps, audio_rows,
            sum(condition_audio_rows, sum(num_audio_latents, endpoint_audio - audio_steps)),
            audio_steps);
  require_finite(video);
  require_finite(audio);
  const std::array<std::uint64_t, 10U> values = {
      1U, width, height, endpoint, endpoint_audio, frames, steps, audio_steps, 24U, rpf};
  std::array<float, 10U> meta{};
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (values[index] > 16777216U)
      fail("H3 schema-1 motion metadata exceeds exact F32 integer range");
    meta[index] = static_cast<float>(values[index]);
  }
  if (std::filesystem::exists(path))
    fail("refusing to overwrite H3 motion context: " + path.string());
  weights::SafeTensorWriter writer(path, {
      {"video_context_rows", ir::DType::F32, video.dims},
      {"audio_context_rows", ir::DType::F32, audio.dims},
      {"motion_context_meta", ir::DType::F32, {10U}}});
  writer.append("video_context_rows", {video.data(), video.byte_size()});
  writer.append("audio_context_rows", {audio.data(), audio.byte_size()});
  writer.append("motion_context_meta", {
      reinterpret_cast<const std::uint8_t *>(meta.data()), sizeof(meta)});
  (void)writer.finish();
  return frames;
}

H3MotionTrim make_h3_motion_trim(std::uint64_t context_frames,
                                std::uint64_t output_frames) {
  if (context_frames) (void)h3_motion_context_steps(context_frames);
  if (!output_frames)
    fail("H3 continuation requires positive delivered frames");
  return {context_frames, thirds(product(context_frames, 4000U)), output_frames,
          thirds(product(output_frames, 4000U)), sum(context_frames, output_frames)};
}

} // namespace dif::frontend
