#include "dif/frontend/h3_conditioning.hpp"
#include "dif/frontend/h3_latents.hpp"
#include "dif/frontend/h3_motion.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

namespace {
int failures = 0;
void expect(bool ok, const std::string &message) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL " << message << '\n';
  }
}
template <class F> void rejects(F fn, const std::string &message) {
  try { fn(); expect(false, message); } catch (const std::exception &) {}
}
dif::runtime::Tensor rows(std::uint64_t count, std::uint64_t width, float base) {
  dif::runtime::Tensor tensor{dif::ir::DType::F32, {count, width}, {}};
  tensor.bytes.resize(count * width * sizeof(float));
  auto data = tensor.f32();
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = base + static_cast<float>(i) / 1024.0F;
  return tensor;
}
void same_rows(const dif::runtime::Tensor &actual, std::uint64_t dest,
               const dif::runtime::Tensor &source, std::uint64_t start,
               std::uint64_t count, const std::string &label) {
  expect(actual.dims[1] == source.dims[1] &&
             std::memcmp(actual.data() + dest * actual.dims[1] * 4U,
                         source.data() + start * source.dims[1] * 4U,
                         count * actual.dims[1] * 4U) == 0,
         label);
}

void layout_gate() {
  // Analytic golden cases from motion_context.mojo: 24 spatial rows/frame,
  // seven text rows, separate immutable conditions, and shared target origin.
  const std::array<std::int32_t, 7U> tags{1, 1, 0, 1, 1, 1, 1};
  for (const auto frames : {5U, 22U, 39U}) {
    const auto steps = dif::frontend::h3_motion_context_steps(frames);
    const auto ca = dif::frontend::h3_motion_context_audio_latents(frames);
    for (const auto residual : {-1.0 / 3.0, 0.0, 1.0 / 3.0}) {
      const auto layout = dif::frontend::make_h3_motion_context_layout(
          tags, 17U, 8U, 12U, 93U, 1U, 2U, 2U, frames, residual);
      const auto cv = steps * 24U;
      const auto fixed_audio_start = 7U + cv;
      const auto target_audio_start = fixed_audio_start + 2U * ca;
      const auto target_video_start = target_audio_start + 186U;
      const auto origin = static_cast<float>(7U + ca);
      expect(layout.sequence_length == target_video_start + 408U, "sequence count");
      expect(layout.num_condition_video_rows == cv &&
                 layout.num_condition_audio_rows == 2U * ca, "condition counts");
      expect(layout.video_indices[0] == 7 &&
                 layout.audio_indices[0] == static_cast<int>(fixed_audio_start) &&
                 layout.video_indices[cv] == static_cast<int>(target_video_start),
             "physical text/fixed-video/fixed-audio/target-audio/target-video order");
      expect(layout.position_ids[7U * 3U] == origin &&
                 layout.position_ids[target_video_start * 3U] == origin &&
                 layout.position_ids[target_audio_start * 3U] == origin,
             "vacated audio reference span retained before shared target origin");
      expect(layout.position_ids[(7U + 24U) * 3U] ==
                 static_cast<float>(static_cast<double>(origin) + 5.0 / 3.0),
             "phase-zero second video latent is one pixel frame later");
      const auto expected_audio_origin = origin +
          static_cast<float>(std::round(frames * (5.0 / 3.0) + residual)) - ca;
      expect(layout.position_ids[fixed_audio_start * 3U] == expected_audio_origin &&
                 layout.position_ids[(fixed_audio_start + ca) * 3U] == expected_audio_origin,
             "stereo tails share rounded endpoint");
      for (std::size_t i = 0U; i < layout.video_indices.size(); ++i)
        expect(layout.video_map[layout.video_indices[i]] == static_cast<int>(i),
               "video inverse mapping");
      for (std::size_t i = 0U; i < layout.audio_indices.size(); ++i)
        expect(layout.audio_map[layout.audio_indices[i]] == static_cast<int>(i),
               "audio inverse mapping");
      const auto plan = dif::frontend::make_h3_row_timestep_plan(
          layout, 0.25F, 0.5F, 0.999F, 1.0F);
      expect(plan.timesteps[plan.timestep_indices[7]] == 0.999F &&
                 plan.timesteps[plan.timestep_indices[fixed_audio_start]] == 1.0F &&
                 plan.timesteps[plan.timestep_indices[target_video_start]] == 0.25F &&
                 plan.timesteps[plan.timestep_indices[target_audio_start]] == 0.5F,
             "separate fixed video/audio and generated timestep masks");
    }
  }
  const auto native = dif::frontend::make_h3_motion_context_layout(
      tags, 37U, 30U, 52U, 207U, 1U, 2U, 2U, 22U, 1.0 / 3.0);
  expect(native.num_condition_video_rows == 2730U &&
             native.sequence_length == 17655U, "832x480 native spatial geometry");
  rejects([&] { dif::frontend::make_h3_motion_context_layout(
      tags, 17, 8, 12, 93, 1, 2, 2, 22, std::numeric_limits<double>::quiet_NaN()); },
      "nonfinite source residual rejected");
  // Existing task layouts remain independently usable after adding continuation.
  const std::array anchors{dif::frontend::H3KeyframeAnchor::First,
                            dif::frontend::H3KeyframeAnchor::Last};
  const auto keyframes = dif::frontend::make_h3_t2va_layout(
      tags, 17, 8, 12, 93, 1, 2, 2, anchors);
  expect(keyframes.num_condition_video_rows == 48U &&
             keyframes.num_condition_audio_rows == 0U, "first/last layout preserved");
}

void artifact_gate(const std::filesystem::path &directory) {
  // Prepend sentinels to emulate a previous conditioned run. Saved tails must
  // come from generated rows; stereo channel offsets must exclude prefixes.
  auto video = rows(3U + 17U * 24U, 96U, 10.0F);
  auto audio = rows(5U + 2U * 93U, 32U, -20.0F);
  const auto path = directory / "tail.safetensors";
  expect(dif::frontend::save_h3_motion_context_tail(
             path, video, audio, 17U, 93U, 192U, 128U, 42U, 3U, 5U) == 39U,
         "delivered cut 42 selects native endpoint 39 and largest tail");
  for (const auto frames : {5U, 22U, 39U}) {
    const auto context = dif::frontend::load_h3_motion_context(path, 192, 128, frames);
    const auto steps = dif::frontend::h3_motion_context_steps(frames);
    const auto ca = dif::frontend::h3_motion_context_audio_latents(frames);
    expect(context.source_video_latent_frames == 12U &&
               context.source_audio_latents == 65U &&
               context.source_audio_overhang == 0.0, "saved cut carries correct clocks");
    same_rows(context.video_rows, 0U, video, 3U + (12U - steps) * 24U,
               steps * 24U, "video tail byte-exact with prefix stripped");
    same_rows(context.audio_rows, 0U, audio, 5U + 65U - ca, ca,
               "left audio tail byte-exact");
    same_rows(context.audio_rows, ca, audio, 5U + 93U + 65U - ca, ca,
               "right audio tail byte-exact and channel-major");
    const auto cn = rows(context.video_rows.dims[0], 96U, 0.125F);
    const auto vn = rows(17U * 24U, 96U, -0.5F);
    const auto an = rows(2U * 93U, 32U, 0.5F);
    auto state = dif::frontend::prepare_h3_motion_context_state(context, cn, vn, an);
    for (std::size_t i = 0; i < context.video_rows.f32().size(); ++i) {
      const volatile float clean = context.video_rows.f32()[i] * 0.999F;
      const volatile float noise = cn.f32()[i] * 0.001F;
      expect(state.video_rows.f32()[i] == clean + noise,
             "motion augmentation preserves literal0.001 and two F32 multiply stores");
    }
    auto zero_context = context;
    std::fill(zero_context.video_rows.f32().begin(), zero_context.video_rows.f32().end(), 0.0F);
    const auto coefficient_probe = dif::frontend::prepare_h3_motion_context_state(
        zero_context, rows(context.video_rows.dims[0], 96U, 1.0F), vn, an);
    expect(coefficient_probe.video_rows.f32()[0] == 0.001F &&
               coefficient_probe.video_rows.f32()[0] != (1.0F - 0.999F),
           "Mojo motion coefficient is not silently substituted by generic reference augmentation");
    same_rows(state.audio_rows, 0, context.audio_rows, 0, 2U * ca,
               "prepared audio condition has no noise augmentation");
    same_rows(state.video_rows, steps * 24U, vn, 0, vn.dims[0],
               "video target noise unmodified");
    same_rows(state.audio_rows, 2U * ca, an, 0, an.dims[0],
               "audio target noise unmodified");
    const auto before_video = state.video_rows;
    const auto before_audio = state.audio_rows;
    const auto velocity_video = rows(state.video_rows.dims[0], 96, 1.0F);
    const auto velocity_audio = rows(state.audio_rows.dims[0], 32, 1.0F);
    dif::sampling::h3_euler_step_in_place(state.video_rows.f32(), velocity_video.f32(),
                                          96U, steps * 24U, 0.25F, 0.75F, 0.5F);
    dif::sampling::h3_euler_step_in_place(state.audio_rows.f32(), velocity_audio.f32(),
                                          32U, 2U * ca, 0.5F, 0.5F, 0.25F);
    same_rows(state.video_rows, 0, before_video, 0, steps * 24U,
               "real shared Euler freezes entire video context prefix");
    same_rows(state.audio_rows, 0, before_audio, 0, 2U * ca,
               "real shared Euler freezes both audio channel prefixes");
    expect(state.video_rows.f32()[steps * 24U * 96U] !=
               before_video.f32()[steps * 24U * 96U], "generated target advances");
  }
  rejects([&] { dif::frontend::load_h3_motion_context(path, 128, 192, 22); },
          "same pixel count with wrong spatial resolution rejected");
  rejects([&] { dif::frontend::load_h3_motion_context(path, 192, 128, 6); },
          "unsupported overlap rejected");
  rejects([&] { dif::frontend::save_h3_motion_context_tail(
      path, video, audio, 17, 93, 192, 128, 42, 3, 5); }, "existing artifact protected");

  const auto full_video = rows(7U * 24U, 96U, 0.25F);
  const auto full_audio = rows(2U * 37U, 32U, 0.5F);
  const auto full_path = directory / "full.safetensors";
  dif::frontend::write_h3_latent_handoff(full_path, full_video, full_audio);
  const auto from_full = dif::frontend::load_h3_motion_context(full_path, 192, 128, 22);
  expect(std::abs(from_full.source_audio_overhang - 1.0 / 3.0) < 1e-12,
         "full handoff retains signed source A/V residual");
  same_rows(from_full.video_rows, 0, full_video, 0, full_video.dims[0],
             "full handoff context extraction exact");
  const auto small = directory / "small.safetensors";
  expect(dif::frontend::save_h3_motion_context_tail(
      small, full_video, full_audio, 7, 37, 192, 128, 22) == 22U,
      "short source saves 22-frame compact tail");
  rejects([&] { dif::frontend::load_h3_motion_context(small, 192, 128, 39); },
          "cannot grow compact tail beyond saved coverage");
}
} // namespace

int main() {
  try {
    expect(dif::frontend::h3_motion_context_audio_latents(5U) == 8U &&
               dif::frontend::h3_motion_context_audio_latents(22U) == 37U &&
               dif::frontend::h3_motion_context_audio_latents(39U) == 65U,
           "native 24/40 Hz overlap clocks");
    expect(dif::frontend::h3_motion_context_endpoint_steps(360U, 107U) == 107U &&
               dif::frontend::h3_motion_context_endpoint_steps(382U, 117U) == 112U,
           "15-second source and continuation snap to nearest native endpoint");
    rejects([] { dif::frontend::h3_motion_context_endpoint_steps(10, 16); },
            "unaligned source endpoint rejected");
    const auto trim = dif::frontend::make_h3_motion_trim(22U, 360U);
    expect(trim.trim_start_frames == 22U && trim.trim_start_audio_samples == 29333U &&
               trim.output_audio_samples == 480000U &&
               trim.continuation_endpoint_frames == 382U, "native overlap trim contract");
    const auto directory = std::filesystem::temp_directory_path() /
        ("dif-h3-motion-tests-" + std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    layout_gate();
    artifact_gate(directory);
    std::cout << "H3_MOTION_CPU " << (failures ? "FAIL" : "PASS")
              << " overlaps=5,22,39 persistence=byte-exact prefix_masks=shared-Euler"
              << " evidence=" << directory << " decoded_gate=not-run\n";
    return failures ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << "H3_MOTION_CPU error: " << error.what() << '\n';
    return 1;
  }
}
