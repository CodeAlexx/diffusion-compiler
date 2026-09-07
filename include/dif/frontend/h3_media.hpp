#pragma once

#include "dif/runtime/tensor.hpp"
#include "dif/frontend/h3_motion.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace dif::frontend {

struct H3Rgb24Video {
  std::uint64_t frames{};
  std::uint64_t height{};
  std::uint64_t width{};
  float minimum{};
  float maximum{};
  std::vector<std::uint8_t> bytes;
};

// Converts source-decoded [1,3,F,H,W] F32 unit-range video into the interleaved
// RGB24 stream consumed by the retained Serenity ffmpeg handoff.
// A zero output_frames means every available frame after trim_start_frames.
// Selecting a window copies only those pixel frames; it does not interpolate.
H3Rgb24Video make_h3_rgb24_video(const runtime::Tensor &decoded,
                               std::uint64_t trim_start_frames = 0,
                               std::uint64_t output_frames = 0);

// Slice the canonical PCM16 WAV emitted by the native H3 decoder, preserving
// interleaved sample bytes exactly. Requires32kHz and adequate source coverage;
// never resamples, pads, or overwrites. AAC compression is a later mux stage.
void write_h3_trimmed_audio_wav(const std::filesystem::path &source,
                               const std::filesystem::path &destination,
                               const H3MotionTrim &trim);

} // namespace dif::frontend
