#pragma once

#include "dif/runtime/tensor.hpp"

#include <cstdint>
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
H3Rgb24Video make_h3_rgb24_video(const runtime::Tensor &decoded);

} // namespace dif::frontend
