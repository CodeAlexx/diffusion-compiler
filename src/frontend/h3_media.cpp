#include "dif/frontend/h3_media.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace dif::frontend {

H3Rgb24Video make_h3_rgb24_video(const runtime::Tensor &decoded) {
  decoded.validate();
  if (decoded.dtype != ir::DType::F32 || decoded.dims.size() != 5U ||
      decoded.dims[0] != 1U || decoded.dims[1] != 3U)
    fail("H3 media handoff requires F32 [1,3,F,H,W] video");
  H3Rgb24Video output;
  output.frames = decoded.dims[2];
  output.height = decoded.dims[3];
  output.width = decoded.dims[4];
  output.minimum = std::numeric_limits<float>::infinity();
  output.maximum = -std::numeric_limits<float>::infinity();
  output.bytes.resize(static_cast<std::size_t>(
      output.frames * output.height * output.width * 3U));
  const auto values = decoded.f32();
  const auto plane = output.frames * output.height * output.width;
  for (std::uint64_t frame = 0U; frame < output.frames; ++frame) {
    for (std::uint64_t y = 0U; y < output.height; ++y) {
      for (std::uint64_t x = 0U; x < output.width; ++x) {
        const auto pixel = (frame * output.height + y) * output.width + x;
        for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
          const auto value = values[static_cast<std::size_t>(channel * plane +
                                                             pixel)];
          if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            fail("H3 decoded video is nonfinite or outside [0,1]");
          output.minimum = std::min(output.minimum, value);
          output.maximum = std::max(output.maximum, value);
          output.bytes[static_cast<std::size_t>(pixel * 3U + channel)] =
              static_cast<std::uint8_t>(std::lround(value * 255.0F));
        }
      }
    }
  }
  return output;
}

} // namespace dif::frontend
