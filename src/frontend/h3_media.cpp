#include "dif/frontend/h3_media.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>

namespace dif::frontend {

H3Rgb24Video make_h3_rgb24_video(const runtime::Tensor &decoded,
                               std::uint64_t trim_start_frames,
                               std::uint64_t output_frames) {
  decoded.validate();
  if (decoded.dtype != ir::DType::F32 || decoded.dims.size() != 5U ||
      decoded.dims[0] != 1U || decoded.dims[1] != 3U)
    fail("H3 media handoff requires F32 [1,3,F,H,W] video");
  H3Rgb24Video output;
  const auto source_frames = decoded.dims[2];
  if (trim_start_frames >= source_frames)
    fail("H3 media overlap leaves no delivered video frames");
  const auto available = source_frames - trim_start_frames;
  output.frames = output_frames ? output_frames : available;
  if (output.frames > available)
    fail("H3 decoded video does not cover requested delivery after overlap trim");
  output.height = decoded.dims[3];
  output.width = decoded.dims[4];
  output.minimum = std::numeric_limits<float>::infinity();
  output.maximum = -std::numeric_limits<float>::infinity();
  output.bytes.resize(static_cast<std::size_t>(
      output.frames * output.height * output.width * 3U));
  const auto values = decoded.f32();
  const auto plane = source_frames * output.height * output.width;
  for (std::uint64_t frame = 0U; frame < output.frames; ++frame) {
    for (std::uint64_t y = 0U; y < output.height; ++y) {
      for (std::uint64_t x = 0U; x < output.width; ++x) {
        const auto pixel = (frame * output.height + y) * output.width + x;
        for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
          const auto source_pixel = ((frame + trim_start_frames) * output.height + y) * output.width + x;
          const auto value = values[static_cast<std::size_t>(channel * plane + source_pixel)];
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

void write_h3_trimmed_audio_wav(const std::filesystem::path &source,
                               const std::filesystem::path &destination,
                               const H3MotionTrim &trim) {
  const auto expected = make_h3_motion_trim(trim.trim_start_frames, trim.output_frames);
  if (trim.trim_start_audio_samples != expected.trim_start_audio_samples ||
      trim.output_audio_samples != expected.output_audio_samples)
    fail("H3 audio trim metadata disagrees with native24FPS/32kHz clock");
  if (std::filesystem::exists(destination))
    fail("refusing to overwrite trimmed H3 WAV: " + destination.string());
  std::ifstream input(source, std::ios::binary);
  std::array<std::uint8_t, 44> header{};
  if (!input.read(reinterpret_cast<char *>(header.data()), header.size()))
    fail("H3 audio trim cannot read native PCM16 WAV header");
  const auto u16 = [&](std::size_t i) {
    return static_cast<std::uint32_t>(header[i]) | (static_cast<std::uint32_t>(header[i + 1]) << 8);
  };
  const auto u32 = [&](std::size_t i) {
    return u16(i) | (u16(i + 2) << 16);
  };
  const auto channels = u16(22);
  const auto alignment = u16(32);
  const auto data_bytes = u32(40);
  if (std::memcmp(header.data(), "RIFF", 4) || std::memcmp(header.data() + 8, "WAVEfmt ", 8) ||
      std::memcmp(header.data() + 36, "data", 4) || u32(16) != 16 || u16(20) != 1 ||
      (channels != 1 && channels != 2) || u32(24) != 32000 || u16(34) != 16 ||
      alignment != channels * 2 || u32(28) != 32000 * alignment ||
      static_cast<std::uint64_t>(u32(4)) + 8 != 44ULL + data_bytes ||
      std::filesystem::file_size(source) != 44ULL + data_bytes || data_bytes % alignment)
    fail("H3 audio trim requires the native canonical32kHz PCM16 mono/stereo WAV");
  const auto samples = data_bytes / alignment;
  if (trim.trim_start_audio_samples > samples ||
      trim.output_audio_samples > samples - trim.trim_start_audio_samples)
    fail("H3 decoded audio does not cover the requested sample-exact delivery");
  const auto bytes = trim.output_audio_samples * alignment;
  if (bytes > std::numeric_limits<std::uint32_t>::max() - 36U)
    fail("H3 trimmed WAV exceeds RIFF size limit");
  const auto put_u32 = [&](std::size_t i, std::uint32_t value) {
    for (std::size_t j = 0; j < 4; ++j) header[i + j] = static_cast<std::uint8_t>(value >> (j * 8));
  };
  put_u32(4, static_cast<std::uint32_t>(36 + bytes));
  put_u32(40, static_cast<std::uint32_t>(bytes));
  input.seekg(static_cast<std::streamoff>(44 + trim.trim_start_audio_samples * alignment));
  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!output.write(reinterpret_cast<const char *>(header.data()), header.size()))
    fail("cannot write trimmed H3 PCM16 header");
  std::array<char, 65536> buffer{};
  auto remaining = bytes;
  while (remaining) {
    const auto count = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
    if (!input.read(buffer.data(), count) || !output.write(buffer.data(), count))
      fail("cannot copy sample-exact H3 PCM16 delivery");
    remaining -= static_cast<std::uint64_t>(count);
  }
}

} // namespace dif::frontend
