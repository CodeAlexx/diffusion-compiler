#include "dif/support/wav.hpp"

#include "dif/support/error.hpp"

#include <cstdint>
#include <fstream>
#include <vector>

namespace dif::support {
namespace {

void push_u16(std::vector<std::uint8_t> &out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void push_u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
}

void push_tag(std::vector<std::uint8_t> &out, const char (&tag)[5]) {
  for (unsigned i = 0; i < 4U; ++i)
    out.push_back(static_cast<std::uint8_t>(tag[i]));
}

} // namespace

void write_wav_pcm16(const std::filesystem::path &path,
                     std::span<const float> waveform, std::uint32_t channels,
                     std::uint64_t samples_per_channel,
                     std::uint32_t sample_rate) {
  if (channels == 0U || samples_per_channel == 0U ||
      waveform.size() != channels * samples_per_channel)
    fail("write_wav_pcm16: waveform size does not match geometry");
  const auto data_bytes =
      static_cast<std::uint32_t>(samples_per_channel * channels * 2U);

  std::vector<std::uint8_t> file;
  file.reserve(44U + data_bytes);
  // RIFF chunk descriptor + fmt + data sub-chunks — byte-for-byte the
  // layout wav.mojo emits (PCM, 16-bit).
  push_tag(file, "RIFF");
  push_u32(file, 36U + data_bytes);
  push_tag(file, "WAVE");
  push_tag(file, "fmt ");
  push_u32(file, 16U);
  push_u16(file, 1U);
  push_u16(file, channels);
  push_u32(file, sample_rate);
  push_u32(file, sample_rate * channels * 2U);
  push_u16(file, channels * 2U);
  push_u16(file, 16U);
  push_tag(file, "data");
  push_u32(file, data_bytes);

  for (std::uint64_t sample = 0; sample < samples_per_channel; ++sample) {
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
      float value = waveform[channel * samples_per_channel + sample];
      if (value > 1.0F)
        value = 1.0F;
      else if (value < -1.0F)
        value = -1.0F;
      const float scaled = value * 32767.0F;
      // Round half away from zero, exactly wav.mojo's expression.
      int quantized = scaled >= 0.0F ? static_cast<int>(scaled + 0.5F)
                                     : static_cast<int>(scaled - 0.5F);
      if (quantized > 32767)
        quantized = 32767;
      if (quantized < -32768)
        quantized = -32768;
      const auto bits = static_cast<std::uint32_t>(quantized & 0xFFFF);
      file.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
      file.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xFFU));
    }
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("write_wav_pcm16: cannot create " + path.string());
  output.write(reinterpret_cast<const char *>(file.data()),
               static_cast<std::streamsize>(file.size()));
  if (!output)
    fail("write_wav_pcm16: cannot write " + path.string());
}

} // namespace dif::support
