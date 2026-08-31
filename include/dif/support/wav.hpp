#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace dif::support {

// 16-bit PCM RIFF/WAVE writer matching serenitymojo/audio/wav.mojo exactly:
// 44-byte header, interleaved channels, clamp to [-1,1] then
// round-half-away-from-zero to int16 (i16 = int(x*32767 +/- 0.5)).
// `waveform` is channel-major: channel 0's samples then channel 1's.
void write_wav_pcm16(const std::filesystem::path &path,
                     std::span<const float> waveform,
                     std::uint32_t channels, std::uint64_t samples_per_channel,
                     std::uint32_t sample_rate);

} // namespace dif::support
