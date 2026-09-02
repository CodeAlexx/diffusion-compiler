#pragma once

// Generic decoded-artifact metrics for the quality gate assistant. Scalar
// metrics support admission; they never replace the required perceptual
// review, and difquality says so in every verdict.

#include "dif/support/png.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dif::quality {

struct ImageSanity {
  bool decodable{};
  std::uint32_t width{};
  std::uint32_t height{};
  // Fraction of pixels equal to the top-left pixel; 1.0 is a constant image.
  double constant_fraction{};
  double mean_luma{};
  double luma_stddev{};
  std::string problem;
};

struct ImageComparison {
  bool comparable{};
  std::string problem;
  double psnr_db{};
  // Mean SSIM over non-overlapping 8x8 grayscale windows (K1 0.01, K2 0.03).
  double ssim_8x8{};
  double mean_absolute_difference{};
  std::uint32_t max_absolute_difference{};
  double identical_pixel_fraction{};
  bool bit_identical{};
};

ImageSanity image_sanity(const RgbImage &image);
ImageComparison compare_images(const RgbImage &reference,
                               const RgbImage &candidate);

struct Waveform {
  std::uint32_t sample_rate{};
  std::uint32_t channels{};
  std::uint64_t samples_per_channel{};
  // Interleaved float samples in [-1, 1].
  std::vector<float> samples;
};

// 16-bit PCM RIFF/WAVE reader (the format write_wav_pcm16 produces).
Waveform read_wav_pcm16(const std::filesystem::path &path);

struct AudioSanity {
  bool decodable{};
  std::uint32_t sample_rate{};
  std::uint32_t channels{};
  std::uint64_t samples_per_channel{};
  double duration_seconds{};
  double rms{};
  double peak{};
  double silence_fraction{};
  double clipping_fraction{};
  std::string problem;
};

struct AudioComparison {
  bool comparable{};
  std::string problem;
  double snr_db{};
  double relative_l2{};
  double max_absolute_difference{};
  std::uint64_t compared_samples{};
  bool bit_identical{};
};

AudioSanity audio_sanity(const Waveform &waveform);
AudioComparison compare_audio(const Waveform &reference,
                              const Waveform &candidate);

} // namespace dif::quality
