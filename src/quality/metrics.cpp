#include "dif/quality/metrics.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace dif::quality {
namespace {

double luma(const std::uint8_t *pixel) {
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}

} // namespace

ImageSanity image_sanity(const RgbImage &image) {
  ImageSanity sanity;
  if (image.width == 0U || image.height == 0U ||
      image.pixels.size() != static_cast<std::size_t>(image.width) *
                                 image.height * 3U) {
    sanity.problem = "image has no decodable pixels";
    return sanity;
  }
  sanity.decodable = true;
  sanity.width = image.width;
  sanity.height = image.height;
  const auto count = static_cast<std::uint64_t>(image.width) * image.height;
  std::uint64_t constant = 0U;
  double sum = 0.0;
  double sum_squares = 0.0;
  for (std::uint64_t index = 0; index < count; ++index) {
    const auto *pixel = image.pixels.data() + index * 3U;
    if (std::memcmp(pixel, image.pixels.data(), 3U) == 0)
      ++constant;
    const auto value = luma(pixel);
    sum += value;
    sum_squares += value * value;
  }
  sanity.constant_fraction =
      static_cast<double>(constant) / static_cast<double>(count);
  sanity.mean_luma = sum / static_cast<double>(count);
  const auto variance =
      std::max(0.0, sum_squares / static_cast<double>(count) -
                        sanity.mean_luma * sanity.mean_luma);
  sanity.luma_stddev = std::sqrt(variance);
  if (sanity.constant_fraction >= 0.999)
    sanity.problem = "image is constant (every pixel equals the first)";
  return sanity;
}

ImageComparison compare_images(const RgbImage &reference,
                               const RgbImage &candidate) {
  ImageComparison comparison;
  if (reference.width != candidate.width ||
      reference.height != candidate.height) {
    comparison.problem = "image geometry differs";
    return comparison;
  }
  if (reference.width == 0U || reference.height == 0U) {
    comparison.problem = "empty image";
    return comparison;
  }
  comparison.comparable = true;
  const auto count =
      static_cast<std::uint64_t>(reference.width) * reference.height;
  double squared_error = 0.0;
  double absolute_error = 0.0;
  std::uint64_t identical = 0U;
  for (std::uint64_t index = 0; index < count * 3U; ++index) {
    const int difference = static_cast<int>(reference.pixels[index]) -
                           static_cast<int>(candidate.pixels[index]);
    squared_error += static_cast<double>(difference * difference);
    absolute_error += std::abs(difference);
    comparison.max_absolute_difference = std::max(
        comparison.max_absolute_difference,
        static_cast<std::uint32_t>(std::abs(difference)));
  }
  for (std::uint64_t index = 0; index < count; ++index)
    if (std::memcmp(reference.pixels.data() + index * 3U,
                    candidate.pixels.data() + index * 3U, 3U) == 0)
      ++identical;
  const auto mse = squared_error / static_cast<double>(count * 3U);
  comparison.mean_absolute_difference =
      absolute_error / static_cast<double>(count * 3U);
  comparison.identical_pixel_fraction =
      static_cast<double>(identical) / static_cast<double>(count);
  comparison.bit_identical = identical == count;
  comparison.psnr_db = mse == 0.0
                           ? std::numeric_limits<double>::infinity()
                           : 10.0 * std::log10(255.0 * 255.0 / mse);
  // SSIM over 8x8 non-overlapping grayscale windows.
  const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
  const double c2 = (0.03 * 255.0) * (0.03 * 255.0);
  double ssim_sum = 0.0;
  std::uint64_t windows = 0U;
  for (std::uint32_t y = 0; y + 8U <= reference.height; y += 8U) {
    for (std::uint32_t x = 0; x + 8U <= reference.width; x += 8U) {
      double mean_a = 0.0;
      double mean_b = 0.0;
      double values_a[64];
      double values_b[64];
      std::size_t slot = 0U;
      for (std::uint32_t dy = 0; dy < 8U; ++dy) {
        for (std::uint32_t dx = 0; dx < 8U; ++dx) {
          const auto offset =
              (static_cast<std::uint64_t>(y + dy) * reference.width + x + dx) *
              3U;
          values_a[slot] = luma(reference.pixels.data() + offset);
          values_b[slot] = luma(candidate.pixels.data() + offset);
          mean_a += values_a[slot];
          mean_b += values_b[slot];
          ++slot;
        }
      }
      mean_a /= 64.0;
      mean_b /= 64.0;
      double var_a = 0.0;
      double var_b = 0.0;
      double covariance = 0.0;
      for (std::size_t index = 0; index < 64U; ++index) {
        var_a += (values_a[index] - mean_a) * (values_a[index] - mean_a);
        var_b += (values_b[index] - mean_b) * (values_b[index] - mean_b);
        covariance += (values_a[index] - mean_a) * (values_b[index] - mean_b);
      }
      var_a /= 63.0;
      var_b /= 63.0;
      covariance /= 63.0;
      const auto ssim = ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)) /
                        ((mean_a * mean_a + mean_b * mean_b + c1) *
                         (var_a + var_b + c2));
      ssim_sum += ssim;
      ++windows;
    }
  }
  comparison.ssim_8x8 = windows == 0U ? 1.0 : ssim_sum / static_cast<double>(windows);
  return comparison;
}

Waveform read_wav_pcm16(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open WAV file " + path.string());
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 12U || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    fail("not a RIFF/WAVE file: " + path.string());
  const auto u16 = [&](std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U);
  };
  const auto u32 = [&](std::size_t offset) {
    return u16(offset) | (u16(offset + 2) << 16U);
  };
  Waveform waveform;
  std::uint32_t bits = 0U;
  std::uint32_t format = 0U;
  bool have_format = false;
  std::size_t offset = 12U;
  while (offset + 8U <= bytes.size()) {
    const std::string id(reinterpret_cast<const char *>(bytes.data() + offset), 4);
    const auto size = u32(offset + 4);
    const auto body = offset + 8U;
    if (id == "fmt " && body + 16U <= bytes.size()) {
      format = u16(body);
      waveform.channels = u16(body + 2);
      waveform.sample_rate = u32(body + 4);
      bits = u16(body + 14);
      have_format = true;
    } else if (id == "data") {
      if (!have_format)
        fail("WAV data chunk precedes fmt chunk: " + path.string());
      if (format != 1U || bits != 16U)
        fail("only 16-bit PCM WAV is supported: " + path.string());
      const auto available = std::min<std::size_t>(size, bytes.size() - body);
      const auto frames = available / (2U * waveform.channels);
      waveform.samples_per_channel = frames;
      waveform.samples.resize(frames * waveform.channels);
      for (std::size_t index = 0; index < frames * waveform.channels; ++index) {
        const auto raw = static_cast<std::int16_t>(u16(body + index * 2U));
        waveform.samples[index] = static_cast<float>(raw) / 32767.0F;
      }
      return waveform;
    }
    offset = body + size + (size & 1U);
  }
  fail("WAV file has no data chunk: " + path.string());
}

AudioSanity audio_sanity(const Waveform &waveform) {
  AudioSanity sanity;
  if (waveform.channels == 0U || waveform.sample_rate == 0U ||
      waveform.samples.empty()) {
    sanity.problem = "waveform has no samples";
    return sanity;
  }
  sanity.decodable = true;
  sanity.sample_rate = waveform.sample_rate;
  sanity.channels = waveform.channels;
  sanity.samples_per_channel = waveform.samples_per_channel;
  sanity.duration_seconds = static_cast<double>(waveform.samples_per_channel) /
                            static_cast<double>(waveform.sample_rate);
  double energy = 0.0;
  std::uint64_t silent = 0U;
  std::uint64_t clipped = 0U;
  for (const auto sample : waveform.samples) {
    const auto magnitude = std::fabs(static_cast<double>(sample));
    energy += magnitude * magnitude;
    sanity.peak = std::max(sanity.peak, magnitude);
    if (magnitude < 1.0e-4)
      ++silent;
    if (magnitude >= 0.999)
      ++clipped;
  }
  sanity.rms = std::sqrt(energy / static_cast<double>(waveform.samples.size()));
  sanity.silence_fraction =
      static_cast<double>(silent) / static_cast<double>(waveform.samples.size());
  sanity.clipping_fraction =
      static_cast<double>(clipped) / static_cast<double>(waveform.samples.size());
  if (sanity.silence_fraction >= 0.999)
    sanity.problem = "waveform is silent";
  return sanity;
}

AudioComparison compare_audio(const Waveform &reference,
                              const Waveform &candidate) {
  AudioComparison comparison;
  if (reference.sample_rate != candidate.sample_rate ||
      reference.channels != candidate.channels) {
    comparison.problem = "sample rate or channel count differs";
    return comparison;
  }
  if (reference.samples.empty() || candidate.samples.empty()) {
    comparison.problem = "empty waveform";
    return comparison;
  }
  if (reference.samples.size() != candidate.samples.size())
    comparison.problem = "length differs; compared the common prefix";
  comparison.comparable = true;
  const auto count = std::min(reference.samples.size(), candidate.samples.size());
  double signal = 0.0;
  double noise = 0.0;
  bool identical = reference.samples.size() == candidate.samples.size();
  for (std::size_t index = 0; index < count; ++index) {
    const auto want = static_cast<double>(reference.samples[index]);
    const auto got = static_cast<double>(candidate.samples[index]);
    signal += want * want;
    noise += (got - want) * (got - want);
    comparison.max_absolute_difference =
        std::max(comparison.max_absolute_difference, std::fabs(got - want));
    identical = identical && reference.samples[index] == candidate.samples[index];
  }
  comparison.compared_samples = count;
  comparison.bit_identical = identical;
  comparison.relative_l2 = signal == 0.0
                               ? (noise == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                               : std::sqrt(noise / signal);
  comparison.snr_db = noise == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : (signal == 0.0 ? -std::numeric_limits<double>::infinity()
                                           : 10.0 * std::log10(signal / noise));
  return comparison;
}

} // namespace dif::quality
