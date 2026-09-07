#include "dif/support/image_resize.hpp"
#include "dif/support/error.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace dif {
namespace {
constexpr int precision = 22;
struct Row { int start; std::vector<std::int32_t> weights; };
double sinc(double x) {
  if (x == 0.0) return 1.0;
  const auto p = x * std::numbers::pi;
  return std::sin(p) / p;
}
std::vector<Row> coefficients(int input, int output) {
  const auto scale = double(input) / double(output);
  const auto filter_scale = std::max(1.0, scale);
  const auto support = 3.0 * filter_scale;
  std::vector<Row> rows;
  rows.reserve(output);
  for (int i = 0; i < output; ++i) {
    const auto center = (double(i) + 0.5) * scale;
    const auto start = std::max(0, int(center - support + 0.5));
    const auto end = std::min(input, int(center + support + 0.5));
    std::vector<double> values;
    double sum = 0.0;
    for (int j = start; j < end; ++j) {
      const auto x = (double(j) - center + 0.5) / filter_scale;
      const auto value = x >= -3.0 && x < 3.0 ? sinc(x) * sinc(x / 3.0) : 0.0;
      values.push_back(value);
      sum += value;
    }
    Row row{start, {}};
    for (const auto value : values) {
      const auto normalized = sum != 0.0 ? value / sum : value;
      row.weights.push_back(static_cast<std::int32_t>(
          normalized * double(1 << precision) + (normalized < 0.0 ? -0.5 : 0.5)));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}
std::size_t bytes(std::uint32_t w, std::uint32_t h) {
  if (!w || !h || w > INT32_MAX / 8U || h > INT32_MAX / 8U ||
      std::uint64_t(w) * h > std::numeric_limits<std::size_t>::max() / 3U)
    fail("invalid RGB resize dimensions");
  return std::size_t(w) * h * 3U;
}
std::uint8_t clip(std::int64_t v) {
  return static_cast<std::uint8_t>(std::clamp<std::int64_t>(v >> precision, 0, 255));
}
}
RgbImage resize_rgb8_lanczos(const RgbImage &image, std::uint32_t width,
                           std::uint32_t height) {
  if (image.pixels.size() != bytes(image.width, image.height))
    fail("RGB resize input payload does not match dimensions");
  (void)bytes(width, height);
  RgbImage current = image;
  if (width != current.width) {
    const auto table = coefficients(current.width, width);
    RgbImage next{width, current.height, std::vector<std::uint8_t>(bytes(width, current.height))};
    for (std::uint32_t y = 0; y < current.height; ++y)
      for (std::uint32_t x = 0; x < width; ++x)
        for (unsigned c = 0; c < 3; ++c) {
          std::int64_t sum = 1 << (precision - 1);
          const auto &row = table[x];
          for (std::size_t k = 0; k < row.weights.size(); ++k)
            sum += std::int64_t(current.pixels[(std::size_t(y) * current.width + row.start + k) * 3 + c]) * row.weights[k];
          next.pixels[(std::size_t(y) * width + x) * 3 + c] = clip(sum);
        }
    current = std::move(next);
  }
  if (height != current.height) {
    const auto table = coefficients(current.height, height);
    RgbImage next{width, height, std::vector<std::uint8_t>(bytes(width, height))};
    for (std::uint32_t y = 0; y < height; ++y)
      for (std::uint32_t x = 0; x < width; ++x)
        for (unsigned c = 0; c < 3; ++c) {
          std::int64_t sum = 1 << (precision - 1);
          const auto &row = table[y];
          for (std::size_t k = 0; k < row.weights.size(); ++k)
            sum += std::int64_t(current.pixels[((std::size_t(row.start) + k) * width + x) * 3 + c]) * row.weights[k];
          next.pixels[(std::size_t(y) * width + x) * 3 + c] = clip(sum);
        }
    current = std::move(next);
  }
  return current;
}
}
