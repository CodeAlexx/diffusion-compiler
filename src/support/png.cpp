#include "dif/support/png.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <vector>

namespace dif {
namespace {

void append_be32(std::vector<std::uint8_t> &output, std::uint32_t value) {
  output.push_back(static_cast<std::uint8_t>(value >> 24U));
  output.push_back(static_cast<std::uint8_t>(value >> 16U));
  output.push_back(static_cast<std::uint8_t>(value >> 8U));
  output.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t crc32(std::span<const std::uint8_t> bytes) {
  auto crc = std::uint32_t{0xffffffffU};
  for (const auto byte : bytes) {
    crc ^= byte;
    for (unsigned bit = 0U; bit < 8U; ++bit)
      crc = (crc >> 1U) ^ (0xedb88320U &
                           (0U - static_cast<std::uint32_t>(crc & 1U)));
  }
  return ~crc;
}

std::uint32_t adler32(std::span<const std::uint8_t> bytes) {
  constexpr auto modulus = std::uint32_t{65521U};
  auto a = std::uint32_t{1U};
  auto b = std::uint32_t{0U};
  for (const auto byte : bytes) {
    a = (a + byte) % modulus;
    b = (b + a) % modulus;
  }
  return (b << 16U) | a;
}

void append_chunk(std::vector<std::uint8_t> &png,
                  const std::array<std::uint8_t, 4> &type,
                  std::span<const std::uint8_t> data) {
  if (data.size() > std::numeric_limits<std::uint32_t>::max())
    fail("PNG chunk exceeds 32-bit length");
  append_be32(png, static_cast<std::uint32_t>(data.size()));
  const auto crc_start = png.size();
  png.insert(png.end(), type.begin(), type.end());
  png.insert(png.end(), data.begin(), data.end());
  append_be32(png, crc32(std::span<const std::uint8_t>(png).subspan(
                       crc_start, type.size() + data.size())));
}

} // namespace

void write_png_rgb8(const std::filesystem::path &path, std::uint32_t width,
                    std::uint32_t height,
                    std::span<const std::uint8_t> rgb) {
  if (width == 0U || height == 0U ||
      static_cast<std::uint64_t>(width) * height >
          std::numeric_limits<std::size_t>::max() / 3U ||
      rgb.size() != static_cast<std::size_t>(width) * height * 3U)
    fail("invalid RGB PNG dimensions or byte count");
  const auto row_bytes = static_cast<std::size_t>(width) * 3U;
  if (static_cast<std::uint64_t>(row_bytes + 1U) * height >
      std::numeric_limits<std::size_t>::max())
    fail("PNG scanline buffer exceeds host size range");
  std::vector<std::uint8_t> scanlines;
  scanlines.reserve((row_bytes + 1U) * height);
  for (std::uint32_t y = 0U; y < height; ++y) {
    scanlines.push_back(0U);
    const auto begin = static_cast<std::size_t>(y) * row_bytes;
    scanlines.insert(scanlines.end(), rgb.begin() + begin,
                     rgb.begin() + begin + row_bytes);
  }

  std::vector<std::uint8_t> deflate{0x78U, 0x01U};
  std::size_t offset = 0U;
  while (offset < scanlines.size()) {
    const auto count = std::min<std::size_t>(65535U, scanlines.size() - offset);
    const auto final = offset + count == scanlines.size();
    deflate.push_back(final ? 0x01U : 0x00U);
    const auto length = static_cast<std::uint16_t>(count);
    const auto inverse = static_cast<std::uint16_t>(~length);
    deflate.push_back(static_cast<std::uint8_t>(length));
    deflate.push_back(static_cast<std::uint8_t>(length >> 8U));
    deflate.push_back(static_cast<std::uint8_t>(inverse));
    deflate.push_back(static_cast<std::uint8_t>(inverse >> 8U));
    deflate.insert(deflate.end(), scanlines.begin() + offset,
                   scanlines.begin() + offset + count);
    offset += count;
  }
  append_be32(deflate, adler32(scanlines));

  std::vector<std::uint8_t> png{0x89U, 'P', 'N', 'G', 0x0dU, 0x0aU, 0x1aU,
                                0x0aU};
  std::vector<std::uint8_t> header;
  append_be32(header, width);
  append_be32(header, height);
  header.insert(header.end(), {8U, 2U, 0U, 0U, 0U});
  append_chunk(png, {'I', 'H', 'D', 'R'}, header);
  append_chunk(png, {'I', 'D', 'A', 'T'}, deflate);
  append_chunk(png, {'I', 'E', 'N', 'D'}, {});

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create PNG file: " + path.string());
  output.write(reinterpret_cast<const char *>(png.data()),
               static_cast<std::streamsize>(png.size()));
  if (!output)
    fail("cannot write PNG file: " + path.string());
}

} // namespace dif
