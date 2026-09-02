#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace dif {

struct RgbImage {
  std::uint32_t width{};
  std::uint32_t height{};
  std::vector<std::uint8_t> pixels;
};

// Decode any standards-compliant PNG into tightly packed RGB8 pixels. The
// decoder deliberately normalizes palette, grayscale, alpha, and 16-bit PNGs
// at this native media-input boundary so model frontends receive one stable
// representation.
RgbImage read_png_rgb8(const std::filesystem::path &path);

// Write a standards-compliant, non-interlaced 8-bit RGB PNG. The native
// encoder uses deterministic uncompressed DEFLATE blocks so media output has
// no third-party runtime dependency.
void write_png_rgb8(const std::filesystem::path &path, std::uint32_t width,
                    std::uint32_t height,
                    std::span<const std::uint8_t> rgb);

} // namespace dif
