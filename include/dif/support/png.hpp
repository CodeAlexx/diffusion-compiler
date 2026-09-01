#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace dif {

// Write a standards-compliant, non-interlaced 8-bit RGB PNG. The native
// encoder uses deterministic uncompressed DEFLATE blocks so media output has
// no third-party runtime dependency.
void write_png_rgb8(const std::filesystem::path &path, std::uint32_t width,
                    std::uint32_t height,
                    std::span<const std::uint8_t> rgb);

} // namespace dif
