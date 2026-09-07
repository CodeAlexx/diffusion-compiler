#pragma once
#include "dif/support/png.hpp"

namespace dif {
// PIL RGB8 LANCZOS semantics: normalized 22-bit fixed-point coefficients,
// horizontal then vertical passes, including the quantized RGB8 intermediate.
RgbImage resize_rgb8_lanczos(const RgbImage &image, std::uint32_t width,
                           std::uint32_t height);
}
