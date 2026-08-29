#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

void append_u16(std::vector<std::uint8_t> &bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void append_u32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void write_bmp(const std::filesystem::path &path, std::uint32_t width,
               std::uint32_t height, const std::vector<std::uint8_t> &rgb) {
  const auto pixels = static_cast<std::uint64_t>(width) * height;
  if (width == 0U || height == 0U || rgb.size() != pixels * 3U)
    dif::fail("invalid BMP pixel buffer");
  const auto row_bytes = static_cast<std::uint64_t>(width) * 3U;
  const auto stride = (row_bytes + 3U) & ~std::uint64_t{3U};
  const auto image_bytes = stride * height;
  const auto file_bytes = 54U + image_bytes;
  if (file_bytes > std::numeric_limits<std::uint32_t>::max())
    dif::fail("BMP output exceeds the format size limit");

  std::vector<std::uint8_t> encoded;
  encoded.reserve(static_cast<std::size_t>(file_bytes));
  encoded.push_back('B');
  encoded.push_back('M');
  append_u32(encoded, static_cast<std::uint32_t>(file_bytes));
  append_u16(encoded, 0U);
  append_u16(encoded, 0U);
  append_u32(encoded, 54U);
  append_u32(encoded, 40U);
  append_u32(encoded, width);
  append_u32(encoded, height);
  append_u16(encoded, 1U);
  append_u16(encoded, 24U);
  append_u32(encoded, 0U);
  append_u32(encoded, static_cast<std::uint32_t>(image_bytes));
  append_u32(encoded, 2835U);
  append_u32(encoded, 2835U);
  append_u32(encoded, 0U);
  append_u32(encoded, 0U);
  for (std::uint32_t output_y = 0U; output_y < height; ++output_y) {
    const auto input_y = height - 1U - output_y;
    for (std::uint32_t x = 0U; x < width; ++x) {
      const auto index =
          (static_cast<std::uint64_t>(input_y) * width + x) * 3U;
      encoded.push_back(rgb[index + 2U]);
      encoded.push_back(rgb[index + 1U]);
      encoded.push_back(rgb[index]);
    }
    encoded.insert(encoded.end(), static_cast<std::size_t>(stride - row_bytes),
                   0U);
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    dif::fail("cannot create BMP file: " + path.string());
  output.write(reinterpret_cast<const char *>(encoded.data()),
               static_cast<std::streamsize>(encoded.size()));
  if (!output)
    dif::fail("cannot write BMP file: " + path.string());
}

std::string frame_name(std::uint64_t frame) {
  std::ostringstream name;
  name << "frame-" << std::setfill('0') << std::setw(4) << frame << ".bmp";
  return name.str();
}

void usage() {
  std::cerr << "usage: difimage INPUT.diftensor OUTPUT_DIRECTORY\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      usage();
      return 2;
    }
    const auto tensor = dif::runtime::read_tensor(argv[1]);
    if (!dif::runtime::is_float_dtype(tensor.dtype))
      dif::fail("difimage requires a floating-point tensor");
    if (tensor.dims.size() != 5U || tensor.dims[0] != 1U ||
        tensor.dims[1] != 3U)
      dif::fail("difimage requires decoded NCTHW shape [1,3,T,H,W]");
    const auto frames = tensor.dims[2];
    const auto height = tensor.dims[3];
    const auto width = tensor.dims[4];
    if (frames == 0U || width == 0U || height == 0U ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max())
      dif::fail("difimage received unsupported decoded dimensions");
    const auto output = std::filesystem::path(argv[2]);
    std::filesystem::create_directories(output);

    const auto frame_pixels = width * height;
    std::vector<std::vector<std::uint8_t>> images;
    images.reserve(static_cast<std::size_t>(frames));
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    std::uint64_t nonfinite = 0U;
    std::uint64_t outside = 0U;
    for (std::uint64_t frame = 0U; frame < frames; ++frame) {
      std::vector<std::uint8_t> rgb(static_cast<std::size_t>(frame_pixels * 3U));
      for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
        for (std::uint64_t y = 0U; y < height; ++y) {
          for (std::uint64_t x = 0U; x < width; ++x) {
            const auto tensor_index =
                ((channel * frames + frame) * height + y) * width + x;
            const auto value = static_cast<double>(
                dif::runtime::load_float(tensor, tensor_index));
            if (!std::isfinite(value)) {
              ++nonfinite;
              continue;
            }
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            if (value < 0.0 || value > 1.0)
              ++outside;
            const auto encoded = static_cast<std::uint8_t>(std::lround(
                std::clamp(value, 0.0, 1.0) * 255.0));
            const auto pixel_index = (y * width + x) * 3U + channel;
            rgb[static_cast<std::size_t>(pixel_index)] = encoded;
          }
        }
      }
      write_bmp(output / frame_name(frame), static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height), rgb);
      images.push_back(std::move(rgb));
    }

    const auto columns = static_cast<std::uint64_t>(
        std::ceil(std::sqrt(static_cast<double>(frames))));
    const auto rows = (frames + columns - 1U) / columns;
    const auto sheet_width = columns * width;
    const auto sheet_height = rows * height;
    if (sheet_width > std::numeric_limits<std::uint32_t>::max() ||
        sheet_height > std::numeric_limits<std::uint32_t>::max())
      dif::fail("difimage contact sheet dimensions are too large");
    std::vector<std::uint8_t> sheet(
        static_cast<std::size_t>(sheet_width * sheet_height * 3U), 0U);
    for (std::uint64_t frame = 0U; frame < frames; ++frame) {
      const auto origin_x = (frame % columns) * width;
      const auto origin_y = (frame / columns) * height;
      for (std::uint64_t y = 0U; y < height; ++y) {
        for (std::uint64_t x = 0U; x < width; ++x) {
          for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
            const auto source = (y * width + x) * 3U + channel;
            const auto target =
                ((origin_y + y) * sheet_width + origin_x + x) * 3U + channel;
            sheet[static_cast<std::size_t>(target)] =
                images[static_cast<std::size_t>(frame)]
                      [static_cast<std::size_t>(source)];
          }
        }
      }
    }
    write_bmp(output / "contact-sheet.bmp",
              static_cast<std::uint32_t>(sheet_width),
              static_cast<std::uint32_t>(sheet_height), sheet);
    std::cout << "DECODED_IMAGE PASS input=" << argv[1] << " shape=[1,3,"
              << frames << ',' << height << ',' << width << "] range=["
              << minimum << ',' << maximum << "] nonfinite=" << nonfinite
              << " outside_0_1=" << outside << " frames=" << frames
              << " contact_sheet=" << (output / "contact-sheet.bmp") << "\n";
    return nonfinite == 0U && outside == 0U ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "difimage: " << error.what() << "\n";
    return 1;
  }
}
