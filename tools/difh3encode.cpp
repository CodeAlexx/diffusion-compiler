#include "dif/backend/plugin.hpp"
#include "dif/frontend/h3_constants.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/png.hpp"
#include "dif/support/torch_cpu_rng.hpp"
#include "dif/weights/bundle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TilePlan {
  std::vector<std::uint64_t> starts;
  std::vector<std::uint64_t> lengths;
  std::vector<std::uint64_t> overlaps;
};

struct Options {
  std::string backend{"cpu"};
  std::filesystem::path backend_plugin;
  std::filesystem::path program;
  std::filesystem::path weight_bundle;
  std::filesystem::path image;
  std::filesystem::path output_moments;
  std::filesystem::path output_latent;
  std::filesystem::path output_rows;
  std::uint32_t pixels_id{};
  std::uint32_t moments_id{};
  std::uint64_t tile_size{256U};
  std::uint64_t tile_overlap{64U};
  std::uint64_t posterior_seed{42U};
  bool verify_shards{false};
  dif::runtime::RunOptions run;
};

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

void usage() {
  std::cerr
      << "usage: difh3encode --backend cpu|cuda --program TILE.difir"
         " --weight-bundle FILE.difbind --image FILE.png --pixels-id ID"
         " --moments-id ID --output-moments FILE.diftensor"
         " --output-latent FILE.diftensor --output-rows FILE.diftensor"
         " [--backend-plugin FILE.so] [--verify-shards] [--tile-size N]"
         " [--tile-overlap N] [--posterior-seed N] [--cache-dir DIR]"
         " [--min-free-mib N]\n";
}

Options parse(int argc, char **argv) {
  Options options;
  options.run.warmups = 0U;
  options.run.iterations = 1U;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *name) -> std::string {
      if (index + 1 >= argc)
        dif::fail(std::string("missing value for ") + name);
      return argv[++index];
    };
    if (option == "--backend")
      options.backend = value("--backend");
    else if (option == "--backend-plugin")
      options.backend_plugin = value("--backend-plugin");
    else if (option == "--program")
      options.program = value("--program");
    else if (option == "--weight-bundle")
      options.weight_bundle = value("--weight-bundle");
    else if (option == "--image")
      options.image = value("--image");
    else if (option == "--pixels-id")
      options.pixels_id = static_cast<std::uint32_t>(
          number(value("--pixels-id"), "pixels id"));
    else if (option == "--moments-id")
      options.moments_id = static_cast<std::uint32_t>(
          number(value("--moments-id"), "moments id"));
    else if (option == "--output-moments")
      options.output_moments = value("--output-moments");
    else if (option == "--output-latent")
      options.output_latent = value("--output-latent");
    else if (option == "--output-rows")
      options.output_rows = value("--output-rows");
    else if (option == "--tile-size")
      options.tile_size = number(value("--tile-size"), "tile size");
    else if (option == "--tile-overlap")
      options.tile_overlap =
          number(value("--tile-overlap"), "tile overlap");
    else if (option == "--posterior-seed")
      options.posterior_seed =
          number(value("--posterior-seed"), "posterior seed");
    else if (option == "--cache-dir")
      options.run.cache_directory = value("--cache-dir");
    else if (option == "--min-free-mib")
      options.run.minimum_free_bytes =
          number(value("--min-free-mib"), "minimum free memory") * 1024ULL *
          1024ULL;
    else if (option == "--verify-shards")
      options.verify_shards = true;
    else {
      usage();
      dif::fail("invalid difh3encode command line");
    }
  }
  if (options.program.empty() || options.weight_bundle.empty() ||
      options.image.empty() || options.output_moments.empty() ||
      options.output_latent.empty() || options.output_rows.empty() ||
      options.pixels_id == 0U || options.moments_id == 0U) {
    usage();
    dif::fail("difh3encode is missing a required argument");
  }
  if (options.tile_size == 0U || options.tile_overlap >= options.tile_size)
    dif::fail("invalid H3 VAE encode tiling policy");
  return options;
}

TilePlan split_tiles(std::uint64_t input_length, std::uint64_t tile_size,
                     std::uint64_t overlap_min, std::uint64_t ratio) {
  if (tile_size >= input_length)
    return {{0U}, {input_length}, {}};
  auto count = (input_length + tile_size - 1U) / tile_size;
  std::vector<std::uint64_t> overlaps;
  std::uint64_t remaining = 0U;
  for (;;) {
    overlaps.assign(static_cast<std::size_t>(count - 1U), overlap_min);
    const auto total_overlap =
        std::accumulate(overlaps.begin(), overlaps.end(), std::uint64_t{0U});
    const auto covered = tile_size * count;
    if (covered >= total_overlap + input_length) {
      remaining = covered - total_overlap - input_length;
      break;
    }
    ++count;
  }
  if (remaining % ratio != 0U)
    dif::fail("tile remainder is not aligned to H3 VAE spatial ratio");
  const auto units = remaining / ratio;
  for (std::uint64_t unit = 0U; unit < units; ++unit)
    overlaps[static_cast<std::size_t>(unit % (count - 1U))] += ratio;
  std::vector<std::uint64_t> starts{0U};
  starts.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0U; index + 1U < count; ++index)
    starts.push_back(starts.back() + tile_size - overlaps[index]);
  std::vector<std::uint64_t> lengths(static_cast<std::size_t>(count),
                                     tile_size);
  return {std::move(starts), std::move(lengths), std::move(overlaps)};
}

std::uint64_t ncthw_index(std::uint64_t channels, std::uint64_t frames,
                          std::uint64_t height, std::uint64_t width,
                          std::uint64_t channel, std::uint64_t frame,
                          std::uint64_t y, std::uint64_t x) {
  (void)channels;
  return (((channel * frames + frame) * height + y) * width + x);
}

dif::runtime::Tensor extract_pixels(const dif::RgbImage &image,
                                    std::uint64_t y_start,
                                    std::uint64_t x_start,
                                    std::uint64_t height,
                                    std::uint64_t width) {
  constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
  constexpr std::array<float, 3> standard_deviation{0.229F, 0.224F, 0.225F};
  if (y_start + height > image.height || x_start + width > image.width)
    dif::fail("H3 encoder tile exceeds source image");
  dif::runtime::Tensor tensor{dif::ir::DType::F32,
                              {1U, 3U, 1U, height, width}, {}};
  tensor.bytes.resize(
      static_cast<std::size_t>(tensor.element_count() * sizeof(float)));
  auto output = tensor.f32();
  for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
    for (std::uint64_t y = 0U; y < height; ++y) {
      for (std::uint64_t x = 0U; x < width; ++x) {
        const auto source =
            ((y_start + y) * image.width + x_start + x) * 3U + channel;
        const auto target = ncthw_index(3U, 1U, height, width, channel, 0U, y,
                                        x);
        const auto pixel = static_cast<float>(image.pixels[source]) / 255.0F;
        output[static_cast<std::size_t>(target)] =
            (pixel - mean[channel]) / standard_deviation[channel];
      }
    }
  }
  return tensor;
}

float blend(float left, float right, std::uint64_t index,
            std::uint64_t extent) {
  if (extent == 0U)
    return right;
  const auto alpha = static_cast<float>(index) / static_cast<float>(extent);
  return left * (1.0F - alpha) + right * alpha;
}

dif::runtime::Tensor stitch_moments(
    const std::vector<dif::runtime::Tensor> &tiles, const TilePlan &y_plan,
    const TilePlan &x_plan, std::uint64_t output_h, std::uint64_t output_w) {
  constexpr std::uint64_t channels = 48U;
  const auto rows = static_cast<std::uint64_t>(y_plan.starts.size());
  const auto columns = static_cast<std::uint64_t>(x_plan.starts.size());
  if (tiles.size() != rows * columns)
    dif::fail("H3 encoder tile count mismatch");
  dif::runtime::Tensor output{dif::ir::DType::F32,
                              {1U, channels, 1U, output_h, output_w}, {}};
  output.bytes.resize(
      static_cast<std::size_t>(output.element_count() * sizeof(float)));
  auto target = output.f32();
  std::uint64_t output_y = 0U;
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto tile_h = tiles[static_cast<std::size_t>(row * columns)].dims[3];
    const auto crop_h =
        row + 1U < rows ? tile_h - y_plan.overlaps[row] / 16U : tile_h;
    std::uint64_t output_x = 0U;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto tile_at = row * columns + column;
      const auto &current_tensor = tiles[static_cast<std::size_t>(tile_at)];
      if (current_tensor.dtype != dif::ir::DType::F32 ||
          current_tensor.dims !=
              std::vector<std::uint64_t>{1U, channels, 1U, tile_h,
                                         current_tensor.dims[4]})
        dif::fail("inconsistent H3 encoder moments tiles");
      const auto tile_w = current_tensor.dims[4];
      const auto crop_w = column + 1U < columns
                              ? tile_w - x_plan.overlaps[column] / 16U
                              : tile_w;
      const auto current = current_tensor.f32();
      for (std::uint64_t channel = 0U; channel < channels; ++channel) {
        for (std::uint64_t y = 0U; y < crop_h; ++y) {
          for (std::uint64_t x = 0U; x < crop_w; ++x) {
            auto value = current[static_cast<std::size_t>(ncthw_index(
                channels, 1U, tile_h, tile_w, channel, 0U, y, x))];
            if (row > 0U && y < y_plan.overlaps[row - 1U] / 16U) {
              const auto &above_tensor =
                  tiles[static_cast<std::size_t>((row - 1U) * columns +
                                                 column)];
              const auto above_h = above_tensor.dims[3];
              const auto above_w = above_tensor.dims[4];
              const auto overlap = y_plan.overlaps[row - 1U] / 16U;
              const auto above = above_tensor.f32();
              const auto previous = above[static_cast<std::size_t>(ncthw_index(
                  channels, 1U, above_h, above_w, channel, 0U,
                  above_h - overlap + y, x))];
              value = blend(previous, value, y, overlap);
            }
            if (column > 0U && x < x_plan.overlaps[column - 1U] / 16U) {
              const auto &left_tensor =
                  tiles[static_cast<std::size_t>(tile_at - 1U)];
              const auto left_h = left_tensor.dims[3];
              const auto left_w = left_tensor.dims[4];
              const auto overlap = x_plan.overlaps[column - 1U] / 16U;
              const auto left = left_tensor.f32();
              const auto previous = left[static_cast<std::size_t>(ncthw_index(
                  channels, 1U, left_h, left_w, channel, 0U, y,
                  left_w - overlap + x))];
              value = blend(previous, value, x, overlap);
            }
            target[static_cast<std::size_t>(ncthw_index(
                channels, 1U, output_h, output_w, channel, 0U, output_y + y,
                output_x + x))] = value;
          }
        }
      }
      output_x += crop_w;
    }
    if (output_x != output_w)
      dif::fail("H3 encoder tile width assembly mismatch");
    output_y += crop_h;
  }
  if (output_y != output_h)
    dif::fail("H3 encoder tile height assembly mismatch");
  return output;
}

std::pair<dif::runtime::Tensor, dif::runtime::Tensor>
sample_and_patchify(const dif::runtime::Tensor &moments,
                    std::uint64_t posterior_seed) {
  if (moments.dtype != dif::ir::DType::F32 || moments.dims.size() != 5U ||
      moments.dims[0] != 1U || moments.dims[1] != 48U ||
      moments.dims[2] != 1U || moments.dims[3] % 2U != 0U ||
      moments.dims[4] % 2U != 0U)
    dif::fail("H3 moments must be F32 [1,48,1,H,W] with even H/W");
  const auto height = moments.dims[3];
  const auto width = moments.dims[4];
  const auto latent_values = 24U * height * width;
  const auto noise = dif::torch_cpu_normal(
      static_cast<std::size_t>(latent_values), posterior_seed);
  dif::runtime::Tensor latent{dif::ir::DType::F32,
                              {1U, 24U, 1U, height, width}, {}};
  latent.bytes.resize(
      static_cast<std::size_t>(latent_values * sizeof(float)));
  const auto source = moments.f32();
  auto sampled = latent.f32();
  for (std::uint64_t index = 0U; index < latent_values; ++index) {
    const auto mean = source[static_cast<std::size_t>(index)];
    const auto log_variance = std::clamp(
        source[static_cast<std::size_t>(latent_values + index)], -30.0F,
        20.0F);
    const auto value = mean + std::exp(0.5F * log_variance) *
                                  noise[static_cast<std::size_t>(index)];
    sampled[static_cast<std::size_t>(index)] =
        dif::runtime::f16_to_float(dif::runtime::float_to_f16(value));
  }

  const auto rows_count = (height / 2U) * (width / 2U);
  dif::runtime::Tensor rows{dif::ir::DType::F32, {rows_count, 96U}, {}};
  rows.bytes.resize(
      static_cast<std::size_t>(rows.element_count() * sizeof(float)));
  auto output = rows.f32();
  std::uint64_t row = 0U;
  for (std::uint64_t y = 0U; y < height; y += 2U) {
    for (std::uint64_t x = 0U; x < width; x += 2U) {
      std::uint64_t column = 0U;
      for (std::uint64_t channel = 0U; channel < 24U; ++channel) {
        for (std::uint64_t patch_y = 0U; patch_y < 2U; ++patch_y) {
          for (std::uint64_t patch_x = 0U; patch_x < 2U; ++patch_x) {
            const auto at = ncthw_index(24U, 1U, height, width, channel, 0U,
                                        y + patch_y, x + patch_x);
            output[static_cast<std::size_t>(row * 96U + column++)] =
                (sampled[static_cast<std::size_t>(at)] -
                 dif::frontend::kH3VideoLatentMean[channel]) /
                dif::frontend::kH3VideoLatentStd[channel];
          }
        }
      }
      ++row;
    }
  }
  return {std::move(latent), std::move(rows)};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    constexpr std::uint64_t ratio = 16U;
    if (options.tile_size % ratio != 0U ||
        options.tile_overlap % ratio != 0U)
      dif::fail("H3 VAE tile size and overlap must align to ratio 16");
    const auto image = dif::read_png_rgb8(options.image);
    if (image.height % ratio != 0U || image.width % ratio != 0U)
      dif::fail("H3 reference image dimensions must be divisible by 16");
    const auto program = dif::ir::read_file(options.program);
    const auto *pixels_desc = program.tensor(options.pixels_id);
    const auto *moments_desc = program.tensor(options.moments_id);
    if (!pixels_desc || !moments_desc ||
        pixels_desc->dims !=
            std::vector<std::uint64_t>{1U, 3U, 1U, options.tile_size,
                                       options.tile_size} ||
        pixels_desc->dtype != dif::ir::DType::F32 ||
        moments_desc->dims !=
            std::vector<std::uint64_t>{1U, 48U, 1U,
                                       options.tile_size / ratio,
                                       options.tile_size / ratio} ||
        moments_desc->dtype != dif::ir::DType::F32)
      dif::fail("H3 encoder program does not match requested tile geometry");

    const auto y_plan = split_tiles(image.height, options.tile_size,
                                    options.tile_overlap, ratio);
    const auto x_plan = split_tiles(image.width, options.tile_size,
                                    options.tile_overlap, ratio);
    auto bindings = dif::weights::load_weight_bundle(
        dif::weights::read_weight_bundle(options.weight_bundle), program,
        options.verify_shards);
    if (bindings.contains(options.pixels_id))
      dif::fail("weight bundle unexpectedly binds H3 pixel input");
    bindings.emplace(options.pixels_id,
                     extract_pixels(image, y_plan.starts.front(),
                                    x_plan.starts.front(), options.tile_size,
                                    options.tile_size));

    std::unique_ptr<dif::runtime::Executor> executor;
    if (!options.backend_plugin.empty())
      executor = dif::backend::make_plugin_executor(options.backend_plugin);
    else if (options.backend == "cpu")
      executor = dif::runtime::make_cpu_executor();
    else if (options.backend == "cuda")
      executor = dif::runtime::make_cuda_executor();
    else
      dif::fail("unknown backend: " + options.backend);
    auto prepared = executor->prepare(program, bindings, options.run);

    std::vector<dif::runtime::Tensor> tiles;
    tiles.reserve(y_plan.starts.size() * x_plan.starts.size());
    double kernel_milliseconds = 0.0;
    std::string backend_name;
    std::string device_name;
    std::uint64_t free_before = 0U;
    std::uint64_t free_after = 0U;
    const auto wall_start = std::chrono::steady_clock::now();
    for (const auto y : y_plan.starts) {
      for (const auto x : x_plan.starts) {
        auto inputs = bindings;
        inputs[options.pixels_id] =
            extract_pixels(image, y, x, options.tile_size, options.tile_size);
        const auto result = prepared->run(inputs, options.run);
        const auto found = result.outputs.find(options.moments_id);
        if (found == result.outputs.end())
          dif::fail("H3 encoder execution did not return moments");
        tiles.push_back(found->second);
        kernel_milliseconds += result.mean_milliseconds;
        if (backend_name.empty()) {
          backend_name = result.backend_name;
          device_name = result.device_name;
          free_before = result.free_bytes_before;
        }
        free_after = result.free_bytes_after;
      }
    }
    auto moments = stitch_moments(tiles, y_plan, x_plan, image.height / ratio,
                                  image.width / ratio);
    auto [latent, rows] = sample_and_patchify(moments, options.posterior_seed);
    dif::runtime::write_tensor(moments, options.output_moments);
    dif::runtime::write_tensor(latent, options.output_latent);
    dif::runtime::write_tensor(rows, options.output_rows);
    const auto wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start)
            .count();
    std::cout << "H3_REFERENCE_ENCODE PASS backend=" << backend_name
              << " device=\"" << device_name << "\" image=" << image.width
              << 'x' << image.height << " moments=[1,48,1,"
              << moments.dims[3] << ',' << moments.dims[4] << "] rows="
              << rows.dims[0] << " tiles=" << tiles.size()
              << " prepare_ms=" << prepared->preparation_milliseconds()
              << " kernel_sum_ms=" << kernel_milliseconds
              << " wall_ms=" << wall_milliseconds
              << " resident_bytes=" << prepared->resident_bytes()
              << " free_before=" << free_before << " free_after=" << free_after
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3encode: " << error.what() << '\n';
    return 1;
  }
}
