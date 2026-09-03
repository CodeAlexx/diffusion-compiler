#include "dif/backend/plugin.hpp"
#include "dif/frontend/h3_media.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/sha256.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/bundle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <thread>
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
  std::filesystem::path input;
  std::filesystem::path output_raw;
  std::filesystem::path output_decoded;
  std::filesystem::path output_rgb;
  std::uint32_t latent_id{};
  std::uint32_t raw_id{};
  std::uint64_t clip_length{17U};
  std::uint64_t token_drop{3U};
  std::uint64_t tile_size{256U};
  std::uint64_t tile_overlap{64U};
  bool verify_shards{false};
  std::filesystem::path tile_digests; // --tile-digests FILE
  dif::runtime::RunOptions run;
};

std::size_t g_worker_override = 0U; // --workers N (0 = hardware concurrency)

std::size_t parallel_worker_count(std::size_t count) {
  const auto hardware = g_worker_override != 0U
                            ? static_cast<unsigned>(g_worker_override)
                            : std::thread::hardware_concurrency();
  return std::min<std::size_t>(count, hardware == 0U ? 1U : hardware);
}

template <class Function>
void parallel_ranges(std::size_t count, Function function) {
  const auto workers = parallel_worker_count(count);
  if (workers <= 1U) {
    if (count != 0U)
      function(0U, count, 0U);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t worker = 0U; worker < workers; ++worker) {
    const auto begin = count * worker / workers;
    const auto end = count * (worker + 1U) / workers;
    threads.emplace_back([=, &function] { function(begin, end, worker); });
  }
  for (auto &thread : threads)
    thread.join();
}

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

void usage() {
  std::cerr
      << "usage: difvaedecode --backend cpu|cuda --program TILE.difir"
         " --weight-bundle FILE.difbind --input LATENT.diftensor"
         " --latent-id ID --raw-id ID"
         " (--output-raw FILE.diftensor --output-decoded FILE.diftensor"
         " | --output-rgb FILE.rgb) [--backend-plugin FILE.so]"
         " [--verify-shards] [--clip-length N] [--token-drop N]"
         " [--tile-size N] [--tile-overlap N] [--warmups N]"
         " [--iterations N] [--min-free-mib N] [--cache-dir DIR]"
         " [--convrot-int8-checkpoint FILE]"
         " [--convrot-int8-linear-count N] [--convrot-int8-resident] [--deterministic-conv] [--trace-ops] [--workers N] [--tile-digests FILE] [--digest-tensor ID ...]\n";
}

Options parse(int argc, char **argv) {
  Options options;
  options.run.warmups = 0U;
  options.run.iterations = 1U;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--backend" && i + 1 < argc)
      options.backend = argv[++i];
    else if (option == "--backend-plugin" && i + 1 < argc)
      options.backend_plugin = argv[++i];
    else if (option == "--program" && i + 1 < argc)
      options.program = argv[++i];
    else if (option == "--weight-bundle" && i + 1 < argc)
      options.weight_bundle = argv[++i];
    else if (option == "--input" && i + 1 < argc)
      options.input = argv[++i];
    else if (option == "--output-raw" && i + 1 < argc)
      options.output_raw = argv[++i];
    else if (option == "--output-decoded" && i + 1 < argc)
      options.output_decoded = argv[++i];
    else if (option == "--output-rgb" && i + 1 < argc)
      options.output_rgb = argv[++i];
    else if (option == "--latent-id" && i + 1 < argc)
      options.latent_id =
          static_cast<std::uint32_t>(number(argv[++i], "latent id"));
    else if (option == "--raw-id" && i + 1 < argc)
      options.raw_id =
          static_cast<std::uint32_t>(number(argv[++i], "raw output id"));
    else if (option == "--clip-length" && i + 1 < argc)
      options.clip_length = number(argv[++i], "clip length");
    else if (option == "--token-drop" && i + 1 < argc)
      options.token_drop = number(argv[++i], "token drop");
    else if (option == "--tile-size" && i + 1 < argc)
      options.tile_size = number(argv[++i], "tile size");
    else if (option == "--tile-overlap" && i + 1 < argc)
      options.tile_overlap = number(argv[++i], "tile overlap");
    else if (option == "--warmups" && i + 1 < argc)
      options.run.warmups =
          static_cast<std::uint32_t>(number(argv[++i], "warmups"));
    else if (option == "--iterations" && i + 1 < argc)
      options.run.iterations =
          static_cast<std::uint32_t>(number(argv[++i], "iterations"));
    else if (option == "--min-free-mib" && i + 1 < argc)
      options.run.minimum_free_bytes =
          number(argv[++i], "minimum free memory") * 1024ULL * 1024ULL;
    else if (option == "--cache-dir" && i + 1 < argc)
      options.run.cache_directory = argv[++i];
    else if (option == "--convrot-int8-checkpoint" && i + 1 < argc)
      options.run.convrot_int8_checkpoint = argv[++i];
    else if (option == "--convrot-int8-linear-count" && i + 1 < argc)
      options.run.convrot_int8_linear_count = static_cast<std::uint32_t>(
          number(argv[++i], "ConvRot INT8 Linear count"));
    else if (option == "--convrot-int8-resident")
      options.run.convrot_int8_resident = true;
    else if (option == "--deterministic-conv")
      options.run.deterministic_convolution_algorithms = true;
    else if (option == "--workers" && i + 1 < argc)
      g_worker_override = static_cast<std::size_t>(number(argv[++i], "workers"));
    else if (option == "--tile-digests" && i + 1 < argc)
      options.tile_digests = argv[++i];
    else if (option == "--digest-tensor" && i + 1 < argc)
      options.run.capture_intermediate_tensors.push_back(
          static_cast<std::uint32_t>(number(argv[++i], "digest tensor")));
    else if (option == "--trace-ops")
      options.run.trace_operations = true;
    else if (option == "--verify-shards")
      options.verify_shards = true;
    else {
      usage();
      dif::fail("invalid difvaedecode command line");
    }
  }
  if (options.program.empty() || options.weight_bundle.empty() ||
      options.input.empty() || options.latent_id == 0U ||
      options.raw_id == 0U) {
    usage();
    dif::fail("difvaedecode is missing a required argument");
  }
  const auto tensor_output = !options.output_raw.empty() &&
                             !options.output_decoded.empty();
  const auto rgb_output = !options.output_rgb.empty();
  if (tensor_output == rgb_output ||
      options.output_raw.empty() != options.output_decoded.empty())
    dif::fail("select exactly one VAE delivery: raw+decoded tensors or RGB24");
  if (options.run.iterations == 0U || options.clip_length == 0U ||
      options.tile_size == 0U || options.tile_overlap >= options.tile_size)
    dif::fail("invalid VAE decode policy");
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
    dif::fail("tile remainder is not aligned to the VAE spatial ratio");
  const auto units = remaining / ratio;
  for (std::uint64_t unit = 0U; unit < units; ++unit)
    overlaps[static_cast<std::size_t>(unit % (count - 1U))] += ratio;
  std::vector<std::uint64_t> starts{0U};
  starts.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0U; index + 1U < count; ++index)
    starts.push_back(starts.back() + tile_size - overlaps[index]);
  return {std::move(starts),
          std::vector<std::uint64_t>(static_cast<std::size_t>(count),
                                     tile_size),
          std::move(overlaps)};
}

std::uint64_t tensor_index(std::uint64_t channels, std::uint64_t frames,
                           std::uint64_t height, std::uint64_t width,
                           std::uint64_t channel, std::uint64_t frame,
                           std::uint64_t y, std::uint64_t x) {
  (void)channels;
  return ((channel * frames + frame) * height + y) * width + x;
}

dif::runtime::Tensor extract_tile(const dif::runtime::Tensor &latent,
                                  std::uint64_t temporal_start,
                                  std::uint64_t clip_tokens,
                                  std::uint64_t spatial_y,
                                  std::uint64_t spatial_x,
                                  std::uint64_t tile_h,
                                  std::uint64_t tile_w) {
  const auto channels = latent.dims[1];
  const auto source_frames = latent.dims[2];
  const auto source_h = latent.dims[3];
  const auto source_w = latent.dims[4];
  dif::runtime::Tensor tile{
      dif::ir::DType::F32, {1U, channels, clip_tokens, tile_h, tile_w}, {}};
  tile.bytes.resize(static_cast<std::size_t>(tile.element_count() * sizeof(float)));
  const auto source = latent.f32();
  auto target = tile.f32();
  for (std::uint64_t channel = 0U; channel < channels; ++channel) {
    for (std::uint64_t frame = 0U; frame < clip_tokens; ++frame) {
      const auto source_frame =
          std::min(temporal_start + frame, source_frames - 1U);
      for (std::uint64_t y = 0U; y < tile_h; ++y) {
        for (std::uint64_t x = 0U; x < tile_w; ++x) {
          const auto source_at = tensor_index(
              channels, source_frames, source_h, source_w, channel,
              source_frame, spatial_y + y, spatial_x + x);
          const auto target_at = tensor_index(channels, clip_tokens, tile_h,
                                              tile_w, channel, frame, y, x);
          target[static_cast<std::size_t>(target_at)] =
              source[static_cast<std::size_t>(source_at)];
        }
      }
    }
  }
  return tile;
}

float blended(float a, float b, std::uint64_t position,
              std::uint64_t extent) {
  const auto weight_b = static_cast<float>(position) /
                        static_cast<float>(extent);
  const auto weight_a = 1.0F - weight_b;
  return a * weight_a + b * weight_b;
}

dif::runtime::Tensor stitch_spatial(
    const std::vector<dif::runtime::Tensor> &tiles, const TilePlan &y_plan,
    const TilePlan &x_plan, std::uint64_t output_h, std::uint64_t output_w) {
  const auto rows = y_plan.starts.size();
  const auto columns = x_plan.starts.size();
  if (tiles.size() != rows * columns || tiles.empty())
    dif::fail("spatial tile result count mismatch");
  const auto &first = tiles.front();
  if (first.dtype != dif::ir::DType::F32 || first.dims.size() != 5U ||
      first.dims[0] != 1U || first.dims[1] != 3U)
    dif::fail("VAE raw tile output must be F32 [1,3,T,H,W]");
  const auto frames = first.dims[2];
  dif::runtime::Tensor output{
      dif::ir::DType::F32, {1U, 3U, frames, output_h, output_w}, {}};
  output.bytes.resize(
      static_cast<std::size_t>(output.element_count() * sizeof(float)));
  auto target = output.f32();
  std::vector<std::uint64_t> tile_heights(tiles.size());
  std::vector<std::uint64_t> tile_widths(tiles.size());
  std::vector<std::uint64_t> crop_heights(tiles.size());
  std::vector<std::uint64_t> crop_widths(tiles.size());
  std::vector<std::uint64_t> output_ys(tiles.size());
  std::vector<std::uint64_t> output_xs(tiles.size());
  std::uint64_t output_y = 0U;
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto tile_h = tiles[static_cast<std::size_t>(row * columns)].dims[3];
    const auto crop_h =
        row + 1U < rows ? tile_h - y_plan.overlaps[row] : tile_h;
    std::uint64_t output_x = 0U;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto tile_at = row * columns + column;
      const auto &current_tensor = tiles[static_cast<std::size_t>(tile_at)];
      if (current_tensor.dtype != dif::ir::DType::F32 ||
          current_tensor.dims.size() != 5U || current_tensor.dims[0] != 1U ||
          current_tensor.dims[1] != 3U ||
          current_tensor.dims[2] != frames ||
          current_tensor.dims[3] != tile_h)
        dif::fail("inconsistent VAE raw tile outputs");
      const auto tile_w = current_tensor.dims[4];
      const auto crop_w = column + 1U < columns
                              ? tile_w - x_plan.overlaps[column]
                              : tile_w;
      const auto index = static_cast<std::size_t>(tile_at);
      tile_heights[index] = tile_h;
      tile_widths[index] = tile_w;
      crop_heights[index] = crop_h;
      crop_widths[index] = crop_w;
      output_ys[index] = output_y;
      output_xs[index] = output_x;
      output_x += crop_w;
    }
    if (output_x != output_w)
      dif::fail("spatial tile width assembly mismatch");
    output_y += crop_h;
  }
  if (output_y != output_h)
    dif::fail("spatial tile height assembly mismatch");

  const auto tasks = tiles.size() * 3U * static_cast<std::size_t>(frames);
  parallel_ranges(tasks, [&](std::size_t begin, std::size_t end,
                             std::size_t) {
    for (auto task = begin; task < end; ++task) {
      const auto tile_at = task / (3U * static_cast<std::size_t>(frames));
      const auto local = task % (3U * static_cast<std::size_t>(frames));
      const auto channel = static_cast<std::uint64_t>(local / frames);
      const auto frame = static_cast<std::uint64_t>(local % frames);
      const auto row = tile_at / columns;
      const auto column = tile_at % columns;
      const auto &current_tensor = tiles[tile_at];
      const auto current = current_tensor.f32();
      const auto tile_h = tile_heights[tile_at];
      const auto tile_w = tile_widths[tile_at];
      const auto crop_h = crop_heights[tile_at];
      const auto crop_w = crop_widths[tile_at];
      for (std::uint64_t y = 0U; y < crop_h; ++y) {
        for (std::uint64_t x = 0U; x < crop_w; ++x) {
          auto value = current[static_cast<std::size_t>(tensor_index(
              3U, frames, tile_h, tile_w, channel, frame, y, x))];
          if (row > 0U && y < y_plan.overlaps[row - 1U]) {
            const auto &above_tensor = tiles[(row - 1U) * columns + column];
            const auto above_h = above_tensor.dims[3];
            const auto above_w = above_tensor.dims[4];
            const auto above = above_tensor.f32();
            const auto a = above[static_cast<std::size_t>(tensor_index(
                3U, frames, above_h, above_w, channel, frame,
                above_h - y_plan.overlaps[row - 1U] + y, x))];
            value = blended(a, value, y, y_plan.overlaps[row - 1U]);
          }
          if (column > 0U && x < x_plan.overlaps[column - 1U]) {
            const auto &left_tensor = tiles[tile_at - 1U];
            const auto left_h = left_tensor.dims[3];
            const auto left_w = left_tensor.dims[4];
            const auto left = left_tensor.f32();
            const auto a = left[static_cast<std::size_t>(tensor_index(
                3U, frames, left_h, left_w, channel, frame, y,
                left_w - x_plan.overlaps[column - 1U] + x))];
            value = blended(a, value, x, x_plan.overlaps[column - 1U]);
          }
          const auto output_at = tensor_index(
              3U, frames, output_h, output_w, channel, frame,
              output_ys[tile_at] + y, output_xs[tile_at] + x);
          target[static_cast<std::size_t>(output_at)] = value;
        }
      }
    }
  });
  return output;
}

std::uint64_t pad_frames(std::uint64_t original_tokens,
                         std::uint64_t padding_tokens,
                         std::uint64_t tokens_per_chunk,
                         std::uint64_t clip_length,
                         std::uint64_t temporal_ratio) {
  if (padding_tokens == 0U)
    return 0U;
  const auto tail = clip_length % temporal_ratio;
  if (tail == 0U)
    return padding_tokens * temporal_ratio;
  std::uint64_t frames = 0U;
  for (std::uint64_t token = 0U; token < padding_tokens; ++token)
    frames += (original_tokens + token) % tokens_per_chunk == 0U
                  ? tail
                  : temporal_ratio;
  return frames;
}

dif::runtime::Tensor trim_temporal(const dif::runtime::Tensor &input,
                                   std::uint64_t frames) {
  if (input.dtype != dif::ir::DType::F32 || input.dims.size() != 5U ||
      frames > input.dims[2])
    dif::fail("invalid temporal trim");
  dif::runtime::Tensor output{
      dif::ir::DType::F32,
      {input.dims[0], input.dims[1], frames, input.dims[3], input.dims[4]},
      {}};
  output.bytes.resize(
      static_cast<std::size_t>(output.element_count() * sizeof(float)));
  const auto source = input.f32();
  auto target = output.f32();
  const auto frame_values = input.dims[3] * input.dims[4];
  parallel_ranges(static_cast<std::size_t>(input.dims[1]),
                  [&](std::size_t begin, std::size_t end, std::size_t) {
    for (auto channel = begin; channel < end; ++channel) {
      const auto source_offset = channel * input.dims[2] * frame_values;
      const auto target_offset = channel * frames * frame_values;
      std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(source_offset),
                  static_cast<std::ptrdiff_t>(frames * frame_values),
                  target.begin() + static_cast<std::ptrdiff_t>(target_offset));
    }
  });
  return output;
}

dif::runtime::Tensor pixel_denormalize(const dif::runtime::Tensor &raw,
                                       double &minimum, double &maximum,
                                       std::uint64_t &nonfinite) {
  constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
  constexpr std::array<float, 3> standard_deviation = {0.229F, 0.224F,
                                                       0.225F};
  dif::runtime::Tensor decoded{dif::ir::DType::F32, raw.dims, {}};
  decoded.bytes.resize(
      static_cast<std::size_t>(decoded.element_count() * sizeof(float)));
  const auto source = raw.f32();
  auto target = decoded.f32();
  const auto channel_values = raw.dims[2] * raw.dims[3] * raw.dims[4];
  const auto value_count = static_cast<std::size_t>(3U * channel_values);
  const auto workers = parallel_worker_count(value_count);
  std::vector<double> minima(workers,
                             std::numeric_limits<double>::infinity());
  std::vector<double> maxima(workers,
                             -std::numeric_limits<double>::infinity());
  std::vector<std::uint64_t> nonfinite_counts(workers, 0U);
  parallel_ranges(value_count, [&](std::size_t begin, std::size_t end,
                                   std::size_t worker) {
    auto local_minimum = minima[worker];
    auto local_maximum = maxima[worker];
    auto local_nonfinite = nonfinite_counts[worker];
    for (auto at = begin; at < end; ++at) {
      const auto channel = at / channel_values;
      auto value = source[static_cast<std::size_t>(at)] *
                       standard_deviation[channel] +
                   mean[channel];
      value = std::clamp(value, 0.0F, 1.0F);
      if (!std::isfinite(value)) {
        ++local_nonfinite;
      } else {
        local_minimum = std::min(local_minimum, static_cast<double>(value));
        local_maximum = std::max(local_maximum, static_cast<double>(value));
      }
      target[static_cast<std::size_t>(at)] = value;
    }
    minima[worker] = local_minimum;
    maxima[worker] = local_maximum;
    nonfinite_counts[worker] = local_nonfinite;
  });
  minimum = *std::min_element(minima.begin(), minima.end());
  maximum = *std::max_element(maxima.begin(), maxima.end());
  nonfinite = std::accumulate(nonfinite_counts.begin(), nonfinite_counts.end(),
                              std::uint64_t{0U});
  return decoded;
}

dif::frontend::H3Rgb24Video pixel_denormalize_rgb24(
    const dif::runtime::Tensor &raw, double &minimum, double &maximum,
    std::uint64_t &nonfinite) {
  constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
  constexpr std::array<float, 3> standard_deviation = {0.229F, 0.224F,
                                                       0.225F};
  if (raw.dtype != dif::ir::DType::F32 || raw.dims.size() != 5U ||
      raw.dims[0] != 1U || raw.dims[1] != 3U)
    dif::fail("VAE RGB24 handoff requires F32 [1,3,F,H,W]");
  dif::frontend::H3Rgb24Video output;
  output.frames = raw.dims[2];
  output.height = raw.dims[3];
  output.width = raw.dims[4];
  const auto channel_values =
      output.frames * output.height * output.width;
  output.bytes.resize(static_cast<std::size_t>(channel_values * 3U));
  const auto source = raw.f32();
  const auto value_count = static_cast<std::size_t>(channel_values * 3U);
  const auto workers = parallel_worker_count(value_count);
  std::vector<double> minima(workers,
                             std::numeric_limits<double>::infinity());
  std::vector<double> maxima(workers,
                             -std::numeric_limits<double>::infinity());
  std::vector<std::uint64_t> nonfinite_counts(workers, 0U);
  parallel_ranges(value_count, [&](std::size_t begin, std::size_t end,
                                   std::size_t worker) {
    auto local_minimum = minima[worker];
    auto local_maximum = maxima[worker];
    auto local_nonfinite = nonfinite_counts[worker];
    for (auto at = begin; at < end; ++at) {
      const auto channel = at / channel_values;
      const auto pixel = at % channel_values;
      auto value = source[at] * standard_deviation[channel] + mean[channel];
      value = std::clamp(value, 0.0F, 1.0F);
      volatile float storage_rounded_value = value;
      const auto delivered_value = storage_rounded_value;
      if (!std::isfinite(delivered_value)) {
        ++local_nonfinite;
        output.bytes[pixel * 3U + channel] = 0U;
      } else {
        local_minimum =
            std::min(local_minimum, static_cast<double>(delivered_value));
        local_maximum =
            std::max(local_maximum, static_cast<double>(delivered_value));
        output.bytes[pixel * 3U + channel] = static_cast<std::uint8_t>(
            std::lround(delivered_value * 255.0F));
      }
    }
    minima[worker] = local_minimum;
    maxima[worker] = local_maximum;
    nonfinite_counts[worker] = local_nonfinite;
  });
  minimum = *std::min_element(minima.begin(), minima.end());
  maximum = *std::max_element(maxima.begin(), maxima.end());
  nonfinite = std::accumulate(nonfinite_counts.begin(), nonfinite_counts.end(),
                              std::uint64_t{0U});
  output.minimum = static_cast<float>(minimum);
  output.maximum = static_cast<float>(maximum);
  return output;
}

struct SpatialAxisLookup {
  std::vector<std::uint64_t> tile;
  std::vector<std::uint64_t> local;
};

SpatialAxisLookup make_spatial_axis_lookup(const TilePlan &plan,
                                           std::uint64_t output_length) {
  SpatialAxisLookup lookup;
  lookup.tile.resize(static_cast<std::size_t>(output_length));
  lookup.local.resize(static_cast<std::size_t>(output_length));
  auto output = std::uint64_t{0U};
  for (std::uint64_t tile = 0U; tile < plan.starts.size(); ++tile) {
    const auto crop = tile + 1U < plan.starts.size()
                          ? plan.lengths[static_cast<std::size_t>(tile)] -
                                plan.overlaps[static_cast<std::size_t>(tile)]
                          : plan.lengths[static_cast<std::size_t>(tile)];
    if (output + crop > output_length)
      dif::fail("spatial VAE fused lookup exceeds output geometry");
    for (std::uint64_t local = 0U; local < crop; ++local) {
      lookup.tile[static_cast<std::size_t>(output + local)] = tile;
      lookup.local[static_cast<std::size_t>(output + local)] = local;
    }
    output += crop;
  }
  if (output != output_length)
    dif::fail("spatial VAE fused lookup does not cover output geometry");
  return lookup;
}

float spatial_tile_value(const std::vector<dif::runtime::Tensor> &tiles,
                         const TilePlan &y_plan, const TilePlan &x_plan,
                         const SpatialAxisLookup &y_lookup,
                         const SpatialAxisLookup &x_lookup,
                         std::uint64_t channel, std::uint64_t frame,
                         std::uint64_t y, std::uint64_t x) {
  const auto rows = y_plan.starts.size();
  const auto columns = x_plan.starts.size();
  const auto row = y_lookup.tile[static_cast<std::size_t>(y)];
  const auto column = x_lookup.tile[static_cast<std::size_t>(x)];
  if (row >= rows || column >= columns || tiles.size() != rows * columns)
    dif::fail("spatial VAE fused tile lookup is invalid");
  const auto local_y = y_lookup.local[static_cast<std::size_t>(y)];
  const auto local_x = x_lookup.local[static_cast<std::size_t>(x)];
  const auto tile_at = row * columns + column;
  const auto &current_tensor = tiles[static_cast<std::size_t>(tile_at)];
  const auto frames = current_tensor.dims[2];
  const auto tile_h = current_tensor.dims[3];
  const auto tile_w = current_tensor.dims[4];
  const auto current = current_tensor.f32();
  auto value = current[static_cast<std::size_t>(tensor_index(
      3U, frames, tile_h, tile_w, channel, frame, local_y, local_x))];
  if (row > 0U && local_y < y_plan.overlaps[row - 1U]) {
    const auto &above_tensor =
        tiles[static_cast<std::size_t>((row - 1U) * columns + column)];
    const auto above_h = above_tensor.dims[3];
    const auto above_w = above_tensor.dims[4];
    const auto above = above_tensor.f32();
    const auto a = above[static_cast<std::size_t>(tensor_index(
        3U, frames, above_h, above_w, channel, frame,
        above_h - y_plan.overlaps[row - 1U] + local_y, local_x))];
    value = blended(a, value, local_y, y_plan.overlaps[row - 1U]);
  }
  if (column > 0U && local_x < x_plan.overlaps[column - 1U]) {
    const auto &left_tensor =
        tiles[static_cast<std::size_t>(tile_at - 1U)];
    const auto left_h = left_tensor.dims[3];
    const auto left_w = left_tensor.dims[4];
    const auto left = left_tensor.f32();
    const auto a = left[static_cast<std::size_t>(tensor_index(
        3U, frames, left_h, left_w, channel, frame, local_y,
        left_w - x_plan.overlaps[column - 1U] + local_x))];
    value = blended(a, value, local_x, x_plan.overlaps[column - 1U]);
  }
  return value;
}

struct TemporalFrameLookup {
  std::uint64_t current_chunk{};
  std::uint64_t current_frame{};
  std::uint64_t previous_chunk{};
  std::uint64_t previous_frame{};
  std::uint64_t blend_position{};
  std::uint64_t blend_extent{};
  bool blend{};
};

dif::frontend::H3Rgb24Video assemble_rgb24_fused(
    const std::vector<std::vector<dif::runtime::Tensor>> &temporal_tiles,
    const TilePlan &y_plan, const TilePlan &x_plan, std::uint64_t output_h,
    std::uint64_t output_w, std::uint64_t frame_pre_padding,
    std::uint64_t frame_overlap, std::uint64_t tokens_per_chunk,
    std::uint64_t temporal_ratio, std::uint64_t padding_frames,
    double &minimum, double &maximum, std::uint64_t &nonfinite) {
  constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
  constexpr std::array<float, 3> standard_deviation = {0.229F, 0.224F,
                                                       0.225F};
  if (temporal_tiles.empty() || temporal_tiles.front().empty())
    dif::fail("fused VAE RGB24 assembly requires decoded tiles");
  const auto &first = temporal_tiles.front().front();
  if (first.dtype != dif::ir::DType::F32 || first.dims.size() != 5U ||
      first.dims[0] != 1U || first.dims[1] != 3U)
    dif::fail("fused VAE RGB24 tile must be F32 [1,3,T,H,W]");
  const auto clip_frames = first.dims[2];
  for (const auto &tiles : temporal_tiles) {
    if (tiles.size() != y_plan.starts.size() * x_plan.starts.size())
      dif::fail("fused VAE RGB24 tile count mismatch");
    for (const auto &tile : tiles) {
      if (tile.dtype != dif::ir::DType::F32 || tile.dims.size() != 5U ||
          tile.dims[0] != 1U || tile.dims[1] != 3U ||
          tile.dims[2] != clip_frames)
        dif::fail("fused VAE RGB24 tile geometry mismatch");
    }
  }
  const auto chunk_decoded_frames = tokens_per_chunk * temporal_ratio;
  const auto main_end = std::min(chunk_decoded_frames, clip_frames);
  if (main_end < frame_pre_padding)
    dif::fail("fused VAE RGB24 clip is shorter than its pre-padding");
  const auto main_frames = main_end - frame_pre_padding;
  const auto tail_start =
      std::min(chunk_decoded_frames + frame_pre_padding, clip_frames);
  const auto tail_frames = clip_frames - tail_start;
  const auto total_frames = temporal_tiles.size() * main_frames + tail_frames;
  if (padding_frames >= total_frames)
    dif::fail("fused VAE RGB24 padding removed every decoded frame");
  const auto output_frames = total_frames - padding_frames;
  const auto blend_extent =
      std::min({frame_overlap, tail_frames, main_frames});
  std::vector<TemporalFrameLookup> temporal_lookup(
      static_cast<std::size_t>(output_frames));
  for (std::uint64_t frame = 0U; frame < output_frames; ++frame) {
    auto &entry = temporal_lookup[static_cast<std::size_t>(frame)];
    if (frame < temporal_tiles.size() * main_frames) {
      entry.current_chunk = frame / main_frames;
      const auto local = frame % main_frames;
      entry.current_frame = frame_pre_padding + local;
      if (entry.current_chunk > 0U && local < blend_extent) {
        entry.blend = true;
        entry.previous_chunk = entry.current_chunk - 1U;
        entry.previous_frame =
            tail_start + tail_frames - blend_extent + local;
        entry.blend_position = local;
        entry.blend_extent = blend_extent;
      }
    } else {
      entry.current_chunk = temporal_tiles.size() - 1U;
      entry.current_frame =
          tail_start + frame - temporal_tiles.size() * main_frames;
    }
  }

  const auto y_lookup = make_spatial_axis_lookup(y_plan, output_h);
  const auto x_lookup = make_spatial_axis_lookup(x_plan, output_w);
  dif::frontend::H3Rgb24Video output;
  output.frames = output_frames;
  output.height = output_h;
  output.width = output_w;
  const auto pixels = output_frames * output_h * output_w;
  output.bytes.resize(static_cast<std::size_t>(pixels * 3U));
  const auto tasks = static_cast<std::size_t>(output_frames * output_h);
  const auto workers = parallel_worker_count(tasks);
  std::vector<double> minima(workers,
                             std::numeric_limits<double>::infinity());
  std::vector<double> maxima(workers,
                             -std::numeric_limits<double>::infinity());
  std::vector<std::uint64_t> nonfinite_counts(workers, 0U);
  parallel_ranges(tasks, [&](std::size_t begin, std::size_t end,
                             std::size_t worker) {
    auto local_minimum = minima[worker];
    auto local_maximum = maxima[worker];
    auto local_nonfinite = nonfinite_counts[worker];
    for (auto task = begin; task < end; ++task) {
      const auto frame = static_cast<std::uint64_t>(task) / output_h;
      const auto y = static_cast<std::uint64_t>(task) % output_h;
      const auto &temporal = temporal_lookup[static_cast<std::size_t>(frame)];
      for (std::uint64_t x = 0U; x < output_w; ++x) {
        const auto pixel = (frame * output_h + y) * output_w + x;
        for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
          auto value = spatial_tile_value(
              temporal_tiles[static_cast<std::size_t>(temporal.current_chunk)],
              y_plan, x_plan, y_lookup, x_lookup, channel,
              temporal.current_frame, y, x);
          if (temporal.blend) {
            const auto previous = spatial_tile_value(
                temporal_tiles[static_cast<std::size_t>(
                    temporal.previous_chunk)],
                y_plan, x_plan, y_lookup, x_lookup, channel,
                temporal.previous_frame, y, x);
            value = blended(previous, value, temporal.blend_position,
                            temporal.blend_extent);
          }
          value = value * standard_deviation[channel] + mean[channel];
          value = std::clamp(value, 0.0F, 1.0F);
          volatile float storage_rounded_value = value;
          const auto delivered_value = storage_rounded_value;
          if (!std::isfinite(delivered_value)) {
            ++local_nonfinite;
            output.bytes[static_cast<std::size_t>(pixel * 3U + channel)] = 0U;
          } else {
            local_minimum =
                std::min(local_minimum, static_cast<double>(delivered_value));
            local_maximum =
                std::max(local_maximum, static_cast<double>(delivered_value));
            output.bytes[static_cast<std::size_t>(pixel * 3U + channel)] =
                static_cast<std::uint8_t>(
                    std::lround(delivered_value * 255.0F));
          }
        }
      }
    }
    minima[worker] = local_minimum;
    maxima[worker] = local_maximum;
    nonfinite_counts[worker] = local_nonfinite;
  });
  minimum = *std::min_element(minima.begin(), minima.end());
  maximum = *std::max_element(maxima.begin(), maxima.end());
  nonfinite = std::accumulate(nonfinite_counts.begin(),
                              nonfinite_counts.end(), std::uint64_t{0U});
  output.minimum = static_cast<float>(minimum);
  output.maximum = static_cast<float>(maximum);
  return output;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    constexpr std::uint64_t spatial_ratio = 16U;
    constexpr std::uint64_t temporal_ratio = 4U;
    if (options.tile_size % spatial_ratio != 0U ||
        options.tile_overlap % spatial_ratio != 0U)
      dif::fail("tile size and overlap must align to the VAE spatial ratio 16");

    const auto program = dif::ir::read_file(options.program);
    const auto *latent_desc = program.tensor(options.latent_id);
    const auto *raw_desc = program.tensor(options.raw_id);
    if (!latent_desc || !raw_desc)
      dif::fail("VAE tile program is missing the requested tensor id");
    if (latent_desc->dtype != dif::ir::DType::F32 ||
        latent_desc->dims.size() != 5U || latent_desc->dims[0] != 1U ||
        latent_desc->dims[1] != 24U)
      dif::fail("VAE tile latent must be F32 [1,24,T,H,W]");
    if (raw_desc->dtype != dif::ir::DType::F32 ||
        raw_desc->dims.size() != 5U || raw_desc->dims[0] != 1U ||
        raw_desc->dims[1] != 3U)
      dif::fail("VAE tile raw output must be F32 [1,3,T,H,W]");

    const auto latent = dif::runtime::read_tensor(options.input);
    if (latent.dtype != dif::ir::DType::F32 || latent.dims.size() != 5U ||
        latent.dims[0] != 1U || latent.dims[1] != 24U ||
        latent.dims[2] == 0U || latent.dims[3] == 0U || latent.dims[4] == 0U)
      dif::fail("input latent must be F32 [1,24,T,H,W]");

    const auto tokens_per_chunk =
        (options.clip_length + temporal_ratio - 1U) / temporal_ratio;
    if (options.token_drop >= tokens_per_chunk)
      dif::fail("token drop must be smaller than tokens per chunk");
    const auto token_overlap =
        (tokens_per_chunk - options.token_drop % tokens_per_chunk) %
        tokens_per_chunk;
    const auto frame_pre_padding =
        (temporal_ratio - options.clip_length % temporal_ratio) %
        temporal_ratio;
    const auto frame_overlap =
        std::max(token_overlap * temporal_ratio, frame_pre_padding) -
        frame_pre_padding;
    auto pseudo_tokens = latent.dims[2] + options.token_drop;
    const auto remainder = pseudo_tokens % tokens_per_chunk;
    const auto padding_tokens =
        remainder == 0U ? 0U : tokens_per_chunk - remainder;
    pseudo_tokens += padding_tokens;
    const auto temporal_chunks =
        pseudo_tokens / tokens_per_chunk - (options.token_drop > 0U ? 1U : 0U);
    const auto tile_tokens = tokens_per_chunk + token_overlap;
    if (temporal_chunks == 0U || latent_desc->dims[2] != tile_tokens)
      dif::fail("tile program temporal geometry does not match decode policy");

    const auto output_h = latent.dims[3] * spatial_ratio;
    const auto output_w = latent.dims[4] * spatial_ratio;
    const auto y_plan = split_tiles(output_h, options.tile_size,
                                    options.tile_overlap, spatial_ratio);
    const auto x_plan = split_tiles(output_w, options.tile_size,
                                    options.tile_overlap, spatial_ratio);
    const auto tile_latent_h = y_plan.lengths.front() / spatial_ratio;
    const auto tile_latent_w = x_plan.lengths.front() / spatial_ratio;
    if (latent_desc->dims[3] != tile_latent_h ||
        latent_desc->dims[4] != tile_latent_w)
      dif::fail("tile program spatial geometry does not match decode policy");

    auto bindings = dif::weights::load_weight_bundle(
        dif::weights::read_weight_bundle(options.weight_bundle), program,
        options.verify_shards);
    if (bindings.contains(options.latent_id))
      dif::fail("weight bundle unexpectedly binds the dynamic latent");
    auto representative = extract_tile(latent, 0U, tile_tokens,
                                       y_plan.starts.front() / spatial_ratio,
                                       x_plan.starts.front() / spatial_ratio,
                                       tile_latent_h, tile_latent_w);
    bindings.emplace(options.latent_id, representative);

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
    // --trace-ops: per-operation device timings aggregated over every tile
    // execution, printed by opcode so the decode's launch mix is attributable.
    std::map<std::string, std::pair<std::uint64_t, double>> opcode_timings;

    const auto decoded_tile_count = temporal_chunks * y_plan.starts.size() *
                                    x_plan.starts.size();
    std::vector<dif::runtime::Tensor> temporal_clips;
    temporal_clips.reserve(static_cast<std::size_t>(temporal_chunks));
    std::vector<std::vector<dif::runtime::Tensor>> fused_temporal_tiles;
    if (!options.output_rgb.empty())
      fused_temporal_tiles.reserve(static_cast<std::size_t>(temporal_chunks));
    double kernel_milliseconds = 0.0;
    std::uint64_t free_before = 0U;
    std::uint64_t free_after = 0U;
    std::string backend_name;
    std::string device_name;
    std::string source_hash;
    std::size_t convrot_linear_count = 0U;
    bool convrot_identity_checked = false;
    double tile_extract_milliseconds = 0.0;
    double execution_wall_milliseconds = 0.0;
    double spatial_stitch_milliseconds = 0.0;
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint64_t temporal = 0U; temporal < temporal_chunks; ++temporal) {
      std::vector<dif::runtime::Tensor> tiles;
      tiles.reserve(y_plan.starts.size() * x_plan.starts.size());
      for (const auto y : y_plan.starts) {
        for (const auto x : x_plan.starts) {
          const auto extract_start = std::chrono::steady_clock::now();
          auto tile = extract_tile(latent, temporal * tokens_per_chunk,
                                   tile_tokens, y / spatial_ratio,
                                   x / spatial_ratio, tile_latent_h,
                                   tile_latent_w);
          tile_extract_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - extract_start)
                  .count();
          auto inputs = bindings;
          inputs[options.latent_id] = std::move(tile);
          const auto execution_start = std::chrono::steady_clock::now();
          auto result = prepared->run(inputs, options.run);
          if (!options.tile_digests.empty()) {
            // SHA-256 of every raw tile output exactly as the executor returned
            // it, before any host-side stitching: separates GPU-side from
            // host-side nondeterminism.
            std::ofstream digests(options.tile_digests, std::ios::app);
            for (const auto &[id, tensor] : result.outputs)
              digests << "chunk=" << temporal << " tile_offset=" << tiles.size() << " output=" << id
                      << " sha256=" << dif::hex_digest(dif::sha256(
                             std::span<const std::uint8_t>(tensor.data(), tensor.byte_size())))
                      << "\n";
            for (const auto &[id, tensor] : result.captured_intermediates)
              digests << "chunk=" << temporal << " tile_offset=" << tiles.size() << " intermediate=" << id
                      << " sha256=" << dif::hex_digest(dif::sha256(
                             std::span<const std::uint8_t>(tensor.data(), tensor.byte_size())))
                      << "\n";
          }
          for (const auto &timing : result.operation_timings) {
            auto &slot = opcode_timings[timing.plan.empty()
                                            ? std::string(dif::ir::opcode_name(timing.opcode))
                                            : timing.plan];
            slot.first += 1U;
            slot.second += timing.mean_milliseconds;
          }
          execution_wall_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - execution_start)
                  .count();
          if (!options.run.convrot_int8_checkpoint.empty() &&
              !convrot_identity_checked) {
            if (result.convrot_int8_linears.empty())
              dif::fail("VAE ConvRot INT8 checkpoint produced no admitted Linears");
            if (options.run.convrot_int8_linear_count != 0U &&
                result.convrot_int8_linears.size() !=
                    options.run.convrot_int8_linear_count)
              dif::fail("VAE ConvRot INT8 admitted Linear count mismatch");
            if (!std::all_of(
                    result.convrot_int8_linears.begin(),
                    result.convrot_int8_linears.end(), [](const auto &linear) {
                      return linear.implementation ==
                             "generic_diffir_linear_cutlass_scaled_f16";
                    }))
              dif::fail("VAE ConvRot INT8 did not select the F16 CUTLASS route");
            convrot_linear_count = result.convrot_int8_linears.size();
            convrot_identity_checked = true;
          }
          const auto found = result.outputs.find(options.raw_id);
          if (found == result.outputs.end())
            dif::fail("tile execution did not return the requested raw output");
          tiles.push_back(found->second);
          kernel_milliseconds += result.mean_milliseconds;
          if (backend_name.empty()) {
            backend_name = result.backend_name;
            device_name = result.device_name;
            source_hash = result.generated_source_hash;
            free_before = result.free_bytes_before;
          }
          free_after = result.free_bytes_after;
        }
      }
      if (options.output_rgb.empty()) {
        const auto stitch_start = std::chrono::steady_clock::now();
        temporal_clips.push_back(
            stitch_spatial(tiles, y_plan, x_plan, output_h, output_w));
        spatial_stitch_milliseconds +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - stitch_start)
                .count();
      } else {
        fused_temporal_tiles.push_back(std::move(tiles));
      }
    }

    const auto temporal_assembly_start = std::chrono::steady_clock::now();
    const auto padded_frames =
        pad_frames(latent.dims[2], padding_tokens, tokens_per_chunk,
                   options.clip_length, temporal_ratio);
    double minimum = 0.0;
    double maximum = 0.0;
    std::uint64_t nonfinite = 0U;
    dif::runtime::Tensor raw;
    dif::runtime::Tensor decoded;
    dif::frontend::H3Rgb24Video rgb;
    double temporal_assembly_milliseconds = 0.0;
    double denormalize_milliseconds = 0.0;
    if (options.output_rgb.empty()) {
      const auto clip_frames = temporal_clips.front().dims[2];
      const auto chunk_decoded_frames = tokens_per_chunk * temporal_ratio;
      const auto main_end = std::min(chunk_decoded_frames, clip_frames);
      if (main_end < frame_pre_padding)
        dif::fail("temporal VAE clip is shorter than its pre-padding");
      const auto main_frames = main_end - frame_pre_padding;
      const auto tail_start =
          std::min(chunk_decoded_frames + frame_pre_padding, clip_frames);
      const auto tail_frames = clip_frames - tail_start;
      const auto total_frames = temporal_chunks * main_frames + tail_frames;
      dif::runtime::Tensor assembled{
          dif::ir::DType::F32, {1U, 3U, total_frames, output_h, output_w}, {}};
      assembled.bytes.resize(static_cast<std::size_t>(
          assembled.element_count() * sizeof(float)));
      auto assembled_values = assembled.f32();
      const auto spatial_values = output_h * output_w;
      const auto assembly_tasks = static_cast<std::size_t>(
          temporal_chunks * 3U * main_frames);
      parallel_ranges(assembly_tasks, [&](std::size_t begin, std::size_t end,
                                          std::size_t) {
        for (auto task = begin; task < end; ++task) {
          const auto temporal =
              static_cast<std::uint64_t>(task / (3U * main_frames));
          const auto local =
              static_cast<std::uint64_t>(task % (3U * main_frames));
          const auto channel = local / main_frames;
          const auto frame = local % main_frames;
          const auto current =
              temporal_clips[static_cast<std::size_t>(temporal)].f32();
          const auto *previous =
              temporal == 0U
                  ? nullptr
                  : &temporal_clips[static_cast<std::size_t>(temporal - 1U)];
          const auto blend_extent =
              previous
                  ? std::min({frame_overlap, tail_frames, main_frames})
                  : 0U;
          for (std::uint64_t pixel = 0U; pixel < spatial_values; ++pixel) {
            const auto current_at =
                ((channel * clip_frames + frame_pre_padding + frame) *
                     spatial_values +
                 pixel);
            auto value = current[static_cast<std::size_t>(current_at)];
            if (previous && frame < blend_extent) {
              const auto previous_values = previous->f32();
              const auto previous_at =
                  ((channel * clip_frames + tail_start + tail_frames -
                    blend_extent + frame) *
                       spatial_values +
                   pixel);
              value = blended(
                  previous_values[static_cast<std::size_t>(previous_at)],
                  value, frame, blend_extent);
            }
            const auto target_at =
                ((channel * total_frames + temporal * main_frames + frame) *
                     spatial_values +
                 pixel);
            assembled_values[static_cast<std::size_t>(target_at)] = value;
          }
        }
      });
      auto write_frame = temporal_chunks * main_frames;
      if (tail_frames > 0U) {
        const auto last = temporal_clips.back().f32();
        parallel_ranges(3U, [&](std::size_t begin, std::size_t end,
                                std::size_t) {
          for (auto channel = begin; channel < end; ++channel) {
            const auto source_at =
                (channel * clip_frames + tail_start) * spatial_values;
            const auto target_at =
                (channel * total_frames + write_frame) * spatial_values;
            std::copy_n(
                last.begin() + static_cast<std::ptrdiff_t>(source_at),
                static_cast<std::ptrdiff_t>(tail_frames * spatial_values),
                assembled_values.begin() +
                    static_cast<std::ptrdiff_t>(target_at));
          }
        });
        write_frame += tail_frames;
      }
      if (write_frame != total_frames)
        dif::fail("temporal VAE assembly frame count mismatch");
      if (padded_frames >= total_frames)
        dif::fail("temporal VAE padding removed every decoded frame");
      raw = trim_temporal(assembled, total_frames - padded_frames);
      temporal_assembly_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - temporal_assembly_start)
              .count();
      const auto denormalize_start = std::chrono::steady_clock::now();
      decoded = pixel_denormalize(raw, minimum, maximum, nonfinite);
      denormalize_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - denormalize_start)
              .count();
    } else {
      rgb = assemble_rgb24_fused(
          fused_temporal_tiles, y_plan, x_plan, output_h, output_w,
          frame_pre_padding, frame_overlap, tokens_per_chunk, temporal_ratio,
          padded_frames, minimum, maximum, nonfinite);
      temporal_assembly_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - temporal_assembly_start)
              .count();
    }
    if (nonfinite != 0U)
      dif::fail("decoded VAE output contains nonfinite values");
    const auto output_io_start = std::chrono::steady_clock::now();
    const auto output_class = options.output_rgb.empty() ? "diagnostic-tensors"
                                                         : "product-rgb24";
    if (options.output_rgb.empty()) {
      dif::runtime::write_tensor(raw, options.output_raw);
      dif::runtime::write_tensor(decoded, options.output_decoded);
    } else {
      std::ofstream output(options.output_rgb,
                           std::ios::binary | std::ios::trunc);
      if (!output)
        dif::fail("cannot create VAE RGB24 output");
      output.write(reinterpret_cast<const char *>(rgb.bytes.data()),
                   static_cast<std::streamsize>(rgb.bytes.size()));
      if (!output)
        dif::fail("cannot write VAE RGB24 output");
    }
    const auto output_io_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - output_io_start)
            .count();
    const auto wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start)
            .count();
    const auto delivered_frames =
        options.output_rgb.empty() ? raw.dims[2] : rgb.frames;
    const auto delivered_height =
        options.output_rgb.empty() ? raw.dims[3] : rgb.height;
    const auto delivered_width =
        options.output_rgb.empty() ? raw.dims[4] : rgb.width;
    if (!opcode_timings.empty()) {
      std::vector<std::pair<std::string, std::pair<std::uint64_t, double>>> rows(
          opcode_timings.begin(), opcode_timings.end());
      std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
        return a.second.second > b.second.second;
      });
      double total = 0.0;
      for (const auto &row : rows)
        total += row.second.second;
      for (const auto &row : rows)
        std::cout << "H3_VAE_DECODE_OP opcode=" << row.first
                  << " launches=" << row.second.first
                  << " total_ms=" << row.second.second
                  << " share=" << (total > 0.0 ? row.second.second / total : 0.0)
                  << '\n';
    }
    std::cout << "H3_VAE_DECODE PASS backend=" << backend_name << " device=\""
              << device_name << "\" latent=[1,24," << latent.dims[2] << ','
              << latent.dims[3] << ',' << latent.dims[4] << "] decoded=[1,3,"
              << delivered_frames << ',' << delivered_height << ','
              << delivered_width
              << "] temporal_chunks=" << temporal_chunks
              << " spatial_tiles="
              << y_plan.starts.size() * x_plan.starts.size()
              << " executions=" << decoded_tile_count
              << " prepare_ms=" << prepared->preparation_milliseconds()
              << " kernel_sum_ms=" << kernel_milliseconds
              << " tile_extract_ms=" << tile_extract_milliseconds
              << " execution_wall_ms=" << execution_wall_milliseconds
              << " spatial_stitch_ms=" << spatial_stitch_milliseconds
              << " temporal_assembly_ms=" << temporal_assembly_milliseconds
              << " denormalize_ms=" << denormalize_milliseconds
              << " output_io_ms=" << output_io_milliseconds
              << " output_class=" << output_class
              << " wall_ms=" << wall_milliseconds
              << " convrot_int8_linears=" << convrot_linear_count
              << " resident_bytes=" << prepared->resident_bytes()
              << " free_before=" << free_before << " free_after=" << free_after
              << " range=[" << minimum << ',' << maximum << "]"
              << " source_hash=" << source_hash << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difvaedecode: " << error.what() << "\n";
    return 1;
  }
}
