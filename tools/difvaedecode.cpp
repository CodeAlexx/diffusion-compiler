#include "dif/backend/plugin.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
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
  std::filesystem::path input;
  std::filesystem::path output_raw;
  std::filesystem::path output_decoded;
  std::uint32_t latent_id{};
  std::uint32_t raw_id{};
  std::uint64_t clip_length{17U};
  std::uint64_t token_drop{3U};
  std::uint64_t tile_size{256U};
  std::uint64_t tile_overlap{64U};
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
      << "usage: difvaedecode --backend cpu|cuda --program TILE.difir"
         " --weight-bundle FILE.difbind --input LATENT.diftensor"
         " --latent-id ID --raw-id ID --output-raw FILE.diftensor"
         " --output-decoded FILE.diftensor [--backend-plugin FILE.so]"
         " [--verify-shards] [--clip-length N] [--token-drop N]"
         " [--tile-size N] [--tile-overlap N] [--warmups N]"
         " [--iterations N] [--min-free-mib N] [--cache-dir DIR]\n";
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
    else if (option == "--verify-shards")
      options.verify_shards = true;
    else {
      usage();
      dif::fail("invalid difvaedecode command line");
    }
  }
  if (options.program.empty() || options.weight_bundle.empty() ||
      options.input.empty() || options.output_raw.empty() ||
      options.output_decoded.empty() || options.latent_id == 0U ||
      options.raw_id == 0U) {
    usage();
    dif::fail("difvaedecode is missing a required argument");
  }
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
          current_tensor.dims[2] != frames ||
          current_tensor.dims[3] != tile_h)
        dif::fail("inconsistent VAE raw tile outputs");
      const auto tile_w = current_tensor.dims[4];
      const auto crop_w = column + 1U < columns
                              ? tile_w - x_plan.overlaps[column]
                              : tile_w;
      const auto current = current_tensor.f32();
      for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
        for (std::uint64_t frame = 0U; frame < frames; ++frame) {
          for (std::uint64_t y = 0U; y < crop_h; ++y) {
            for (std::uint64_t x = 0U; x < crop_w; ++x) {
              auto value = current[static_cast<std::size_t>(tensor_index(
                  3U, frames, tile_h, tile_w, channel, frame, y, x))];
              if (row > 0U && y < y_plan.overlaps[row - 1U]) {
                const auto &above_tensor =
                    tiles[static_cast<std::size_t>((row - 1U) * columns +
                                                   column)];
                const auto above_h = above_tensor.dims[3];
                const auto above_w = above_tensor.dims[4];
                const auto above = above_tensor.f32();
                const auto a = above[static_cast<std::size_t>(tensor_index(
                    3U, frames, above_h, above_w, channel, frame,
                    above_h - y_plan.overlaps[row - 1U] + y, x))];
                value = blended(a, value, y, y_plan.overlaps[row - 1U]);
              }
              if (column > 0U && x < x_plan.overlaps[column - 1U]) {
                const auto &left_tensor =
                    tiles[static_cast<std::size_t>(tile_at - 1U)];
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
                  output_y + y, output_x + x);
              target[static_cast<std::size_t>(output_at)] = value;
            }
          }
        }
      }
      output_x += crop_w;
    }
    if (output_x != output_w)
      dif::fail("spatial tile width assembly mismatch");
    output_y += crop_h;
  }
  if (output_y != output_h)
    dif::fail("spatial tile height assembly mismatch");
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
  for (std::uint64_t channel = 0U; channel < input.dims[1]; ++channel) {
    const auto source_offset = channel * input.dims[2] * frame_values;
    const auto target_offset = channel * frames * frame_values;
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(source_offset),
                static_cast<std::ptrdiff_t>(frames * frame_values),
                target.begin() + static_cast<std::ptrdiff_t>(target_offset));
  }
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
  minimum = std::numeric_limits<double>::infinity();
  maximum = -std::numeric_limits<double>::infinity();
  nonfinite = 0U;
  for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
    for (std::uint64_t index = 0U; index < channel_values; ++index) {
      const auto at = channel * channel_values + index;
      auto value = source[static_cast<std::size_t>(at)] *
                       standard_deviation[channel] +
                   mean[channel];
      value = std::clamp(value, 0.0F, 1.0F);
      if (!std::isfinite(value)) {
        ++nonfinite;
      } else {
        minimum = std::min(minimum, static_cast<double>(value));
        maximum = std::max(maximum, static_cast<double>(value));
      }
      target[static_cast<std::size_t>(at)] = value;
    }
  }
  return decoded;
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

    const auto decoded_tile_count = temporal_chunks * y_plan.starts.size() *
                                    x_plan.starts.size();
    std::vector<dif::runtime::Tensor> temporal_clips;
    temporal_clips.reserve(static_cast<std::size_t>(temporal_chunks));
    double kernel_milliseconds = 0.0;
    std::uint64_t free_before = 0U;
    std::uint64_t free_after = 0U;
    std::string backend_name;
    std::string device_name;
    std::string source_hash;
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint64_t temporal = 0U; temporal < temporal_chunks; ++temporal) {
      std::vector<dif::runtime::Tensor> tiles;
      tiles.reserve(y_plan.starts.size() * x_plan.starts.size());
      for (const auto y : y_plan.starts) {
        for (const auto x : x_plan.starts) {
          auto tile = extract_tile(latent, temporal * tokens_per_chunk,
                                   tile_tokens, y / spatial_ratio,
                                   x / spatial_ratio, tile_latent_h,
                                   tile_latent_w);
          auto inputs = bindings;
          inputs[options.latent_id] = std::move(tile);
          auto result = prepared->run(inputs, options.run);
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
      temporal_clips.push_back(
          stitch_spatial(tiles, y_plan, x_plan, output_h, output_w));
    }

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
    assembled.bytes.resize(
        static_cast<std::size_t>(assembled.element_count() * sizeof(float)));
    auto assembled_values = assembled.f32();
    const auto spatial_values = output_h * output_w;
    std::uint64_t write_frame = 0U;
    for (std::uint64_t temporal = 0U; temporal < temporal_chunks; ++temporal) {
      const auto current = temporal_clips[static_cast<std::size_t>(temporal)].f32();
      const auto *previous = temporal == 0U
                                 ? nullptr
                                 : &temporal_clips[static_cast<std::size_t>(
                                       temporal - 1U)];
      const auto blend_extent =
          previous ? std::min({frame_overlap, tail_frames, main_frames}) : 0U;
      for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
        for (std::uint64_t frame = 0U; frame < main_frames; ++frame) {
          for (std::uint64_t pixel = 0U; pixel < spatial_values; ++pixel) {
            const auto current_at =
                ((channel * clip_frames + frame_pre_padding + frame) *
                     spatial_values +
                 pixel);
            auto value = current[static_cast<std::size_t>(current_at)];
            if (previous && frame < blend_extent) {
              const auto previous_values = previous->f32();
              const auto previous_at =
                  ((channel * clip_frames + tail_start +
                    tail_frames - blend_extent + frame) *
                       spatial_values +
                   pixel);
              value = blended(
                  previous_values[static_cast<std::size_t>(previous_at)], value,
                  frame, blend_extent);
            }
            const auto target_at =
                ((channel * total_frames + write_frame + frame) *
                     spatial_values +
                 pixel);
            assembled_values[static_cast<std::size_t>(target_at)] = value;
          }
        }
      }
      write_frame += main_frames;
    }
    if (tail_frames > 0U) {
      const auto last = temporal_clips.back().f32();
      for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
        const auto source_at =
            (channel * clip_frames + tail_start) * spatial_values;
        const auto target_at =
            (channel * total_frames + write_frame) * spatial_values;
        std::copy_n(last.begin() + static_cast<std::ptrdiff_t>(source_at),
                    static_cast<std::ptrdiff_t>(tail_frames * spatial_values),
                    assembled_values.begin() +
                        static_cast<std::ptrdiff_t>(target_at));
      }
      write_frame += tail_frames;
    }
    if (write_frame != total_frames)
      dif::fail("temporal VAE assembly frame count mismatch");
    const auto padded_frames =
        pad_frames(latent.dims[2], padding_tokens, tokens_per_chunk,
                   options.clip_length, temporal_ratio);
    if (padded_frames >= total_frames)
      dif::fail("temporal VAE padding removed every decoded frame");
    auto raw = trim_temporal(assembled, total_frames - padded_frames);
    double minimum = 0.0;
    double maximum = 0.0;
    std::uint64_t nonfinite = 0U;
    auto decoded = pixel_denormalize(raw, minimum, maximum, nonfinite);
    if (nonfinite != 0U)
      dif::fail("decoded VAE output contains nonfinite values");
    dif::runtime::write_tensor(raw, options.output_raw);
    dif::runtime::write_tensor(decoded, options.output_decoded);
    const auto wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start)
            .count();
    std::cout << "H3_VAE_DECODE PASS backend=" << backend_name << " device=\""
              << device_name << "\" latent=[1,24," << latent.dims[2] << ','
              << latent.dims[3] << ',' << latent.dims[4] << "] decoded=[1,3,"
              << decoded.dims[2] << ',' << decoded.dims[3] << ','
              << decoded.dims[4] << "] temporal_chunks=" << temporal_chunks
              << " spatial_tiles="
              << y_plan.starts.size() * x_plan.starts.size()
              << " executions=" << decoded_tile_count
              << " prepare_ms=" << prepared->preparation_milliseconds()
              << " kernel_sum_ms=" << kernel_milliseconds
              << " wall_ms=" << wall_milliseconds
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
