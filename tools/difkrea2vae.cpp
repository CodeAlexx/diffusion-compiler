#include "dif/frontend/krea2_vae.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/png.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path checkpoint, fixture, sampler, reference, config, png,
      output, report, diffir;
};

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::filesystem::path {
      if (++index >= argc)
        dif::fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--checkpoint")
      result.checkpoint = value();
    else if (option == "--fixture")
      result.fixture = value();
    else if (option == "--sampler")
      result.sampler = value();
    else if (option == "--reference")
      result.reference = value();
    else if (option == "--config")
      result.config = value();
    else if (option == "--png")
      result.png = value();
    else if (option == "--output")
      result.output = value();
    else if (option == "--report")
      result.report = value();
    else if (option == "--diffir")
      result.diffir = value();
    else
      dif::fail("invalid difkrea2vae argument: " + option);
  }
  const auto full = !result.sampler.empty();
  if (result.checkpoint.empty() || result.output.empty() ||
      result.report.empty() || result.diffir.empty() ||
      (!full && result.fixture.empty()) ||
      (full && (result.reference.empty() || result.config.empty() ||
                result.png.empty())))
    dif::fail("difkrea2vae tile mode requires checkpoint/fixture/output/report/diffir; full mode replaces fixture with sampler/reference/config/png");
  if (std::filesystem::exists(result.output) ||
      std::filesystem::exists(result.report) ||
      std::filesystem::exists(result.diffir) ||
      (!result.png.empty() && std::filesystem::exists(result.png)))
    dif::fail("refusing to overwrite an existing Krea 2 VAE artifact");
  return result;
}

dif::runtime::Tensor conv3d_last_temporal_slice(
    const dif::runtime::Tensor &source,
    const dif::ir::TensorDesc &destination) {
  if (source.dtype != destination.dtype || source.dims.size() != 5U ||
      destination.dims.size() != 4U ||
      source.dims[0] != destination.dims[0] ||
      source.dims[1] != destination.dims[1] ||
      source.dims[3] != destination.dims[2] ||
      source.dims[4] != destination.dims[3])
    dif::fail("Qwen Image Conv3d slice shape mismatch");
  dif::runtime::Tensor result{destination.dtype, destination.dims, {}};
  result.bytes.resize(static_cast<std::size_t>(
      result.element_count() * dif::ir::dtype_size(result.dtype)));
  const auto outer = source.dims[0] * source.dims[1];
  const auto temporal = source.dims[2];
  const auto plane = source.dims[3] * source.dims[4];
  const auto plane_bytes =
      static_cast<std::size_t>(plane * dif::ir::dtype_size(source.dtype));
  for (std::uint64_t index = 0U; index < outer; ++index) {
    const auto source_offset = (index * temporal + temporal - 1U) * plane;
    std::memcpy(result.mutable_data() + index * plane_bytes,
                source.data() +
                    source_offset * dif::ir::dtype_size(source.dtype),
                plane_bytes);
  }
  result.validate();
  return result;
}

dif::runtime::Tensor bind_weight(
    const dif::weights::SafeTensorFile &checkpoint,
    const dif::frontend::Krea2VaeWeightBinding &binding,
    const dif::ir::TensorDesc &destination, std::uint64_t &converted) {
  auto source =
      dif::weights::map_safetensor(checkpoint, binding.source_name);
  if (source.dtype == dif::ir::DType::F32 &&
      destination.dtype == dif::ir::DType::BF16) {
    source = dif::runtime::convert_float_tensor(source, destination.dtype);
    ++converted;
  }
  switch (binding.transform) {
  case dif::frontend::Krea2VaeWeightTransform::Direct:
    break;
  case dif::frontend::Krea2VaeWeightTransform::FlattenSingletonDimensions:
    if (source.element_count() != destination.element_count())
      dif::fail("Qwen Image RMS gamma element-count mismatch: " +
                binding.source_name);
    source.dims = destination.dims;
    break;
  case dif::frontend::Krea2VaeWeightTransform::Conv3dLastTemporalSlice:
    source = conv3d_last_temporal_slice(source, destination);
    break;
  }
  if (source.dtype != destination.dtype || source.dims != destination.dims)
    dif::fail("Qwen Image checkpoint tensor mismatch: " +
              binding.source_name);
  source.validate();
  return source;
}

struct Metrics {
  double cosine{}, relative_l2{}, max_absolute{}, norm_ratio{};
  std::uint64_t nonfinite{}, bit_mismatches{}, elements{};
};

Metrics measure(const dif::runtime::Tensor &reference,
                const dif::runtime::Tensor &actual) {
  if (reference.dtype != actual.dtype || reference.dims != actual.dims)
    dif::fail("Krea 2 VAE boundary comparison shape/dtype mismatch");
  long double dot = 0.0L;
  long double reference_squared = 0.0L;
  long double actual_squared = 0.0L;
  long double error_squared = 0.0L;
  Metrics result;
  result.elements = reference.element_count();
  const auto width = dif::ir::dtype_size(reference.dtype);
  for (std::uint64_t index = 0U; index < result.elements; ++index) {
    result.bit_mismatches +=
        std::memcmp(reference.data() + index * width,
                    actual.data() + index * width, width) != 0;
    const auto expected =
        static_cast<double>(dif::runtime::load_float(reference, index));
    const auto observed =
        static_cast<double>(dif::runtime::load_float(actual, index));
    if (!std::isfinite(expected) || !std::isfinite(observed)) {
      ++result.nonfinite;
      continue;
    }
    const auto error = observed - expected;
    dot += static_cast<long double>(expected) * observed;
    reference_squared += static_cast<long double>(expected) * expected;
    actual_squared += static_cast<long double>(observed) * observed;
    error_squared += static_cast<long double>(error) * error;
    result.max_absolute = std::max(result.max_absolute, std::abs(error));
  }
  const auto denominator = std::sqrt(reference_squared * actual_squared);
  result.cosine = denominator == 0.0L
                      ? 1.0
                      : static_cast<double>(dot / denominator);
  result.relative_l2 =
      reference_squared == 0.0L
          ? 0.0
          : static_cast<double>(std::sqrt(error_squared / reference_squared));
  result.norm_ratio =
      reference_squared == 0.0L
          ? 1.0
          : static_cast<double>(std::sqrt(actual_squared / reference_squared));
  return result;
}

void emit(std::ostream &out, const Metrics &value) {
  out << "{\"cosine\":" << value.cosine
      << ",\"relative_l2\":" << value.relative_l2
      << ",\"max_absolute\":" << value.max_absolute
      << ",\"norm_ratio\":" << value.norm_ratio
      << ",\"nonfinite\":" << value.nonfinite
      << ",\"bit_mismatches\":" << value.bit_mismatches
      << ",\"elements\":" << value.elements << "}";
}

dif::runtime::Tensor bf16_tensor(std::vector<std::uint64_t> dims,
                                 const std::vector<float> &values) {
  dif::runtime::Tensor result{dif::ir::DType::BF16, std::move(dims), {}};
  if (result.element_count() != values.size())
    dif::fail("BF16 tensor value count mismatch");
  result.bytes.resize(values.size() * sizeof(std::uint16_t));
  for (std::size_t index = 0U; index < values.size(); ++index)
    dif::runtime::store_float(result, index, values[index]);
  result.validate();
  return result;
}

std::vector<float> config_vector(const std::filesystem::path &path,
                                 std::string_view key) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    dif::fail("cannot open Qwen Image VAE config: " + path.string());
  const auto end = input.tellg();
  if (end < 0 || end > 1024 * 1024)
    dif::fail("Qwen Image VAE config has invalid size");
  std::string text(static_cast<std::size_t>(end), '\0');
  input.seekg(0);
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input)
    dif::fail("cannot read Qwen Image VAE config");
  const auto root = dif::json::parse(text);
  const auto *value = root.find(key);
  if (!value || !value->is_array() || value->array().size() != 16U)
    dif::fail("Qwen Image VAE config vector must contain 16 values: " +
              std::string(key));
  std::vector<float> result;
  result.reserve(16U);
  for (const auto &item : value->array())
    result.push_back(static_cast<float>(item.number()));
  return result;
}

dif::runtime::Tensor unpatch(const dif::runtime::Tensor &tokens) {
  if (tokens.dtype != dif::ir::DType::BF16 ||
      tokens.dims != std::vector<std::uint64_t>{1U, 4096U, 64U})
    dif::fail("Krea 2 final tokens must be BF16 [1,4096,64]");
  dif::runtime::Tensor latent{dif::ir::DType::BF16,
                              {1U, 16U, 128U, 128U}, {}};
  latent.bytes.resize(static_cast<std::size_t>(latent.element_count() * 2U));
  for (std::uint64_t patch_y = 0U; patch_y < 64U; ++patch_y)
    for (std::uint64_t patch_x = 0U; patch_x < 64U; ++patch_x) {
      const auto token = patch_y * 64U + patch_x;
      for (std::uint64_t channel = 0U; channel < 16U; ++channel)
        for (std::uint64_t py = 0U; py < 2U; ++py)
          for (std::uint64_t px = 0U; px < 2U; ++px) {
            const auto feature = (channel * 2U + py) * 2U + px;
            const auto target =
                (channel * 128U + patch_y * 2U + py) * 128U +
                patch_x * 2U + px;
            std::memcpy(latent.mutable_data() + target * 2U,
                        tokens.data() + (token * 64U + feature) * 2U, 2U);
          }
    }
  latent.validate();
  return latent;
}

dif::runtime::Tensor extract_tile(const dif::runtime::Tensor &latent,
                                  std::uint64_t start_y,
                                  std::uint64_t start_x,
                                  std::uint64_t height,
                                  std::uint64_t width) {
  dif::runtime::Tensor tile{dif::ir::DType::BF16,
                            {1U, 16U, height, width}, {}};
  tile.bytes.resize(static_cast<std::size_t>(tile.element_count() * 2U));
  for (std::uint64_t channel = 0U; channel < 16U; ++channel)
    for (std::uint64_t y = 0U; y < height; ++y) {
      const auto source = (channel * 128U + start_y + y) * 128U + start_x;
      const auto target = (channel * height + y) * width;
      std::memcpy(tile.mutable_data() + target * 2U,
                  latent.data() + source * 2U,
                  static_cast<std::size_t>(width * 2U));
    }
  tile.validate();
  return tile;
}

float round_bf16(float value) {
  return dif::runtime::bf16_to_float(dif::runtime::float_to_bf16(value));
}

void blend_vertical(const dif::runtime::Tensor &above,
                    dif::runtime::Tensor &current,
                    std::uint64_t requested_extent) {
  const auto extent = std::min({above.dims[2], current.dims[2],
                                requested_extent});
  for (std::uint64_t y = 0U; y < extent; ++y) {
    const auto weight_b = static_cast<float>(y) / static_cast<float>(extent);
    const auto weight_a = 1.0F - weight_b;
    for (std::uint64_t channel = 0U; channel < 3U; ++channel)
      for (std::uint64_t x = 0U; x < current.dims[3]; ++x) {
        const auto source =
            (channel * above.dims[2] + above.dims[2] - extent + y) *
                above.dims[3] +
            x;
        const auto target =
            (channel * current.dims[2] + y) * current.dims[3] + x;
        const auto a = round_bf16(
            dif::runtime::load_float(above, source) * weight_a);
        const auto b = round_bf16(
            dif::runtime::load_float(current, target) * weight_b);
        dif::runtime::store_float(current, target, round_bf16(a + b));
      }
  }
}

void blend_horizontal(const dif::runtime::Tensor &left,
                      dif::runtime::Tensor &current,
                      std::uint64_t requested_extent) {
  const auto extent = std::min({left.dims[3], current.dims[3],
                                requested_extent});
  for (std::uint64_t x = 0U; x < extent; ++x) {
    const auto weight_b = static_cast<float>(x) / static_cast<float>(extent);
    const auto weight_a = 1.0F - weight_b;
    for (std::uint64_t channel = 0U; channel < 3U; ++channel)
      for (std::uint64_t y = 0U; y < current.dims[2]; ++y) {
        const auto source =
            (channel * left.dims[2] + y) * left.dims[3] +
            left.dims[3] - extent + x;
        const auto target =
            (channel * current.dims[2] + y) * current.dims[3] + x;
        const auto a =
            round_bf16(dif::runtime::load_float(left, source) * weight_a);
        const auto b = round_bf16(
            dif::runtime::load_float(current, target) * weight_b);
        dif::runtime::store_float(current, target, round_bf16(a + b));
      }
  }
}

struct PreparedTile {
  dif::frontend::Krea2VaeBuild build;
  dif::runtime::TensorMap bindings;
  std::unique_ptr<dif::runtime::PreparedExecution> execution;
  std::string fingerprint;
};

std::unique_ptr<PreparedTile> prepare_tile(
    dif::runtime::Executor &backend,
    const dif::weights::SafeTensorFile &checkpoint,
    const dif::runtime::Tensor &latent_std,
    const dif::runtime::Tensor &latent_mean, std::uint64_t height,
    std::uint64_t width, const dif::runtime::RunOptions &options,
    std::uint64_t &converted) {
  auto result = std::make_unique<PreparedTile>();
  dif::frontend::Krea2VaeConfig config;
  config.latent_height = height;
  config.latent_width = width;
  config.capture_boundaries = false;
  result->build = dif::frontend::make_krea2_qwen_image_vae(config);
  const auto *input = result->build.program.tensor(result->build.latent_input);
  result->bindings.emplace(result->build.latent_input,
                           dif::runtime::zeros(*input));
  result->bindings.emplace(result->build.latent_std, latent_std);
  result->bindings.emplace(result->build.latent_mean, latent_mean);
  for (const auto &binding : result->build.weights) {
    const auto *destination = result->build.program.tensor(binding.tensor);
    result->bindings.emplace(
        binding.tensor,
        bind_weight(checkpoint, binding, *destination, converted));
  }
  result->fingerprint =
      dif::hex_digest(dif::ir::fingerprint(result->build.program));
  result->execution =
      backend.prepare(result->build.program, result->bindings, options);
  return result;
}

int run_full(const Arguments &arguments) {
  const auto checkpoint =
      dif::weights::read_safetensors(arguments.checkpoint);
  const auto sampler = dif::weights::read_safetensors(arguments.sampler);
  const auto reference = dif::weights::read_safetensors(arguments.reference);
  const auto tokens =
      dif::weights::map_safetensor(sampler, "final_image_tokens");
  const auto latent = unpatch(tokens);
  const auto latent_std =
      bf16_tensor({16U}, config_vector(arguments.config, "latents_std"));
  const auto latent_mean =
      bf16_tensor({16U}, config_vector(arguments.config, "latents_mean"));

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 512ULL * 1024ULL * 1024ULL;
  auto backend = dif::runtime::make_cuda_executor();
  std::array<std::pair<std::uint64_t, std::uint64_t>, 4> shapes{
      std::pair{32U, 32U}, std::pair{32U, 8U}, std::pair{8U, 32U},
      std::pair{8U, 8U}};
  std::array<std::unique_ptr<PreparedTile>, 4> prepared;
  std::uint64_t converted = 0U;
  double preparation_milliseconds = 0.0;
  std::uint64_t resident_bytes = 0U;
  for (std::size_t index = 0U; index < shapes.size(); ++index) {
    prepared[index] = prepare_tile(
        *backend, checkpoint, latent_std, latent_mean, shapes[index].first,
        shapes[index].second, options, converted);
    preparation_milliseconds +=
        prepared[index]->execution->preparation_milliseconds();
    resident_bytes += prepared[index]->execution->resident_bytes();
  }
  dif::ir::write_file(prepared[0]->build.program, arguments.diffir);

  constexpr std::array<std::uint64_t, 6> starts{0U, 24U, 48U, 72U,
                                                96U, 120U};
  std::vector<dif::runtime::Tensor> tiles;
  tiles.reserve(36U);
  double kernel_milliseconds = 0.0;
  std::uint64_t launches = 0U;
  std::uint64_t attention_dispatches = 0U;
  std::uint64_t convolution_dispatches = 0U;
  std::uint64_t free_bytes_before = 0U;
  std::uint64_t free_bytes_after = std::numeric_limits<std::uint64_t>::max();
  std::string device_name;
  const auto wall_started = std::chrono::steady_clock::now();
  for (const auto start_y : starts) {
    for (const auto start_x : starts) {
      const auto height = std::min<std::uint64_t>(32U, 128U - start_y);
      const auto width = std::min<std::uint64_t>(32U, 128U - start_x);
      const auto shape_index = height == 32U ? (width == 32U ? 0U : 1U)
                                             : (width == 32U ? 2U : 3U);
      auto &plan = *prepared[shape_index];
      auto inputs = plan.bindings;
      inputs.insert_or_assign(plan.build.latent_input,
                              extract_tile(latent, start_y, start_x, height,
                                           width));
      auto result = plan.execution->run(inputs, options);
      tiles.push_back(result.outputs.at(plan.build.raw_output));
      kernel_milliseconds += result.mean_milliseconds;
      launches += result.run_telemetry.kernel_launches +
                  result.run_telemetry.cublaslt_matmuls +
                  result.run_telemetry.cudnn_attention_dispatches +
                  result.run_telemetry.cudnn_convolution_dispatches;
      attention_dispatches +=
          result.run_telemetry.cudnn_attention_dispatches;
      convolution_dispatches +=
          result.run_telemetry.cudnn_convolution_dispatches;
      if (device_name.empty()) {
        device_name = result.device_name;
        free_bytes_before = result.free_bytes_before;
      }
      free_bytes_after = std::min(free_bytes_after, result.free_bytes_after);
    }
  }

  dif::runtime::Tensor raw{dif::ir::DType::BF16,
                           {1U, 3U, 1024U, 1024U}, {}};
  raw.bytes.resize(static_cast<std::size_t>(raw.element_count() * 2U));
  for (std::size_t row = 0U; row < starts.size(); ++row) {
    for (std::size_t column = 0U; column < starts.size(); ++column) {
      auto &tile = tiles[row * starts.size() + column];
      if (row != 0U)
        blend_vertical(tiles[(row - 1U) * starts.size() + column], tile, 64U);
      if (column != 0U)
        blend_horizontal(tiles[row * starts.size() + column - 1U], tile, 64U);
      const auto copy_h = std::min<std::uint64_t>(192U, tile.dims[2]);
      const auto copy_w = std::min<std::uint64_t>(192U, tile.dims[3]);
      const auto origin_y = row * 192U;
      const auto origin_x = column * 192U;
      for (std::uint64_t channel = 0U; channel < 3U; ++channel)
        for (std::uint64_t y = 0U; y < copy_h; ++y) {
          const auto source = (channel * tile.dims[2] + y) * tile.dims[3];
          const auto target =
              (channel * 1024U + origin_y + y) * 1024U + origin_x;
          std::memcpy(raw.mutable_data() + target * 2U,
                      tile.data() + source * 2U,
                      static_cast<std::size_t>(copy_w * 2U));
        }
    }
  }
  raw.validate();
  dif::runtime::Tensor clamped{dif::ir::DType::BF16, raw.dims, {}};
  clamped.bytes.resize(raw.byte_size());
  for (std::uint64_t index = 0U; index < raw.element_count(); ++index)
    dif::runtime::store_float(
        clamped, index,
        std::clamp(dif::runtime::load_float(raw, index), -1.0F, 1.0F));
  clamped.validate();
  const auto wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_started)
          .count();

  dif::weights::SafeTensorWriter writer(
      arguments.output,
      {{"raw_output", raw.dtype, raw.dims},
       {"clamped_output", clamped.dtype, clamped.dims}});
  writer.append("raw_output", std::span<const std::uint8_t>(
                                  raw.data(), raw.byte_size()));
  writer.append("clamped_output", std::span<const std::uint8_t>(
                                      clamped.data(), clamped.byte_size()));
  (void)writer.finish();

  std::vector<std::uint8_t> rgb(1024U * 1024U * 3U);
  for (std::uint64_t channel = 0U; channel < 3U; ++channel)
    for (std::uint64_t y = 0U; y < 1024U; ++y)
      for (std::uint64_t x = 0U; x < 1024U; ++x) {
        const auto tensor_index = (channel * 1024U + y) * 1024U + x;
        auto value = round_bf16(
            dif::runtime::load_float(clamped, tensor_index) * 0.5F);
        value = round_bf16(value + 0.5F);
        value = round_bf16(value * 255.0F);
        const auto pixel_index = (y * 1024U + x) * 3U + channel;
        rgb[pixel_index] = static_cast<std::uint8_t>(
            std::clamp(value, 0.0F, 255.0F));
      }
  dif::write_png_rgb8(arguments.png, 1024U, 1024U, rgb);

  const auto raw_reference =
      dif::weights::map_safetensor(reference, "raw_output");
  const auto clamped_reference =
      dif::weights::map_safetensor(reference, "clamped_output");
  const auto raw_metrics = measure(raw_reference, raw);
  const auto clamped_metrics = measure(clamped_reference, clamped);
  const auto admitted = raw_metrics.nonfinite == 0U &&
                        clamped_metrics.nonfinite == 0U &&
                        clamped_metrics.cosine >= 0.9999 &&
                        clamped_metrics.relative_l2 <= 0.02;
  std::ofstream report(arguments.report, std::ios::trunc);
  report << std::setprecision(17)
         << "{\n  \"source_commit\": \"db3984fbc6e13b34c0064990fc2d95ac64d00058\",\n"
         << "  \"checkpoint\": " << std::quoted(arguments.checkpoint.string())
         << ",\n  \"sampler\": " << std::quoted(arguments.sampler.string())
         << ",\n  \"reference\": " << std::quoted(arguments.reference.string())
         << ",\n  \"config\": " << std::quoted(arguments.config.string())
         << ",\n  \"output\": " << std::quoted(arguments.output.string())
         << ",\n  \"png\": " << std::quoted(arguments.png.string())
         << ",\n  \"dtype\": \"BF16\",\n  \"geometry\": \"1024x1024\",\n"
         << "  \"tiles\": 36,\n  \"preparation_ms\": "
         << preparation_milliseconds
         << ",\n  \"tile_kernel_ms\": " << kernel_milliseconds
         << ",\n  \"decode_wall_ms\": " << wall_milliseconds
         << ",\n  \"resident_bytes\": " << resident_bytes
         << ",\n  \"free_bytes_before\": " << free_bytes_before
         << ",\n  \"minimum_free_bytes_after\": " << free_bytes_after
         << ",\n  \"run_launches\": " << launches
         << ",\n  \"cudnn_attention_dispatches\": " << attention_dispatches
         << ",\n  \"cudnn_convolution_dispatches\": "
         << convolution_dispatches << ",\n  \"fingerprints\": [";
  for (std::size_t index = 0U; index < prepared.size(); ++index)
    report << (index == 0U ? "" : ",") << std::quoted(prepared[index]->fingerprint);
  report << "],\n  \"raw_metrics\": ";
  emit(report, raw_metrics);
  report << ",\n  \"clamped_metrics\": ";
  emit(report, clamped_metrics);
  report << ",\n  \"output_sha256\": \""
         << dif::hex_digest(dif::sha256_file(arguments.output))
         << "\",\n  \"png_sha256\": \""
         << dif::hex_digest(dif::sha256_file(arguments.png))
         << "\",\n  \"admitted\": " << (admitted ? "true" : "false")
         << "\n}\n";
  if (!report)
    dif::fail("failed to write full Krea 2 VAE report");
  std::cout << (admitted ? "KREA2_VAE_FULL_PASS" : "KREA2_VAE_FULL_FAIL")
            << " report=" << arguments.report << " png=" << arguments.png
            << " decode_wall_ms=" << wall_milliseconds
            << " cosine=" << clamped_metrics.cosine
            << " rel_l2=" << clamped_metrics.relative_l2 << "\n";
  return admitted ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    if (!arguments.sampler.empty())
      return run_full(arguments);
    const auto build = dif::frontend::make_krea2_qwen_image_vae();
    const auto checkpoint =
        dif::weights::read_safetensors(arguments.checkpoint);
    const auto fixture = dif::weights::read_safetensors(arguments.fixture);
    dif::runtime::TensorMap bindings;
    bindings.emplace(build.latent_input,
                     dif::weights::map_safetensor(fixture, "latent_input"));
    bindings.emplace(build.latent_std,
                     dif::weights::map_safetensor(fixture, "latent_std"));
    bindings.emplace(build.latent_mean,
                     dif::weights::map_safetensor(fixture, "latent_mean"));
    std::uint64_t converted = 0U;
    for (const auto &binding : build.weights) {
      const auto *destination = build.program.tensor(binding.tensor);
      if (!destination)
        dif::fail("Krea 2 VAE binding lost its DiffIR tensor");
      bindings.emplace(binding.tensor,
                       bind_weight(checkpoint, binding, *destination,
                                   converted));
    }
    dif::ir::write_file(build.program, arguments.diffir);

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 512ULL * 1024ULL * 1024ULL;
    options.profile_pipeline = true;
    const auto result = dif::runtime::make_cuda_executor()->run(
        build.program, bindings, options);

    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (const auto &[name, id] : build.boundaries) {
      const auto &tensor = result.outputs.at(id);
      specs.push_back({name, tensor.dtype, tensor.dims});
    }
    dif::weights::SafeTensorWriter writer(arguments.output, std::move(specs));
    for (const auto &[name, id] : build.boundaries) {
      const auto &tensor = result.outputs.at(id);
      writer.append(name, std::span<const std::uint8_t>(tensor.data(),
                                                        tensor.byte_size()));
    }
    (void)writer.finish();

    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"source_commit\": \"db3984fbc6e13b34c0064990fc2d95ac64d00058\",\n"
           << "  \"checkpoint\": " << std::quoted(arguments.checkpoint.string())
           << ",\n  \"fixture\": " << std::quoted(arguments.fixture.string())
           << ",\n  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(build.program))
           << "\",\n  \"operations\": " << build.program.operations.size()
           << ",\n  \"checkpoint_tensors\": " << build.weights.size()
           << ",\n  \"f32_to_bf16_parameters\": " << converted
           << ",\n  \"backend\": " << std::quoted(result.backend_name)
           << ",\n  \"device\": " << std::quoted(result.device_name)
           << ",\n  \"preparation_ms\": " << result.preparation_milliseconds
           << ",\n  \"execution_ms\": " << result.mean_milliseconds
           << ",\n  \"resident_bytes\": " << result.resident_bytes
           << ",\n  \"run_launches\": "
           << (result.run_telemetry.kernel_launches +
               result.run_telemetry.cublaslt_matmuls +
               result.run_telemetry.cudnn_attention_dispatches +
               result.run_telemetry.cudnn_convolution_dispatches)
           << ",\n  \"cudnn_attention_dispatches\": "
           << result.run_telemetry.cudnn_attention_dispatches
           << ",\n  \"cudnn_convolution_dispatches\": "
           << result.run_telemetry.cudnn_convolution_dispatches
           << ",\n  \"boundaries\": {\n";
    bool first = true;
    for (const auto &[name, id] : build.boundaries) {
      if (!first)
        report << ",\n";
      first = false;
      const auto reference =
          dif::weights::map_safetensor(fixture, name);
      report << "    " << std::quoted(name) << ": ";
      emit(report, measure(reference, result.outputs.at(id)));
    }
    report << "\n  }\n}\n";
    if (!report)
      dif::fail("failed to write Krea 2 VAE report");
    std::cout << "KREA2_VAE_TILE_PASS report=" << arguments.report
              << " output=" << arguments.output
              << " fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(build.program))
              << " execution_ms=" << result.mean_milliseconds << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difkrea2vae: " << error.what() << "\n";
    return 1;
  }
}
