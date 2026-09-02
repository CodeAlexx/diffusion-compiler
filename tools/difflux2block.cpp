#include "dif/frontend/flux2.hpp"
#include "dif/frontend/flux2_vae.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path checkpoint;
  std::filesystem::path fixture;
  std::filesystem::path output;
  std::filesystem::path report;
  std::filesystem::path diffir;
  std::uint64_t batch_size{1U};
  std::uint64_t block{};
  std::uint64_t image_tokens{};
  std::uint64_t text_tokens{};
  std::uint64_t double_depth{8U};
  std::uint64_t single_depth{24U};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t attention_implementation{2U};
  std::vector<dif::runtime::LinearAlgorithmChoice> linear_algorithm_choices;
  std::vector<std::string> capture_boundaries;
  bool expand_linear_algorithms{};
  bool single{};
  bool transformer{};
  bool vae{};
};

void usage() {
  std::cerr << "usage: difflux2block --checkpoint MODEL.safetensors "
               "--fixture creator.safetensors --output native.safetensors "
               "--report report.json --diffir block.diffir --image-tokens N "
               "--text-tokens N [--batch-size N] [--block N] [--single | --transformer "
               "--double-depth N --single-depth N | --vae --latent-height N "
               "--latent-width N] [--attention cudnn|flash]"
               " [--select-linear-algorithm OP_ID:HEURISTIC_RANK]"
               " [--capture-boundary NAME ...]"
               " [--expand-linear-algorithms]\n";
}

std::uint64_t parse_u64(const char *value, std::string_view option) {
  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (!end || *end != '\0')
    dif::fail("invalid " + std::string(option));
  return parsed;
}

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto require_value = [&]() -> const char * {
      if (index + 1 >= argc)
        dif::fail("missing value for " + option);
      return argv[++index];
    };
    if (option == "--checkpoint")
      result.checkpoint = require_value();
    else if (option == "--fixture")
      result.fixture = require_value();
    else if (option == "--output")
      result.output = require_value();
    else if (option == "--report")
      result.report = require_value();
    else if (option == "--diffir")
      result.diffir = require_value();
    else if (option == "--batch-size")
      result.batch_size = parse_u64(require_value(), option);
    else if (option == "--block")
      result.block = parse_u64(require_value(), option);
    else if (option == "--image-tokens")
      result.image_tokens = parse_u64(require_value(), option);
    else if (option == "--text-tokens")
      result.text_tokens = parse_u64(require_value(), option);
    else if (option == "--double-depth")
      result.double_depth = parse_u64(require_value(), option);
    else if (option == "--single-depth")
      result.single_depth = parse_u64(require_value(), option);
    else if (option == "--latent-height")
      result.latent_height = parse_u64(require_value(), option);
    else if (option == "--latent-width")
      result.latent_width = parse_u64(require_value(), option);
    else if (option == "--attention") {
      const std::string backend = require_value();
      if (backend == "cudnn")
        result.attention_implementation = 2U;
      else if (backend == "flash")
        result.attention_implementation = 4U;
      else
        dif::fail("invalid attention backend " + backend);
    }
    else if (option == "--select-linear-algorithm") {
      const std::string selection = require_value();
      const auto split = selection.find(':');
      if (split == std::string::npos || split == 0U ||
          split + 1U >= selection.size())
        dif::fail("Linear algorithm choice must be OP_ID:HEURISTIC_RANK");
      result.linear_algorithm_choices.push_back(
          {static_cast<std::uint32_t>(
               parse_u64(selection.substr(0U, split).c_str(), option)),
           static_cast<std::uint32_t>(
               parse_u64(selection.substr(split + 1U).c_str(), option))});
    }
    else if (option == "--capture-boundary")
      result.capture_boundaries.emplace_back(require_value());
    else if (option == "--expand-linear-algorithms")
      result.expand_linear_algorithms = true;
    else if (option == "--single")
      result.single = true;
    else if (option == "--transformer")
      result.transformer = true;
    else if (option == "--vae")
      result.vae = true;
    else {
      usage();
      dif::fail("invalid difflux2block argument: " + option);
    }
  }
  const auto kinds = static_cast<unsigned>(result.single) +
                     static_cast<unsigned>(result.transformer) +
                     static_cast<unsigned>(result.vae);
  if (result.checkpoint.empty() || result.fixture.empty() ||
      result.output.empty() || result.report.empty() || result.diffir.empty() ||
      kinds > 1U || result.batch_size == 0U ||
      (!result.transformer && result.batch_size != 1U) ||
      (result.vae &&
       (result.latent_height == 0U || result.latent_width == 0U)) ||
      (!result.vae &&
       (result.image_tokens == 0U || result.text_tokens == 0U))) {
    usage();
    dif::fail("difflux2block requires every path and both token counts");
  }
  return result;
}

struct Metrics {
  double cosine{};
  double relative_l2{};
  double maximum_absolute{};
  double norm_ratio{};
  std::uint64_t nonfinite{};
  std::uint64_t bit_mismatches{};
  std::uint64_t elements{};
};

Metrics measure(const dif::runtime::Tensor &reference,
                const dif::runtime::Tensor &actual) {
  reference.validate();
  actual.validate();
  if (reference.dtype != actual.dtype || reference.dims != actual.dims)
    dif::fail("FLUX.2 boundary comparison shape/dtype mismatch");
  Metrics result;
  long double dot = 0.0L;
  long double reference_squared = 0.0L;
  long double actual_squared = 0.0L;
  long double error_squared = 0.0L;
  const auto element_bytes = dif::ir::dtype_size(reference.dtype);
  for (std::uint64_t index = 0; index < reference.element_count(); ++index) {
    ++result.elements;
    if (std::memcmp(reference.data() + index * element_bytes,
                    actual.data() + index * element_bytes,
                    element_bytes) != 0)
      ++result.bit_mismatches;
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
    result.maximum_absolute =
        std::max(result.maximum_absolute, std::abs(error));
  }
  const auto cosine_denominator =
      std::sqrt(reference_squared * actual_squared);
  result.cosine = cosine_denominator == 0.0L
                      ? (reference_squared == actual_squared ? 1.0 : 0.0)
                      : static_cast<double>(dot / cosine_denominator);
  result.relative_l2 = reference_squared == 0.0L
                           ? (error_squared == 0.0L
                                  ? 0.0
                                  : std::numeric_limits<double>::infinity())
                           : static_cast<double>(
                                 std::sqrt(error_squared / reference_squared));
  result.norm_ratio = reference_squared == 0.0L
                          ? (actual_squared == 0.0L
                                 ? 1.0
                                 : std::numeric_limits<double>::infinity())
                          : static_cast<double>(
                                std::sqrt(actual_squared / reference_squared));
  return result;
}

void emit_metrics(std::ostream &out, const Metrics &metrics) {
  out << "{\"cosine\":" << metrics.cosine
      << ",\"relative_l2\":" << metrics.relative_l2
      << ",\"max_absolute\":" << metrics.maximum_absolute
      << ",\"norm_ratio\":" << metrics.norm_ratio
      << ",\"nonfinite\":" << metrics.nonfinite
      << ",\"bit_mismatches\":" << metrics.bit_mismatches
      << ",\"elements\":" << metrics.elements << "}";
}

struct ExecutionBuild {
  dif::ir::Program program;
  dif::runtime::TensorMap bindings;
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
  std::vector<dif::frontend::Flux2VaeWeightTransform> checkpoint_transforms;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    const auto checkpoint =
        dif::weights::read_safetensors(arguments.checkpoint);
    const auto fixture = dif::weights::read_safetensors(arguments.fixture);

    ExecutionBuild build;
    if (arguments.vae) {
      dif::frontend::Flux2VaeConfig config;
      config.latent_height = arguments.latent_height;
      config.latent_width = arguments.latent_width;
      config.streamed_constants = false;
      config.capture_boundaries = true;
      auto vae = dif::frontend::make_flux2_vae_decoder(config);
      build.program = std::move(vae.program);
      build.bindings.emplace(
          vae.latent_tokens_input,
          dif::weights::map_safetensor(fixture, "latent_tokens_input"));
      for (auto &weight : vae.weights) {
        build.checkpoint_tensors.push_back(weight.tensor_id);
        build.checkpoint_names.push_back(std::move(weight.name));
        build.checkpoint_transforms.push_back(weight.transform);
      }
      build.boundaries = std::move(vae.boundaries);
    } else if (arguments.transformer) {
      dif::frontend::Flux2KleinTransformerConfig config;
      config.image_tokens = arguments.image_tokens;
      config.text_tokens = arguments.text_tokens;
      config.batch_size = arguments.batch_size;
      config.double_depth = arguments.double_depth;
      config.single_depth = arguments.single_depth;
      config.streamed_constants = true;
      config.attention_implementation = arguments.attention_implementation;
      config.capture_depth_boundaries = true;
      auto transformer =
          dif::frontend::make_flux2_klein_9b_transformer(config);
      build.program = std::move(transformer.program);
      build.bindings = std::move(transformer.generated_constants);
      build.bindings.emplace(
          transformer.latent_input,
          dif::weights::map_safetensor(fixture, "latent_input"));
      build.bindings.emplace(
          transformer.conditioning_input,
          dif::weights::map_safetensor(fixture, "conditioning_input"));
      build.bindings.emplace(
          transformer.timestep_input,
          dif::weights::map_safetensor(fixture, "timestep_input"));
      build.bindings.emplace(
          transformer.position_ids_input,
          dif::weights::map_safetensor(fixture, "position_ids"));
      build.checkpoint_tensors = std::move(transformer.checkpoint_tensors);
      build.checkpoint_names = std::move(transformer.checkpoint_names);
      build.boundaries = std::move(transformer.boundaries);
      build.boundaries.emplace_back("prediction",
                                    transformer.prediction_output);
    } else if (arguments.single) {
      dif::frontend::Flux2KleinSingleBlockConfig config;
      config.tokens = arguments.image_tokens + arguments.text_tokens;
      config.block_index = arguments.block;
      config.attention_implementation = arguments.attention_implementation;
      config.capture_boundaries = true;
      auto single = dif::frontend::make_flux2_klein_9b_single_block(config);
      build.program = std::move(single.program);
      build.bindings = std::move(single.generated_constants);
      build.bindings.emplace(
          single.sequence_input,
          dif::weights::map_safetensor(fixture, "sequence_input"));
      build.bindings.emplace(
          single.position_ids_input,
          dif::weights::map_safetensor(fixture, "position_ids"));
      build.bindings.emplace(
          single.modulation_input,
          dif::weights::map_safetensor(fixture, "modulation"));
      build.checkpoint_tensors = std::move(single.checkpoint_tensors);
      build.checkpoint_names = std::move(single.checkpoint_names);
      build.boundaries = std::move(single.boundaries);
    } else {
      dif::frontend::Flux2KleinDoubleBlockConfig config;
      config.image_tokens = arguments.image_tokens;
      config.text_tokens = arguments.text_tokens;
      config.block_index = arguments.block;
      config.attention_implementation = arguments.attention_implementation;
      config.capture_boundaries = true;
      auto stream = dif::frontend::make_flux2_klein_9b_double_block(config);
      build.program = std::move(stream.program);
      build.bindings = std::move(stream.generated_constants);
      build.bindings.emplace(
          stream.image_input,
          dif::weights::map_safetensor(fixture, "image_input"));
      build.bindings.emplace(
          stream.text_input,
          dif::weights::map_safetensor(fixture, "text_input"));
      build.bindings.emplace(
          stream.position_ids_input,
          dif::weights::map_safetensor(fixture, "position_ids"));
      build.bindings.emplace(
          stream.image_modulation_input,
          dif::weights::map_safetensor(fixture, "image_modulation"));
      build.bindings.emplace(
          stream.text_modulation_input,
          dif::weights::map_safetensor(fixture, "text_modulation"));
      build.checkpoint_tensors = std::move(stream.checkpoint_tensors);
      build.checkpoint_names = std::move(stream.checkpoint_names);
      build.boundaries = std::move(stream.boundaries);
    }
    if (!arguments.capture_boundaries.empty()) {
      std::vector<std::pair<std::string, std::uint32_t>> selected;
      selected.reserve(arguments.capture_boundaries.size());
      for (const auto &requested : arguments.capture_boundaries) {
        const auto boundary = std::find_if(
            build.boundaries.begin(), build.boundaries.end(),
            [&](const auto &candidate) { return candidate.first == requested; });
        if (boundary == build.boundaries.end())
          dif::fail("unknown FLUX.2 capture boundary: " + requested);
        selected.push_back(*boundary);
      }
      build.boundaries = std::move(selected);
    }
    for (std::size_t index = 0; index < build.checkpoint_tensors.size();
         ++index) {
      auto tensor = dif::weights::map_safetensor(
          checkpoint, build.checkpoint_names.at(index));
      if (!build.checkpoint_transforms.empty() &&
          build.checkpoint_transforms.at(index) ==
              dif::frontend::Flux2VaeWeightTransform::
                  BatchNormStandardDeviation) {
        dif::runtime::Tensor transformed{tensor.dtype, tensor.dims, {}};
        transformed.bytes.resize(tensor.byte_size());
        for (std::uint64_t element = 0U; element < tensor.element_count();
             ++element) {
          const auto variance = dif::runtime::load_float(tensor, element);
          dif::runtime::store_float(transformed, element,
                                    std::sqrt(variance + 1.0e-4F));
        }
        transformed.validate();
        tensor = std::move(transformed);
      }
      const auto *description =
          build.program.tensor(build.checkpoint_tensors.at(index));
      if (!description || tensor.dtype != description->dtype ||
          tensor.dims != description->dims)
        dif::fail("checkpoint tensor does not match FLUX.2 DiffIR: " +
                  build.checkpoint_names.at(index));
      build.bindings.emplace(build.checkpoint_tensors.at(index),
                             std::move(tensor));
    }

    const auto encoded = dif::ir::encode(build.program);
    std::filesystem::create_directories(arguments.diffir.parent_path());
    std::ofstream diffir(arguments.diffir, std::ios::binary | std::ios::trunc);
    diffir.write(reinterpret_cast<const char *>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    if (!diffir)
      dif::fail("failed to write FLUX.2 block DiffIR");

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
    options.profile_pipeline = true;
    options.linear_algorithm_choices = arguments.linear_algorithm_choices;
    options.expand_linear_algorithms = arguments.expand_linear_algorithms;
    const auto result = dif::runtime::make_cuda_executor()->run(
        build.program, build.bindings, options);

    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    specs.reserve(build.boundaries.size());
    for (const auto &[name, id] : build.boundaries) {
      const auto &tensor = result.outputs.at(id);
      specs.push_back({name, tensor.dtype, tensor.dims});
    }
    std::filesystem::create_directories(arguments.output.parent_path());
    dif::weights::SafeTensorWriter writer(arguments.output, std::move(specs));
    for (const auto &[name, id] : build.boundaries) {
      const auto &tensor = result.outputs.at(id);
      writer.append(name,
                    std::span<const std::uint8_t>(tensor.data(),
                                                  tensor.byte_size()));
    }
    (void)writer.finish();

    std::filesystem::create_directories(arguments.report.parent_path());
    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"source_commit\": "
              "\"50fe5162777813d869182b139e83b10743caef15\",\n"
           << "  \"model_revision\": "
              "\"32773329fbe7e81a90ef971740e8ba4b0364ecf3\",\n"
           << "  \"checkpoint\": " << std::quoted(arguments.checkpoint.string())
           << ",\n  \"fixture\": " << std::quoted(arguments.fixture.string())
           << ",\n  \"output\": " << std::quoted(arguments.output.string())
           << ",\n  \"block\": " << arguments.block
           << ",\n  \"attention_implementation\": "
           << arguments.attention_implementation
           << ",\n  \"kind\": "
           << std::quoted(
                  arguments.vae
                      ? "vae"
                      : (arguments.transformer
                             ? "transformer"
                             : (arguments.single ? "single" : "double")))
           << ",\n  \"double_depth\": " << arguments.double_depth
           << ",\n  \"batch_size\": " << arguments.batch_size
           << ",\n  \"single_depth\": " << arguments.single_depth
           << ",\n  \"image_tokens\": " << arguments.image_tokens
           << ",\n  \"text_tokens\": " << arguments.text_tokens
           << ",\n  \"latent_height\": " << arguments.latent_height
           << ",\n  \"latent_width\": " << arguments.latent_width
           << ",\n  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(build.program))
           << "\",\n  \"checkpoint_tensors\": "
           << build.checkpoint_tensors.size()
           << ",\n  \"backend\": " << std::quoted(result.backend_name)
           << ",\n  \"device\": " << std::quoted(result.device_name)
           << ",\n  \"preparation_ms\": " << result.preparation_milliseconds
           << ",\n  \"execution_ms\": " << result.mean_milliseconds
           << ",\n  \"resident_bytes\": " << result.resident_bytes
           << ",\n  \"free_bytes_before\": " << result.free_bytes_before
           << ",\n  \"kernel_launches\": "
           << result.run_telemetry.kernel_launches
           << ",\n  \"cublaslt_matmuls\": "
           << result.run_telemetry.cublaslt_matmuls
           << ",\n  \"cudnn_attention_dispatches\": "
           << result.run_telemetry.cudnn_attention_dispatches
           << ",\n  \"boundaries\": {\n";
    bool first = true;
    for (const auto &[name, id] : build.boundaries) {
      if (!first)
        report << ",\n";
      first = false;
      report << "    " << std::quoted(name) << ": ";
      emit_metrics(report,
                   measure(dif::weights::map_safetensor(fixture, name),
                           result.outputs.at(id)));
    }
    report << "\n  }\n}\n";
    if (!report)
      dif::fail("failed to write FLUX.2 native block report");
    std::cout << "FLUX2_BLOCK_PASS kind="
              << (arguments.transformer
                      ? "transformer"
                      : (arguments.vae
                             ? "vae"
                             : (arguments.single ? "single" : "double")))
              << " report=" << arguments.report
              << " output=" << arguments.output << " fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(build.program))
              << " execution_ms=" << result.mean_milliseconds << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difflux2block: " << error.what() << "\n";
    return 1;
  }
}
