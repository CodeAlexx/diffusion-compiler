#include "dif/frontend/krea2.hpp"
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
  std::string sequence_override;
  std::uint64_t block{};
  bool final_only{};
};

void usage() {
  std::cerr << "usage: difkrea2block --checkpoint RAW.safetensors "
               "--fixture creator.safetensors --output native.safetensors "
               "--report report.json --diffir block.diffir [--block N] "
               "[--sequence-override FILE.safetensors::NAME] [--final-only]\n";
}

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--checkpoint" && index + 1 < argc)
      result.checkpoint = argv[++index];
    else if (option == "--fixture" && index + 1 < argc)
      result.fixture = argv[++index];
    else if (option == "--output" && index + 1 < argc)
      result.output = argv[++index];
    else if (option == "--report" && index + 1 < argc)
      result.report = argv[++index];
    else if (option == "--diffir" && index + 1 < argc)
      result.diffir = argv[++index];
    else if (option == "--sequence-override" && index + 1 < argc)
      result.sequence_override = argv[++index];
    else if (option == "--final-only")
      result.final_only = true;
    else if (option == "--block" && index + 1 < argc) {
      char *end = nullptr;
      const auto value = std::strtoull(argv[++index], &end, 10);
      if (!end || *end != '\0')
        dif::fail("invalid --block");
      result.block = value;
    } else {
      usage();
      dif::fail("invalid difkrea2block argument");
    }
  }
  if (result.checkpoint.empty() || result.fixture.empty() ||
      result.output.empty() || result.report.empty() || result.diffir.empty()) {
    usage();
    dif::fail("difkrea2block requires checkpoint, fixture, output, report, and diffir");
  }
  return result;
}

dif::runtime::Tensor i32_tensor(std::vector<std::int32_t> values) {
  dif::runtime::Tensor result{dif::ir::DType::I32,
                              {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  result.validate();
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
                const dif::runtime::Tensor &actual,
                const std::vector<std::uint8_t> *validity) {
  reference.validate();
  actual.validate();
  if (reference.dtype != actual.dtype || reference.dims != actual.dims)
    dif::fail("Krea 2 boundary comparison shape/dtype mismatch");
  const auto trailing = [&](std::size_t start) {
    std::uint64_t result = 1U;
    for (std::size_t axis = start; axis < reference.dims.size(); ++axis)
      result *= reference.dims[axis];
    return result;
  };
  const bool batch_sequence = reference.dims.size() >= 2U &&
                              reference.dims[0] == 1U &&
                              reference.dims[1] == 4608U;
  const bool flat_sequence = reference.dims.size() >= 2U &&
                             reference.dims[0] == 4608U;
  const auto token_width = batch_sequence ? trailing(2U)
                           : flat_sequence ? trailing(1U)
                                           : 0U;
  long double dot = 0.0L;
  long double reference_squared = 0.0L;
  long double actual_squared = 0.0L;
  long double error_squared = 0.0L;
  Metrics result;
  const auto element_bytes = dif::ir::dtype_size(reference.dtype);
  for (std::uint64_t index = 0; index < reference.element_count(); ++index) {
    if (validity && token_width != 0U &&
        (*validity)[static_cast<std::size_t>(index / token_width)] == 0U)
      continue;
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

std::vector<std::pair<std::string, std::uint32_t>>
boundaries(const dif::frontend::Krea2BlockBuild &build, bool final_only) {
  if (final_only)
    return {{"final_output", build.final_output}};
  return {
      {"modulated_parameters", build.modulated_parameters},
      {"input_normalized", build.input_normalized},
      {"attention_input", build.attention_input},
      {"query", build.query},
      {"key", build.key},
      {"value", build.value},
      {"rotary_query", build.rotary_query},
      {"rotary_key", build.rotary_key},
      {"attention_output", build.attention_output},
      {"attention_gate", build.attention_gate},
      {"output_projection", build.output_projection},
      {"attention_residual", build.attention_residual},
      {"mlp_input", build.mlp_input},
      {"mlp_gate", build.mlp_gate},
      {"mlp_up", build.mlp_up},
      {"mlp_gate_activated", build.mlp_gate_activated},
      {"mlp_activation", build.mlp_activation},
      {"mlp_output", build.mlp_output},
      {"final_output", build.final_output},
  };
}

dif::runtime::Tensor load_tensor_spec(std::string_view spec) {
  const auto separator = spec.find("::");
  if (separator == std::string_view::npos || separator == 0U ||
      separator + 2U == spec.size())
    dif::fail("sequence override must be FILE.safetensors::TENSOR_NAME");
  const auto path = std::filesystem::path(spec.substr(0U, separator));
  const auto name = spec.substr(separator + 2U);
  return dif::weights::map_safetensor(dif::weights::read_safetensors(path),
                                      name);
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

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    dif::frontend::Krea2Config config;
    config.streamed_constants = false;
    config.prenorm_reduction_tile = 8192U;
    // This expanded oracle compiles postnorm at a generalized call site and
    // selected a 2048-wide reduction. Production block.forward uses 8192.
    config.postnorm_reduction_tile = 2048U;
    const auto build =
        dif::frontend::make_krea2_block(config, arguments.block,
                                        !arguments.final_only);
    const auto checkpoint =
        dif::weights::read_safetensors(arguments.checkpoint);
    const auto fixture = dif::weights::read_safetensors(arguments.fixture);
    dif::runtime::TensorMap bindings;
    bindings.emplace(
        build.sequence_input,
        arguments.sequence_override.empty()
            ? dif::weights::map_safetensor(fixture, "sequence_input")
            : load_tensor_spec(arguments.sequence_override));
    bindings.emplace(build.modulation_input,
                     dif::weights::map_safetensor(fixture, "modulation_input"));
    bindings.emplace(build.positions_input,
                     dif::weights::map_safetensor(fixture, "positions"));
    bindings.emplace(build.validity_mask_input,
                     dif::weights::map_safetensor(fixture, "validity_mask"));
    std::vector<std::int32_t> pair_axes;
    std::vector<std::int32_t> pair_indices;
    for (std::int32_t axis = 0; axis < 3; ++axis) {
      const std::int32_t dimension = axis == 0 ? 32 : 48;
      for (std::int32_t pair = 0; pair < dimension / 2; ++pair) {
        pair_axes.push_back(axis);
        pair_indices.push_back(pair);
      }
    }
    bindings.emplace(build.rotary_pair_axes, i32_tensor(pair_axes));
    bindings.emplace(build.rotary_pair_indices, i32_tensor(pair_indices));
    bindings.emplace(build.rotary_axis_dims, i32_tensor({32, 48, 48}));

    std::uint64_t converted_parameters = 0U;
    for (std::size_t index = 0; index < build.checkpoint_tensors.size();
         ++index) {
      auto tensor = dif::weights::map_safetensor(
          checkpoint, build.checkpoint_names.at(index));
      if (tensor.dtype == dif::ir::DType::F32) {
        tensor = dif::runtime::convert_float_tensor(
            tensor, dif::ir::DType::BF16);
        ++converted_parameters;
      }
      const auto *description =
          build.program.tensor(build.checkpoint_tensors.at(index));
      if (!description || tensor.dtype != description->dtype ||
          tensor.dims != description->dims)
        dif::fail("checkpoint tensor does not match Krea 2 DiffIR: " +
                  build.checkpoint_names.at(index));
      bindings.emplace(build.checkpoint_tensors.at(index), std::move(tensor));
    }

    const auto encoded = dif::ir::encode(build.program);
    std::filesystem::create_directories(arguments.diffir.parent_path());
    std::ofstream diffir(arguments.diffir, std::ios::binary | std::ios::trunc);
    diffir.write(reinterpret_cast<const char *>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    if (!diffir)
      dif::fail("failed to write Krea 2 block DiffIR");

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
    options.profile_pipeline = true;
    const auto result = dif::runtime::make_cuda_executor()->run(
        build.program, bindings, options);

    const auto named_boundaries = boundaries(build, arguments.final_only);
    std::vector<dif::weights::SafeTensorWriteSpec> output_specs;
    output_specs.reserve(named_boundaries.size());
    for (const auto &[name, id] : named_boundaries) {
      const auto &tensor = result.outputs.at(id);
      output_specs.push_back({name, tensor.dtype, tensor.dims});
    }
    std::filesystem::create_directories(arguments.output.parent_path());
    dif::weights::SafeTensorWriter writer(arguments.output,
                                           std::move(output_specs));
    for (const auto &[name, id] : named_boundaries) {
      const auto &tensor = result.outputs.at(id);
      writer.append(name,
                    std::span<const std::uint8_t>(tensor.data(),
                                                  tensor.byte_size()));
    }
    (void)writer.finish();

    const auto mask_tensor =
        dif::weights::map_safetensor(fixture, "validity_mask");
    std::vector<std::uint8_t> validity(mask_tensor.data(),
                                       mask_tensor.data() +
                                           mask_tensor.byte_size());
    std::filesystem::create_directories(arguments.report.parent_path());
    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"source_commit\": \"db3984fbc6e13b34c0064990fc2d95ac64d00058\",\n"
           << "  \"checkpoint\": " << std::quoted(arguments.checkpoint.string())
           << ",\n  \"fixture\": " << std::quoted(arguments.fixture.string())
           << ",\n  \"output\": " << std::quoted(arguments.output.string())
           << ",\n  \"block\": " << arguments.block
           << ",\n  \"sequence_override\": "
           << std::quoted(arguments.sequence_override)
           << ",\n  \"final_only\": "
           << (arguments.final_only ? "true" : "false")
           << ",\n  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(build.program))
           << "\",\n  \"checkpoint_tensors\": "
           << build.checkpoint_tensors.size()
           << ",\n  \"f32_to_bf16_parameters\": " << converted_parameters
           << ",\n  \"backend\": " << std::quoted(result.backend_name)
           << ",\n  \"device\": " << std::quoted(result.device_name)
           << ",\n  \"preparation_ms\": " << result.preparation_milliseconds
           << ",\n  \"execution_ms\": " << result.mean_milliseconds
           << ",\n  \"resident_bytes\": " << result.resident_bytes
           << ",\n  \"free_bytes_before\": " << result.free_bytes_before
           << ",\n  \"run_launches\": "
           << (result.run_telemetry.kernel_launches +
               result.run_telemetry.cublaslt_matmuls +
               result.run_telemetry.cudnn_attention_dispatches)
           << ",\n  \"cublaslt_matmuls\": "
           << result.run_telemetry.cublaslt_matmuls
           << ",\n  \"cudnn_attention_dispatches\": "
           << result.run_telemetry.cudnn_attention_dispatches
           << ",\n  \"boundaries\": {\n";
    bool first = true;
    for (const auto &[name, id] : named_boundaries) {
      if (!first)
        report << ",\n";
      first = false;
      const auto reference = dif::weights::map_safetensor(fixture, name);
      const auto &actual = result.outputs.at(id);
      report << "    " << std::quoted(name) << ": {\"full\":";
      emit_metrics(report, measure(reference, actual, nullptr));
      report << ",\"valid_tokens\":";
      emit_metrics(report, measure(reference, actual, &validity));
      report << "}";
    }
    report << "\n  }\n}\n";
    if (!report)
      dif::fail("failed to write Krea 2 native block report");
    std::cout << "KREA2_BLOCK_PASS report=" << arguments.report
              << " output=" << arguments.output
              << " fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(build.program))
              << " execution_ms=" << result.mean_milliseconds << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difkrea2block: " << error.what() << "\n";
    return 1;
  }
}
