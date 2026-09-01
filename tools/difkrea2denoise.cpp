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
  std::filesystem::path checkpoint, fixture, output, report, diffir;
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
    if (option == "--checkpoint") result.checkpoint = value();
    else if (option == "--fixture") result.fixture = value();
    else if (option == "--output") result.output = value();
    else if (option == "--report") result.report = value();
    else if (option == "--diffir") result.diffir = value();
    else dif::fail("invalid difkrea2denoise argument: " + option);
  }
  if (result.checkpoint.empty() || result.fixture.empty() ||
      result.output.empty() || result.report.empty() || result.diffir.empty())
    dif::fail("difkrea2denoise requires checkpoint, fixture, output, report, and diffir");
  return result;
}

dif::runtime::Tensor i32_tensor(std::vector<std::int32_t> values) {
  dif::runtime::Tensor result{dif::ir::DType::I32,
                              {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(result.mutable_data(), values.data(), result.byte_size());
  result.validate();
  return result;
}

struct Metrics {
  double cosine{}, relative_l2{}, max_absolute{}, norm_ratio{};
  std::uint64_t nonfinite{}, bit_mismatches{}, elements{};
};

Metrics measure(const dif::runtime::Tensor &reference,
                const dif::runtime::Tensor &actual,
                const std::vector<std::uint8_t> *validity) {
  if (reference.dtype != actual.dtype ||
      reference.element_count() != actual.element_count())
    dif::fail("Krea denoiser boundary comparison shape/dtype mismatch");
  std::uint64_t token_width = 0U;
  if (validity && reference.element_count() % validity->size() == 0U)
    token_width = reference.element_count() / validity->size();
  long double dot = 0, reference_squared = 0, actual_squared = 0,
              error_squared = 0;
  Metrics result;
  const auto width = dif::ir::dtype_size(reference.dtype);
  for (std::uint64_t index = 0; index < reference.element_count(); ++index) {
    if (validity && token_width &&
        (*validity)[static_cast<std::size_t>(index / token_width)] == 0U)
      continue;
    ++result.elements;
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
  result.cosine = denominator == 0 ? 1.0 : static_cast<double>(dot / denominator);
  result.relative_l2 = reference_squared == 0
                           ? 0.0
                           : static_cast<double>(std::sqrt(error_squared /
                                                           reference_squared));
  result.norm_ratio = reference_squared == 0
                          ? 1.0
                          : static_cast<double>(std::sqrt(actual_squared /
                                                          reference_squared));
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

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    dif::frontend::Krea2Config config;
    config.streamed_constants = true;
    const auto build = dif::frontend::make_krea2_denoiser(config, true);
    const auto checkpoint = dif::weights::read_safetensors(arguments.checkpoint);
    const auto fixture = dif::weights::read_safetensors(arguments.fixture);
    dif::runtime::TensorMap bindings;
    bindings.emplace(build.image_tokens_input,
                     dif::weights::map_safetensor(fixture,
                                                   "image_tokens_input"));
    bindings.emplace(build.context_input,
                     dif::weights::map_safetensor(fixture, "context_input"));
    bindings.emplace(build.timestep_input,
                     dif::weights::map_safetensor(fixture, "timestep_input"));
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
    bindings.emplace(build.rotary_pair_axes, i32_tensor(std::move(pair_axes)));
    bindings.emplace(build.rotary_pair_indices,
                     i32_tensor(std::move(pair_indices)));
    bindings.emplace(build.rotary_axis_dims, i32_tensor({32, 48, 48}));

    std::uint64_t converted = 0U;
    std::uint64_t converted_bytes = 0U;
    for (std::size_t index = 0; index < build.checkpoint_tensors.size(); ++index) {
      auto tensor = dif::weights::map_safetensor(
          checkpoint, build.checkpoint_names[index]);
      if (tensor.dtype == dif::ir::DType::F32) {
        tensor = dif::runtime::convert_float_tensor(tensor,
                                                     dif::ir::DType::BF16);
        ++converted;
        converted_bytes += tensor.byte_size();
      }
      const auto *description =
          build.program.tensor(build.checkpoint_tensors[index]);
      if (!description || description->dtype != tensor.dtype ||
          description->dims != tensor.dims)
        dif::fail("Krea denoiser checkpoint mismatch: " +
                  build.checkpoint_names[index]);
      bindings.emplace(build.checkpoint_tensors[index], std::move(tensor));
    }
    dif::ir::write_file(build.program, arguments.diffir);

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 512ULL * 1024ULL * 1024ULL;
    options.profile_pipeline = true;
    // Retain checkpoint pages for repeated denoise evaluations. The transport
    // still stages and copies exact BF16 weights every evaluation; it does not
    // force the OS to reread those pages from disk on each pass.
    options.streamed_release_mapped_pages_per_copy = false;
    const auto result =
        dif::runtime::make_cuda_executor()->run(build.program, bindings, options);

    std::vector<std::pair<std::string, std::uint32_t>> boundaries{
        {"projected_image", build.projected_image},
        {"timestep_embedding", build.timestep_embedding},
        {"timestep_first_linear", build.timestep_first_linear},
        {"timestep_first_activation", build.timestep_first_activation},
        {"timestep_output", build.timestep_output},
        {"timestep_projection_activation",
         build.timestep_projection_activation},
        {"modulation_output", build.modulation_output},
    };
    for (std::size_t index = 0U; index < build.block_outputs.size(); ++index)
      boundaries.emplace_back("block_" + std::to_string(index + 1U),
                              build.block_outputs[index]);
    boundaries.emplace_back("last_modulated", build.last_modulated);
    boundaries.emplace_back("velocity_output", build.velocity_output);
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (const auto &[name, id] : boundaries) {
      const auto &tensor = result.outputs.at(id);
      specs.push_back({name, tensor.dtype, tensor.dims});
    }
    dif::weights::SafeTensorWriter writer(arguments.output, std::move(specs));
    for (const auto &[name, id] : boundaries) {
      const auto &tensor = result.outputs.at(id);
      writer.append(name, std::span<const std::uint8_t>(tensor.data(),
                                                        tensor.byte_size()));
    }
    (void)writer.finish();

    const auto mask =
        dif::weights::map_safetensor(fixture, "validity_mask");
    std::vector<std::uint8_t> validity(mask.data(),
                                       mask.data() + mask.byte_size());
    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"source_commit\": \"db3984fbc6e13b34c0064990fc2d95ac64d00058\",\n"
           << "  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(build.program)) << "\",\n"
           << "  \"operations\": " << build.program.operations.size() << ",\n"
           << "  \"checkpoint_tensors\": "
           << build.checkpoint_tensors.size() << ",\n"
           << "  \"f32_to_bf16_parameters\": " << converted << ",\n"
           << "  \"converted_bf16_bytes\": " << converted_bytes << ",\n"
           << "  \"preparation_ms\": " << result.preparation_milliseconds << ",\n"
           << "  \"execution_ms\": " << result.mean_milliseconds << ",\n"
           << "  \"resident_bytes\": " << result.resident_bytes << ",\n"
           << "  \"streamed_weight_bytes\": "
           << result.pipeline_profile.streamed_weight_bytes << ",\n"
           << "  \"streamed_host_stage_ms\": "
           << result.pipeline_profile.streamed_host_stage_milliseconds << ",\n"
           << "  \"streamed_host_wait_ms\": "
           << result.pipeline_profile.streamed_host_wait_milliseconds << ",\n"
           << "  \"streamed_h2d_ms\": "
           << result.pipeline_profile.streamed_h2d_milliseconds << ",\n"
           << "  \"operation_kernel_ms\": "
           << result.pipeline_profile.operation_kernel_milliseconds << ",\n"
           << "  \"attention_kernel_ms\": "
           << result.pipeline_profile.attention_kernel_milliseconds << ",\n"
           << "  \"run_launches\": "
           << (result.run_telemetry.kernel_launches +
               result.run_telemetry.cublaslt_matmuls +
               result.run_telemetry.cudnn_attention_dispatches)
           << ",\n  \"boundaries\": {\n";
    bool first = true;
    for (const auto &[name, id] : boundaries) {
      if (!first) report << ",\n";
      first = false;
      const auto &actual = result.outputs.at(id);
      const auto *reference_entry = fixture.find(name);
      if (!reference_entry) {
        report << "    " << std::quoted(name)
               << ": {\"reference\":\"not captured by creator fixture\"}";
        continue;
      }
      auto reference = dif::weights::map_safetensor(fixture, name);
      if (reference.dims != actual.dims &&
          reference.element_count() == actual.element_count())
        reference.dims = actual.dims;
      const bool sequence_boundary = name.rfind("block_", 0U) == 0U;
      report << "    " << std::quoted(name) << ": {\"full\":";
      emit(report, measure(reference, actual, nullptr));
      if (sequence_boundary) {
        report << ",\"valid_tokens\":";
        emit(report, measure(reference, actual, &validity));
      }
      report << "}";
    }
    report << "\n  }\n}\n";
    std::cout << "KREA2_DENOISER_PASS fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(build.program))
              << " execution_ms=" << result.mean_milliseconds << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difkrea2denoise: " << error.what() << "\n";
    return 1;
  }
}
