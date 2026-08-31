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
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path checkpoint, fixture, taps, mask_inputs, output, report,
      diffir;
  bool no_compare{};
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
    else if (option == "--taps") result.taps = value();
    else if (option == "--mask-inputs") result.mask_inputs = value();
    else if (option == "--output") result.output = value();
    else if (option == "--report") result.report = value();
    else if (option == "--diffir") result.diffir = value();
    else if (option == "--no-compare") result.no_compare = true;
    else dif::fail("invalid difkrea2text argument: " + option);
  }
  if (result.checkpoint.empty() || result.output.empty() ||
      result.report.empty() || result.diffir.empty() ||
      (result.fixture.empty() &&
       (result.taps.empty() || result.mask_inputs.empty())) ||
      (result.no_compare &&
       (result.taps.empty() || result.mask_inputs.empty())))
    dif::fail("difkrea2text requires checkpoint, output, report, diffir, and "
              "either a fixture or taps+mask-inputs");
  return result;
}

dif::runtime::Tensor pack_taps(const dif::weights::SafeTensorFile &file) {
  constexpr std::uint64_t tokens = 512U;
  constexpr std::uint64_t taps = 12U;
  constexpr std::uint64_t hidden = 2560U;
  dif::runtime::Tensor result{dif::ir::DType::BF16,
                              {1U, tokens, taps, hidden}, {}};
  result.bytes.resize(tokens * taps * hidden * sizeof(std::uint16_t));
  for (std::uint64_t tap = 0U; tap < taps; ++tap) {
    const auto name = "tap_" + (tap < 10U ? std::string("0") : std::string()) +
                      std::to_string(tap);
    const auto source = dif::weights::map_safetensor(file, name);
    if (source.dtype != dif::ir::DType::BF16 ||
        source.dims != std::vector<std::uint64_t>({tokens, hidden}))
      dif::fail("Krea text tap has wrong shape or dtype: " + name);
    for (std::uint64_t token = 0U; token < tokens; ++token) {
      const auto source_offset = token * hidden * sizeof(std::uint16_t);
      const auto destination_offset =
          (token * taps + tap) * hidden * sizeof(std::uint16_t);
      std::memcpy(result.mutable_data() + destination_offset,
                  source.data() + source_offset,
                  hidden * sizeof(std::uint16_t));
    }
  }
  result.validate();
  return result;
}

dif::runtime::Tensor sliced_text_mask(
    const dif::weights::SafeTensorFile &file) {
  constexpr std::uint64_t prefix = 34U;
  constexpr std::uint64_t tokens = 512U;
  const auto source = dif::weights::map_safetensor(file, "attention_mask");
  if (source.dtype != dif::ir::DType::Bool ||
      source.element_count() != prefix + tokens)
    dif::fail("Krea tokenizer attention_mask must be Bool [1,546]");
  dif::runtime::Tensor result{dif::ir::DType::Bool, {1U, tokens}, {}};
  result.bytes.resize(tokens);
  std::memcpy(result.mutable_data(), source.data() + prefix, tokens);
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
    dif::fail("Krea text boundary comparison shape/dtype mismatch");
  std::uint64_t token_width = 0U;
  if (validity && !reference.dims.empty()) {
    if (reference.dims[0] == validity->size())
      token_width = reference.element_count() / reference.dims[0];
    else if (reference.dims.size() > 1U &&
             reference.dims[1] == validity->size())
      token_width = reference.element_count() / reference.dims[1];
  }
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
    const auto expected = static_cast<double>(dif::runtime::load_float(reference, index));
    const auto observed = static_cast<double>(dif::runtime::load_float(actual, index));
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
                           : static_cast<double>(std::sqrt(error_squared / reference_squared));
  result.norm_ratio = reference_squared == 0
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

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    const auto build = dif::frontend::make_krea2_text_fusion(true);
    const auto checkpoint = dif::weights::read_safetensors(arguments.checkpoint);
    std::optional<dif::weights::SafeTensorFile> fixture;
    if (!arguments.fixture.empty())
      fixture.emplace(dif::weights::read_safetensors(arguments.fixture));
    dif::runtime::TensorMap bindings;
    if (!arguments.taps.empty()) {
      const auto taps = dif::weights::read_safetensors(arguments.taps);
      bindings.emplace(build.context_input, pack_taps(taps));
    } else {
      bindings.emplace(
          build.context_input,
          dif::weights::map_safetensor(*fixture, "context_input"));
    }
    if (!arguments.mask_inputs.empty()) {
      const auto mask_inputs =
          dif::weights::read_safetensors(arguments.mask_inputs);
      bindings.emplace(build.validity_mask_input,
                       sliced_text_mask(mask_inputs));
    } else {
      bindings.emplace(
          build.validity_mask_input,
          dif::weights::map_safetensor(*fixture, "validity_mask"));
    }
    std::uint64_t converted = 0U;
    for (std::size_t index = 0; index < build.checkpoint_tensors.size(); ++index) {
      auto tensor = dif::weights::map_safetensor(
          checkpoint, build.checkpoint_names[index]);
      if (tensor.dtype == dif::ir::DType::F32) {
        tensor = dif::runtime::convert_float_tensor(tensor, dif::ir::DType::BF16);
        ++converted;
      }
      const auto *description = build.program.tensor(build.checkpoint_tensors[index]);
      if (!description || description->dtype != tensor.dtype ||
          description->dims != tensor.dims)
        dif::fail("Krea text checkpoint mismatch: " + build.checkpoint_names[index]);
      bindings.emplace(build.checkpoint_tensors[index], std::move(tensor));
    }
    dif::ir::write_file(build.program, arguments.diffir);
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
    options.profile_pipeline = true;
    const auto result =
        dif::runtime::make_cuda_executor()->run(build.program, bindings, options);

    const std::vector<std::pair<std::string, std::uint32_t>> boundaries{
        {"layerwise_0", build.block_outputs[0]},
        {"layerwise_1", build.block_outputs[1]},
        {"projected", build.projected_output},
        {"refiner_0", build.block_outputs[2]},
        {"refiner_1", build.block_outputs[3]},
        {"conditioning_output", build.conditioning_output},
    };
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (const auto &[name, id] : boundaries) {
      const auto &tensor = result.outputs.at(id);
      specs.push_back({name, tensor.dtype, tensor.dims});
    }
    dif::weights::SafeTensorWriter writer(arguments.output, std::move(specs));
    for (const auto &[name, id] : boundaries) {
      const auto &tensor = result.outputs.at(id);
      writer.append(name, {tensor.data(), static_cast<std::size_t>(tensor.byte_size())});
    }
    (void)writer.finish();

    const auto &mask = bindings.at(build.validity_mask_input);
    std::vector<std::uint8_t> validity(mask.data(), mask.data() + mask.byte_size());
    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"source_commit\": \"db3984fbc6e13b34c0064990fc2d95ac64d00058\",\n"
           << "  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(build.program)) << "\",\n"
           << "  \"checkpoint_tensors\": " << build.checkpoint_tensors.size() << ",\n"
           << "  \"f32_to_bf16_parameters\": " << converted << ",\n"
           << "  \"preparation_ms\": " << result.preparation_milliseconds << ",\n"
           << "  \"execution_ms\": " << result.mean_milliseconds << ",\n"
           << "  \"resident_bytes\": " << result.resident_bytes << ",\n"
           << "  \"boundaries\": {\n";
    bool first = true;
    if (!arguments.no_compare) {
      for (const auto &[name, id] : boundaries) {
        if (!first) report << ",\n";
        first = false;
        auto reference = dif::weights::map_safetensor(*fixture, name);
        const auto &actual = result.outputs.at(id);
        if (reference.dims != actual.dims &&
            reference.element_count() == actual.element_count())
          reference.dims = actual.dims;
        report << "    " << std::quoted(name) << ": {\"full\":";
        emit(report, measure(reference, actual, nullptr));
        report << ",\"valid_tokens\":";
        emit(report, measure(reference, actual, &validity));
        report << "}";
      }
    }
    report << "\n  }\n}\n";
    std::cout << "KREA2_TEXT_PASS fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(build.program))
              << " execution_ms=" << result.mean_milliseconds << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difkrea2text: " << error.what() << "\n";
    return 1;
  }
}
