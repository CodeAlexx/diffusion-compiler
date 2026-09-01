// difaudioops — per-case runner for the audio opcode fixture gate. Mirrors
// tools/difditops.cpp: reads the
// fixture tensors exported by tools/export_audio_opcode_fixtures.py, builds
// the single-operation DiffIR program with the case's exact attributes, runs
// it on the selected backend, and writes actual.diftensor for difcompare.
//
// The case table is hardcoded on BOTH sides (exporter and runner), the same
// convention the DiT backward gate uses; geometry comes from the fixture
// tensors, output length from the Conv1d verifier formula.

#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::string case_name;
  std::filesystem::path fixture;
  std::filesystem::path output;
  std::string backend = "cpu";
};

struct ConvSpec {
  std::uint64_t stride = 1;
  std::uint64_t dilation = 1;
  std::uint64_t groups = 1;
  std::uint64_t pad_left = 0;
  std::uint64_t pad_right = 0;
  std::uint64_t pad_mode = 0;
  bool transposed = false;
  std::uint64_t trim_left = 0;
  std::uint64_t trim_right = 0;
  bool bias = true;
};

void usage() {
  std::cerr
      << "usage: difaudioops CASE --fixture DIR --output DIR "
         "--backend cpu|cuda\n"
         "CASE: conv_k1_pointwise | conv_k3_dilated3 | conv_k7_plain |\n"
         "      conv_k11_dilated5 | conv_k9_grouped3 | conv_k3_stride2_nopad "
         "|\n"
         "      conv_k12_depthwise_replicate_asym | conv_k4_transposed_stride2"
         " |\n"
         "      conv_k9_transposed_stride5 |"
         " conv_k12_transposed_depthwise_replicate |\n"
         "      conv_k7_transposed_grouped | snake_beta_c7 | snake_beta_c1 |"
         " snake_beta_c64\n";
}

ConvSpec conv_spec(const std::string &name) {
  ConvSpec spec;
  if (name == "conv_k1_pointwise") {
  } else if (name == "conv_k3_dilated3") {
    spec.dilation = 3;
    spec.pad_left = spec.pad_right = 3;
  } else if (name == "conv_k7_plain") {
    spec.pad_left = spec.pad_right = 3;
  } else if (name == "conv_k11_dilated5") {
    spec.dilation = 5;
    spec.pad_left = spec.pad_right = 25;
  } else if (name == "conv_k9_grouped3") {
    spec.groups = 3;
    spec.pad_left = spec.pad_right = 4;
  } else if (name == "conv_k3_stride2_nopad") {
    spec.stride = 2;
    spec.bias = false;
  } else if (name == "conv_k12_depthwise_replicate_asym") {
    spec.stride = 2;
    spec.groups = 5;
    spec.pad_left = 5;
    spec.pad_right = 6;
    spec.pad_mode = 1;
    spec.bias = false;
  } else if (name == "conv_k4_transposed_stride2") {
    spec.stride = 2;
    spec.transposed = true;
    spec.trim_left = spec.trim_right = 1;
  } else if (name == "conv_k9_transposed_stride5") {
    spec.stride = 5;
    spec.transposed = true;
    spec.trim_left = spec.trim_right = 2;
  } else if (name == "conv_k12_transposed_depthwise_replicate") {
    spec.stride = 2;
    spec.groups = 5;
    spec.pad_left = spec.pad_right = 5;
    spec.pad_mode = 1;
    spec.transposed = true;
    spec.trim_left = spec.trim_right = 15;
    spec.bias = false;
  } else if (name == "conv_k7_transposed_grouped") {
    spec.stride = 3;
    spec.groups = 2;
    spec.transposed = true;
    spec.trim_left = spec.trim_right = 2;
  } else {
    usage();
    dif::fail("unknown conv case: " + name);
  }
  return spec;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    Arguments arguments;
    arguments.case_name = argv[1];
    for (int index = 2; index < argc; ++index) {
      const std::string option = argv[index];
      auto value = [&](const char *label) -> std::string {
        if (++index >= argc)
          dif::fail(std::string("missing value for ") + label);
        return argv[index];
      };
      if (option == "--fixture")
        arguments.fixture = value("--fixture");
      else if (option == "--output")
        arguments.output = value("--output");
      else if (option == "--backend")
        arguments.backend = value("--backend");
      else {
        usage();
        return 2;
      }
    }
    if (arguments.fixture.empty() || arguments.output.empty())
      dif::fail("difaudioops requires --fixture and --output");

    using namespace dif::ir;
    Program program;
    dif::runtime::TensorMap bindings;
    std::uint32_t next_tensor = 1U;
    const auto input_tensor = [&](const std::string &name) {
      auto tensor =
          dif::runtime::read_tensor(arguments.fixture / (name + ".diftensor"));
      const auto id = next_tensor++;
      program.tensors.push_back(
          {id, tensor.dtype,
           static_cast<std::uint32_t>(TensorRole::Input), tensor.dims});
      bindings.emplace(id, std::move(tensor));
      return id;
    };

    std::uint32_t output_id = 0U;
    const bool snake = arguments.case_name.rfind("snake_beta", 0U) == 0U;
    if (snake) {
      const auto input = input_tensor("input");
      const auto alpha = input_tensor("alpha");
      const auto beta = input_tensor("beta");
      const auto &description = *program.tensor(input);
      output_id = next_tensor++;
      program.tensors.push_back(
          {output_id, description.dtype,
           static_cast<std::uint32_t>(TensorRole::Output), description.dims});
      program.operations.push_back(
          {1U, Opcode::SnakeBeta, {input, alpha, beta}, {output_id},
           {Attribute::f64(AttrKey::Epsilon, 1.0e-9)}});
    } else {
      const auto spec = conv_spec(arguments.case_name);
      const auto input = input_tensor("input");
      const auto weight = input_tensor("weight");
      std::vector<std::uint32_t> inputs{input, weight};
      if (spec.bias)
        inputs.push_back(input_tensor("bias"));
      const auto &in_description = *program.tensor(input);
      const auto &weight_description = *program.tensor(weight);
      const auto length = in_description.dims[2];
      const auto kernel = weight_description.dims[2];
      const auto padded = length + spec.pad_left + spec.pad_right;
      std::uint64_t out_channels = 0U;
      std::uint64_t out_length = 0U;
      if (spec.transposed) {
        out_channels = weight_description.dims[1] * spec.groups;
        out_length = (padded - 1U) * spec.stride + kernel - spec.trim_left -
                     spec.trim_right;
      } else {
        out_channels = weight_description.dims[0];
        out_length =
            (padded - (spec.dilation * (kernel - 1U) + 1U)) / spec.stride + 1U;
      }
      output_id = next_tensor++;
      program.tensors.push_back(
          {output_id, in_description.dtype,
           static_cast<std::uint32_t>(TensorRole::Output),
           {in_description.dims[0], out_channels, out_length}});
      std::vector<Attribute> attributes{
          Attribute::u64(AttrKey::Stride, spec.stride),
          Attribute::u64(AttrKey::Dilation, spec.dilation),
          Attribute::u64(AttrKey::Groups, spec.groups),
          Attribute::u64(AttrKey::PadLeft, spec.pad_left),
          Attribute::u64(AttrKey::PadRight, spec.pad_right),
          Attribute::u64(AttrKey::PadMode, spec.pad_mode),
          Attribute::boolean(AttrKey::Transposed, spec.transposed),
          Attribute::u64(AttrKey::TrimLeft, spec.trim_left),
          Attribute::u64(AttrKey::TrimRight, spec.trim_right)};
      program.operations.push_back(
          {1U, Opcode::Conv1d, std::move(inputs), {output_id},
           std::move(attributes)});
    }

    dif::ir::verify(program);
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    std::unique_ptr<dif::runtime::Executor> executor;
    if (arguments.backend == "cpu")
      executor = dif::runtime::make_cpu_executor();
    else if (arguments.backend == "cuda")
      executor = dif::runtime::make_cuda_executor();
    else
      dif::fail("unknown backend: " + arguments.backend);
    const auto result = executor->run(program, bindings, options);
    std::filesystem::create_directories(arguments.output);
    dif::runtime::write_tensor(result.outputs.at(output_id),
                               arguments.output / "actual.diftensor");
    std::cout << "AUDIOOPS PASS case=" << arguments.case_name
              << " backend=" << result.backend_name << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difaudioops: " << error.what() << "\n";
    return 1;
  }
}
