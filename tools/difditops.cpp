// Per-opcode fixture runner for the DiT backward gates.  Reads the PyTorch
// fixture tensors exported by tools/export_dit_backward_fixtures.py, builds
// the single backward operation (attention builds forward Attention +
// AttentionLse + AttentionBackward — the exact decomposed recompute chain the
// autodiff rule emits), runs it on the requested backend, and writes the
// actual gradients as actual-grad-<name>.diftensor for difcompare.

#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Arguments {
  std::string case_name;
  std::filesystem::path fixture;
  std::filesystem::path output;
  std::string backend{"cpu"};
};

void usage() {
  std::cerr << "usage: difditops CASE --fixture DIR --output DIR "
               "--backend cpu|cuda\n"
               "CASE: rms_norm | rms_norm_modulate_weighted | "
               "rms_norm_modulate_plain |\n"
               "      swiglu_gatefirst | swiglu_valuefirst | residual_gate |"
               " layer_norm |\n"
               "      qk_norm_rope_fulltable | qk_norm_rope_halftable |\n"
               "      attention_full | attention_causal\n";
}

struct Builder {
  dif::ir::Program program;
  dif::runtime::TensorMap bindings;
  // gradient name -> tensor id
  std::vector<std::pair<std::string, std::uint32_t>> gradients;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};
  std::filesystem::path fixture;

  std::uint32_t input(const std::string &name) {
    auto tensor = dif::runtime::read_tensor(fixture / (name + ".diftensor"));
    const auto id = next_tensor++;
    program.tensors.push_back(
        {id, tensor.dtype, dif::ir::TensorRole::Input, tensor.dims});
    bindings.emplace(id, std::move(tensor));
    return id;
  }

  std::uint32_t output(const std::string &gradient_name, dif::ir::DType dtype,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back(
        {id, dtype, dif::ir::TensorRole::Output, std::move(dims)});
    gradients.emplace_back(gradient_name, id);
    return id;
  }

  std::uint32_t internal(dif::ir::DType dtype,
                         std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back(
        {id, dtype, dif::ir::TensorRole::Internal, std::move(dims)});
    return id;
  }

  void operation(dif::ir::Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<dif::ir::Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  }

  const dif::ir::TensorDesc &describe(std::uint32_t id) {
    return *program.tensor(id);
  }
};

constexpr double kEpsilon = 1.0e-5;

Builder build_case(const Arguments &arguments) {
  using namespace dif::ir;
  Builder builder;
  builder.fixture = arguments.fixture;
  const auto &name = arguments.case_name;
  const auto epsilon = Attribute::f64(AttrKey::Epsilon, kEpsilon);
  if (name == "rms_norm") {
    const auto grad_output = builder.input("grad-output");
    const auto x = builder.input("x");
    const auto weight = builder.input("weight");
    const auto &x_description = builder.describe(x);
    const auto grad_x =
        builder.output("x", x_description.dtype, x_description.dims);
    const auto grad_weight = builder.output(
        "weight", x_description.dtype, builder.describe(weight).dims);
    builder.operation(Opcode::RmsNormBackward, {grad_output, x, weight},
                      {grad_x, grad_weight}, {epsilon});
    return builder;
  }
  if (name == "rms_norm_modulate_weighted" ||
      name == "rms_norm_modulate_plain") {
    const bool weighted = name == "rms_norm_modulate_weighted";
    const auto grad_output = builder.input("grad-output");
    const auto x = builder.input("x");
    const auto weight = weighted ? builder.input("weight") : 0U;
    const auto scale = builder.input("scale");
    const auto &x_description = builder.describe(x);
    const auto dtype = x_description.dtype;
    const auto dims = x_description.dims;
    const auto grad_x = builder.output("x", dtype, dims);
    const auto grad_scale = builder.output("scale", dtype, dims);
    const auto grad_shift = builder.output("shift", dtype, dims);
    std::vector<std::uint32_t> inputs{grad_output, x};
    std::vector<std::uint32_t> outputs{grad_x, grad_scale, grad_shift};
    if (weighted) {
      // input order is grad_output, x, weight, scale
      inputs.push_back(weight);
      outputs.push_back(builder.output("weight", dtype,
                                       builder.describe(weight).dims));
    }
    inputs.push_back(scale);
    builder.operation(Opcode::RmsNormModulateBackward, inputs, outputs,
                      {epsilon});
    return builder;
  }
  if (name == "swiglu_gatefirst" || name == "swiglu_valuefirst") {
    const bool gate_first = name == "swiglu_gatefirst";
    const auto grad_output = builder.input("grad-output");
    const auto x = builder.input("x");
    const auto &x_description = builder.describe(x);
    const auto grad_x =
        builder.output("x", x_description.dtype, x_description.dims);
    builder.operation(Opcode::SwiGluBackward, {grad_output, x}, {grad_x},
                      {Attribute::boolean(AttrKey::GateFirst, gate_first)});
    return builder;
  }
  if (name == "residual_gate") {
    const auto grad_output = builder.input("grad-output");
    const auto branch = builder.input("branch");
    const auto gate = builder.input("gate");
    const auto &description = builder.describe(branch);
    const auto grad_branch =
        builder.output("branch", description.dtype, description.dims);
    const auto grad_gate =
        builder.output("gate", description.dtype, description.dims);
    builder.operation(Opcode::ResidualGateBackward,
                      {grad_output, branch, gate}, {grad_branch, grad_gate});
    return builder;
  }
  if (name == "layer_norm") {
    const auto grad_output = builder.input("grad-output");
    const auto x = builder.input("x");
    const auto weight = builder.input("weight");
    const auto &x_description = builder.describe(x);
    const auto grad_x =
        builder.output("x", x_description.dtype, x_description.dims);
    const auto grad_weight = builder.output(
        "weight", x_description.dtype, builder.describe(weight).dims);
    const auto grad_bias = builder.output(
        "bias", x_description.dtype, builder.describe(weight).dims);
    builder.operation(Opcode::LayerNormBackward, {grad_output, x, weight},
                      {grad_x, grad_weight, grad_bias}, {epsilon});
    return builder;
  }
  if (name == "qk_norm_rope_fulltable" || name == "qk_norm_rope_halftable") {
    const auto grad_output = builder.input("grad-output");
    const auto x = builder.input("x");
    const auto weight = builder.input("weight");
    const auto cos = builder.input("cos");
    const auto sin = builder.input("sin");
    const auto &x_description = builder.describe(x);
    const auto grad_x =
        builder.output("x", x_description.dtype, x_description.dims);
    const auto grad_weight = builder.output(
        "weight", x_description.dtype, builder.describe(weight).dims);
    const auto table_width = builder.describe(cos).dims[1];
    const auto rotary = name == "qk_norm_rope_fulltable"
                            ? table_width
                            : table_width * 2U;
    builder.operation(Opcode::QkNormPartialRopeBackward,
                      {grad_output, x, weight, cos, sin},
                      {grad_x, grad_weight},
                      {epsilon, Attribute::u64(AttrKey::RotaryDim, rotary)});
    return builder;
  }
  if (name == "attention_full" || name == "attention_causal") {
    const bool causal = name == "attention_causal";
    const auto grad_output = builder.input("grad-output");
    const auto q = builder.input("q");
    const auto k = builder.input("k");
    const auto v = builder.input("v");
    const auto &q_description = builder.describe(q);
    const auto dtype = q_description.dtype;
    const auto dims = q_description.dims;
    const auto forward_output = builder.internal(dtype, dims);
    const auto lse = builder.internal(dif::ir::DType::F32,
                                      {dims[0], dims[1]});
    const auto grad_q = builder.output("q", dtype, dims);
    const auto grad_k = builder.output("k", dtype, dims);
    const auto grad_v = builder.output("v", dtype, dims);
    const auto causal_attribute = Attribute::boolean(AttrKey::Causal, causal);
    builder.operation(Opcode::Attention, {q, k, v}, {forward_output},
                      {causal_attribute});
    builder.operation(Opcode::AttentionLse, {q, k}, {lse},
                      {causal_attribute});
    builder.operation(Opcode::AttentionBackward,
                      {grad_output, q, k, v, forward_output, lse},
                      {grad_q, grad_k, grad_v}, {causal_attribute});
    return builder;
  }
  usage();
  dif::fail("unknown case: " + name);
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
      dif::fail("difditops requires --fixture and --output");

    auto builder = build_case(arguments);
    dif::ir::verify(builder.program);
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
    const auto result =
        executor->run(builder.program, builder.bindings, options);
    std::filesystem::create_directories(arguments.output);
    for (const auto &[gradient_name, id] : builder.gradients)
      dif::runtime::write_tensor(result.outputs.at(id),
                                 arguments.output /
                                     ("actual-grad-" + gradient_name +
                                      ".diftensor"));
    std::cout << "DITOPS PASS case=" << arguments.case_name
              << " backend=" << result.backend_name
              << " gradients=" << builder.gradients.size() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difditops: " << error.what() << "\n";
    return 1;
  }
}
