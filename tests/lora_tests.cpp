#include "dif/frontend/lora.hpp"
#include "dif/frontend/training.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"
#include "dif/training/checkpoint.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::F32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

std::size_t count_operations(const dif::ir::Program &program,
                             dif::ir::Opcode opcode) {
  return static_cast<std::size_t>(
      std::count_if(program.operations.begin(), program.operations.end(),
                    [&](const auto &operation) {
                      return operation.opcode == opcode;
                    }));
}

dif::frontend::LoraFlowTrainingConfig small_config() {
  dif::frontend::LoraFlowTrainingConfig config;
  config.rows = 4U;
  config.latent_width = 4U;
  config.timestep_width = 2U;
  config.hidden_width = 8U;
  config.rank = 2U;
  config.alpha = 4.0; // scale 2.0 exercises a non-unit alpha/rank
  config.learning_rate = 1.0e-2;
  config.weight_decay = 1.0e-2;
  return config;
}

std::vector<float> pattern(std::size_t count, float base, float step,
                           float modulus) {
  std::vector<float> values(count);
  for (std::size_t index = 0U; index < count; ++index)
    values[index] =
        base + step * static_cast<float>(index) -
        modulus * std::floor((base + step * static_cast<float>(index)) /
                             modulus);
  return values;
}

dif::runtime::TensorMap
small_inputs(const dif::frontend::LoraFlowTrainingBuild &build,
             std::uint64_t seed) {
  const auto *clean_description = build.program.tensor(build.clean_input);
  const auto rows = clean_description->dims[0];
  const auto latent = clean_description->dims[1];
  const auto timestep =
      build.program.tensor(build.timestep_features_input)->dims[1];
  dif::runtime::TensorMap inputs;
  inputs.emplace(build.clean_input,
                 f32_tensor({rows, latent},
                            pattern(rows * latent, -0.8F, 0.13F, 1.7F)));
  inputs.emplace(build.noise_input,
                 f32_tensor({rows, latent},
                            pattern(rows * latent, 0.6F, -0.09F, 1.3F)));
  std::vector<float> clean_scale(rows * latent);
  std::vector<float> noise_scale(rows * latent);
  for (std::size_t index = 0U; index < clean_scale.size(); ++index) {
    clean_scale[index] =
        0.1F + 0.8F * static_cast<float>(index) /
                   static_cast<float>(clean_scale.size());
    noise_scale[index] = 1.0F - clean_scale[index];
  }
  inputs.emplace(build.clean_scale_input,
                 f32_tensor({rows, latent}, clean_scale));
  inputs.emplace(build.noise_scale_input,
                 f32_tensor({rows, latent}, noise_scale));
  inputs.emplace(build.timestep_features_input,
                 f32_tensor({rows, timestep},
                            pattern(rows * timestep, 0.05F, 0.11F, 0.9F)));
  std::vector<float> target(rows * latent);
  const auto clean_values = inputs.at(build.clean_input).f32();
  const auto noise_values = inputs.at(build.noise_input).f32();
  for (std::size_t index = 0U; index < target.size(); ++index)
    target[index] = clean_values[index] - noise_values[index];
  inputs.emplace(build.target_velocity_input,
                 f32_tensor({rows, latent}, target));
  float constant_seed = -0.15F;
  for (const auto &frozen : build.frozen_constants) {
    const auto *description = build.program.tensor(frozen.tensor_id);
    inputs.emplace(frozen.tensor_id,
                   f32_tensor(description->dims,
                              pattern(description->element_count(),
                                      constant_seed, 0.021F, 0.5F)));
    constant_seed += 0.07F;
  }
  auto adapters = dif::frontend::default_lora_adapter_init(build, seed);
  for (auto &[id, tensor] : adapters)
    inputs.emplace(id, std::move(tensor));
  for (const auto &binding : build.optimizer_bindings) {
    inputs.emplace(binding.first_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.first_moment_input)));
    inputs.emplace(binding.second_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.second_moment_input)));
  }
  inputs.emplace(build.step_input, i32_tensor({1U}, {0}));
  return inputs;
}

void test_lora_builder_structure() {
  const auto config = small_config();
  const auto build = dif::frontend::make_lora_flow_training(config);
  expect(build.adapters.size() == 3U &&
             build.optimizer_bindings.size() == 6U &&
             build.frozen_constants.size() == 5U,
         "LoRA builder exposes three adapters, six optimizer bindings, and "
         "five frozen constants");
  for (const auto &frozen : build.frozen_constants) {
    const auto *description = build.program.tensor(frozen.tensor_id);
    expect(description &&
               description->roles == dif::ir::TensorRole::Constant,
           "LoRA base weights are pure frozen constants");
  }
  for (const auto &adapter : build.adapters) {
    const auto *a_description = build.program.tensor(adapter.lora_a);
    const auto *b_description = build.program.tensor(adapter.lora_b);
    expect(a_description && b_description &&
               a_description->has_role(dif::ir::TensorRole::Parameter) &&
               a_description->has_role(dif::ir::TensorRole::Input) &&
               b_description->has_role(dif::ir::TensorRole::Parameter) &&
               a_description->dims[0] == config.rank &&
               b_description->dims[1] == config.rank,
           "LoRA adapters are trainable A [rank,in] and B [out,rank]");
  }
  expect(build.adapters[0].name == "latent_proj" &&
             build.adapters[1].name == "time_proj" &&
             build.adapters[2].name == "out_proj",
         "LoRA adapter name map is stable");
  const auto fill =
      std::find_if(build.program.operations.begin(),
                   build.program.operations.end(), [](const auto &operation) {
                     return operation.opcode == dif::ir::Opcode::Fill;
                   });
  expect(fill != build.program.operations.end() &&
             fill->f64(dif::ir::AttrKey::Value, 0.0) ==
                 config.alpha / static_cast<double>(config.rank),
         "alpha/rank scale is a fingerprinted in-graph Fill");
  const auto rebuilt = dif::frontend::make_lora_flow_training(config);
  expect(dif::ir::fingerprint(build.program) ==
             dif::ir::fingerprint(rebuilt.program),
         "LoRA builder is deterministic under fingerprint");
}

void test_frozen_weight_gradient_economy() {
  const auto build = dif::frontend::make_lora_flow_training(small_config());
  expect(count_operations(build.program, dif::ir::Opcode::Linear) == 9U,
         "LoRA graph has three base and six adapter Linears");
  expect(count_operations(build.program,
                          dif::ir::Opcode::LinearBackwardInput) == 9U,
         "every Linear keeps its input gradient path");
  expect(count_operations(build.program,
                          dif::ir::Opcode::LinearBackwardWeight) == 6U,
         "frozen base weights emit no LinearBackwardWeight — only the six "
         "adapters do");
  expect(count_operations(build.program, dif::ir::Opcode::AdamWUpdate) == 6U,
         "optimizer updates cover exactly the six adapter parameters");

  // Canonical all-parameter graphs must be emitted unchanged.
  dif::frontend::MlpTrainingConfig mlp;
  mlp.rows = 2U;
  mlp.input_width = 2U;
  mlp.hidden_width = 3U;
  mlp.output_width = 1U;
  expect(dif::frontend::make_mlp_training(mlp).program.operations.size() ==
             19U,
         "canonical MLP training graph emission is unchanged");

  // Transitive case: a weight produced by another operation still gets
  // LinearBackwardWeight because its gradient reaches requested leaves.
  dif::ir::Program forward;
  constexpr auto input = dif::ir::TensorRole::Input;
  constexpr auto parameter =
      dif::ir::TensorRole::Input | dif::ir::TensorRole::Parameter;
  forward.tensors = {
      {1U, dif::ir::DType::F32, input, {2U, 3U}},
      {2U, dif::ir::DType::F32, parameter, {4U, 3U}},
      {3U, dif::ir::DType::F32, parameter, {4U, 3U}},
      {4U, dif::ir::DType::F32, dif::ir::TensorRole::Internal, {4U, 3U}},
      {5U, dif::ir::DType::F32, dif::ir::TensorRole::Internal, {2U, 4U}},
      {6U, dif::ir::DType::F32, input, {2U, 4U}},
      {7U, dif::ir::DType::F32, dif::ir::TensorRole::Output, {1U}},
  };
  forward.operations = {
      {1U, dif::ir::Opcode::Add, {2U, 3U}, {4U}, {}},
      {2U, dif::ir::Opcode::Linear, {1U, 4U}, {5U}, {}},
      {3U, dif::ir::Opcode::MseLoss, {5U, 6U}, {7U}, {}},
  };
  const std::array<std::uint32_t, 2> targets = {2U, 3U};
  const auto differentiated =
      dif::training::differentiate(forward, 7U, targets);
  expect(count_operations(differentiated.program,
                          dif::ir::Opcode::LinearBackwardWeight) == 1U &&
             differentiated.gradients.contains(2U) &&
             differentiated.gradients.contains(3U),
         "a produced weight keeps LinearBackwardWeight for transitive "
         "gradients");
}

void test_lora_step_one_semantics_and_cuda_parity() {
  const auto config = small_config();
  const auto build = dif::frontend::make_lora_flow_training(config);
  auto inputs = small_inputs(build, 42U);

  for (const auto &adapter : build.adapters) {
    const auto values = inputs.at(adapter.lora_a).f32();
    const auto in_features =
        build.program.tensor(adapter.lora_a)->dims[1];
    const auto bound =
        1.0F / std::sqrt(static_cast<float>(in_features));
    expect(std::any_of(values.begin(), values.end(),
                       [](float value) { return value != 0.0F; }) &&
               std::all_of(values.begin(), values.end(),
                           [&](float value) {
                             return std::abs(value) <= bound;
                           }),
           "default A init is nonzero Kaiming-uniform within 1/sqrt(in)");
    const auto b_values = inputs.at(adapter.lora_b).f32();
    expect(std::all_of(b_values.begin(), b_values.end(),
                       [](float value) { return value == 0.0F; }),
           "default B init is exactly zero");
  }

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);
  const auto loss = reference.outputs.at(build.loss_output).f32()[0];
  expect(std::isfinite(loss) && loss > 0.0F,
         "LoRA training graph produces a finite positive loss");

  // Flame ordering contract: while B == 0, dL/dA == 0 EXACTLY (the delta
  // path is dead through B), and dL/dB is already nonzero (driven by A's
  // low-rank activations).  B must learn first.
  for (std::size_t index = 0U; index < build.adapters.size(); ++index) {
    const auto &adapter = build.adapters[index];
    const auto &a_binding = build.optimizer_bindings[2U * index];
    const auto &b_binding = build.optimizer_bindings[2U * index + 1U];
    expect(a_binding.parameter_input == adapter.lora_a &&
               b_binding.parameter_input == adapter.lora_b,
           "optimizer bindings pair (A,B) per adapter in order");
    const auto a_gradient =
        reference.outputs.at(a_binding.gradient_output).f32();
    expect(std::all_of(a_gradient.begin(), a_gradient.end(),
                       [](float value) { return value == 0.0F; }),
           "step-1 dL/dA is exactly zero while B == 0");
    const auto b_gradient =
        reference.outputs.at(b_binding.gradient_output).f32();
    expect(std::any_of(b_gradient.begin(), b_gradient.end(),
                       [](float value) { return value != 0.0F; }),
           "step-1 dL/dB is nonzero");
    const auto b_updated =
        reference.outputs.at(b_binding.parameter_output).f32();
    expect(std::any_of(b_updated.begin(), b_updated.end(),
                       [](float value) { return value != 0.0F; }),
           "AdamW moves B away from zero at step 1");
    // Zero-gradient A still shrinks by decoupled weight decay only.
    const auto a_before = inputs.at(a_binding.parameter_input).f32();
    const auto a_updated =
        reference.outputs.at(a_binding.parameter_output).f32();
    const auto decay = static_cast<float>(
        1.0 - config.learning_rate * config.weight_decay);
    float worst = 0.0F;
    for (std::size_t value = 0U; value < a_before.size(); ++value)
      worst = std::max(worst,
                       std::abs(a_updated[value] - a_before[value] * decay));
    expect(worst <= 1.0e-7F,
           "zero-gradient A moves only by decoupled weight decay at step 1");
  }

  if (dif::runtime::cuda_available()) {
    const auto candidate =
        dif::runtime::make_cuda_executor()->run(build.program, inputs,
                                                options);
    float maximum_absolute_error = 0.0F;
    for (const auto &[id, expected_tensor] : reference.outputs) {
      const auto expected = expected_tensor.f32();
      const auto actual = candidate.outputs.at(id).f32();
      for (std::size_t index = 0U; index < expected.size(); ++index)
        maximum_absolute_error =
            std::max(maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    expect(maximum_absolute_error <= 1.0e-5F,
           "CUDA LoRA forward, backward, and adapter AdamW match CPU");
    std::cout << "GATE lora_training_one_step backend="
              << candidate.backend_name << " device="
              << candidate.device_name
              << " max_abs=" << maximum_absolute_error << "\n";
  }
}

void test_lora_export_and_alpha_regression() {
  const auto config = small_config();
  const auto build = dif::frontend::make_lora_flow_training(config);
  auto inputs = small_inputs(build, 7U);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto result =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);

  dif::training::Checkpoint checkpoint;
  checkpoint.program_fingerprint = dif::ir::fingerprint(build.program);
  checkpoint.completed_steps = 1U;
  for (const auto &binding : build.optimizer_bindings) {
    checkpoint.state.emplace(binding.parameter_input,
                             result.outputs.at(binding.parameter_output));
    checkpoint.state.emplace(binding.first_moment_input,
                             result.outputs.at(binding.first_moment_output));
    checkpoint.state.emplace(
        binding.second_moment_input,
        result.outputs.at(binding.second_moment_output));
  }

  const auto temporary =
      std::filesystem::temp_directory_path() /
      ("dif-lora-export-" +
       std::to_string(std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count()));
  std::filesystem::create_directories(temporary);
  const auto exported = temporary / "adapters.safetensors";
  dif::frontend::export_lora_adapters(build, checkpoint, exported);
  bool validated = true;
  try {
    dif::frontend::validate_lora_export(exported, build.adapters);
  } catch (const dif::Error &) {
    validated = false;
  }
  expect(validated, "exported adapters pass the .alpha validation gate");

  const auto file = dif::weights::read_safetensors(exported);
  for (const auto &adapter : build.adapters) {
    const auto stored_a_tensor =
        dif::weights::map_safetensor(file, adapter.name + ".lora_A.weight");
    const auto stored_a = stored_a_tensor.f32();
    const auto trained_a =
        checkpoint.state.at(adapter.lora_a).f32();
    expect(stored_a.size() == trained_a.size() &&
               std::equal(stored_a.begin(), stored_a.end(),
                          trained_a.begin()),
           "exported lora_A bytes equal trained checkpoint state");
    const auto stored_alpha_tensor =
        dif::weights::map_safetensor(file, adapter.name + ".alpha");
    const auto stored_alpha = stored_alpha_tensor.f32();
    expect(stored_alpha.size() == 1U &&
               stored_alpha[0] == static_cast<float>(config.alpha),
           "exported .alpha carries the exact training alpha");
  }

  // Regression for flame's 2026-05-27 incident: an export missing .alpha
  // must be rejected — loaders would silently fall back to scale=1.0.
  const auto alphaless = temporary / "alphaless.safetensors";
  {
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (const auto &adapter : build.adapters) {
      specs.push_back({adapter.name + ".lora_A.weight", dif::ir::DType::F32,
                       build.program.tensor(adapter.lora_a)->dims});
      specs.push_back({adapter.name + ".lora_B.weight", dif::ir::DType::F32,
                       build.program.tensor(adapter.lora_b)->dims});
    }
    dif::weights::SafeTensorWriter writer(alphaless, std::move(specs));
    for (const auto &adapter : build.adapters) {
      const auto &a_state = checkpoint.state.at(adapter.lora_a);
      const auto &b_state = checkpoint.state.at(adapter.lora_b);
      writer.append(adapter.name + ".lora_A.weight",
                    {a_state.data(), a_state.byte_size()});
      writer.append(adapter.name + ".lora_B.weight",
                    {b_state.data(), b_state.byte_size()});
    }
    writer.finish();
  }
  bool rejected = false;
  try {
    dif::frontend::validate_lora_export(alphaless, build.adapters);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected,
         "an export without .alpha scalars is rejected by the gate");

  dif::training::Checkpoint mismatched = checkpoint;
  mismatched.program_fingerprint[0] ^= 1U;
  bool fingerprint_rejected = false;
  try {
    dif::frontend::export_lora_adapters(build, mismatched,
                                        temporary / "mismatch.safetensors");
  } catch (const dif::Error &) {
    fingerprint_rejected = true;
  }
  expect(fingerprint_rejected,
         "export rejects a checkpoint from a different program");
  std::filesystem::remove_all(temporary);
}

} // namespace

int main() {
  try {
    test_lora_builder_structure();
    test_frozen_weight_gradient_economy();
    test_lora_step_one_semantics_and_cuda_parity();
    test_lora_export_and_alpha_regression();
  } catch (const std::exception &error) {
    std::cerr << "unhandled: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " LoRA test expectation(s) failed\n";
    return 1;
  }
  std::cout << "OK dif_lora_tests\n";
  return 0;
}
