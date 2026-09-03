#include "dif/frontend/dit_lora.hpp"
#include "dif/frontend/lora.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/checkpoint.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

dif::runtime::Tensor float_tensor(dif::ir::DType dtype,
                                  std::vector<std::uint64_t> dims,
                                  const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dtype, std::move(dims), {}};
  tensor.bytes.resize(values.size() * dif::ir::dtype_size(dtype));
  tensor.validate();
  for (std::size_t index = 0U; index < values.size(); ++index)
    dif::runtime::store_float(tensor, index, values[index]);
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

std::vector<float> float_values(const dif::runtime::Tensor &tensor) {
  std::vector<float> values(tensor.element_count());
  for (std::size_t index = 0U; index < values.size(); ++index)
    values[index] = dif::runtime::load_float(tensor, index);
  return values;
}

std::size_t count_operations(const dif::ir::Program &program,
                             dif::ir::Opcode opcode) {
  return static_cast<std::size_t>(
      std::count_if(program.operations.begin(), program.operations.end(),
                    [&](const auto &operation) {
                      return operation.opcode == opcode;
                    }));
}

std::vector<float> pattern(std::size_t count, float base, float step,
                           float amplitude, float offset) {
  std::vector<float> values(count);
  for (std::size_t index = 0U; index < count; ++index)
    values[index] =
        offset +
        amplitude * std::sin(base + step * static_cast<float>(index));
  return values;
}

dif::frontend::DitLoraTrainingConfig small_config(bool bf16) {
  dif::frontend::DitLoraTrainingConfig config;
  config.sequence = 8U;
  config.heads = 2U;
  config.head_dim = 4U;
  config.mlp_width = 8U;
  config.blocks = 1U;
  config.rotary_dim = 4U;
  config.rank = 2U;
  config.alpha = 4.0; // scale 2.0: a defaulted/dropped scale cannot pass
  config.compute_dtype = bf16 ? dif::ir::DType::BF16 : dif::ir::DType::F32;
  return config;
}

dif::runtime::TensorMap
small_inputs(const dif::frontend::DitLoraTrainingBuild &build,
             std::uint64_t seed) {
  const auto compute = build.config.compute_dtype;
  dif::runtime::TensorMap inputs;
  float phase = 0.1F * static_cast<float>(seed % 97U);
  const auto data = [&](std::uint32_t id, float amplitude, float offset) {
    const auto *description = build.program.tensor(id);
    inputs.insert_or_assign(
        id, float_tensor(description->dtype, description->dims,
                         pattern(description->element_count(), phase, 0.7F,
                                 amplitude, offset)));
    phase += 1.3F;
  };
  data(build.x_input, 0.8F, 0.0F);
  data(build.cos_input, 1.0F, 0.0F);
  data(build.sin_input, 1.0F, 0.0F);
  data(build.target_input, 0.8F, 0.0F);
  for (const auto &modulation : build.modulation_inputs) {
    for (const auto id :
         {modulation.scale1, modulation.shift1, modulation.gate1,
          modulation.scale2, modulation.shift2, modulation.gate2})
      data(id, 0.2F, 0.0F);
  }
  for (const auto &frozen : build.frozen_constants) {
    const auto *description = build.program.tensor(frozen.tensor_id);
    const bool norm_weight =
        frozen.name.find("norm") != std::string::npos;
    const bool matrix = description->dims.size() == 2U;
    const auto amplitude =
        norm_weight ? 0.2F
        : matrix    ? 1.0F / std::sqrt(static_cast<float>(
                                description->dims[1]))
                    : 0.05F;
    data(frozen.tensor_id, amplitude, norm_weight ? 1.0F : 0.0F);
  }
  for (const auto &adapter : build.adapters) {
    const auto *a_description = build.program.tensor(adapter.lora_a);
    const auto bound =
        1.0F / std::sqrt(static_cast<float>(a_description->dims[1]));
    inputs.insert_or_assign(
        adapter.lora_a,
        float_tensor(dif::ir::DType::F32, a_description->dims,
                     pattern(a_description->element_count(), phase, 0.9F,
                             0.9F * bound, 0.0F)));
    phase += 0.7F;
    inputs.insert_or_assign(
        adapter.lora_b,
        dif::runtime::zeros(*build.program.tensor(adapter.lora_b)));
  }
  for (const auto &binding : build.optimizer_bindings) {
    inputs.emplace(binding.first_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.first_moment_input)));
    inputs.emplace(binding.second_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.second_moment_input)));
  }
  inputs.emplace(build.step_input, i32_tensor({1U}, {0}));
  static_cast<void>(compute);
  return inputs;
}

void test_builder_structure() {
  for (const bool bf16 : {true, false}) {
    const auto config = small_config(bf16);
    const auto build = dif::frontend::make_dit_lora_training(config);
    expect(build.adapters.size() == 6U &&
               build.optimizer_bindings.size() == 12U &&
               build.frozen_constants.size() == 16U,
           "DiT LoRA builder exposes 6 sites, 12 adapters, 16 frozen "
           "tensors per block");
    std::unordered_set<std::uint32_t> frozen_ids;
    for (const auto &frozen : build.frozen_constants) {
      frozen_ids.insert(frozen.tensor_id);
      const auto *description = build.program.tensor(frozen.tensor_id);
      expect(description &&
                 description->roles == dif::ir::TensorRole::Constant &&
                 description->dtype == config.compute_dtype,
             "frozen base tensors are pure Constants in the compute dtype");
    }
    for (const auto &adapter : build.adapters) {
      const auto *a_description = build.program.tensor(adapter.lora_a);
      const auto *b_description = build.program.tensor(adapter.lora_b);
      expect(a_description && b_description &&
                 a_description->dtype == dif::ir::DType::F32 &&
                 b_description->dtype == dif::ir::DType::F32 &&
                 a_description->has_role(dif::ir::TensorRole::Parameter) &&
                 b_description->has_role(dif::ir::TensorRole::Parameter) &&
                 a_description->dims[0] == config.rank &&
                 b_description->dims[1] == config.rank,
             "adapters are F32 parameters A [rank,in] / B [out,rank]");
    }
    expect(build.adapters[0].name == "block0.q" &&
               build.adapters[5].name == "block0.fc2",
           "adapter name map follows the canonical site order");
    expect(count_operations(build.program, dif::ir::Opcode::Linear) == 18U,
           "6 base + 12 adapter Linears per block");
    expect(count_operations(build.program,
                            dif::ir::Opcode::LinearBackwardInput) == 18U,
           "every Linear keeps its input-gradient path");
    expect(count_operations(build.program,
                            dif::ir::Opcode::LinearBackwardWeight) == 12U,
           "only the 12 adapter Linears emit weight gradients (frozen-dW "
           "economy on the base)");
    const auto casts =
        count_operations(build.program, dif::ir::Opcode::Cast);
    expect(bf16 ? casts == 24U : casts == 0U,
           "BF16 builds 12 forward adapter Casts + 12 gradient Casts; F32 "
           "builds none");
    expect(build.program.tensor(build.loss_output)->dtype ==
                   dif::ir::DType::F32 &&
               build.program.tensor(build.prediction_output)->dtype ==
                   config.compute_dtype,
           "loss is F32; prediction carries the compute dtype");
    for (const auto &operation : build.program.operations) {
      for (const auto output : operation.outputs)
        expect(!frozen_ids.contains(output),
               "no operation writes a frozen base tensor");
      if (operation.opcode == dif::ir::Opcode::AdamWUpdate)
        for (const auto input : operation.inputs)
          expect(!frozen_ids.contains(input),
                 "the optimizer never touches a frozen base tensor");
    }
    const auto rebuilt = dif::frontend::make_dit_lora_training(config);
    expect(dif::ir::fingerprint(build.program) ==
               dif::ir::fingerprint(rebuilt.program),
           "DiT LoRA builder is deterministic under fingerprint");
  }
}

void test_step_one_semantics_and_cuda_parity() {
  const auto config = small_config(true);
  const auto build = dif::frontend::make_dit_lora_training(config);
  auto inputs = small_inputs(build, 42U);

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);
  const auto loss = reference.outputs.at(build.loss_output).f32()[0];
  expect(std::isfinite(loss) && loss > 0.0F,
         "BF16 DiT LoRA graph produces a finite positive loss");

  for (std::size_t site = 0U; site < build.adapters.size(); ++site) {
    const auto &a_binding = build.optimizer_bindings[2U * site];
    const auto &b_binding = build.optimizer_bindings[2U * site + 1U];
    expect(a_binding.parameter_input == build.adapters[site].lora_a &&
               b_binding.parameter_input == build.adapters[site].lora_b,
           "optimizer bindings pair (A,B) per site in order");
    const auto a_gradient =
        reference.outputs.at(a_binding.gradient_output).f32();
    expect(std::all_of(a_gradient.begin(), a_gradient.end(),
                       [](float value) { return value == 0.0F; }),
           "step-1 dL/dA is exactly zero while B == 0 (BF16 block)");
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
      const auto expected = float_values(expected_tensor);
      const auto actual = float_values(candidate.outputs.at(id));
      for (std::size_t index = 0U; index < expected.size(); ++index)
        maximum_absolute_error =
            std::max(maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    // Bar from the recorded DiT BF16 CPU-vs-CUDA precedent (worst 3.9e-3 =
    // one BF16 ulp at gradient magnitude; bar 1.6e-2).
    expect(maximum_absolute_error <= 1.6e-2F,
           "CUDA BF16 DiT LoRA one-step matches CPU within the recorded "
           "BF16 bar");
    std::cout << "GATE dit_lora_one_step backend=" << candidate.backend_name
              << " device=" << candidate.device_name
              << " max_abs=" << maximum_absolute_error << "\n";
  }
}

// Grad-flow gate: a fused inference plan on a differentiated program must be
// refused at prepare (the plan declares no backward), not silently executed.
void test_fused_plan_refused_on_training_program() {
  if (!dif::runtime::cuda_available())
    return;
  const auto config = small_config(true);
  const auto build = dif::frontend::make_dit_lora_training(config);
  auto inputs = small_inputs(build, 42U);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  std::uint32_t linear_id = 0U;
  for (const auto &operation : build.program.operations)
    if (operation.opcode == dif::ir::Opcode::Linear) {
      linear_id = operation.id;
      break;
    }
  options.fuse_linear_swiglu_operations = {linear_id};
  bool refused = false;
  std::string message;
  try {
    (void)dif::runtime::make_cuda_executor()->run(build.program, inputs,
                                                  options);
  } catch (const std::exception &error) {
    refused = true;
    message = error.what();
  }
  expect(refused && message.find("declare no backward") != std::string::npos,
         "a fused inference plan on a training program is refused at prepare");
  std::cout << "GATE dit_lora_fused_plan_refused message=" << message << "\n";
}

void test_export_and_alpha_regression() {
  const auto config = small_config(true);
  const auto build = dif::frontend::make_dit_lora_training(config);
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
      ("dif-dit-lora-export-" +
       std::to_string(std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count()));
  std::filesystem::create_directories(temporary);
  const auto exported = temporary / "adapters.safetensors";
  dif::frontend::export_lora_adapters(build.program, build.adapters,
                                      checkpoint, exported);
  bool validated = true;
  try {
    dif::frontend::validate_lora_export(exported, build.adapters);
  } catch (const dif::Error &) {
    validated = false;
  }
  expect(validated,
         "DiT LoRA export passes the .alpha validation gate");
  const auto file = dif::weights::read_safetensors(exported);
  expect(file.find("block0.q.lora_A.weight") != nullptr &&
             file.find("block0.fc2.lora_B.weight") != nullptr &&
             file.find("block0.out.alpha") != nullptr,
         "export carries per-site lora_A/lora_B/.alpha names");
  {
    const auto stored_a_tensor =
        dif::weights::map_safetensor(file, "block0.q.lora_A.weight");
    const auto stored_a = stored_a_tensor.f32();
    const auto trained_a =
        checkpoint.state.at(build.adapters[0].lora_a).f32();
    expect(stored_a.size() == trained_a.size() &&
               std::equal(stored_a.begin(), stored_a.end(),
                          trained_a.begin()),
           "exported lora_A bytes equal trained checkpoint state");
  }
  dif::training::Checkpoint mismatched = checkpoint;
  mismatched.program_fingerprint[0] ^= 1U;
  bool rejected = false;
  try {
    dif::frontend::export_lora_adapters(build.program, build.adapters,
                                        mismatched,
                                        temporary / "bad.safetensors");
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "export rejects a checkpoint from a different program");
  std::filesystem::remove_all(temporary);
}

} // namespace

int main() {
  try {
    test_builder_structure();
    test_step_one_semantics_and_cuda_parity();
    test_fused_plan_refused_on_training_program();
    test_export_and_alpha_regression();
  } catch (const std::exception &error) {
    std::cerr << "unhandled: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " DiT LoRA test expectation(s) failed\n";
    return 1;
  }
  std::cout << "OK dif_dit_lora_tests\n";
  return 0;
}
