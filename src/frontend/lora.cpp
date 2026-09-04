#include "dif/frontend/lora.hpp"

#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"
#include "dif/training/step.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace dif::frontend {
namespace {

// Deterministic SplitMix64; documented generator so adapter init is
// reproducible across platforms without depending on a library
// distribution's implementation details.
std::uint64_t splitmix64_next(std::uint64_t &state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t mixed = state;
  mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
  return mixed ^ (mixed >> 31U);
}

} // namespace

LoraFlowTrainingBuild
make_lora_flow_training(const LoraFlowTrainingConfig &config) {
  if (config.rows == 0U || config.latent_width == 0U ||
      config.timestep_width == 0U || config.hidden_width == 0U ||
      config.rank == 0U)
    fail("LoRA training dimensions and rank must be positive");
  if (!(config.alpha > 0.0))
    fail("LoRA alpha must be positive");

  LoraFlowTrainingBuild build;
  auto &forward = build.program;
  constexpr auto input = ir::TensorRole::Input;
  constexpr auto output = ir::TensorRole::Output;
  constexpr auto internal = ir::TensorRole::Internal;
  constexpr auto constant = ir::TensorRole::Constant;
  constexpr auto parameter = ir::TensorRole::Input | ir::TensorRole::Parameter;
  const auto rows = config.rows;
  const auto latent = config.latent_width;
  const auto timestep = config.timestep_width;
  const auto hidden = config.hidden_width;
  const auto rank = config.rank;
  const double scale = config.alpha / static_cast<double>(config.rank);

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto add_tensor = [&](std::uint32_t role, std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    forward.tensors.push_back({id, ir::DType::F32, role, std::move(dims)});
    return id;
  };
  auto add_operation = [&](ir::Opcode opcode,
                           std::vector<std::uint32_t> inputs,
                           std::vector<std::uint32_t> outputs,
                           std::vector<ir::Attribute> attributes = {}) {
    forward.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs),
         std::move(attributes)});
  };

  build.clean_input = add_tensor(input, {rows, latent});
  build.noise_input = add_tensor(input, {rows, latent});
  build.clean_scale_input = add_tensor(input, {rows, latent});
  build.noise_scale_input = add_tensor(input, {rows, latent});
  build.timestep_features_input = add_tensor(input, {rows, timestep});
  build.target_velocity_input = add_tensor(input, {rows, latent});

  // Frozen base model: role Constant (bundle-bindable, never updated).
  const auto base_latent_weight = add_tensor(constant, {hidden, latent});
  const auto base_latent_bias = add_tensor(constant, {hidden});
  const auto base_time_weight = add_tensor(constant, {hidden, timestep});
  const auto base_out_weight = add_tensor(constant, {latent, hidden});
  const auto base_out_bias = add_tensor(constant, {latent});
  build.frozen_constants = {
      {base_latent_weight, "latent_proj.base.weight"},
      {base_latent_bias, "latent_proj.base.bias"},
      {base_time_weight, "time_proj.base.weight"},
      {base_out_weight, "out_proj.base.weight"},
      {base_out_bias, "out_proj.base.bias"},
  };

  // Trainable adapters: A [rank, in], B [out, rank].
  const auto latent_a = add_tensor(parameter, {rank, latent});
  const auto latent_b = add_tensor(parameter, {hidden, rank});
  const auto time_a = add_tensor(parameter, {rank, timestep});
  const auto time_b = add_tensor(parameter, {hidden, rank});
  const auto out_a = add_tensor(parameter, {rank, hidden});
  const auto out_b = add_tensor(parameter, {latent, rank});
  build.adapters = {
      {"latent_proj", base_latent_weight, latent_a, latent_b, rank,
       config.alpha},
      {"time_proj", base_time_weight, time_a, time_b, rank, config.alpha},
      {"out_proj", base_out_weight, out_a, out_b, rank, config.alpha},
  };
  const std::array<std::uint32_t, 6> adapter_parameters = {
      latent_a, latent_b, time_a, time_b, out_a, out_b};

  // One LoRA-augmented Linear:
  //   out = Linear(x, W_base) + Fill(alpha/rank) * Linear(Linear(x, A), B)
  // The low-rank path stays explicit; the dense delta is never formed.
  auto lora_linear = [&](std::uint32_t x, std::uint32_t base_weight,
                         std::uint32_t lora_a, std::uint32_t lora_b,
                         std::uint64_t out_width) {
    const auto base = add_tensor(internal, {rows, out_width});
    add_operation(ir::Opcode::Linear, {x, base_weight}, {base});
    const auto low = add_tensor(internal, {rows, rank});
    add_operation(ir::Opcode::Linear, {x, lora_a}, {low});
    const auto delta = add_tensor(internal, {rows, out_width});
    add_operation(ir::Opcode::Linear, {low, lora_b}, {delta});
    const auto delta_scale = add_tensor(internal, {rows, out_width});
    add_operation(ir::Opcode::Fill, {}, {delta_scale},
                  {ir::Attribute::f64(ir::AttrKey::Value, scale)});
    const auto scaled = add_tensor(internal, {rows, out_width});
    add_operation(ir::Opcode::Multiply, {delta, delta_scale}, {scaled});
    const auto combined = add_tensor(internal, {rows, out_width});
    add_operation(ir::Opcode::Add, {base, scaled}, {combined});
    return combined;
  };

  const auto scaled_clean = add_tensor(internal, {rows, latent});
  add_operation(ir::Opcode::Multiply,
                {build.clean_input, build.clean_scale_input}, {scaled_clean});
  const auto scaled_noise = add_tensor(internal, {rows, latent});
  add_operation(ir::Opcode::Multiply,
                {build.noise_input, build.noise_scale_input}, {scaled_noise});
  const auto noised = add_tensor(internal, {rows, latent});
  add_operation(ir::Opcode::Add, {scaled_clean, scaled_noise}, {noised});

  const auto latent_hidden =
      lora_linear(noised, base_latent_weight, latent_a, latent_b, hidden);
  const auto biased_hidden = add_tensor(internal, {rows, hidden});
  add_operation(ir::Opcode::BiasAdd, {latent_hidden, base_latent_bias},
                {biased_hidden});
  const auto time_hidden = lora_linear(build.timestep_features_input,
                                       base_time_weight, time_a, time_b,
                                       hidden);
  const auto conditioned = add_tensor(internal, {rows, hidden});
  add_operation(ir::Opcode::Add, {biased_hidden, time_hidden}, {conditioned});
  const auto activated = add_tensor(internal, {rows, hidden});
  add_operation(ir::Opcode::SiLU, {conditioned}, {activated});
  const auto projected =
      lora_linear(activated, base_out_weight, out_a, out_b, latent);
  build.prediction_output = add_tensor(output, {rows, latent});
  add_operation(ir::Opcode::BiasAdd, {projected, base_out_bias},
                {build.prediction_output});
  build.loss_output = add_tensor(output, {1U});
  add_operation(ir::Opcode::MseLoss,
                {build.prediction_output, build.target_velocity_input},
                {build.loss_output});

  // Forward, backward and the AdamW update compose into ONE program with one
  // tensor namespace, so the memory planner sees the activations, the
  // gradients and the optimizer state at the same time.
  const training::OptimizerHyperparameters hyperparameters{
      config.learning_rate, config.beta1, config.beta2, config.epsilon,
      config.weight_decay};
  auto step = training::build_training_step(forward, build.loss_output,
                                            adapter_parameters, hyperparameters);
  build.program = std::move(step.program);
  build.step_input = step.step_input;
  for (const auto &binding : step.bindings)
    build.optimizer_bindings.push_back(
        {binding.parameter_input, binding.gradient_output,
         binding.first_moment_input, binding.second_moment_input,
         binding.parameter_output, binding.first_moment_output,
         binding.second_moment_output});
  ir::verify(build.program);
  return build;
}

runtime::TensorMap
default_lora_adapter_init(const LoraFlowTrainingBuild &build,
                          std::uint64_t seed) {
  runtime::TensorMap initial;
  std::uint64_t state = seed;
  for (const auto &adapter : build.adapters) {
    const auto *a_description = build.program.tensor(adapter.lora_a);
    const auto *b_description = build.program.tensor(adapter.lora_b);
    if (!a_description || !b_description ||
        a_description->dims.size() != 2U)
      fail("LoRA build adapter tensors are malformed");
    const auto in_features = a_description->dims[1];
    const auto bound =
        1.0 / std::sqrt(static_cast<double>(in_features));
    const auto elements = a_description->element_count();
    runtime::Tensor lora_a{ir::DType::F32, a_description->dims,
                           std::vector<std::uint8_t>(elements * 4U)};
    auto *values = reinterpret_cast<float *>(lora_a.mutable_data());
    for (std::uint64_t index = 0U; index < elements; ++index) {
      const auto unit = static_cast<double>(splitmix64_next(state) >> 11U) *
                        0x1.0p-53;
      values[index] = static_cast<float>((2.0 * unit - 1.0) * bound);
    }
    initial.emplace(adapter.lora_a, std::move(lora_a));
    initial.emplace(adapter.lora_b, runtime::zeros(*b_description));
  }
  return initial;
}

void export_lora_adapters(const ir::Program &program,
                          std::span<const LoraAdapterBinding> adapters,
                          const training::Checkpoint &checkpoint,
                          const std::filesystem::path &path) {
  if (adapters.empty())
    fail("LoRA export requires at least one adapter");
  if (checkpoint.program_fingerprint != ir::fingerprint(program))
    fail("LoRA export checkpoint targets a different program fingerprint");
  std::vector<weights::SafeTensorWriteSpec> specs;
  for (const auto &adapter : adapters) {
    const auto *a_description = program.tensor(adapter.lora_a);
    const auto *b_description = program.tensor(adapter.lora_b);
    if (!a_description || !b_description)
      fail("LoRA export adapter tensor is not in the program");
    specs.push_back({adapter.name + ".lora_A.weight", ir::DType::F32,
                     a_description->dims});
    specs.push_back({adapter.name + ".lora_B.weight", ir::DType::F32,
                     b_description->dims});
    // .alpha is export metadata, NOT trainable state.  Omitting it makes
    // loaders assume scale=1.0 (flame's 16x over-application incident).
    specs.push_back({adapter.name + ".alpha", ir::DType::F32, {1U}});
  }
  weights::SafeTensorWriter writer(path, std::move(specs));
  for (const auto &adapter : adapters) {
    for (const auto &[suffix, id] :
         {std::pair<const char *, std::uint32_t>{".lora_A.weight",
                                                 adapter.lora_a},
          std::pair<const char *, std::uint32_t>{".lora_B.weight",
                                                 adapter.lora_b}}) {
      const auto found = checkpoint.state.find(id);
      if (found == checkpoint.state.end())
        fail("LoRA export checkpoint is missing adapter tensor " +
             std::to_string(id));
      writer.append(adapter.name + suffix,
                    {found->second.data(), found->second.byte_size()});
    }
    const auto alpha_value = static_cast<float>(adapter.alpha);
    std::array<std::uint8_t, 4> alpha_bytes{};
    std::memcpy(alpha_bytes.data(), &alpha_value, sizeof(alpha_value));
    writer.append(adapter.name + ".alpha", alpha_bytes);
  }
  writer.finish();
}

void export_lora_adapters(const LoraFlowTrainingBuild &build,
                          const training::Checkpoint &checkpoint,
                          const std::filesystem::path &path) {
  export_lora_adapters(build.program, build.adapters, checkpoint, path);
}

void validate_lora_export(const std::filesystem::path &path,
                          std::span<const LoraAdapterBinding> adapters) {
  if (adapters.empty())
    fail("LoRA export validation requires at least one adapter");
  const auto file = weights::read_safetensors(path);
  for (const auto &adapter : adapters) {
    for (const auto suffix : {".lora_A.weight", ".lora_B.weight"}) {
      const auto *entry = file.find(adapter.name + suffix);
      if (!entry || entry->dtype != ir::DType::F32)
        fail("LoRA export is missing F32 tensor " + adapter.name + suffix);
    }
    const auto alpha_name = adapter.name + ".alpha";
    const auto *alpha = file.find(alpha_name);
    if (!alpha)
      fail("LoRA export is missing required scalar " + alpha_name +
           " — external loaders would fall back to scale=1.0 and "
           "mis-apply the adapter");
    std::uint64_t elements = 1U;
    for (const auto dimension : alpha->dims)
      elements *= dimension;
    if (alpha->dtype != ir::DType::F32 || elements != 1U)
      fail("LoRA export " + alpha_name + " must be a single F32 value");
    const auto alpha_tensor = weights::map_safetensor(file, alpha_name);
    const auto stored = alpha_tensor.f32()[0];
    if (stored != static_cast<float>(adapter.alpha))
      fail("LoRA export " + alpha_name + " does not match the build alpha");
  }
}

} // namespace dif::frontend
