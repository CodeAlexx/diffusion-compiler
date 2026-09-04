#include "dif/frontend/dit_block.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"
#include "dif/training/step.hpp"

#include <algorithm>
#include <string>

namespace dif::frontend {

DitBlockTrainingBuild
make_dit_block_training(const DitBlockTrainingConfig &config) {
  if (config.sequence == 0U || config.heads == 0U || config.head_dim == 0U ||
      config.mlp_width == 0U || config.blocks == 0U)
    fail("DiT block training dimensions must be positive");
  if (config.rotary_dim == 0U || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0U)
    fail("DiT block rotary_dim must be even and at most head_dim");
  if (config.sequence > 4096U)
    fail("DiT block training admits sequence <= 4096 (naive attention)");

  DitBlockTrainingBuild build;
  build.config = config;
  auto &program = build.program;
  const auto sequence = config.sequence;
  const auto hidden = config.heads * config.head_dim;
  const auto table_width =
      config.full_rope_table ? config.rotary_dim : config.rotary_dim / 2U;
  const auto dtype = ir::DType::F32;
  constexpr auto input_role = ir::TensorRole::Input;
  constexpr auto parameter_role =
      ir::TensorRole::Input | ir::TensorRole::Parameter;

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto add_tensor = [&](std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_operation = [&](ir::Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<ir::Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto parameter = [&](std::uint64_t block, const char *name,
                             std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(parameter_role, std::move(dims));
    build.parameters.push_back(id);
    build.parameter_names.push_back("block" + std::to_string(block) + "." +
                                    name);
    return id;
  };

  build.x_input = add_tensor(input_role, {sequence, hidden});
  build.cos_input = add_tensor(input_role, {sequence, table_width});
  build.sin_input = add_tensor(input_role, {sequence, table_width});
  build.target_input = add_tensor(input_role, {sequence, hidden});

  const auto norm_epsilon =
      ir::Attribute::f64(ir::AttrKey::Epsilon, config.epsilon_norm);
  const auto rotary_attribute =
      ir::Attribute::u64(ir::AttrKey::RotaryDim, config.rotary_dim);
  const auto causal_attribute =
      ir::Attribute::boolean(ir::AttrKey::Causal, config.causal);

  auto x = build.x_input;
  for (std::uint64_t block = 0U; block < config.blocks; ++block) {
    DitBlockModulationInputs modulation;
    modulation.scale1 = add_tensor(input_role, {sequence, hidden});
    modulation.shift1 = add_tensor(input_role, {sequence, hidden});
    modulation.gate1 = add_tensor(input_role, {sequence, hidden});
    modulation.scale2 = add_tensor(input_role, {sequence, hidden});
    modulation.shift2 = add_tensor(input_role, {sequence, hidden});
    modulation.gate2 = add_tensor(input_role, {sequence, hidden});
    build.modulation_inputs.push_back(modulation);

    const auto norm1_weight = parameter(block, "norm1_w", {hidden});
    const auto q_weight = parameter(block, "q_w", {hidden, hidden});
    const auto q_bias = parameter(block, "q_b", {hidden});
    const auto k_weight = parameter(block, "k_w", {hidden, hidden});
    const auto k_bias = parameter(block, "k_b", {hidden});
    const auto v_weight = parameter(block, "v_w", {hidden, hidden});
    const auto v_bias = parameter(block, "v_b", {hidden});
    const auto q_norm_weight =
        parameter(block, "q_norm_w", {config.head_dim});
    const auto k_norm_weight =
        parameter(block, "k_norm_w", {config.head_dim});
    const auto out_weight = parameter(block, "out_w", {hidden, hidden});
    const auto out_bias = parameter(block, "out_b", {hidden});
    const auto norm2_weight = parameter(block, "norm2_w", {hidden});
    const auto fc1_weight =
        parameter(block, "fc1_w", {2U * config.mlp_width, hidden});
    const auto fc1_bias = parameter(block, "fc1_b", {2U * config.mlp_width});
    const auto fc2_weight =
        parameter(block, "fc2_w", {hidden, config.mlp_width});
    const auto fc2_bias = parameter(block, "fc2_b", {hidden});

    const auto modulated1 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::RmsNormModulate,
                  {x, norm1_weight, modulation.scale1, modulation.shift1},
                  {modulated1}, {norm_epsilon});
    const auto q = add_tensor(ir::TensorRole::Internal,
                              {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::Linear, {modulated1, q_weight, q_bias}, {q});
    const auto k = add_tensor(ir::TensorRole::Internal,
                              {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::Linear, {modulated1, k_weight, k_bias}, {k});
    const auto v = add_tensor(ir::TensorRole::Internal,
                              {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::Linear, {modulated1, v_weight, v_bias}, {v});
    const auto q_rotated = add_tensor(
        ir::TensorRole::Internal, {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::QkNormPartialRope,
                  {q, q_norm_weight, build.cos_input, build.sin_input},
                  {q_rotated}, {norm_epsilon, rotary_attribute});
    const auto k_rotated = add_tensor(
        ir::TensorRole::Internal, {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::QkNormPartialRope,
                  {k, k_norm_weight, build.cos_input, build.sin_input},
                  {k_rotated}, {norm_epsilon, rotary_attribute});
    const auto attended = add_tensor(
        ir::TensorRole::Internal, {sequence, config.heads, config.head_dim});
    add_operation(ir::Opcode::Attention, {q_rotated, k_rotated, v},
                  {attended}, {causal_attribute});
    const auto projected =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::Linear, {attended, out_weight, out_bias},
                  {projected});
    const auto gated1 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::ResidualGate, {x, projected, modulation.gate1},
                  {gated1});
    const auto modulated2 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::RmsNormModulate,
                  {gated1, norm2_weight, modulation.scale2,
                   modulation.shift2},
                  {modulated2}, {norm_epsilon});
    const auto expanded = add_tensor(ir::TensorRole::Internal,
                                     {sequence, 2U * config.mlp_width});
    add_operation(ir::Opcode::Linear, {modulated2, fc1_weight, fc1_bias},
                  {expanded});
    const auto activated = add_tensor(ir::TensorRole::Internal,
                                      {sequence, config.mlp_width});
    add_operation(ir::Opcode::SwiGlu, {expanded}, {activated},
                  {ir::Attribute::boolean(ir::AttrKey::GateFirst, true)});
    const auto contracted =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::Linear, {activated, fc2_weight, fc2_bias},
                  {contracted});
    const auto gated2 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::ResidualGate,
                  {gated1, contracted, modulation.gate2}, {gated2});
    x = gated2;
  }

  // Mark the final block output as the observable prediction.
  for (auto &tensor : program.tensors)
    if (tensor.id == x)
      tensor.roles |= ir::TensorRole::Output;
  build.prediction_output = x;
  build.loss_output = add_tensor(ir::TensorRole::Output, {1U});
  {
    auto &loss_tensor = program.tensors.back();
    loss_tensor.dtype = ir::DType::F32;
    loss_tensor.dims = {1U};
  }
  add_operation(ir::Opcode::MseLoss, {x, build.target_input},
                {build.loss_output});

  // Forward, backward and the AdamW update compose into ONE program with one
  // tensor namespace, so the memory planner sees the activations, the
  // gradients and the optimizer state at the same time.
  const training::OptimizerHyperparameters hyperparameters{
      config.learning_rate, config.beta1, config.beta2, config.epsilon_adam,
      config.weight_decay};
  auto step = training::build_training_step(program, build.loss_output,
                                            build.parameters, hyperparameters);
  build.program = std::move(step.program);
  build.step_input = step.step_input;
  // The bindings are difcore's own type now, so the master-weight fields
  // travel with them instead of being dropped by a field-by-field copy.
  build.optimizer_bindings = std::move(step.bindings);
  ir::verify(build.program);
  return build;
}

} // namespace dif::frontend
