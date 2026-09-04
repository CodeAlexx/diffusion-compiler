#include "dif/frontend/dit_lora.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"
#include "dif/training/step.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace dif::frontend {

DitLoraTrainingBuild
make_dit_lora_training(const DitLoraTrainingConfig &config) {
  if (config.sequence == 0U || config.heads == 0U || config.head_dim == 0U ||
      config.mlp_width == 0U || config.blocks == 0U || config.rank == 0U)
    fail("DiT LoRA training dimensions and rank must be positive");
  if (config.rotary_dim == 0U || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0U)
    fail("DiT LoRA rotary_dim must be even and at most head_dim");
  if (config.sequence > 4096U)
    fail("DiT LoRA training admits sequence <= 4096 (naive attention)");
  if (config.compute_dtype != ir::DType::F32 &&
      config.compute_dtype != ir::DType::BF16)
    fail("DiT LoRA compute dtype must be F32 or BF16");
  if (!(config.alpha > 0.0))
    fail("DiT LoRA alpha must be positive");

  DitLoraTrainingBuild build;
  build.config = config;
  auto &program = build.program;
  const auto sequence = config.sequence;
  const auto hidden = config.heads * config.head_dim;
  const auto rank = config.rank;
  const auto table_width =
      config.full_rope_table ? config.rotary_dim : config.rotary_dim / 2U;
  const auto compute = config.compute_dtype;
  const bool mixed = compute != ir::DType::F32;
  const double scale = config.alpha / static_cast<double>(config.rank);
  constexpr auto input_role = ir::TensorRole::Input;
  constexpr auto constant_role = ir::TensorRole::Constant;
  constexpr auto parameter_role =
      ir::TensorRole::Input | ir::TensorRole::Parameter;

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto add_typed = [&](ir::DType dtype, std::uint32_t roles,
                             std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_tensor = [&](std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    return add_typed(compute, roles, std::move(dims));
  };
  const auto add_operation = [&](ir::Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<ir::Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto frozen = [&](std::uint64_t block, const char *name,
                          std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(constant_role, std::move(dims));
    build.frozen_constants.push_back(
        {id, "block" + std::to_string(block) + "." + name});
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

  struct SiteIds {
    std::uint32_t base_weight{};
    std::uint32_t base_bias{};
    std::uint32_t lora_a{};
    std::uint32_t lora_b{};
  };
  std::vector<std::uint32_t> adapter_parameters;

  // One LoRA-augmented Linear.  The adapters are stored F32 and cross an
  // explicit Cast into the compute dtype in mixed mode (autograd-aware:
  // their gradients Cast back to F32).  The low-rank path stays explicit;
  // the dense delta is never formed.  The alpha/rank scale is an in-graph
  // fingerprinted Fill multiplied into the delta.
  const auto lora_linear = [&](std::uint32_t x, const SiteIds &site,
                               std::vector<std::uint64_t> out_dims,
                               std::uint64_t out_width) {
    const auto base = add_tensor(ir::TensorRole::Internal, out_dims);
    add_operation(ir::Opcode::Linear, {x, site.base_weight, site.base_bias},
                  {base});
    auto weight_a = site.lora_a;
    auto weight_b = site.lora_b;
    if (mixed) {
      const auto a_dims = program.tensor(site.lora_a)->dims;
      const auto b_dims = program.tensor(site.lora_b)->dims;
      weight_a = add_tensor(ir::TensorRole::Internal, a_dims);
      add_operation(ir::Opcode::Cast, {site.lora_a}, {weight_a});
      weight_b = add_tensor(ir::TensorRole::Internal, b_dims);
      add_operation(ir::Opcode::Cast, {site.lora_b}, {weight_b});
    }
    const auto low = add_tensor(ir::TensorRole::Internal, {sequence, rank});
    add_operation(ir::Opcode::Linear, {x, weight_a}, {low});
    const auto delta = add_tensor(ir::TensorRole::Internal, out_dims);
    add_operation(ir::Opcode::Linear, {low, weight_b}, {delta});
    const auto delta_scale = add_tensor(ir::TensorRole::Internal, out_dims);
    add_operation(ir::Opcode::Fill, {}, {delta_scale},
                  {ir::Attribute::f64(ir::AttrKey::Value, scale)});
    const auto scaled = add_tensor(ir::TensorRole::Internal, out_dims);
    add_operation(ir::Opcode::Multiply, {delta, delta_scale}, {scaled});
    const auto combined = add_tensor(ir::TensorRole::Internal, out_dims);
    add_operation(ir::Opcode::Add, {base, scaled}, {combined});
    static_cast<void>(out_width);
    return combined;
  };

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

    // Frozen base, canonical make_dit_block_training order.
    const auto norm1_weight = frozen(block, "norm1_w", {hidden});
    const auto q_weight = frozen(block, "q_w", {hidden, hidden});
    const auto q_bias = frozen(block, "q_b", {hidden});
    const auto k_weight = frozen(block, "k_w", {hidden, hidden});
    const auto k_bias = frozen(block, "k_b", {hidden});
    const auto v_weight = frozen(block, "v_w", {hidden, hidden});
    const auto v_bias = frozen(block, "v_b", {hidden});
    const auto q_norm_weight = frozen(block, "q_norm_w", {config.head_dim});
    const auto k_norm_weight = frozen(block, "k_norm_w", {config.head_dim});
    const auto out_weight = frozen(block, "out_w", {hidden, hidden});
    const auto out_bias = frozen(block, "out_b", {hidden});
    const auto norm2_weight = frozen(block, "norm2_w", {hidden});
    const auto fc1_weight = frozen(block, "fc1_w",
                                   {2U * config.mlp_width, hidden});
    const auto fc1_bias = frozen(block, "fc1_b", {2U * config.mlp_width});
    const auto fc2_weight = frozen(block, "fc2_w",
                                   {hidden, config.mlp_width});
    const auto fc2_bias = frozen(block, "fc2_b", {hidden});

    // Adapters, canonical site order q,k,v,out,fc1,fc2; A [rank,in] and
    // B [out,rank], stored F32 regardless of the compute dtype.
    const auto site = [&](const char *name, std::uint32_t base_weight,
                          std::uint32_t base_bias, std::uint64_t in_width,
                          std::uint64_t out_width) {
      SiteIds ids;
      ids.base_weight = base_weight;
      ids.base_bias = base_bias;
      ids.lora_a =
          add_typed(ir::DType::F32, parameter_role, {rank, in_width});
      ids.lora_b =
          add_typed(ir::DType::F32, parameter_role, {out_width, rank});
      adapter_parameters.push_back(ids.lora_a);
      adapter_parameters.push_back(ids.lora_b);
      build.adapters.push_back(
          {"block" + std::to_string(block) + "." + name, base_weight,
           ids.lora_a, ids.lora_b, rank, config.alpha});
      return ids;
    };
    const auto q_site = site("q", q_weight, q_bias, hidden, hidden);
    const auto k_site = site("k", k_weight, k_bias, hidden, hidden);
    const auto v_site = site("v", v_weight, v_bias, hidden, hidden);
    const auto out_site = site("out", out_weight, out_bias, hidden, hidden);
    const auto fc1_site =
        site("fc1", fc1_weight, fc1_bias, hidden, 2U * config.mlp_width);
    const auto fc2_site =
        site("fc2", fc2_weight, fc2_bias, config.mlp_width, hidden);

    const auto modulated1 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::RmsNormModulate,
                  {x, norm1_weight, modulation.scale1, modulation.shift1},
                  {modulated1}, {norm_epsilon});
    const auto q =
        lora_linear(modulated1, q_site,
                    {sequence, config.heads, config.head_dim}, hidden);
    const auto k =
        lora_linear(modulated1, k_site,
                    {sequence, config.heads, config.head_dim}, hidden);
    const auto v =
        lora_linear(modulated1, v_site,
                    {sequence, config.heads, config.head_dim}, hidden);
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
        lora_linear(attended, out_site, {sequence, hidden}, hidden);
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
    const auto expanded =
        lora_linear(modulated2, fc1_site,
                    {sequence, 2U * config.mlp_width}, 2U * config.mlp_width);
    const auto activated = add_tensor(ir::TensorRole::Internal,
                                      {sequence, config.mlp_width});
    add_operation(ir::Opcode::SwiGlu, {expanded}, {activated},
                  {ir::Attribute::boolean(ir::AttrKey::GateFirst, true)});
    const auto contracted =
        lora_linear(activated, fc2_site, {sequence, hidden}, hidden);
    const auto gated2 =
        add_tensor(ir::TensorRole::Internal, {sequence, hidden});
    add_operation(ir::Opcode::ResidualGate,
                  {gated1, contracted, modulation.gate2}, {gated2});
    x = gated2;
  }

  for (auto &tensor : program.tensors)
    if (tensor.id == x)
      tensor.roles |= ir::TensorRole::Output;
  build.prediction_output = x;
  // BF16 prediction feeds MseLoss directly; the loss is F32 [1] per the
  // mixed-precision training semantics.
  build.loss_output =
      add_typed(ir::DType::F32, ir::TensorRole::Output, {1U});
  add_operation(ir::Opcode::MseLoss, {x, build.target_input},
                {build.loss_output});

  // Forward, backward and the AdamW update compose into ONE program with one
  // tensor namespace, so the memory planner sees the activations, the
  // gradients and the optimizer state at the same time.
  const training::OptimizerHyperparameters hyperparameters{
      config.learning_rate, config.beta1, config.beta2, config.epsilon_adam,
      config.weight_decay};
  auto step = training::build_training_step(program, build.loss_output,
                                            adapter_parameters, hyperparameters);
  build.program = std::move(step.program);
  build.step_input = step.step_input;
  // The bindings are difcore's own type now, so the master-weight fields
  // travel with them instead of being dropped by a field-by-field copy.
  build.optimizer_bindings = std::move(step.bindings);
  ir::verify(build.program);
  // Flame lesson (FLUX_SKEPTIC: 152 adapters shipped against 418 canonical,
  // a 64% miss): the adapter set is a declarative contract, asserted here.
  constexpr std::size_t sites_per_block = 6U;
  if (build.adapters.size() != config.blocks * sites_per_block ||
      build.optimizer_bindings.size() != build.adapters.size() * 2U)
    fail("DiT LoRA builder produced " + std::to_string(build.adapters.size()) +
         " adapters and " + std::to_string(build.optimizer_bindings.size()) +
         " optimizer bindings; the canonical contract is " +
         std::to_string(config.blocks * sites_per_block) + " adapters (" +
         std::to_string(sites_per_block) + " sites x " +
         std::to_string(config.blocks) + " blocks) and two bindings each");
  return build;
}

} // namespace dif::frontend
