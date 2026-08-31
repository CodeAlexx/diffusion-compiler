#pragma once

#include "dif/frontend/training.hpp"
#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// A DiT-style transformer block training graph over the new backward opcode
// set: per block
//   RmsNormModulate -> {q,k,v} Linear(+bias) -> QkNormPartialRope on q,k ->
//   Attention -> out Linear(+bias) -> ResidualGate ->
//   RmsNormModulate -> fc1 Linear(+bias) -> SwiGlu(GateFirst) ->
//   fc2 Linear(+bias) -> ResidualGate
// stacked `blocks` times, closed by MseLoss against a target, differentiated
// wrt every parameter, and driven by per-parameter AdamWUpdate ops (the
// established MLP/LoRA training-graph shape).  Modulation vectors
// (scale/shift/gate per sub-block) and the RoPE tables are non-trainable
// data inputs; DiffIR's ResidualGate gate is per-token full shape.
struct DitBlockTrainingConfig {
  std::uint64_t sequence{16U};
  std::uint64_t heads{2U};
  std::uint64_t head_dim{8U};
  std::uint64_t mlp_width{16U};
  std::uint64_t blocks{1U};
  std::uint64_t rotary_dim{8U};   // even, <= head_dim
  bool full_rope_table{true};     // cos/sin [S, rotary] vs [S, rotary/2]
  bool causal{false};
  double learning_rate{5.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon_adam{1.0e-8};
  double weight_decay{1.0e-2};
  double epsilon_norm{1.0e-5};
};

// Canonical per-block parameter order (fixture naming depends on it):
// norm1_w, q_w, q_b, k_w, k_b, v_w, v_b, q_norm_w, k_norm_w, out_w, out_b,
// norm2_w, fc1_w, fc1_b, fc2_w, fc2_b  -> 16 parameters per block.
constexpr std::uint64_t kDitBlockParameterCount = 16U;

struct DitBlockModulationInputs {
  std::uint32_t scale1{};
  std::uint32_t shift1{};
  std::uint32_t gate1{};
  std::uint32_t scale2{};
  std::uint32_t shift2{};
  std::uint32_t gate2{};
};

struct DitBlockTrainingBuild {
  ir::Program program;
  DitBlockTrainingConfig config;
  std::uint32_t x_input{};
  std::uint32_t cos_input{};
  std::uint32_t sin_input{};
  std::uint32_t target_input{};
  std::uint32_t step_input{};
  std::uint32_t prediction_output{};
  std::uint32_t loss_output{};
  std::vector<DitBlockModulationInputs> modulation_inputs;  // per block
  // Parameter tensor ids, block-major in the canonical order above.
  std::vector<std::uint32_t> parameters;
  // Parameter names, same order, e.g. "block0.q_w".
  std::vector<std::string> parameter_names;
  // One binding per parameter, same order.
  std::vector<OptimizerBinding> optimizer_bindings;
};

DitBlockTrainingBuild
make_dit_block_training(const DitBlockTrainingConfig &config);

} // namespace dif::frontend
