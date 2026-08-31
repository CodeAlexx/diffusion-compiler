#pragma once

#include "dif/frontend/dit_block.hpp"
#include "dif/frontend/lora.hpp"
#include "dif/frontend/training.hpp"
#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// LoRA-augmented DiT transformer block training (M2): the
// make_dit_block_training topology with activation-path LoRA on the block's
// six Linears (q, k, v, out, fc1, fc2), the base model FROZEN, and only the
// adapters trained.
//
// Mixed-precision contract (docs/MIXED_PRECISION_TRAINING_GATE_2026-08-31.md
// + flame lora.rs):
//   - compute_dtype BF16 (default): every base weight, activation,
//     modulation vector, and RoPE table is BF16 storage (F32 register math,
//     one round per stored tensor — note the BF16 cos/sin tables are the
//     same precision floor flame's BF16-RoPE audit recorded).  Adapters A/B
//     are STORED F32 (what AdamW expects) and enter the block through
//     explicit autograd-aware Cast ops, so their gradients land in F32.
//     The MseLoss consumes the BF16 prediction directly and produces the
//     F32 loss; moments are F32 ALWAYS.
//   - compute_dtype F32: the same graph with no Cast boundaries (the
//     debugging/ablation variant).
//   - A [rank,in] (Kaiming-uniform bound 1/sqrt(in) by contract — the gate
//     fixture supplies the values), B [out,rank] zeros, delta scaled by a
//     fingerprinted in-graph Fill(alpha/rank), low-rank path explicit.
//   - Base weights/biases/norm weights are role Constant: no optimizer
//     state, no LinearBackwardWeight (frozen-dW economy), and DiffIR's
//     verifier forbids any operation from writing them.
struct DitLoraTrainingConfig {
  std::uint64_t sequence{16U};
  std::uint64_t heads{2U};
  std::uint64_t head_dim{8U};
  std::uint64_t mlp_width{16U};
  std::uint64_t blocks{2U};
  std::uint64_t rotary_dim{8U};
  bool full_rope_table{true};
  bool causal{false};
  std::uint64_t rank{4U};
  double alpha{8.0};
  ir::DType compute_dtype{ir::DType::BF16};
  double learning_rate{5.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon_adam{1.0e-8};
  double weight_decay{1.0e-2};
  double epsilon_norm{1.0e-5};
};

// Canonical per-block LoRA site order (adapter and fixture naming depend on
// it): q, k, v, out, fc1, fc2 -> 6 sites, 12 adapter parameters per block.
constexpr std::uint64_t kDitLoraSitesPerBlock = 6U;

struct DitLoraTrainingBuild {
  ir::Program program;
  DitLoraTrainingConfig config;
  std::uint32_t x_input{};
  std::uint32_t cos_input{};
  std::uint32_t sin_input{};
  std::uint32_t target_input{};
  std::uint32_t step_input{};
  std::uint32_t prediction_output{};
  std::uint32_t loss_output{};
  std::vector<DitBlockModulationInputs> modulation_inputs;  // per block
  // Frozen base tensors, block-major in make_dit_block_training's canonical
  // 16-parameter order, named "block<b>.<name>"; role Constant.
  std::vector<LoraConstantBinding> frozen_constants;
  // LoRA sites, block-major in the canonical site order, named
  // "block<b>.<site>"; lora_a/lora_b are the F32 Parameter ids.
  std::vector<LoraAdapterBinding> adapters;
  // Two bindings per site (A then B), same order as `adapters` expanded.
  std::vector<OptimizerBinding> optimizer_bindings;
};

DitLoraTrainingBuild
make_dit_lora_training(const DitLoraTrainingConfig &config);

} // namespace dif::frontend
