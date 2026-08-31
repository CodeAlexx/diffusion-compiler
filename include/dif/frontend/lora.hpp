#pragma once

#include "dif/frontend/training.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/training/checkpoint.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace dif::frontend {

// F32 activation-path LoRA training over the rectified-flow MLP objective.
//
// Contract (flame-core / EriDiffusion-v2 lora.rs, FLAME_PORT_SOURCE_NOTES.md
// section 6):
//   - A is [rank, in_features], Kaiming-uniform init with
//     bound = 1/sqrt(in_features) (torch nn.Linear default).
//   - B is [out_features, rank], zero init.
//   - forward delta = (x @ A^T @ B^T) * (alpha / rank); the low-rank
//     projection stays explicit as two Linear operations — the dense
//     [out, in] delta is never materialized.
//   - The alpha/rank scale lives IN-GRAPH as a fingerprinted `Fill` of the
//     delta's shape multiplied into the delta (`Multiply`), mirroring the
//     rectified-flow builder's fingerprinted loss-scale `Fill`. It is not
//     caller-supplied runtime policy.
//   - Base weights are role `Constant` (frozen, bundle-bindable); adapters
//     A/B are role `Input|Parameter` and are the only differentiation and
//     optimizer targets.
//   - Export writes `<name>.lora_A.weight`, `<name>.lora_B.weight`, AND
//     `<name>.alpha` (F32 [1] scalar; metadata, NOT trainable). A missing
//     .alpha makes external loaders fall back to scale=1.0, which
//     over-applies adapters trained with alpha != rank (flame's 2026-05-27
//     ~16x over-application incident).
struct LoraFlowTrainingConfig {
  std::uint64_t rows{16U};
  std::uint64_t latent_width{8U};
  std::uint64_t timestep_width{4U};
  std::uint64_t hidden_width{16U};
  std::uint64_t rank{4U};
  double alpha{4.0};
  double learning_rate{5.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon{1.0e-8};
  double weight_decay{1.0e-2};
};

// Export/bundle name map entry for one LoRA-augmented Linear. Mirrors the
// H3VideoVaeBinding id<->name pattern.
struct LoraAdapterBinding {
  std::string name;
  std::uint32_t base_weight{};
  std::uint32_t lora_a{};
  std::uint32_t lora_b{};
  std::uint64_t rank{};
  double alpha{};
};

struct LoraConstantBinding {
  std::uint32_t tensor_id{};
  std::string name;
};

struct LoraFlowTrainingBuild {
  ir::Program program;
  std::uint32_t clean_input{};
  std::uint32_t noise_input{};
  std::uint32_t clean_scale_input{};
  std::uint32_t noise_scale_input{};
  std::uint32_t timestep_features_input{};
  std::uint32_t target_velocity_input{};
  // Frozen base weights and biases, role Constant.
  std::vector<LoraConstantBinding> frozen_constants;
  // The three LoRA-augmented Linears, in graph order.
  std::vector<LoraAdapterBinding> adapters;
  // Six bindings: (lora_A, lora_B) per adapter, ascending parameter id.
  std::vector<OptimizerBinding> optimizer_bindings;
  std::uint32_t step_input{};
  std::uint32_t prediction_output{};
  std::uint32_t loss_output{};
};

LoraFlowTrainingBuild
make_lora_flow_training(const LoraFlowTrainingConfig &config);

// Deterministic default adapter initialization: Kaiming-uniform A
// (bound = 1/sqrt(in_features), SplitMix64 stream over adapters in build
// order) and zero B. Returns one tensor per adapter parameter id.
runtime::TensorMap
default_lora_adapter_init(const LoraFlowTrainingBuild &build,
                          std::uint64_t seed);

// Writes trained adapters as SafeTensors: for every adapter,
// `<name>.lora_A.weight` [rank,in], `<name>.lora_B.weight` [out,rank], and
// `<name>.alpha` F32 [1]. Fails closed if the checkpoint's program
// fingerprint does not match the build or any adapter state is missing.
void export_lora_adapters(const LoraFlowTrainingBuild &build,
                          const training::Checkpoint &checkpoint,
                          const std::filesystem::path &path);

// Regression gate for the flame .alpha export lesson: fails unless every
// adapter has lora_A.weight, lora_B.weight, and an F32 single-element
// .alpha whose value matches the build's alpha.
void validate_lora_export(const std::filesystem::path &path,
                          std::span<const LoraAdapterBinding> adapters);

} // namespace dif::frontend
