#pragma once

#include "dif/ir/ir.hpp"
#include "dif/training/step.hpp"

#include <cstdint>
#include <vector>

namespace dif::frontend {

struct MlpTrainingConfig {
  std::uint64_t rows{32U};
  std::uint64_t input_width{8U};
  std::uint64_t hidden_width{16U};
  std::uint64_t output_width{4U};
  double learning_rate{1.0e-2};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon{1.0e-8};
  double weight_decay{1.0e-2};
  // Storage dtype for parameters and activations.  F32 keeps the historical
  // graph byte-for-byte (fingerprint-stable).  BF16 builds the flame-style
  // mixed-precision graph: BF16 storage crossing a Cast boundary into an
  // F32 loss, BF16 gradients (F32 accumulation inside kernels), and F32
  // optimizer moments.
  ir::DType compute_dtype{ir::DType::F32};
};

// The binding difcore already defines. It used to be re-declared here with
// the master-weight fields dropped, so every frontend copied it field by
// field out of the training step and quietly lost them.
using OptimizerBinding = training::ParameterBinding;

struct MlpTrainingBuild {
  ir::Program program;
  std::uint32_t features_input{};
  std::uint32_t target_input{};
  std::uint32_t step_input{};
  std::uint32_t prediction_output{};
  std::uint32_t loss_output{};
  std::vector<OptimizerBinding> optimizer_bindings;
};

MlpTrainingBuild make_mlp_training(const MlpTrainingConfig &config);

struct RectifiedFlowTrainingConfig {
  std::uint64_t rows{16U};
  std::uint64_t latent_width{8U};
  std::uint64_t timestep_width{4U};
  std::uint64_t hidden_width{16U};
  std::uint64_t accumulation_steps{2U};
  double learning_rate{5.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon{1.0e-8};
  double weight_decay{1.0e-2};
};

struct RectifiedFlowMicrobatch {
  std::uint32_t clean_input{};
  std::uint32_t noise_input{};
  std::uint32_t clean_scale_input{};
  std::uint32_t noise_scale_input{};
  std::uint32_t timestep_features_input{};
  std::uint32_t target_velocity_input{};
  std::uint32_t prediction_output{};
  std::uint32_t loss_tensor{};
};

struct RectifiedFlowTrainingBuild {
  ir::Program program;
  std::vector<RectifiedFlowMicrobatch> microbatches;
  std::uint32_t loss_scale_tensor{};
  std::uint32_t step_input{};
  std::uint32_t loss_output{};
  std::vector<OptimizerBinding> optimizer_bindings;
};

RectifiedFlowTrainingBuild
make_rectified_flow_training(const RectifiedFlowTrainingConfig &config);

} // namespace dif::frontend
