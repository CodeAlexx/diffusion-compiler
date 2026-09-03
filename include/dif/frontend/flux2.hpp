#pragma once

#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

// FLUX.2 [klein] 9B creator Qwen3-8B observable conditioner. The creator
// requests all hidden states and consumes raw states 9, 18, and 27. Layers
// after 27 are dead with respect to that observable output and are omitted by
// the frontend without changing semantics.
Qwen3VlConditionerConfig
make_flux2_klein_9b_conditioner_config(std::uint64_t executed_layers = 27U);
// FLUX.2 [dev] conditioner: the Mistral Small 3.1 24B language tower
// (40 layers, hidden 5120, 32/8 heads, head_dim 128, MLP 32768, vocab
// 131072, eps 1e-5, rope theta 1e9, no QK-norm, keys under
// language_model.model.). The pipeline takes hidden states 10, 20 and 30
// (outputs after layers 9, 19, 29) concatenated to 15360, so 30 executed
// layers suffice. Prompt = system message + user prompt through the Mistral
// chat template, 512 tokens.
Qwen3VlConditionerConfig
make_flux2_dev_conditioner_config(std::uint64_t executed_layers = 30U);

// FLUX.2 transformer geometry. Defaults are FLUX.2 [klein] Base 9B, which
// keeps every existing Klein program byte-identical; flux2_dev_geometry()
// selects FLUX.2 [dev] (32B: hidden 6144, 48 heads, MLP 18432, 8 + 48
// blocks, Mistral context width 15360, guidance embedding added to the
// timestep vector). Verified against the checkpoint headers 2026-09-03.
struct Flux2Geometry {
  std::uint64_t hidden{4096U};
  std::uint64_t heads{32U};
  std::uint64_t head_dim{128U};
  std::uint64_t mlp{12288U};
  std::uint64_t double_depth{8U};
  std::uint64_t single_depth{24U};
  std::uint64_t context_width{12288U};
  bool guidance_embedding{false};
};
Flux2Geometry flux2_klein_9b_geometry();
Flux2Geometry flux2_dev_geometry();

struct Flux2KleinDoubleBlockConfig {
  Flux2Geometry geometry{};
  std::uint64_t batch_size{1U};
  std::uint64_t image_tokens{4096U};
  std::uint64_t text_tokens{512U};
  std::uint64_t block_index{};
  std::uint64_t attention_implementation{2U};
  bool capture_boundaries{true};
};

struct Flux2KleinDoubleBlockBuild {
  ir::Program program;
  runtime::TensorMap generated_constants;
  std::uint32_t image_input{};
  std::uint32_t text_input{};
  std::uint32_t position_ids_input{};
  std::uint32_t image_modulation_input{};
  std::uint32_t text_modulation_input{};
  std::uint32_t image_output{};
  std::uint32_t text_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// One creator-faithful real-width double-stream block. Model identity and
// checkpoint names stay here; every operation is a shared DiffIR semantic.
Flux2KleinDoubleBlockBuild make_flux2_klein_9b_double_block(
    const Flux2KleinDoubleBlockConfig &config = {});

struct Flux2KleinSingleBlockConfig {
  Flux2Geometry geometry{};
  std::uint64_t batch_size{1U};
  std::uint64_t tokens{4608U};
  std::uint64_t block_index{};
  std::uint64_t attention_implementation{2U};
  bool capture_boundaries{true};
};

struct Flux2KleinSingleBlockBuild {
  ir::Program program;
  runtime::TensorMap generated_constants;
  std::uint32_t sequence_input{};
  std::uint32_t position_ids_input{};
  std::uint32_t modulation_input{};
  std::uint32_t sequence_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

Flux2KleinSingleBlockBuild make_flux2_klein_9b_single_block(
    const Flux2KleinSingleBlockConfig &config = {});

struct Flux2KleinTransformerConfig {
  Flux2Geometry geometry{};
  std::uint64_t batch_size{1U};
  std::uint64_t image_tokens{4096U};
  std::uint64_t text_tokens{512U};
  std::uint64_t double_depth{8U};
  std::uint64_t single_depth{24U};
  std::uint64_t attention_implementation{2U};
  bool streamed_constants{true};
  bool capture_depth_boundaries{false};
  bool capture_first_block_boundaries{false};
};

struct Flux2KleinTransformerBuild {
  ir::Program program;
  runtime::TensorMap generated_constants;
  std::uint32_t latent_input{};
  std::uint32_t conditioning_input{};
  std::uint32_t timestep_input{};
  // Nonzero only for a geometry with guidance_embedding: BF16 [batch]
  // guidance value embedded like the timestep and added to the vector.
  std::uint32_t guidance_input{};
  std::uint32_t position_ids_input{};
  std::uint32_t prediction_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// One shared DiffIR for the complete FLUX.2 transformer. Partial depths are
// parity ladders; the production shape is 8 double + 24 single blocks.
Flux2KleinTransformerBuild make_flux2_klein_9b_transformer(
    const Flux2KleinTransformerConfig &config = {});

// Base-model classifier-free guidance followed by the creator's eager BF16
// Euler update.  The policy is FLUX.2 frontend semantics; its implementation
// is composed entirely from shared DiffIR elementwise operations plus the
// generic EulerVelocityStep semantic.
struct Flux2KleinCfgEulerBuild {
  ir::Program program;
  std::uint32_t sample_input{};
  std::uint32_t conditional_velocity_input{};
  std::uint32_t unconditional_velocity_input{};
  std::uint32_t guidance_input{};
  std::uint32_t current_timestep_input{};
  std::uint32_t next_timestep_input{};
  std::uint32_t negative_one_constant{};
  std::uint32_t guided_velocity_output{};
  std::uint32_t sample_output{};
};

Flux2KleinCfgEulerBuild make_flux2_klein_base_cfg_euler_step(
    std::vector<std::uint64_t> sample_shape = {4096U, 128U});

} // namespace dif::frontend
