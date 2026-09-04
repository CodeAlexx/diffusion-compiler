#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

// How one checkpoint tensor maps onto a program constant. The single-file
// SDXL checkpoint keeps CLIP-L in the HuggingFace layout and OpenCLIP-G in
// the OpenCLIP layout (fused in_proj, text_projection stored as [in,out]);
// the binder applies the transform so the program sees one topology.
enum class ClipWeightTransform : std::uint32_t {
  Direct = 1,
  // Rows [index*C, (index+1)*C) of a fused [3C, C] / [3C] in_proj tensor.
  FusedRowsQ = 2,
  FusedRowsK = 3,
  FusedRowsV = 4,
  // [K,N] stored for x @ W; the Linear weight is its transpose [N,K].
  Transpose = 5,
};

struct ClipWeightBinding {
  std::uint32_t tensor{};
  std::string source_name;
  ClipWeightTransform transform{ClipWeightTransform::Direct};
};

enum class ClipCheckpointLayout : std::uint32_t { HuggingFace = 1, OpenClip = 2 };

struct ClipTextTowerConfig {
  std::uint64_t hidden_size{768};
  std::uint64_t layers{12};
  std::uint64_t heads{12};
  std::uint64_t intermediate_size{3072};
  std::uint64_t vocabulary{49408};
  std::uint64_t positions{77};
  ir::GeluApproximation activation{ir::GeluApproximation::QuickSigmoid};
  double layer_norm_epsilon{1.0e-5};
  // Layers executed. The hidden-state tap follows the reference's
  // hidden_states[k] convention: the raw residual after `hidden_layers`
  // layers (SDXL reads the penultimate state: layers - 1), without the
  // final layer norm.
  std::uint64_t executed_layers{11};
  std::uint64_t hidden_layers{11};
  // Pooled output: final layer norm of the fully executed stack, the row at
  // the end-of-text position, then the text projection (OpenCLIP-G).
  bool pooled_output{false};
  ClipCheckpointLayout layout{ClipCheckpointLayout::HuggingFace};
  std::string checkpoint_prefix{
      "conditioner.embedders.0.transformer.text_model."};
  // The reference runs the towers in F32 (its manual-cast ops promote the
  // F16 weights to the F32 embeddings). F32 is the parity form.
  ir::DType dtype{ir::DType::F32};
  // 1 = generated exact attention (F32, unbatched); 2 = cuDNN (bf16/f16).
  std::uint64_t attention_implementation{1};
  bool capture_boundaries{true};
};

struct ClipTextTowerBuild {
  ir::Program program;
  ClipTextTowerConfig config;
  std::vector<ClipWeightBinding> weights;
  // I32 [positions]: the token ids of one 77-token prompt chunk.
  std::uint32_t token_ids_input{};
  // I32 [1]: the end-of-text row the pooled output reads (valid tokens - 1).
  std::uint32_t pooled_row_input{};
  // [positions, hidden] raw residual after `hidden_layers` layers.
  std::uint32_t hidden_output{};
  // [1, hidden] projected pooled vector (zero when not requested).
  std::uint32_t pooled_output{};
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// The reference CLIP text model (comfy/clip_model.py CLIPTextModel_):
// token embedding + position embedding, then per layer LayerNorm ->
// q/k/v Linear -> causal Attention -> out Linear -> Add -> LayerNorm ->
// fc1 -> activation -> fc2 -> Add; final LayerNorm and text projection on
// the EOS row for the pooled vector.
ClipTextTowerBuild make_clip_text_tower(const ClipTextTowerConfig &config);

// SDXL base 1.0: CLIP-L (12 layers, quick-GELU, penultimate hidden state)
// and OpenCLIP-G (32 layers, exact GELU, penultimate hidden state, pooled
// projection from the end-of-text row).
ClipTextTowerConfig sdxl_clip_l_config();
ClipTextTowerConfig sdxl_clip_g_config();

} // namespace dif::frontend
