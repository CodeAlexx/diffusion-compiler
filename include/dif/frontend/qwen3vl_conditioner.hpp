#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// MiniMax-H3's text conditioner: the Qwen3-VL-32B TEXT tower, executed to a
// fixed depth and read as the RAW residual stream (pre-final-norm).
//
// Every field below is taken from the checkpoint's own text_encoder
// config.json, not from secondary documentation.
// `executed_layers` is the extraction rule: transformers' hidden_states[k] is
// the raw state AFTER k layers, so H3's "hidden layer 50" means layers 0..49
// run and `model.norm` is NOT applied.
struct Qwen3VlConditionerConfig {
  std::uint64_t hidden_size{5120};
  std::uint64_t executed_layers{50};
  std::uint64_t attention_heads{64};
  std::uint64_t key_value_heads{8};
  std::uint64_t head_dim{128};
  std::uint64_t intermediate_size{25600};
  std::uint64_t vocabulary{151936};
  double rms_norm_epsilon{1.0e-6};
  double rope_theta{5.0e6};
  // 2 = cuDNN SDPA (BF16, the accepted denoiser's exact-attention class);
  // 1 = the generated exact kernel (admitted for S <= 4096).
  std::uint64_t attention_implementation{2};

  // Optional generic extraction/masking contract used by diffusion
  // conditioners such as Krea 2. Hidden-state index k means the raw residual
  // after k decoder layers, matching transformers output_hidden_states.
  // Empty selection preserves the historical single final-residual output.
  std::vector<std::uint64_t> selected_hidden_states;
  std::uint64_t output_slice_start{0};
  std::uint64_t output_sequence_length{0};
  bool use_attention_mask{false};
  bool dynamic_position_ids{false};
  bool mask_padding_queries{true};

  // Optional multimodal row-splice contract. The vision tower remains a
  // separate prepared DiffIR program; its merged rows replace image-pad token
  // embeddings before layer 0, and its deep-stack rows are added immediately
  // after the listed decoder layers. Zero preserves the text-only program.
  std::uint64_t vision_token_count{0};
  std::vector<std::uint64_t> deepstack_language_layers{0U, 1U, 2U};

  // Literal checkpoint prefix for the shared Qwen decoder topology. Qwen3-VL
  // text towers live under model.language_model.; ordinary Qwen3 CausalLM
  // checkpoints live under model. The model frontend chooses the name while
  // the DiffIR and executor stay shared.
  std::string checkpoint_prefix{"model.language_model."};

  // Optionally concatenate selected raw hidden states across their final
  // dimension inside DiffIR. Individual taps remain outputs for parity.
  bool concatenate_selected_hidden_states{false};

  // Parity-only observability. This changes tensor roles, not computation.
  bool capture_first_layer_boundaries{false};
};

// One checkpoint tensor the program consumes. `name` is the literal
// safetensors key in text_encoder/model.safetensors.index.json, so the
// bundle binder can resolve it without a translation table.
struct Qwen3VlConditionerBinding {
  std::uint32_t tensor_id{};
  std::string name;
};

struct Qwen3VlConditionerBuild {
  ir::Program program;
  // Streamed checkpoint weights, in first-use order (the memory plan's
  // prefetch distance walks this order).
  std::vector<Qwen3VlConditionerBinding> bindings;
  // Compiler-generated constants (rotary positions and inverse frequencies):
  // derived from the config, so the program stays self-contained.
  runtime::TensorMap generated_constants;
  std::uint32_t token_ids_input_id{};
  std::uint32_t attention_mask_input_id{};
  std::uint32_t position_ids_input_id{};
  std::uint32_t vision_embeddings_input_id{};
  std::uint32_t vision_destination_map_input_id{};
  std::uint32_t visual_positions_input_id{};
  std::vector<std::uint32_t> vision_deepstack_input_ids;
  std::uint32_t conditioning_output_id{};
  std::vector<std::uint32_t> conditioning_output_ids;
  std::vector<std::pair<std::string, std::uint32_t>> first_layer_boundaries;
  std::uint64_t linear_operations{};
  std::uint64_t attention_operations{};
};

// Build the conditioner as ONE DiffIR program over generic opcodes:
// GatherRows embedding, RotaryPosition tables, and per layer RmsNorm ->
// Linear q/k/v -> QkNormPartialRope -> grouped-query Attention (KvHeads) ->
// Linear o -> Add -> RmsNorm -> Linear gate/up -> SiLU -> Multiply ->
// Linear down -> Add. No Qwen-specific opcode exists or is needed; the
// model's identity lives entirely in this frontend and its weight names.
Qwen3VlConditionerBuild
build_qwen3vl_conditioner_program(std::uint64_t sequence_length,
                                  const Qwen3VlConditionerConfig &config = {});

} // namespace dif::frontend
