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
// config.json (docs/QWEN3VL_CONDITIONER_PLAN.md §1), not from documentation.
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
  std::uint32_t conditioning_output_id{};
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
