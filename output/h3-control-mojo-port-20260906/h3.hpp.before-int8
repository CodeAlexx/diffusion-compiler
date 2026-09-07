#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>

namespace dif::frontend {

struct H3DenoiserConfig {
  std::uint64_t video_tokens{};
  std::uint64_t audio_tokens{};
  std::uint64_t text_tokens{};
  std::uint64_t timestep_tables{};
  std::uint64_t hidden{5376};
  std::uint64_t heads{56};
  std::uint64_t head_dim{128};
  std::uint64_t ffn{14336};
  std::uint64_t rotary{96};
  std::uint64_t layers{50};
  std::uint64_t refiner_layers{2};
  std::uint64_t video_input_dim{96};
  std::uint64_t audio_input_dim{32};
  std::uint64_t text_input_dim{5120};
  std::uint64_t time_input_dim{256};
  std::uint64_t time_hidden_dim{5376};
  std::uint64_t time_embed_dim{2688};
  std::uint64_t block_size{256};
  std::uint64_t attention_implementation{2};
  bool streamed_constants{};
};

// Builds the complete mixed-precision H3 denoiser from raw timesteps and packed
// three-axis position IDs. Generic DiffIR preprocessing produces the sinusoidal
// timestep features and rotary cosine/sine tables on the selected backend.
ir::Program make_h3_denoiser(const H3DenoiserConfig &config);

ir::Program make_h3_stack_bf16(std::uint64_t sequence,
                               std::uint64_t hidden,
                               std::uint64_t heads,
                               std::uint64_t head_dim,
                               std::uint64_t ffn,
                               std::uint64_t rotary,
                               std::uint64_t layers,
                               std::uint64_t block_size,
                               bool streamed_constants);

ir::Program make_h3_transformer_bf16(std::uint64_t sequence,
                                     std::uint64_t hidden,
                                     std::uint64_t heads,
                                     std::uint64_t head_dim,
                                     std::uint64_t ffn,
                                     std::uint64_t rotary,
                                     std::uint64_t layers,
                                     std::uint64_t timestep_tables,
                                     std::uint64_t time_embed_dim,
                                     std::uint64_t block_size,
                                     bool streamed_constants,
                                     bool source_shaped_qkv = true,
                                     std::uint64_t attention_implementation = 1);

ir::Program make_h3_token_refiner_bf16(std::uint64_t sequence,
                                       std::uint64_t hidden,
                                       std::uint64_t heads,
                                       std::uint64_t head_dim,
                                       std::uint64_t ffn,
                                       std::uint64_t layers,
                                       std::uint64_t block_size,
                                       bool streamed_constants);

ir::Program make_h3_block_raw_bf16(std::uint64_t sequence,
                                   std::uint64_t hidden,
                                   std::uint64_t heads,
                                   std::uint64_t head_dim,
                                   std::uint64_t ffn,
                                   std::uint64_t rotary,
                                   std::uint64_t block_size,
                                   bool streamed_constants);

} // namespace dif::frontend
