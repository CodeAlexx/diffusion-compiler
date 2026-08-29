#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

struct H3VideoVaeConfig {
  std::uint64_t latent_frames{};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t layers{36};
  std::uint64_t latent_channels{24};
  std::uint64_t output_channels{3};
  std::uint64_t heads{32};
  std::uint64_t head_dim{64};
  std::uint64_t ffn{8192};
  std::uint64_t register_tokens{4};
  std::uint64_t patch_t{4};
  std::uint64_t patch_h{16};
  std::uint64_t patch_w{16};
  std::uint64_t rotary_dim{48};
  std::uint64_t block_size{256};
  std::uint64_t attention_implementation{2};
  double rope_theta{100.0};
  bool streamed_constants{};
};

struct H3VideoVaeBinding {
  std::uint32_t tensor_id{};
  std::string name;
  // Empty for compiler-generated geometry/normalization constants.
  std::string source_name;
};

struct H3VideoVaeBuild {
  ir::Program program;
  std::vector<H3VideoVaeBinding> bindings;
  runtime::TensorMap generated_constants;
  std::uint32_t raw_output_id{};
  std::uint32_t decoded_output_id{};
};

// Released MiniMax-H3 non-causal video ViT decoder, including latent
// denormalization, post_quant_conv, the learned decoder, ImageNet pixel
// denormalization, and [0,1] clamp. Batch one is deliberate: DiffIR Attention
// represents one attention document per operation.
H3VideoVaeBuild make_h3_video_vae_decoder(const H3VideoVaeConfig &config);

} // namespace dif::frontend
