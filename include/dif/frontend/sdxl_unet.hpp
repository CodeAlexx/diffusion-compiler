#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

// One checkpoint tensor the UNet consumes. `source_name` is the literal
// single-file SDXL checkpoint key (model.diffusion_model.*). The checkpoint
// stores F16; the F16 program maps it without conversion.
struct SdxlUnetWeightBinding {
  std::uint32_t tensor{};
  std::string source_name;
};

struct SdxlUnetConfig {
  // Rows of the denoiser batch: 2 for classifier-free guidance (cond and
  // uncond in one pass, the reference's calc_cond_batch form).
  std::uint64_t batch{2};
  std::uint64_t latent_height{128};
  std::uint64_t latent_width{128};
  // 77 tokens per encoder chunk; the reference concatenates the CLIP-L and
  // OpenCLIP-G penultimate states to a 2048-wide context.
  std::uint64_t context_tokens{77};
  // The reference runs the SDXL UNet in F16 (its weight dtype) with F32
  // accumulation inside cuBLAS/cuDNN; that is the parity form.
  ir::DType dtype{ir::DType::F16};
  bool streamed_constants{};
  bool capture_boundaries{true};
  std::string checkpoint_prefix{"model.diffusion_model."};
  // 2 = cuDNN SDPA (the production attention for bf16/f16).
  std::uint64_t attention_implementation{2};
};

struct SdxlUnetBuild {
  ir::Program program;
  SdxlUnetConfig config;
  std::vector<SdxlUnetWeightBinding> weights;
  // [B,4,h,w] in the program dtype: the reference's calculate_input result
  // (x / sqrt(sigma^2 + 1)).
  std::uint32_t latent_input{};
  // F32 [B]: the integer-valued timesteps the reference feeds the UNet.
  std::uint32_t timestep_input{};
  // [B, tokens, 2048] cross-attention context.
  std::uint32_t context_input{};
  // [B, 2816] pooled text + size conditioning vector (encode_adm).
  std::uint32_t vector_input{};
  // [B,4,h,w] epsilon prediction.
  std::uint32_t output{};
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// SDXL base 1.0 UNet (the reference's UNetModel with model_channels 320,
// channel_mult (1,2,4), two res blocks per level, transformer depths
// (0,0,2,2,10,10), context 2048, adm 2816, linear projections): time and
// label embeddings, conv_in, the input/middle/output block stacks with skip
// concatenation, and the GroupNorm + SiLU + conv out head. Every learned
// operation runs through shared opcodes.
SdxlUnetBuild make_sdxl_unet(const SdxlUnetConfig &config = {});

} // namespace dif::frontend
