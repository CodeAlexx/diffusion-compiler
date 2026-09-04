#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

// One checkpoint tensor the decoder consumes. `source_name` is the literal
// single-file SDXL checkpoint key (first_stage_model.*); the checkpoint
// stores F16 and the binder converts to the program dtype.
struct SdxlVaeWeightBinding {
  std::uint32_t tensor{};
  std::string source_name;
};

struct SdxlVaeConfig {
  std::uint64_t batch{1};
  std::uint64_t latent_height{128};
  std::uint64_t latent_width{128};
  // The reference sampler decodes SDXL's KL VAE in BF16 on this class of
  // GPU (F16 overflows in its mid block); F32 is the parity form.
  ir::DType dtype{ir::DType::BF16};
  bool streamed_constants{};
  bool capture_boundaries{true};
  std::string checkpoint_prefix{"first_stage_model."};
};

struct SdxlVaeBuild {
  ir::Program program;
  SdxlVaeConfig config;
  // The decoder consumes the UNSCALED latent: the sampler already divided
  // by the latent format's 0.13025 (the reference's process_latent_out).
  std::uint32_t latent_input{};
  // [B,3,8h,8w] in the program dtype, the reference decoder's raw output
  // before its (x + 1) / 2 clamp to [0, 1].
  std::uint32_t raw_output{};
  std::vector<SdxlVaeWeightBinding> weights;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// SDXL base 1.0 KL VAE decoder as one DiffIR program: post_quant_conv,
// conv_in, mid (resnet, attention, resnet), four up levels of three resnets
// each (512, 512, 256, 128 channels) with nearest-2x + conv upsampling
// between them, GroupNorm(32, eps 1e-6) + SiLU + conv_out. Every learned
// operation runs through shared opcodes (Conv2d, GroupNorm, SiLU, Attention,
// UpsampleNearest2d, Add).
SdxlVaeBuild make_sdxl_vae_decoder(const SdxlVaeConfig &config = {});

} // namespace dif::frontend
