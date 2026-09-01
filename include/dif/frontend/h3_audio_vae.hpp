#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// Released MiniMax-H3 BigVGAN audio decoder configuration, sourced from
// audio_vae/metadata.json and config.yaml.
struct AudioBigVganConfig {
  std::uint64_t latent_channels{32};
  std::uint64_t latent_dim{2048};
  std::uint64_t decoder_dim{1024};
  std::vector<std::uint64_t> upsample_rates{5, 5, 2, 2, 2, 2, 2};
  std::vector<std::uint64_t> upsample_kernels{9, 9, 4, 4, 4, 4, 4};
  std::vector<std::uint64_t> resblock_kernels{3, 7, 11};
  std::vector<std::uint64_t> resblock_dilations{1, 3, 5};
  std::uint64_t resample_kernel{12};
  std::uint64_t resample_ratio{2};
};

struct AudioBigVganBinding {
  std::uint32_t tensor_id{};
  // Folded-checkpoint tensor name (difimport fold-audio-weight-norm output);
  // empty source_name marks a compiler-generated constant (latent denorm).
  std::string name;
  std::string source_name;
};

struct AudioBigVganBuild {
  ir::Program program;
  std::vector<AudioBigVganBinding> bindings;
  runtime::TensorMap generated_constants;
  std::uint32_t latent_input_id{};
  std::uint32_t waveform_output_id{};
  std::uint64_t conv1d_operations{};
  std::uint64_t snake_beta_operations{};
};

// One self-contained DiffIR program: [B, 32, T] state-space latents in
// (denormalization is IN-PROGRAM as a depthwise K=1 conv over the baked
// latents_mean/latents_std, integrator decision 4), waveform [B, 1, 800*T]
// out, clamped to [-1, 1]. `stages` truncates after the given number of
// upsample stages for parity isolation: 0 = after conv_pre ("pre"),
// 1..7 = after that stage's block average, 8 (or >= rates+1) = the full
// decoder including activation_post/conv_post/clamp.
AudioBigVganBuild build_audio_bigvgan_program(
    std::uint64_t batch, std::uint64_t latent_frames, std::uint64_t stages,
    const AudioBigVganConfig &config = {});

} // namespace dif::frontend
