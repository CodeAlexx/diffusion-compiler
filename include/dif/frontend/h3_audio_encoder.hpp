#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/json.hpp"
#include "dif/weights/safetensors.hpp"

#include <string>
#include <vector>

namespace dif::frontend {

// Translated from serenitymojo/models/minimax_h3_device/audio_encoder_device.mojo.
// Architecture values come from the checkpoint's JSON, not machine defaults.
struct H3AudioEncoderConfig {
  std::uint64_t encoder_dim{};
  std::vector<std::uint64_t> encoder_rates;
  std::uint64_t latent_dim{}, latent_channels{}, num_attention_heads{};
  double eps{};
};
H3AudioEncoderConfig h3_audio_encoder_config(const json::Value &document);

struct H3AudioEncoderBinding {
  std::uint32_t tensor_id{};
  std::string name;
  bool weight_normalized{};
};
struct H3AudioEncoderBuild {
  ir::Program program;
  std::vector<H3AudioEncoderBinding> bindings;
  runtime::TensorMap generated_constants;
  std::uint32_t waveform_input_id{}, trunk_output_id{}, preblock_output_id{},
      mean_output_id{};
};

// F32 mono waveform [B,1,S] -> F32 posterior mean [B,C,ceil(S/hop)].
// Each stereo channel is an independent batch item. Preserves the native
// port's BF16 Q/K/V -> causal cuDNN SDPA -> F32 attention boundary.
H3AudioEncoderBuild build_h3_audio_encoder_program(
    std::uint64_t batch, std::uint64_t samples,
    const H3AudioEncoderConfig &config);

// Maps only encoder/pre_block/mean_proj and folds DAC weight norm once.
// Decoder weights and the unused stochastic-posterior head are not loaded.
runtime::TensorMap bind_h3_audio_encoder_weights(
    const H3AudioEncoderBuild &build, const weights::SafeTensorFile &checkpoint);

} // namespace dif::frontend
