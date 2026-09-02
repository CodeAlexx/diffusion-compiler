#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

struct H3VideoEncoderConfig {
  std::uint64_t batch{1U};
  std::uint64_t frames{1U};
  std::uint64_t height{256U};
  std::uint64_t width{256U};
  bool streamed_constants{};
  bool capture_boundaries{true};
};

struct H3VideoEncoderBinding {
  std::uint32_t tensor{};
  std::string source_name;
};

struct H3VideoEncoderBuild {
  ir::Program program;
  H3VideoEncoderConfig config;
  std::uint32_t pixels_input{};
  std::uint32_t moments_output{};
  std::vector<H3VideoEncoderBinding> weights;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// Build the released MiniMax-H3 causal 3D CNN encoder plus quant_conv for one
// spatial tile. Image/video tiling, posterior sampling, latent normalization,
// and patchification remain H3 frontend orchestration; every learned operation
// executes through shared DiffIR and the common runtime.
H3VideoEncoderBuild
make_h3_video_encoder(const H3VideoEncoderConfig &config = {});

} // namespace dif::frontend
