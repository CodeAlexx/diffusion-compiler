#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// Released Krea 2 Raw/Turbo MMDiT architecture.  Runtime geometry is
// configurable, but checkpoint dimensions are deliberately fixed to the
// creator model instead of exposing knobs that could describe a wrong graph.
struct Krea2Config {
  static constexpr std::uint64_t kFeatures = 6144U;
  static constexpr std::uint64_t kTimestepDim = 256U;
  static constexpr std::uint64_t kTextDim = 2560U;
  static constexpr std::uint64_t kHeads = 48U;
  static constexpr std::uint64_t kKvHeads = 12U;
  static constexpr std::uint64_t kHeadDim = 128U;
  static constexpr std::uint64_t kMlpDim = 16384U;
  static constexpr std::uint64_t kLayers = 28U;
  static constexpr std::uint64_t kPatch = 2U;
  static constexpr std::uint64_t kLatentChannels = 16U;
  static constexpr std::uint64_t kTextLayers = 12U;
  static constexpr std::uint64_t kTextHeads = 20U;
  static constexpr std::uint64_t kTextKvHeads = 20U;
  static constexpr std::uint64_t kVaeCompression = 8U;
  static constexpr std::uint64_t kSequenceAlignment = 256U;

  std::uint64_t batch{1U};
  std::uint64_t width{1024U};
  std::uint64_t height{1024U};
  std::uint64_t text_tokens{512U};
  bool streamed_constants{true};
};

struct Krea2Architecture {
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t image_grid_height{};
  std::uint64_t image_grid_width{};
  std::uint64_t image_tokens{};
  std::uint64_t combined_tokens{};
  std::uint64_t padded_tokens{};
  std::uint64_t patch_input_dim{};
  std::uint64_t patch_output_dim{};
};

Krea2Architecture inspect_krea2_architecture(const Krea2Config &config);

// Source-faithful real-dimension scaffold for mmdit.py:388-389:
// BF16 t -> F32 -> temb(256, period=1e4, tfactor=1e3, cos then sin) -> BF16
// -> Linear(256,6144)+bias -> tanh GELU -> Linear(6144,6144)+bias
// -> tanh GELU -> Linear(6144,36864)+bias.
//
// This is an admitted frontend slice, not a claim that the complete Krea 2
// block is expressible yet.  Masked attention and creator-style three-axis
// interleaved RoPE remain explicit full-block dependencies.
struct Krea2TimeConditioningBuild {
  ir::Program program;
  Krea2Config config;
  std::uint32_t timestep_input{};
  std::uint32_t timestep_embedding{};
  std::uint32_t timestep_output{};
  std::uint32_t modulation_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
};

Krea2TimeConditioningBuild
make_krea2_time_conditioning(const Krea2Config &config = {});

} // namespace dif::frontend
