#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

enum class Krea2VaeWeightTransform : std::uint32_t {
  Direct = 1,
  FlattenSingletonDimensions = 2,
  Conv3dLastTemporalSlice = 3,
};

struct Krea2VaeWeightBinding {
  std::uint32_t tensor{};
  std::string source_name;
  Krea2VaeWeightTransform transform{Krea2VaeWeightTransform::Direct};
};

struct Krea2VaeConfig {
  std::uint64_t batch{1};
  std::uint64_t latent_height{32};
  std::uint64_t latent_width{32};
  bool streamed_constants{};
  bool capture_boundaries{true};
};

struct Krea2VaeBuild {
  ir::Program program;
  Krea2VaeConfig config;
  std::uint32_t latent_input{};
  std::uint32_t latent_std{};
  std::uint32_t latent_mean{};
  std::uint32_t raw_output{};
  std::uint32_t clamped_output{};
  std::vector<Krea2VaeWeightBinding> weights;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// Build one source-faithful Qwen-Image decoder tile. Spatial tiling and blend
// policy remain in the Krea frontend/tool; every learned operation executes
// through shared DiffIR and the common runtime.
Krea2VaeBuild make_krea2_qwen_image_vae(const Krea2VaeConfig &config = {});

} // namespace dif::frontend
