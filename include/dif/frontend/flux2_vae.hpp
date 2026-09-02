#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {

enum class Flux2VaeWeightTransform : std::uint32_t {
  Direct = 1,
  BatchNormStandardDeviation = 2,
};

struct Flux2VaeWeightBinding {
  std::uint32_t tensor_id{};
  std::string name;
  Flux2VaeWeightTransform transform{Flux2VaeWeightTransform::Direct};
};

struct Flux2VaeConfig {
  std::uint64_t latent_height{64U};
  std::uint64_t latent_width{64U};
  bool streamed_constants{false};
  bool capture_boundaries{false};
};

struct Flux2VaeBuild {
  ir::Program program;
  Flux2VaeConfig config;
  std::uint32_t latent_tokens_input{};
  std::uint32_t raw_output{};
  std::uint32_t clamped_output{};
  std::vector<Flux2VaeWeightBinding> weights;
  std::vector<std::pair<std::string, std::uint32_t>> boundaries;
};

// Creator FLUX.2 autoencoder decoder expressed only with shared DiffIR math.
Flux2VaeBuild make_flux2_vae_decoder(const Flux2VaeConfig &config = {});

} // namespace dif::frontend
