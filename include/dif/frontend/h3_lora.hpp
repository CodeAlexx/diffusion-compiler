#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/weights/bundle.hpp"
#include <filesystem>
#include <vector>

namespace dif::frontend {

struct H3LoraRequest {
  std::filesystem::path path;
  float multiplier{1.0F};
};

struct H3LoraResult {
  ir::Program program;
  runtime::TensorMap constants;
  std::size_t adapters{};
  std::size_t projections{};
};

// The sealed base bundle names establish model ownership. QKV adapter B is
// logical [Q;K;V] even though the base checkpoint is head-interleaved. FC1 is
// [gate;value] here, unlike Mojo's swapped runtime representation.
H3LoraResult add_h3_lora(const ir::Program &program,
                         const weights::WeightBundle &base_bundle,
                         const std::vector<H3LoraRequest> &requests);

// Bind the existing native H256 cache to base Linear operation ids, never to
// inserted adapter or ControlNet linears. This compatibility route deliberately
// uses generic projections rather than the overlay-unaware fused H3 MLP.
std::vector<runtime::RunOptions::ConvRotLinearBinding>
h3_lora_convrot_bindings(const ir::Program &base_program,
                         const weights::WeightBundle &base_bundle,
                         const std::filesystem::path &cache,
                         std::uint32_t resident_layers,
                         std::uint32_t attention_layers,
                         std::uint32_t mlp_layers);

} // namespace dif::frontend
