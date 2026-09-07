#pragma once

#include "dif/frontend/h3.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/weights/safetensors.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace dif::frontend {

// Frontend semantics of Alibaba's released dense Union adapter. All neural
// operations are ordinary DiffIR executed by the shared runtime. The adapter
// extends a canonical, unoptimized make_h3_denoiser graph without renumbering
// any existing tensor, so immutable base bundles remain usable.
struct H3ControlConfig {
  std::uint64_t patch_features{196};
  std::vector<std::uint64_t> injection_layers{0, 10, 20, 30, 40};
  std::uint32_t controls{1};
  bool apply_audio{false};
  bool streamed_constants{true};
};

struct H3ControlWeight {
  std::uint32_t tensor_id{};
  std::string name;
};

struct H3ControlGraph {
  ir::Program program;
  std::vector<H3ControlWeight> weights;
  std::vector<std::uint32_t> rows_inputs;     // each [video_tokens,patch_features] F32
  std::vector<std::uint32_t> strength_inputs; // each [1] F32 scalar
  std::vector<std::uint32_t> injected_outputs;
  std::vector<std::uint32_t> side_outputs;
  std::uint64_t mapped_weight_bytes{};
};

struct H3ControlSchedule {
  float strength{1.0F};
  float start{0.0F};
  float end{1.0F};
};

// Progress is the H3 data-ward video timestep, not the evaluation index or
// noise sigma. Bounds are inclusive, matching SerenityFlow's DiT payload.
float h3_control_strength(const H3ControlSchedule &schedule, float progress);

// Recover and validate the actual canonical frontend geometry; never infer
// checkpoint architecture from an adapter filename or assume official widths.
H3DenoiserConfig h3_control_denoiser_config(const ir::Program &base);

H3ControlGraph add_h3_control(const ir::Program &base,
                             const H3DenoiserConfig &denoiser,
                             const H3ControlConfig &control = {});

// Reads only the header and validates the complete actual tensor inventory.
// Quantized / curve-form / partially compatible files are rejected. No
// checkpoint payload is copied or reordered: separate Q/K/V and value-first
// SwiGLU stay as released, with the order expressed in the graph.
void validate_h3_control_checkpoint(const weights::SafeTensorFile &checkpoint,
                                    const H3ControlGraph &graph);
runtime::TensorMap map_h3_control_weights(
    const weights::SafeTensorFile &checkpoint, const H3ControlGraph &graph);

// Native row preparation for an already VAE-encoded guide. Input is the
// canonical [1,C,T,H,W] F32 latent, where C=24 for ordinary controls or C=49
// for [control,visibility,masked_source] inpainting. Features are padded AFTER
// patchification, as in the source, never interleaved before it.
runtime::Tensor h3_control_rows(const runtime::Tensor &latent,
                               std::uint64_t patch_height = 2,
                               std::uint64_t patch_width = 2,
                               std::uint64_t patch_features = 196);

} // namespace dif::frontend
