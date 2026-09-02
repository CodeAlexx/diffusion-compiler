#pragma once

#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/frontend/provenance.hpp"
#include "dif/ir/ir.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
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
  // The real creator block.forward path selects static 8192-wide reductions
  // for both norms. Keep the attributes separate so expanded development
  // oracles can record a different compiler-selected call-site shape.
  std::uint64_t prenorm_reduction_tile{8192U};
  std::uint64_t postnorm_reduction_tile{8192U};
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

// Official sampling.py schedule policy. Raw uses the resolution-derived shift;
// Turbo may pin mu explicitly. Values preserve torch.linspace and eager F32
// tensor-operation boundaries rather than using an algebraically simplified
// shifted-sigma expression.
struct Krea2ScheduleConfig {
  std::uint32_t steps{52U};
  std::uint64_t minimum_resolution{256U};
  std::uint64_t maximum_resolution{1280U};
  double minimum_mu{0.5};
  double maximum_mu{1.15};
  double sigma{1.0};
  std::optional<double> fixed_mu;
};

struct Krea2Schedule {
  double mu{};
  std::vector<float> timesteps;
};

Krea2Schedule make_krea2_schedule(
    const Krea2Config &model = {},
    const Krea2ScheduleConfig &schedule = {});

// Source-ordered Raw CFG and Euler update:
//   difference = cond - uncond
//   guided = difference * guidance
//   velocity = cond + guided
//   next_sample = sample + (next-current) * velocity
// Each intermediate is stored in the sample dtype, matching eager BF16.
struct Krea2CfgEulerBuild {
  ir::Program program;
  std::uint32_t sample_input{};
  std::uint32_t conditional_velocity_input{};
  std::uint32_t unconditional_velocity_input{};
  std::uint32_t guidance_input{};
  std::uint32_t current_timestep_input{};
  std::uint32_t next_timestep_input{};
  std::uint32_t negative_one_constant{};
  std::uint32_t difference_output{};
  std::uint32_t guided_delta_output{};
  std::uint32_t velocity_output{};
  std::uint32_t sample_output{};
};

Krea2CfgEulerBuild
make_krea2_cfg_euler_step(std::vector<std::uint64_t> sample_shape);

// Source-ordered Turbo Euler update when classifier-free guidance is disabled:
//   next_sample = sample + (next-current) * velocity
// This deliberately omits the unconditional branch and all CFG tensor work.
struct Krea2EulerBuild {
  ir::Program program;
  std::uint32_t sample_input{};
  std::uint32_t velocity_input{};
  std::uint32_t current_timestep_input{};
  std::uint32_t next_timestep_input{};
  std::uint32_t sample_output{};
};

Krea2EulerBuild
make_krea2_euler_step(std::vector<std::uint64_t> sample_shape);

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

// One released SingleStreamBlock expressed only with shared DiffIR semantics.
// The input is the already-combined text/image sequence and the timestep
// projection is accepted as an input so block parity can be localized before
// admitting the conditioner and outer model. Checkpoint parameters are bound
// in BF16 exactly as creator `model.to(torch.bfloat16)` materializes them;
// source F32 tensors therefore require the runtime's explicit conversion.
// Krea AI's official release is the semantic oracle for this frontend. The
// pinned revision travels with every provenance table the frontend writes.
inline constexpr std::string_view kKrea2Creator = "krea-ai/krea-2";
inline constexpr std::string_view kKrea2CreatorRevision =
    "db3984fbc6e13b34c0064990fc2d95ac64d00058";

struct Krea2BlockBuild {
  ir::Program program;
  Krea2Config config;
  std::uint64_t block_index{};
  // Creator module / section per operation, recorded by this builder from
  // the boundary tensors it produced.
  ProvenanceTable provenance;
  std::uint32_t sequence_input{};
  std::uint32_t modulation_input{};
  std::uint32_t positions_input{};
  std::uint32_t validity_mask_input{};
  std::uint32_t modulated_parameters{};
  std::uint32_t input_normalized{};
  std::uint32_t attention_input{};
  std::uint32_t query{};
  std::uint32_t key{};
  std::uint32_t value{};
  std::uint32_t rotary_query{};
  std::uint32_t rotary_key{};
  std::uint32_t attention_output{};
  std::uint32_t attention_gate{};
  std::uint32_t output_projection{};
  std::uint32_t attention_residual{};
  std::uint32_t mlp_input{};
  std::uint32_t mlp_gate{};
  std::uint32_t mlp_up{};
  std::uint32_t mlp_gate_activated{};
  std::uint32_t mlp_activation{};
  std::uint32_t mlp_output{};
  std::uint32_t final_output{};
  std::uint32_t rotary_pair_axes{};
  std::uint32_t rotary_pair_indices{};
  std::uint32_t rotary_axis_dims{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
};

Krea2BlockBuild make_krea2_block(const Krea2Config &config = {},
                                 std::uint64_t block_index = 0U,
                                 bool capture_boundaries = true);

// Exact official Krea 2 Qwen3-VL-4B text-tower contract. The encoder consumes
// 541 prefix/prompt/padding tokens plus the fixed five-token assistant suffix,
// executes all 36 layers with a padding-aware causal mask, then returns the
// 12 raw residual taps sliced after the 34-token system/user prefix.
Qwen3VlConditionerConfig make_krea2_conditioner_config();

// The shared Krea text-fusion path consumes the 12 selected Qwen residual
// taps, refines across the tap axis, projects 12 -> 1, refines across text
// tokens with the exact padding mask, and maps 2560 -> the DiT width 6144.
struct Krea2TextFusionBuild {
  ir::Program program;
  std::uint32_t context_input{};
  std::uint32_t validity_mask_input{};
  std::vector<std::uint32_t> block_outputs;
  std::vector<std::pair<std::string, std::uint32_t>> first_block_boundaries;
  std::uint32_t projected_output{};
  std::uint32_t conditioning_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
};

Krea2TextFusionBuild make_krea2_text_fusion(bool capture_boundaries = true,
                                            bool capture_first_block = false);

// Complete source-faithful Raw MMDiT evaluation after text fusion. The graph
// owns the patch projection, timestep tower, all 28 shared-runtime blocks,
// and the final 6144 -> 64 velocity head. It intentionally accepts packed
// 2x2 image tokens: deterministic noise creation and patch packing are sampler
// frontend responsibilities, while every learned operation remains in DiffIR.
struct Krea2DenoiserBuild {
  ir::Program program;
  Krea2Config config;
  ProvenanceTable provenance;
  std::uint32_t image_tokens_input{};
  std::uint32_t context_input{};
  std::uint32_t timestep_input{};
  std::uint32_t positions_input{};
  std::uint32_t validity_mask_input{};
  std::uint32_t rotary_pair_axes{};
  std::uint32_t rotary_pair_indices{};
  std::uint32_t rotary_axis_dims{};
  std::uint32_t projected_image{};
  std::uint32_t timestep_embedding{};
  std::uint32_t timestep_first_linear{};
  std::uint32_t timestep_first_activation{};
  std::uint32_t timestep_output{};
  std::uint32_t timestep_projection_activation{};
  std::uint32_t modulation_output{};
  std::vector<std::uint32_t> block_outputs;
  std::uint32_t last_modulated{};
  std::uint32_t velocity_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
};

Krea2DenoiserBuild make_krea2_denoiser(const Krea2Config &config = {},
                                       bool capture_block_outputs = false);

// One complete Turbo denoise execution plan. The 28-block denoiser is
// repeated inside a single verified DiffIR program and the source-ordered
// Euler update connects one evaluation to the next. Checkpoint constants and
// prompt inputs keep one tensor identity across every evaluation; only
// timestep inputs and transient activations are cloned. This is deliberately
// an execution-plan frontend, not a Krea-specific runtime path.
struct Krea2TurboExecutionBuild {
  ir::Program program;
  Krea2Config config;
  std::uint32_t initial_image_input{};
  std::uint32_t context_input{};
  std::uint32_t positions_input{};
  std::uint32_t validity_mask_input{};
  std::uint32_t rotary_pair_axes{};
  std::uint32_t rotary_pair_indices{};
  std::uint32_t rotary_axis_dims{};
  std::vector<std::uint32_t> model_timestep_inputs;
  std::vector<std::uint32_t> current_timestep_inputs;
  std::vector<std::uint32_t> next_timestep_inputs;
  std::vector<std::uint32_t> velocity_outputs;
  std::vector<std::uint32_t> image_outputs;
  std::uint32_t final_image_output{};
  std::vector<std::uint32_t> checkpoint_tensors;
  std::vector<std::string> checkpoint_names;
  // Per-evaluation streamed aliases map new DiffIR tensor ids back to the
  // canonical checkpoint tensor id whose mapped host storage they share.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> constant_sources;
};

Krea2TurboExecutionBuild
make_krea2_turbo_execution(const Krea2Config &config, std::uint32_t steps,
                           bool capture_trajectory,
                           const std::vector<std::uint32_t>
                               &reusable_resident_constants);

} // namespace dif::frontend
