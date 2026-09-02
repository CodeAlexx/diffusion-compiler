#include "dif/frontend/krea2.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace dif::frontend {

Krea2Architecture inspect_krea2_architecture(const Krea2Config &config) {
  constexpr auto alignment =
      Krea2Config::kVaeCompression * Krea2Config::kPatch;
  if (config.batch == 0U || config.width == 0U || config.height == 0U ||
      config.text_tokens == 0U || config.text_tokens > 512U ||
      (config.prenorm_reduction_tile != 2048U &&
       config.prenorm_reduction_tile != 8192U) ||
      (config.postnorm_reduction_tile != 2048U &&
       config.postnorm_reduction_tile != 8192U) ||
      config.width % alignment != 0U || config.height % alignment != 0U)
    fail("Krea 2 requires a positive batch, 1..512 text tokens, a creator "
         "RMS reduction tile, and image dimensions divisible by VAE "
         "compression * patch (16)");

  Krea2Architecture result;
  result.latent_height = config.height / Krea2Config::kVaeCompression;
  result.latent_width = config.width / Krea2Config::kVaeCompression;
  result.image_grid_height = result.latent_height / Krea2Config::kPatch;
  result.image_grid_width = result.latent_width / Krea2Config::kPatch;
  if (result.image_grid_height >
      std::numeric_limits<std::uint64_t>::max() / result.image_grid_width)
    fail("Krea 2 image-token geometry overflows");
  result.image_tokens = result.image_grid_height * result.image_grid_width;
  if (result.image_tokens >
      std::numeric_limits<std::uint64_t>::max() - config.text_tokens)
    fail("Krea 2 combined sequence overflows");
  result.combined_tokens = result.image_tokens + config.text_tokens;
  if (result.combined_tokens >
      std::numeric_limits<std::uint64_t>::max() -
          (Krea2Config::kSequenceAlignment - 1U))
    fail("Krea 2 padded sequence overflows");
  result.padded_tokens =
      ((result.combined_tokens + Krea2Config::kSequenceAlignment - 1U) /
       Krea2Config::kSequenceAlignment) *
      Krea2Config::kSequenceAlignment;
  result.patch_input_dim = Krea2Config::kLatentChannels *
                           Krea2Config::kPatch * Krea2Config::kPatch;
  result.patch_output_dim = result.patch_input_dim;
  return result;
}

Krea2Schedule make_krea2_schedule(const Krea2Config &model,
                                  const Krea2ScheduleConfig &schedule_config) {
  const auto architecture = inspect_krea2_architecture(model);
  constexpr auto alignment = Krea2Config::kVaeCompression * Krea2Config::kPatch;
  if (schedule_config.steps == 0U || schedule_config.minimum_resolution == 0U ||
      schedule_config.maximum_resolution == 0U ||
      schedule_config.minimum_resolution > schedule_config.maximum_resolution ||
      !std::isfinite(schedule_config.minimum_mu) ||
      !std::isfinite(schedule_config.maximum_mu) ||
      !(schedule_config.sigma > 0.0) || !std::isfinite(schedule_config.sigma) ||
      (schedule_config.fixed_mu.has_value() &&
       !std::isfinite(*schedule_config.fixed_mu)))
    fail("Krea 2 schedule configuration is invalid");

  const auto minimum_grid = schedule_config.minimum_resolution / alignment;
  const auto maximum_grid = schedule_config.maximum_resolution / alignment;
  if (minimum_grid == 0U || maximum_grid == 0U ||
      minimum_grid > std::numeric_limits<std::uint64_t>::max() / minimum_grid ||
      maximum_grid > std::numeric_limits<std::uint64_t>::max() / maximum_grid)
    fail("Krea 2 schedule resolution geometry overflows");
  const auto x1 = minimum_grid * minimum_grid;
  const auto x2 = maximum_grid * maximum_grid;
  if (x1 == x2)
    fail("Krea 2 schedule interpolation endpoints must differ");

  Krea2Schedule result;
  if (schedule_config.fixed_mu.has_value()) {
    result.mu = *schedule_config.fixed_mu;
  } else {
    const auto slope =
        (schedule_config.maximum_mu - schedule_config.minimum_mu) /
        static_cast<double>(x2 - x1);
    result.mu = slope * static_cast<double>(architecture.image_tokens) +
                (schedule_config.minimum_mu - slope * static_cast<double>(x1));
  }

  const auto points = schedule_config.steps + 1U;
  const auto step = -1.0F / static_cast<float>(schedule_config.steps);
  const auto halfway = points / 2U;
  const auto shifted = static_cast<float>(std::exp(result.mu));
  const auto exponent = static_cast<float>(schedule_config.sigma);
  result.timesteps.reserve(points);
  for (std::uint32_t index = 0U; index < points; ++index) {
    // torch.linspace constructs the F32 CPU grid from both endpoints with an
    // FMA. Preserve each subsequent eager tensor-op rounding boundary from
    // sampling.py instead of simplifying shift/(shift + (1/t - 1)^sigma).
    const auto base = index < halfway
                          ? std::fma(step, static_cast<float>(index), 1.0F)
                          : std::fma(-step,
                                     static_cast<float>(points - index - 1U),
                                     0.0F);
    volatile float reciprocal = 1.0F / base;
    volatile float unshifted = reciprocal - 1.0F;
    volatile float powered = std::pow(unshifted, exponent);
    volatile float denominator = shifted + powered;
    // PyTorch's scalar/tensor reverse-divide kernel materializes the tensor
    // reciprocal and then multiplies by the scalar. That eager boundary differs
    // from one scalar IEEE division at 15/53 Raw schedule points.
    volatile float inverse_denominator = 1.0F / denominator;
    volatile float timestep = shifted * inverse_denominator;
    result.timesteps.push_back(static_cast<float>(timestep));
  }
  if (result.timesteps.size() != points || result.timesteps.front() != 1.0F ||
      result.timesteps.back() != 0.0F)
    fail("Krea 2 schedule lost its 1-to-0 endpoints");
  return result;
}

Krea2CfgEulerBuild
make_krea2_cfg_euler_step(std::vector<std::uint64_t> sample_shape) {
  using namespace ir;
  if (sample_shape.empty())
    fail("Krea 2 CFG/Euler sample shape must be non-empty");
  std::uint64_t elements = 1U;
  for (const auto dimension : sample_shape) {
    if (dimension == 0U ||
        elements > std::numeric_limits<std::uint64_t>::max() / dimension)
      fail("Krea 2 CFG/Euler sample shape is invalid");
    elements *= dimension;
  }
  (void)elements;

  Krea2CfgEulerBuild build;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_operation = [&](Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs) {
    program.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs), {}});
  };

  build.sample_input = add_tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.conditional_velocity_input =
      add_tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.unconditional_velocity_input =
      add_tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.guidance_input = add_tensor(DType::BF16, TensorRole::Input, {1U});
  build.current_timestep_input =
      add_tensor(DType::F32, TensorRole::Input, {1U});
  build.next_timestep_input = add_tensor(DType::F32, TensorRole::Input, {1U});
  build.negative_one_constant =
      add_tensor(DType::BF16, TensorRole::Constant, {1U});

  const auto negative_one =
      add_tensor(DType::BF16, TensorRole::Internal, sample_shape);
  add_operation(Opcode::BroadcastTo, {build.negative_one_constant},
                {negative_one});
  const auto negative_unconditional =
      add_tensor(DType::BF16, TensorRole::Internal, sample_shape);
  add_operation(Opcode::Multiply,
                {build.unconditional_velocity_input, negative_one},
                {negative_unconditional});
  build.difference_output =
      add_tensor(DType::BF16, TensorRole::Output, sample_shape);
  add_operation(Opcode::Add,
                {build.conditional_velocity_input, negative_unconditional},
                {build.difference_output});

  const auto guidance =
      add_tensor(DType::BF16, TensorRole::Internal, sample_shape);
  add_operation(Opcode::BroadcastTo, {build.guidance_input}, {guidance});
  build.guided_delta_output =
      add_tensor(DType::BF16, TensorRole::Output, sample_shape);
  add_operation(Opcode::Multiply, {build.difference_output, guidance},
                {build.guided_delta_output});
  build.velocity_output =
      add_tensor(DType::BF16, TensorRole::Output, sample_shape);
  add_operation(Opcode::Add,
                {build.conditional_velocity_input, build.guided_delta_output},
                {build.velocity_output});
  build.sample_output =
      add_tensor(DType::BF16, TensorRole::Output, sample_shape);
  add_operation(Opcode::EulerVelocityStep,
                {build.sample_input, build.velocity_output,
                 build.current_timestep_input, build.next_timestep_input},
                {build.sample_output});

  verify(program);
  return build;
}

Krea2EulerBuild
make_krea2_euler_step(std::vector<std::uint64_t> sample_shape) {
  using namespace ir;
  if (sample_shape.empty())
    fail("Krea 2 Euler sample shape must be non-empty");
  std::uint64_t elements = 1U;
  for (const auto dimension : sample_shape) {
    if (dimension == 0U ||
        elements > std::numeric_limits<std::uint64_t>::max() / dimension)
      fail("Krea 2 Euler sample shape is invalid");
    elements *= dimension;
  }
  (void)elements;

  Krea2EulerBuild build;
  auto &program = build.program;
  build.sample_input = 1U;
  build.velocity_input = 2U;
  build.current_timestep_input = 3U;
  build.next_timestep_input = 4U;
  build.sample_output = 5U;
  program.tensors = {
      {build.sample_input, DType::BF16, TensorRole::Input, sample_shape},
      {build.velocity_input, DType::BF16, TensorRole::Input, sample_shape},
      {build.current_timestep_input, DType::F32, TensorRole::Input, {1U}},
      {build.next_timestep_input, DType::F32, TensorRole::Input, {1U}},
      {build.sample_output, DType::BF16, TensorRole::Output, sample_shape},
  };
  program.operations = {
      {1U,
       Opcode::EulerVelocityStep,
       {build.sample_input, build.velocity_input,
        build.current_timestep_input, build.next_timestep_input},
       {build.sample_output},
       {}},
  };
  verify(program);
  return build;
}

Krea2TimeConditioningBuild
make_krea2_time_conditioning(const Krea2Config &config) {
  (void)inspect_krea2_architecture(config);
  using namespace ir;
  Krea2TimeConditioningBuild build;
  build.config = config;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed
                                 : TensorRole::Internal));
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_operation = [&](Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto add_checkpoint = [&](const char *name,
                                  std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(DType::BF16, constant_roles, std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.emplace_back(name);
    return id;
  };

  build.timestep_input =
      add_tensor(DType::BF16, TensorRole::Input, {config.batch});
  const auto timestep_f32 =
      add_tensor(DType::F32, TensorRole::Internal, {config.batch});
  add_operation(Opcode::Cast, {build.timestep_input}, {timestep_f32});
  build.timestep_embedding = add_tensor(
      DType::F32, TensorRole::Internal,
      {config.batch, Krea2Config::kTimestepDim});
  add_operation(
      Opcode::SinusoidalTimestep, {timestep_f32},
      {build.timestep_embedding},
      {Attribute::boolean(AttrKey::FlipSinToCos, true),
       Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
       Attribute::f64(AttrKey::Scale, 1000.0),
       Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  const auto embedding_bf16 = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kTimestepDim});
  add_operation(Opcode::Cast, {build.timestep_embedding}, {embedding_bf16});

  const auto tmlp0_weight = add_checkpoint(
      "tmlp.0.weight",
      {Krea2Config::kFeatures, Krea2Config::kTimestepDim});
  const auto tmlp0_bias =
      add_checkpoint("tmlp.0.bias", {Krea2Config::kFeatures});
  const auto tmlp0 = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {embedding_bf16, tmlp0_weight, tmlp0_bias}, {tmlp0},
                {Attribute::u64(
                    AttrKey::LinearBiasMode,
                    static_cast<std::uint64_t>(LinearBiasMode::Addmm))});
  const auto tmlp0_activated = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  const auto gelu_attrs = std::vector<Attribute>{Attribute::u64(
      AttrKey::Approximation,
      static_cast<std::uint64_t>(GeluApproximation::Tanh))};
  add_operation(Opcode::Gelu, {tmlp0}, {tmlp0_activated}, gelu_attrs);

  const auto tmlp2_weight = add_checkpoint(
      "tmlp.2.weight", {Krea2Config::kFeatures, Krea2Config::kFeatures});
  const auto tmlp2_bias =
      add_checkpoint("tmlp.2.bias", {Krea2Config::kFeatures});
  build.timestep_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {tmlp0_activated, tmlp2_weight, tmlp2_bias},
                {build.timestep_output},
                {Attribute::u64(
                    AttrKey::LinearBiasMode,
                    static_cast<std::uint64_t>(LinearBiasMode::Addmm))});

  const auto tproj_activated = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Gelu, {build.timestep_output}, {tproj_activated},
                gelu_attrs);
  const auto tproj1_weight = add_checkpoint(
      "tproj.1.weight",
      {6U * Krea2Config::kFeatures, Krea2Config::kFeatures});
  const auto tproj1_bias =
      add_checkpoint("tproj.1.bias", {6U * Krea2Config::kFeatures});
  build.modulation_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {config.batch, 6U * Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {tproj_activated, tproj1_weight, tproj1_bias},
                {build.modulation_output},
                {Attribute::u64(
                    AttrKey::LinearBiasMode,
                    static_cast<std::uint64_t>(LinearBiasMode::Addmm))});

  verify(program);
  return build;
}

Krea2BlockBuild make_krea2_block(const Krea2Config &config,
                                 std::uint64_t block_index,
                                 bool capture_boundaries) {
  const auto architecture = inspect_krea2_architecture(config);
  if (block_index >= Krea2Config::kLayers)
    fail("Krea 2 block index is outside the released 28-block model");
  using namespace ir;
  Krea2BlockBuild build;
  build.config = config;
  build.block_index = block_index;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto checkpoint_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed
                                 : TensorRole::Internal));
  const auto boundary_roles = capture_boundaries
                                  ? static_cast<std::uint32_t>(TensorRole::Output)
                                  : static_cast<std::uint32_t>(TensorRole::Internal);
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_bf16 = [&](std::vector<std::uint64_t> dims,
                            bool boundary = false) {
    return add_tensor(DType::BF16,
                      boundary ? boundary_roles : TensorRole::Internal,
                      std::move(dims));
  };
  const auto add_operation = [&](Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto prefix = "blocks." + std::to_string(block_index) + ".";
  const auto add_checkpoint = [&](std::string name,
                                  std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(DType::BF16, checkpoint_roles, std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.push_back(prefix + std::move(name));
    return id;
  };

  const auto batch = config.batch;
  const auto sequence = architecture.padded_tokens;
  const auto rows = batch * sequence;
  const auto features = Krea2Config::kFeatures;
  const auto heads = Krea2Config::kHeads;
  const auto kv_heads = Krea2Config::kKvHeads;
  const auto head_dim = Krea2Config::kHeadDim;
  const auto mlp = Krea2Config::kMlpDim;
  const std::vector<std::uint64_t> sequence_shape{batch, sequence, features};

  build.sequence_input =
      add_tensor(DType::BF16, TensorRole::Input, sequence_shape);
  build.modulation_input = add_tensor(
      DType::BF16, TensorRole::Input, {batch, 6U * features});
  build.positions_input =
      add_tensor(DType::F32, TensorRole::Input, {batch, sequence, 3U});
  build.validity_mask_input =
      add_tensor(DType::Bool, TensorRole::Input, {batch, sequence});

  const auto modulation_delta =
      add_checkpoint("mod.lin", {6U * features});
  const auto modulation_delta_2d = add_bf16({1U, 6U * features});
  add_operation(Opcode::Reshape, {modulation_delta}, {modulation_delta_2d});
  build.modulated_parameters =
      add_bf16({batch, 6U * features}, true);
  add_operation(Opcode::Add,
                {build.modulation_input, modulation_delta_2d},
                {build.modulated_parameters});

  const auto modulation_chunk = [&](std::uint64_t chunk) {
    const auto sliced = add_bf16({batch, features});
    add_operation(Opcode::Slice, {build.modulated_parameters}, {sliced},
                  {Attribute::u64(AttrKey::Axis, 1U),
                   Attribute::u64(AttrKey::Start, chunk * features)});
    const auto vector = add_bf16({features});
    add_operation(Opcode::Reshape, {sliced}, {vector});
    return vector;
  };
  const auto prescale = modulation_chunk(0U);
  const auto preshift = modulation_chunk(1U);
  const auto pregate_vector = modulation_chunk(2U);
  const auto postscale = modulation_chunk(3U);
  const auto postshift = modulation_chunk(4U);
  const auto postgate_vector = modulation_chunk(5U);

  const auto broadcast_modulation = [&](std::uint32_t vector) {
    const auto expanded = add_bf16({batch, 1U, features});
    add_operation(Opcode::Reshape, {vector}, {expanded});
    const auto broadcast = add_bf16(sequence_shape);
    add_operation(Opcode::BroadcastTo, {expanded}, {broadcast});
    return broadcast;
  };
  const auto pregate = broadcast_modulation(pregate_vector);
  const auto postgate = broadcast_modulation(postgate_vector);

  const auto ones = add_bf16({features});
  add_operation(Opcode::Fill, {}, {ones},
                {Attribute::f64(AttrKey::Value, 1.0)});
  const auto prenorm_weight = add_checkpoint("prenorm.scale", {features});
  build.input_normalized = add_bf16(sequence_shape, true);
  add_operation(Opcode::RmsNorm,
                {build.sequence_input, prenorm_weight},
                {build.input_normalized},
                {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                 Attribute::f64(AttrKey::WeightOffset, 1.0),
                 Attribute::u64(AttrKey::BlockSize, 512U),
                 Attribute::u64(AttrKey::ReductionTileSize,
                                config.prenorm_reduction_tile)});
  const auto prescale_plus_one = add_bf16({features});
  add_operation(Opcode::Add, {ones, prescale}, {prescale_plus_one});
  build.attention_input = add_bf16(sequence_shape, true);
  add_operation(Opcode::AffineLastDim,
                {build.input_normalized, prescale_plus_one, preshift},
                {build.attention_input});
  const auto attention_input_flat = add_bf16({rows, features});
  add_operation(Opcode::Reshape, {build.attention_input},
                {attention_input_flat});

  const auto wq = add_checkpoint("attn.wq.weight", {features, features});
  const auto wk = add_checkpoint("attn.wk.weight",
                                 {kv_heads * head_dim, features});
  const auto wv = add_checkpoint("attn.wv.weight",
                                 {kv_heads * head_dim, features});
  const auto gate_weight =
      add_checkpoint("attn.gate.weight", {features, features});
  const auto q_flat = add_bf16({rows, heads * head_dim});
  const auto k_flat = add_bf16({rows, kv_heads * head_dim});
  const auto v_flat = add_bf16({rows, kv_heads * head_dim});
  add_operation(Opcode::Linear, {attention_input_flat, wq}, {q_flat});
  add_operation(Opcode::Linear, {attention_input_flat, wk}, {k_flat});
  add_operation(Opcode::Linear, {attention_input_flat, wv}, {v_flat});
  const auto q_shaped = add_bf16({batch, sequence, heads, head_dim});
  const auto k_shaped = add_bf16({batch, sequence, kv_heads, head_dim});
  build.value = add_bf16({batch, sequence, kv_heads, head_dim}, true);
  add_operation(Opcode::Reshape, {q_flat}, {q_shaped});
  add_operation(Opcode::Reshape, {k_flat}, {k_shaped});
  add_operation(Opcode::Reshape, {v_flat}, {build.value});
  const auto qnorm =
      add_checkpoint("attn.qknorm.qnorm.scale", {head_dim});
  const auto knorm =
      add_checkpoint("attn.qknorm.knorm.scale", {head_dim});
  build.query = add_bf16({batch, sequence, heads, head_dim}, true);
  build.key = add_bf16({batch, sequence, kv_heads, head_dim}, true);
  const auto qk_norm_attributes =
      std::vector<Attribute>{Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                             Attribute::f64(AttrKey::WeightOffset, 1.0),
                             Attribute::u64(AttrKey::BlockSize, 128U)};
  add_operation(Opcode::RmsNorm, {q_shaped, qnorm}, {build.query},
                qk_norm_attributes);
  add_operation(Opcode::RmsNorm, {k_shaped, knorm}, {build.key},
                qk_norm_attributes);

  build.rotary_pair_axes =
      add_tensor(DType::I32, TensorRole::Constant, {head_dim / 2U});
  build.rotary_pair_indices =
      add_tensor(DType::I32, TensorRole::Constant, {head_dim / 2U});
  build.rotary_axis_dims = add_tensor(DType::I32, TensorRole::Constant, {3U});
  const auto rotary_cosine =
      add_tensor(DType::F32, TensorRole::Internal,
                 {batch, sequence, head_dim / 2U});
  const auto rotary_sine =
      add_tensor(DType::F32, TensorRole::Internal,
                 {batch, sequence, head_dim / 2U});
  add_operation(Opcode::RotaryFrequency,
                {build.positions_input, build.rotary_pair_axes,
                 build.rotary_pair_indices, build.rotary_axis_dims},
                {rotary_cosine, rotary_sine},
                {Attribute::f64(AttrKey::Theta, 1000.0),
                 Attribute::f64(AttrKey::Ntk, 1.0)});
  const auto rotary_attributes = std::vector<Attribute>{Attribute::u64(
      AttrKey::RotaryLayout,
      static_cast<std::uint64_t>(RotaryLayout::Interleaved))};
  build.rotary_query =
      add_bf16({batch, sequence, heads, head_dim}, true);
  build.rotary_key =
      add_bf16({batch, sequence, kv_heads, head_dim}, true);
  add_operation(Opcode::RotaryApply,
                {build.query, rotary_cosine, rotary_sine},
                {build.rotary_query}, rotary_attributes);
  add_operation(Opcode::RotaryApply,
                {build.key, rotary_cosine, rotary_sine},
                {build.rotary_key}, rotary_attributes);

  const auto attention_bias =
      add_bf16({batch, 1U, sequence, sequence});
  add_operation(Opcode::BooleanMaskToBias, {build.validity_mask_input},
                {attention_bias});
  const auto attention_4d =
      add_bf16({batch, sequence, heads, head_dim});
  add_operation(Opcode::Attention,
                {build.rotary_query, build.rotary_key, build.value,
                 attention_bias},
                {attention_4d},
                {Attribute::u64(AttrKey::KvHeads, kv_heads),
                 Attribute::u64(AttrKey::Implementation, 2U),
                 Attribute::f64(AttrKey::AttentionScale,
                                1.0 / 11.313708498984761)});
  build.attention_output = add_bf16({rows, features}, true);
  add_operation(Opcode::Reshape, {attention_4d}, {build.attention_output});
  const auto gate_logits = add_bf16({rows, features});
  add_operation(Opcode::Linear, {attention_input_flat, gate_weight},
                {gate_logits});
  build.attention_gate = add_bf16({rows, features}, true);
  add_operation(Opcode::Sigmoid, {gate_logits}, {build.attention_gate});
  const auto gated_attention = add_bf16({rows, features});
  add_operation(Opcode::Multiply,
                {build.attention_output, build.attention_gate},
                {gated_attention});
  const auto wo = add_checkpoint("attn.wo.weight", {features, features});
  build.output_projection = add_bf16({rows, features}, true);
  add_operation(Opcode::Linear, {gated_attention, wo},
                {build.output_projection});
  const auto projected_sequence = add_bf16(sequence_shape);
  add_operation(Opcode::Reshape, {build.output_projection},
                {projected_sequence});
  const auto gated_projection = add_bf16(sequence_shape);
  add_operation(Opcode::Multiply, {pregate, projected_sequence},
                {gated_projection});
  build.attention_residual = add_bf16(sequence_shape, true);
  add_operation(Opcode::Add, {build.sequence_input, gated_projection},
                {build.attention_residual});

  const auto postnorm_weight = add_checkpoint("postnorm.scale", {features});
  const auto post_normalized = add_bf16(sequence_shape);
  add_operation(Opcode::RmsNorm,
                {build.attention_residual, postnorm_weight},
                {post_normalized},
                {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                 Attribute::f64(AttrKey::WeightOffset, 1.0),
                 Attribute::u64(AttrKey::BlockSize, 512U),
                 Attribute::u64(AttrKey::ReductionTileSize,
                                config.postnorm_reduction_tile)});
  const auto postscale_plus_one = add_bf16({features});
  add_operation(Opcode::Add, {ones, postscale}, {postscale_plus_one});
  build.mlp_input = add_bf16(sequence_shape, true);
  add_operation(Opcode::AffineLastDim,
                {post_normalized, postscale_plus_one, postshift},
                {build.mlp_input});
  const auto mlp_input_flat = add_bf16({rows, features});
  add_operation(Opcode::Reshape, {build.mlp_input}, {mlp_input_flat});
  const auto mlp_gate_weight =
      add_checkpoint("mlp.gate.weight", {mlp, features});
  const auto mlp_up_weight =
      add_checkpoint("mlp.up.weight", {mlp, features});
  const auto mlp_down_weight =
      add_checkpoint("mlp.down.weight", {features, mlp});
  build.mlp_gate = add_bf16({rows, mlp}, true);
  build.mlp_up = add_bf16({rows, mlp}, true);
  add_operation(Opcode::Linear, {mlp_input_flat, mlp_gate_weight},
                {build.mlp_gate});
  add_operation(Opcode::Linear, {mlp_input_flat, mlp_up_weight},
                {build.mlp_up});
  build.mlp_gate_activated = add_bf16({rows, mlp}, true);
  add_operation(Opcode::SiLU, {build.mlp_gate},
                {build.mlp_gate_activated});
  build.mlp_activation = add_bf16({rows, mlp}, true);
  add_operation(Opcode::Multiply,
                {build.mlp_gate_activated, build.mlp_up},
                {build.mlp_activation});
  build.mlp_output = add_bf16({rows, features}, true);
  add_operation(Opcode::Linear, {build.mlp_activation, mlp_down_weight},
                {build.mlp_output});
  const auto mlp_sequence = add_bf16(sequence_shape);
  add_operation(Opcode::Reshape, {build.mlp_output}, {mlp_sequence});
  const auto gated_mlp = add_bf16(sequence_shape);
  add_operation(Opcode::Multiply, {postgate, mlp_sequence}, {gated_mlp});
  build.final_output = add_tensor(DType::BF16, TensorRole::Output,
                                  sequence_shape);
  add_operation(Opcode::Add, {build.attention_residual, gated_mlp},
                {build.final_output});

  // Record the creator section each operation was lowered from, driven by the
  // boundary tensors this builder itself produced (never by tensor names).
  build.provenance.frontend = "krea2";
  build.provenance.creator = std::string(kKrea2Creator);
  build.provenance.creator_revision = std::string(kKrea2CreatorRevision);
  {
    std::string section = "modulation";
    for (const auto &op : program.operations) {
      const auto produces = [&](std::uint32_t id) {
        return std::find(op.outputs.begin(), op.outputs.end(), id) !=
               op.outputs.end();
      };
      if (op.opcode == Opcode::Attention ||
          op.opcode == Opcode::BooleanMaskToBias)
        section = "attention.core";
      else if (op.opcode == Opcode::RotaryFrequency ||
               op.opcode == Opcode::RotaryApply)
        section = "attention.rotary";
      else if (produces(build.query) || produces(build.key))
        section = "attention.qknorm";
      else if (produces(build.attention_gate))
        section = "attention.gate";
      else if (produces(build.output_projection))
        section = "attention.out";
      else if (produces(build.attention_residual))
        section = "attention.residual";
      else if (produces(build.mlp_gate) || produces(build.mlp_up))
        section = "mlp.gate_up";
      else if (produces(build.mlp_gate_activated) ||
               produces(build.mlp_activation))
        section = "mlp.activation";
      else if (produces(build.mlp_output))
        section = "mlp.down";
      else if (produces(build.final_output))
        section = "mlp.residual";
      std::string module = "blocks." + std::to_string(block_index) + ".";
      if (section.rfind("attention", 0U) == 0U)
        module += "attn";
      else if (section.rfind("mlp", 0U) == 0U)
        module += "mlp";
      else
        module += "mod";
      build.provenance.records.push_back(
          {op.id, std::move(module), static_cast<std::int64_t>(block_index),
           section});
      // Transitions that take effect after a boundary producer.
      if (produces(build.attention_input))
        section = "attention.qkv";
      else if (produces(build.attention_output))
        section = "attention.gate";
      else if (produces(build.attention_residual))
        section = "modulation.post";
      else if (produces(build.mlp_input))
        section = "mlp.gate_up";
      else if (produces(build.mlp_output))
        section = "mlp.residual";
    }
  }
  for (std::size_t index = 0U; index < build.checkpoint_tensors.size(); ++index)
    build.provenance.weight_names.emplace_back(build.checkpoint_tensors[index],
                                               build.checkpoint_names[index]);

  verify(program);
  return build;
}

Qwen3VlConditionerConfig make_krea2_conditioner_config() {
  Qwen3VlConditionerConfig config;
  config.hidden_size = 2560U;
  config.executed_layers = 36U;
  config.attention_heads = 32U;
  config.key_value_heads = 8U;
  config.head_dim = 128U;
  config.intermediate_size = 9728U;
  config.vocabulary = 151936U;
  config.rms_norm_epsilon = 1.0e-6;
  config.rope_theta = 5.0e6;
  config.attention_implementation = 2U;
  config.selected_hidden_states = {2U,  5U,  8U,  11U, 14U, 17U,
                                   20U, 23U, 26U, 29U, 32U, 35U};
  config.output_slice_start = 34U;
  config.output_sequence_length = 512U;
  config.use_attention_mask = true;
  config.dynamic_position_ids = true;
  return config;
}

Krea2TextFusionBuild make_krea2_text_fusion(bool capture_boundaries,
                                            bool capture_first_block) {
  using namespace ir;
  constexpr std::uint64_t batch = 1U;
  constexpr std::uint64_t tokens = 512U;
  constexpr std::uint64_t taps = 12U;
  constexpr std::uint64_t hidden = Krea2Config::kTextDim;
  constexpr std::uint64_t heads = Krea2Config::kTextHeads;
  constexpr std::uint64_t head_dim = hidden / heads;
  constexpr std::uint64_t mlp = 6912U;

  Krea2TextFusionBuild build;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto output_roles = capture_boundaries
                                ? static_cast<std::uint32_t>(TensorRole::Output)
                                : static_cast<std::uint32_t>(TensorRole::Internal);
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto bf16 = [&](std::vector<std::uint64_t> dims,
                        bool boundary = false) {
    return add_tensor(DType::BF16,
                      boundary ? output_roles : TensorRole::Internal,
                      std::move(dims));
  };
  const auto operation = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                             std::vector<std::uint32_t> outputs,
                             std::vector<Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  };
  const auto checkpoint = [&](std::string name,
                              std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(
        DType::BF16, static_cast<std::uint32_t>(TensorRole::Constant),
        std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.push_back(std::move(name));
    return id;
  };
  const auto norm_attributes = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::f64(AttrKey::WeightOffset, 1.0),
      Attribute::u64(AttrKey::BlockSize, 128U)};

  build.context_input = add_tensor(
      DType::BF16, TensorRole::Input, {batch, tokens, taps, hidden});
  build.validity_mask_input =
      add_tensor(DType::Bool, TensorRole::Input, {batch, tokens});

  const auto text_block = [&](std::uint32_t input, const std::string &prefix,
                              std::uint64_t block_batch,
                              std::uint64_t block_tokens,
                              std::uint32_t attention_bias,
                              bool capture_first_block) {
    const auto capture = [&](std::string name, std::uint32_t id) {
      if (capture_first_block)
        build.first_block_boundaries.emplace_back(std::move(name), id);
    };
    const std::vector<std::uint64_t> shape{block_batch, block_tokens, hidden};
    const auto prenorm_weight =
        checkpoint(prefix + "prenorm.scale", {hidden});
    const auto prenorm = bf16(shape, capture_first_block);
    operation(Opcode::RmsNorm, {input, prenorm_weight}, {prenorm},
              norm_attributes);
    capture("first_prenorm", prenorm);
    const auto rows = block_batch * block_tokens;
    const auto prenorm_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {prenorm}, {prenorm_flat});
    const auto wq = checkpoint(prefix + "attn.wq.weight", {hidden, hidden});
    const auto wk = checkpoint(prefix + "attn.wk.weight", {hidden, hidden});
    const auto wv = checkpoint(prefix + "attn.wv.weight", {hidden, hidden});
    const auto q_flat = bf16({rows, hidden}, capture_first_block);
    const auto k_flat = bf16({rows, hidden}, capture_first_block);
    const auto v_flat = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Linear, {prenorm_flat, wq}, {q_flat});
    operation(Opcode::Linear, {prenorm_flat, wk}, {k_flat});
    operation(Opcode::Linear, {prenorm_flat, wv}, {v_flat});
    capture("first_q", q_flat);
    capture("first_k", k_flat);
    capture("first_v", v_flat);
    const auto q = bf16({block_batch, block_tokens, heads, head_dim});
    const auto k = bf16({block_batch, block_tokens, heads, head_dim});
    const auto v = bf16({block_batch, block_tokens, heads, head_dim});
    operation(Opcode::Reshape, {q_flat}, {q});
    operation(Opcode::Reshape, {k_flat}, {k});
    operation(Opcode::Reshape, {v_flat}, {v});
    const auto qnorm_weight =
        checkpoint(prefix + "attn.qknorm.qnorm.scale", {head_dim});
    const auto knorm_weight =
        checkpoint(prefix + "attn.qknorm.knorm.scale", {head_dim});
    const auto qnorm = bf16({block_batch, block_tokens, heads, head_dim},
                            capture_first_block);
    const auto knorm = bf16({block_batch, block_tokens, heads, head_dim},
                            capture_first_block);
    operation(Opcode::RmsNorm, {q, qnorm_weight}, {qnorm}, norm_attributes);
    operation(Opcode::RmsNorm, {k, knorm_weight}, {knorm}, norm_attributes);
    capture("first_qnorm", qnorm);
    capture("first_knorm", knorm);
    const auto attended = bf16({block_batch, block_tokens, heads, head_dim},
                               capture_first_block);
    auto attention_inputs = std::vector<std::uint32_t>{qnorm, knorm, v};
    if (attention_bias != 0U)
      attention_inputs.push_back(attention_bias);
    operation(Opcode::Attention, std::move(attention_inputs), {attended},
              {Attribute::u64(AttrKey::KvHeads, heads),
               Attribute::u64(AttrKey::Implementation, 2U)});
    capture("first_attention", attended);
    const auto attended_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {attended}, {attended_flat});
    const auto gate_weight =
        checkpoint(prefix + "attn.gate.weight", {hidden, hidden});
    const auto gate_logits = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Linear, {prenorm_flat, gate_weight}, {gate_logits});
    capture("first_gate_logits", gate_logits);
    const auto gate = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Sigmoid, {gate_logits}, {gate});
    capture("first_gate", gate);
    const auto gated = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Multiply, {attended_flat, gate}, {gated});
    capture("first_gated_attention", gated);
    const auto wo = checkpoint(prefix + "attn.wo.weight", {hidden, hidden});
    const auto projected_flat = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Linear, {gated, wo}, {projected_flat});
    capture("first_attention_projection", projected_flat);
    const auto projected = bf16(shape);
    operation(Opcode::Reshape, {projected_flat}, {projected});
    const auto attention_residual = bf16(shape, capture_first_block);
    operation(Opcode::Add, {input, projected}, {attention_residual});
    capture("first_attention_residual", attention_residual);

    const auto postnorm_weight =
        checkpoint(prefix + "postnorm.scale", {hidden});
    const auto postnorm = bf16(shape, capture_first_block);
    operation(Opcode::RmsNorm, {attention_residual, postnorm_weight},
              {postnorm}, norm_attributes);
    capture("first_postnorm", postnorm);
    const auto postnorm_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {postnorm}, {postnorm_flat});
    const auto gate_mlp_weight =
        checkpoint(prefix + "mlp.gate.weight", {mlp, hidden});
    const auto up_mlp_weight =
        checkpoint(prefix + "mlp.up.weight", {mlp, hidden});
    const auto down_mlp_weight =
        checkpoint(prefix + "mlp.down.weight", {hidden, mlp});
    const auto gate_mlp = bf16({rows, mlp}, capture_first_block);
    const auto up_mlp = bf16({rows, mlp}, capture_first_block);
    operation(Opcode::Linear, {postnorm_flat, gate_mlp_weight}, {gate_mlp});
    operation(Opcode::Linear, {postnorm_flat, up_mlp_weight}, {up_mlp});
    capture("first_mlp_gate", gate_mlp);
    capture("first_mlp_up", up_mlp);
    const auto activated = bf16({rows, mlp}, capture_first_block);
    operation(Opcode::SiLU, {gate_mlp}, {activated});
    capture("first_mlp_gate_activated", activated);
    const auto multiplied = bf16({rows, mlp}, capture_first_block);
    operation(Opcode::Multiply, {activated, up_mlp}, {multiplied});
    capture("first_mlp_activation", multiplied);
    const auto down_flat = bf16({rows, hidden}, capture_first_block);
    operation(Opcode::Linear, {multiplied, down_mlp_weight}, {down_flat});
    capture("first_mlp_output", down_flat);
    const auto down = bf16(shape);
    operation(Opcode::Reshape, {down_flat}, {down});
    const auto output = bf16(shape, true);
    operation(Opcode::Add, {attention_residual, down}, {output});
    build.block_outputs.push_back(output);
    return output;
  };

  auto context = bf16({tokens, taps, hidden});
  operation(Opcode::Reshape, {build.context_input}, {context});
  for (std::uint64_t index = 0; index < 2U; ++index)
    context = text_block(
        context, "txtfusion.layerwise_blocks." + std::to_string(index) + ".",
        tokens, taps, 0U,
        capture_boundaries && capture_first_block && index == 0U);

  const auto context_4d = bf16({batch, tokens, taps, hidden});
  operation(Opcode::Reshape, {context}, {context_4d});
  const auto context_permuted = bf16({batch, tokens, hidden, taps});
  operation(Opcode::Permute, {context_4d}, {context_permuted},
            {Attribute::u64(AttrKey::Permutation0, 0U),
             Attribute::u64(AttrKey::Permutation1, 1U),
             Attribute::u64(AttrKey::Permutation2, 3U),
             Attribute::u64(AttrKey::Permutation3, 2U)});
  const auto projector =
      checkpoint("txtfusion.projector.weight", {1U, taps});
  const auto projector_input = bf16({batch * tokens * hidden, taps});
  operation(Opcode::Reshape, {context_permuted}, {projector_input});
  const auto projector_output = bf16({batch * tokens * hidden, 1U});
  operation(Opcode::Linear, {projector_input, projector}, {projector_output});
  const auto projected_4d = bf16({batch, tokens, hidden, 1U});
  operation(Opcode::Reshape, {projector_output}, {projected_4d});
  build.projected_output = bf16({batch, tokens, hidden}, true);
  operation(Opcode::Reshape, {projected_4d}, {build.projected_output});

  const auto text_bias = bf16({batch, 1U, tokens, tokens});
  operation(Opcode::BooleanMaskToBias, {build.validity_mask_input},
            {text_bias});
  context = build.projected_output;
  for (std::uint64_t index = 0; index < 2U; ++index)
    context = text_block(
        context, "txtfusion.refiner_blocks." + std::to_string(index) + ".",
        batch, tokens, text_bias, false);

  const auto txt_norm_weight = checkpoint("txtmlp.0.scale", {hidden});
  const auto txt_norm = bf16({batch, tokens, hidden});
  operation(Opcode::RmsNorm, {context, txt_norm_weight}, {txt_norm},
            norm_attributes);
  const auto txt_norm_flat = bf16({batch * tokens, hidden});
  operation(Opcode::Reshape, {txt_norm}, {txt_norm_flat});
  const auto txt_linear_weight =
      checkpoint("txtmlp.1.weight", {Krea2Config::kFeatures, hidden});
  const auto txt_linear_bias =
      checkpoint("txtmlp.1.bias", {Krea2Config::kFeatures});
  const auto txt_linear_flat =
      bf16({batch * tokens, Krea2Config::kFeatures});
  operation(Opcode::Linear,
            {txt_norm_flat, txt_linear_weight, txt_linear_bias},
            {txt_linear_flat});
  const auto txt_gelu =
      bf16({batch * tokens, Krea2Config::kFeatures});
  operation(Opcode::Gelu, {txt_linear_flat}, {txt_gelu},
            {Attribute::u64(
                AttrKey::Approximation,
                static_cast<std::uint64_t>(GeluApproximation::Tanh))});
  const auto txt_output_weight = checkpoint(
      "txtmlp.3.weight", {Krea2Config::kFeatures, Krea2Config::kFeatures});
  const auto txt_output_bias =
      checkpoint("txtmlp.3.bias", {Krea2Config::kFeatures});
  const auto conditioning_flat =
      bf16({batch * tokens, Krea2Config::kFeatures});
  operation(Opcode::Linear, {txt_gelu, txt_output_weight, txt_output_bias},
            {conditioning_flat});
  build.conditioning_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {batch, tokens, Krea2Config::kFeatures});
  operation(Opcode::Reshape, {conditioning_flat},
            {build.conditioning_output});

  verify(build.program);
  return build;
}

Krea2DenoiserBuild make_krea2_denoiser(const Krea2Config &config,
                                       bool capture_block_outputs) {
  using namespace ir;
  const auto architecture = inspect_krea2_architecture(config);
  if (architecture.combined_tokens != architecture.padded_tokens)
    fail("the admitted Krea 2 denoiser currently requires naturally aligned "
         "text+image geometry; generic sequence padding is not yet lowered");

  Krea2DenoiserBuild build;
  build.config = config;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto checkpoint_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed
                                 : TensorRole::Internal));
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto bf16 = [&](std::vector<std::uint64_t> dims,
                        std::uint32_t roles = TensorRole::Internal) {
    return add_tensor(DType::BF16, roles, std::move(dims));
  };
  const auto operation = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                             std::vector<std::uint32_t> outputs,
                             std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto checkpoint = [&](std::string name,
                              std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(DType::BF16, checkpoint_roles, std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.push_back(std::move(name));
    return id;
  };

  const auto batch = config.batch;
  const auto image_tokens = architecture.image_tokens;
  const auto sequence = architecture.padded_tokens;
  const auto features = Krea2Config::kFeatures;
  const auto patch_features = architecture.patch_input_dim;
  const std::vector<std::uint64_t> sequence_shape{batch, sequence, features};

  build.image_tokens_input = add_tensor(
      DType::BF16, TensorRole::Input,
      {batch, image_tokens, patch_features});
  build.context_input = add_tensor(
      DType::BF16, TensorRole::Input,
      {batch, config.text_tokens, features});
  build.timestep_input =
      add_tensor(DType::BF16, TensorRole::Input, {batch});
  build.positions_input = add_tensor(
      DType::F32, TensorRole::Input, {batch, sequence, 3U});
  build.validity_mask_input =
      add_tensor(DType::Bool, TensorRole::Input, {batch, sequence});
  build.rotary_pair_axes = add_tensor(
      DType::I32, TensorRole::Constant, {Krea2Config::kHeadDim / 2U});
  build.rotary_pair_indices = add_tensor(
      DType::I32, TensorRole::Constant, {Krea2Config::kHeadDim / 2U});
  build.rotary_axis_dims =
      add_tensor(DType::I32, TensorRole::Constant, {3U});

  const auto image_flat = bf16({batch * image_tokens, patch_features});
  operation(Opcode::Reshape, {build.image_tokens_input}, {image_flat});
  const auto first_weight =
      checkpoint("first.weight", {features, patch_features});
  const auto first_bias = checkpoint("first.bias", {features});
  const auto projected_image_flat =
      bf16({batch * image_tokens, features});
  operation(Opcode::Linear, {image_flat, first_weight, first_bias},
            {projected_image_flat});
  build.projected_image = bf16(
      {batch, image_tokens, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Reshape, {projected_image_flat}, {build.projected_image});

  const auto timestep_f32 =
      add_tensor(DType::F32, TensorRole::Internal, {batch});
  operation(Opcode::Cast, {build.timestep_input}, {timestep_f32});
  const auto timestep_embedding_f32 = add_tensor(
      DType::F32, TensorRole::Internal,
      {batch, Krea2Config::kTimestepDim});
  operation(Opcode::SinusoidalTimestep, {timestep_f32},
            {timestep_embedding_f32},
            {Attribute::boolean(AttrKey::FlipSinToCos, true),
             Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
             Attribute::f64(AttrKey::Scale, 1000.0),
             Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  build.timestep_embedding = bf16(
      {batch, Krea2Config::kTimestepDim},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Cast, {timestep_embedding_f32},
            {build.timestep_embedding});
  const auto tmlp0_weight = checkpoint(
      "tmlp.0.weight", {features, Krea2Config::kTimestepDim});
  const auto tmlp0_bias = checkpoint("tmlp.0.bias", {features});
  build.timestep_first_linear = bf16(
      {batch, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Linear,
            {build.timestep_embedding, tmlp0_weight, tmlp0_bias},
            {build.timestep_first_linear},
            {Attribute::u64(
                AttrKey::LinearBiasMode,
                static_cast<std::uint64_t>(LinearBiasMode::Addmm))});
  const auto gelu_attributes = std::vector<Attribute>{Attribute::u64(
      AttrKey::Approximation,
      static_cast<std::uint64_t>(GeluApproximation::Tanh))};
  build.timestep_first_activation = bf16(
      {batch, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Gelu, {build.timestep_first_linear},
            {build.timestep_first_activation}, gelu_attributes);
  const auto tmlp2_weight =
      checkpoint("tmlp.2.weight", {features, features});
  const auto tmlp2_bias = checkpoint("tmlp.2.bias", {features});
  build.timestep_output = bf16(
      {batch, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Linear,
            {build.timestep_first_activation, tmlp2_weight, tmlp2_bias},
            {build.timestep_output},
            {Attribute::u64(
                AttrKey::LinearBiasMode,
                static_cast<std::uint64_t>(LinearBiasMode::Addmm))});
  build.timestep_projection_activation = bf16(
      {batch, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Gelu, {build.timestep_output},
            {build.timestep_projection_activation},
            gelu_attributes);
  const auto tproj_weight =
      checkpoint("tproj.1.weight", {6U * features, features});
  const auto tproj_bias = checkpoint("tproj.1.bias", {6U * features});
  build.modulation_output = bf16(
      {batch, 6U * features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Linear,
            {build.timestep_projection_activation, tproj_weight, tproj_bias},
            {build.modulation_output},
            {Attribute::u64(
                AttrKey::LinearBiasMode,
                static_cast<std::uint64_t>(LinearBiasMode::Addmm))});

  auto combined = bf16(sequence_shape);
  operation(Opcode::Concat, {build.context_input, build.projected_image},
            {combined}, {Attribute::u64(AttrKey::Axis, 1U)});

  build.provenance.frontend = "krea2";
  build.provenance.creator = std::string(kKrea2Creator);
  build.provenance.creator_revision = std::string(kKrea2CreatorRevision);
  // Operations lowered so far are the patch projection, the timestep tower,
  // and the sequence concat; classify them from the tensors they produced.
  {
    std::string section = "patch.first";
    std::string module = "first";
    for (const auto &op : program.operations) {
      const auto produces = [&](std::uint32_t id) {
        return std::find(op.outputs.begin(), op.outputs.end(), id) !=
               op.outputs.end();
      };
      if (op.opcode == Opcode::Concat) {
        section = "sequence.concat";
        module = "denoiser";
      }
      build.provenance.records.push_back({op.id, module, -1, section});
      if (produces(build.projected_image)) {
        section = "timestep.tower";
        module = "tmlp/tproj";
      }
    }
  }

  for (std::uint64_t layer = 0U; layer < Krea2Config::kLayers; ++layer) {
    const auto block = make_krea2_block(config, layer, false);
    const auto layer_first_operation = next_operation;
    for (const auto &record : block.provenance.records)
      build.provenance.records.push_back(
          {layer_first_operation + (record.operation_id - 1U),
           record.creator_module, record.block, record.semantic_tag});
    std::unordered_map<std::uint32_t, std::uint32_t> remap{
        {block.sequence_input, combined},
        {block.modulation_input, build.modulation_output},
        {block.positions_input, build.positions_input},
        {block.validity_mask_input, build.validity_mask_input},
        {block.rotary_pair_axes, build.rotary_pair_axes},
        {block.rotary_pair_indices, build.rotary_pair_indices},
        {block.rotary_axis_dims, build.rotary_axis_dims},
    };
    std::unordered_map<std::uint32_t, std::string> checkpoint_names;
    for (std::size_t index = 0U; index < block.checkpoint_tensors.size();
         ++index)
      checkpoint_names.emplace(block.checkpoint_tensors[index],
                               block.checkpoint_names[index]);
    for (const auto &description : block.program.tensors) {
      if (remap.contains(description.id))
        continue;
      const auto found = checkpoint_names.find(description.id);
      if (found != checkpoint_names.end()) {
        remap.emplace(description.id,
                      checkpoint(found->second, description.dims));
        continue;
      }
      auto roles = static_cast<std::uint32_t>(TensorRole::Internal);
      if (capture_block_outputs && description.id == block.final_output)
        roles = TensorRole::Output;
      remap.emplace(description.id,
                    add_tensor(description.dtype, roles, description.dims));
    }
    for (const auto &source : block.program.operations) {
      auto inputs = source.inputs;
      auto outputs = source.outputs;
      for (auto &id : inputs)
        id = remap.at(id);
      for (auto &id : outputs)
        id = remap.at(id);
      operation(source.opcode, std::move(inputs), std::move(outputs),
                source.attributes);
    }
    combined = remap.at(block.final_output);
    build.block_outputs.push_back(combined);
  }
  const auto after_blocks_operation = next_operation;

  const auto combined_flat = bf16({batch * sequence, features});
  operation(Opcode::Reshape, {combined}, {combined_flat});
  const auto last_norm_weight = checkpoint("last.norm.scale", {features});
  const auto last_modulation =
      checkpoint("last.modulation.lin", {2U, features});
  // The official compiled last layer keeps RMSNorm and its shared scale/shift
  // modulation in one F32 region. Materializing the intermediate BF16 tensors
  // changes the released velocity, so encode the generic fused contract.
  build.last_modulated = bf16(
      {batch * sequence, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(
      Opcode::RmsNormModulate,
      {combined_flat, last_norm_weight, build.timestep_output, last_modulation},
      {build.last_modulated},
      {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
       Attribute::f64(AttrKey::WeightOffset, 1.0),
       Attribute::u64(AttrKey::BlockSize, 512U),
       Attribute::u64(AttrKey::ReductionTileSize, 8192U),
       Attribute::u64(
           AttrKey::ModulationLayout,
           static_cast<std::uint64_t>(ModulationLayout::SharedVectorDelta))});
  const auto last_weight =
      checkpoint("last.linear.weight", {architecture.patch_output_dim, features});
  const auto last_bias =
      checkpoint("last.linear.bias", {architecture.patch_output_dim});
  const auto final_flat =
      bf16({batch * sequence, architecture.patch_output_dim});
  operation(Opcode::Linear, {build.last_modulated, last_weight, last_bias},
            {final_flat},
            {Attribute::u64(AttrKey::WorkspaceLimitBytes,
                            1U * 1024U * 1024U)});
  const auto final_sequence =
      bf16({batch, sequence, architecture.patch_output_dim});
  operation(Opcode::Reshape, {final_flat}, {final_sequence});
  build.velocity_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {batch, image_tokens, architecture.patch_output_dim});
  operation(Opcode::Slice, {final_sequence}, {build.velocity_output},
            {Attribute::u64(AttrKey::Axis, 1U),
             Attribute::u64(AttrKey::Start, config.text_tokens)});
  for (const auto &op : program.operations)
    if (op.id >= after_blocks_operation)
      build.provenance.records.push_back({op.id, "last", -1, "final.head"});
  for (std::size_t index = 0U; index < build.checkpoint_tensors.size(); ++index)
    build.provenance.weight_names.emplace_back(build.checkpoint_tensors[index],
                                               build.checkpoint_names[index]);

  verify(program);
  return build;
}

Krea2TurboExecutionBuild
make_krea2_turbo_execution(const Krea2Config &config, std::uint32_t steps,
                           bool capture_trajectory,
                           const std::vector<std::uint32_t>
                               &reusable_resident_constants) {
  using namespace ir;
  if (steps == 0U)
    fail("Krea 2 Turbo execution requires at least one evaluation");

  const auto denoiser = make_krea2_denoiser(config, false);
  Krea2TurboExecutionBuild build;
  build.config = config;
  build.initial_image_input = denoiser.image_tokens_input;
  build.context_input = denoiser.context_input;
  build.positions_input = denoiser.positions_input;
  build.validity_mask_input = denoiser.validity_mask_input;
  build.rotary_pair_axes = denoiser.rotary_pair_axes;
  build.rotary_pair_indices = denoiser.rotary_pair_indices;
  build.rotary_axis_dims = denoiser.rotary_axis_dims;
  build.checkpoint_tensors = denoiser.checkpoint_tensors;
  build.checkpoint_names = denoiser.checkpoint_names;
  const std::unordered_set<std::uint32_t> reusable(
      reusable_resident_constants.begin(),
      reusable_resident_constants.end());

  const auto shared = [&](std::uint32_t id) {
    return id == denoiser.image_tokens_input ||
           id == denoiser.context_input || id == denoiser.positions_input ||
           id == denoiser.validity_mask_input;
  };
  for (const auto &tensor : denoiser.program.tensors) {
    if (tensor.has_role(TensorRole::Constant)) {
      if (tensor.has_role(TensorRole::Streamed) &&
          !reusable.contains(tensor.id))
        continue;
      build.program.tensors.push_back(tensor);
    } else if (shared(tensor.id)) {
      build.program.tensors.push_back(tensor);
    }
  }

  auto next_tensor = std::uint32_t{1U};
  for (const auto &tensor : denoiser.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  auto next_operation = std::uint32_t{1U};
  for (const auto &operation : denoiser.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              const std::vector<std::uint64_t> &dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, dims});
    return id;
  };

  auto sample = denoiser.image_tokens_input;
  for (std::uint32_t step = 0U; step < steps; ++step) {
    std::unordered_map<std::uint32_t, std::uint32_t> remap;
    for (const auto &tensor : denoiser.program.tensors) {
      if (tensor.has_role(TensorRole::Constant)) {
        if (!tensor.has_role(TensorRole::Streamed) ||
            reusable.contains(tensor.id)) {
          remap.emplace(tensor.id, tensor.id);
        } else {
          const auto id = add_tensor(tensor.dtype, tensor.roles, tensor.dims);
          build.constant_sources.emplace_back(id, tensor.id);
          remap.emplace(tensor.id, id);
        }
        continue;
      }
      if (tensor.id == denoiser.context_input ||
          tensor.id == denoiser.positions_input ||
          tensor.id == denoiser.validity_mask_input) {
        remap.emplace(tensor.id, tensor.id);
        continue;
      }
      if (tensor.id == denoiser.image_tokens_input) {
        remap.emplace(tensor.id, sample);
        continue;
      }
      if (tensor.id == denoiser.timestep_input) {
        const auto id = add_tensor(tensor.dtype, TensorRole::Input,
                                   tensor.dims);
        build.model_timestep_inputs.push_back(id);
        remap.emplace(tensor.id, id);
        continue;
      }
      auto roles = static_cast<std::uint32_t>(TensorRole::Internal);
      if (capture_trajectory && tensor.id == denoiser.velocity_output)
        roles = TensorRole::Output;
      const auto id = add_tensor(tensor.dtype, roles, tensor.dims);
      remap.emplace(tensor.id, id);
    }

    for (const auto &source : denoiser.program.operations) {
      auto inputs = source.inputs;
      auto outputs = source.outputs;
      for (auto &id : inputs)
        id = remap.at(id);
      for (auto &id : outputs)
        id = remap.at(id);
      build.program.operations.push_back(
          {next_operation++, source.opcode, std::move(inputs),
           std::move(outputs), source.attributes});
    }

    const auto current = add_tensor(DType::F32, TensorRole::Input, {1U});
    const auto next = add_tensor(DType::F32, TensorRole::Input, {1U});
    build.current_timestep_inputs.push_back(current);
    build.next_timestep_inputs.push_back(next);
    const auto output_roles =
        capture_trajectory || step + 1U == steps
            ? static_cast<std::uint32_t>(TensorRole::Output)
            : static_cast<std::uint32_t>(TensorRole::Internal);
    const auto image = add_tensor(
        DType::BF16, output_roles,
        denoiser.program.tensor(denoiser.image_tokens_input)->dims);
    const auto velocity = remap.at(denoiser.velocity_output);
    build.program.operations.push_back(
        {next_operation++, Opcode::EulerVelocityStep,
         {sample, velocity, current, next}, {image}, {}});
    build.velocity_outputs.push_back(velocity);
    build.image_outputs.push_back(image);
    sample = image;
  }
  build.final_image_output = sample;
  verify(build.program);
  return build;
}

} // namespace dif::frontend
