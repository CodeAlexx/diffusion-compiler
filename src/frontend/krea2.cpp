#include "dif/frontend/krea2.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>

namespace dif::frontend {

Krea2Architecture inspect_krea2_architecture(const Krea2Config &config) {
  constexpr auto alignment =
      Krea2Config::kVaeCompression * Krea2Config::kPatch;
  if (config.batch == 0U || config.width == 0U || config.height == 0U ||
      config.text_tokens == 0U || config.text_tokens > 512U ||
      config.width % alignment != 0U || config.height % alignment != 0U)
    fail("Krea 2 requires a positive batch, 1..512 text tokens, and image "
         "dimensions divisible by VAE compression * patch (16)");

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
                {embedding_bf16, tmlp0_weight, tmlp0_bias}, {tmlp0});
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
                {build.timestep_output});

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
                {build.modulation_output});

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
  const auto modulation_delta_batch =
      add_bf16({batch, 6U * features});
  add_operation(Opcode::BroadcastTo, {modulation_delta_2d},
                {modulation_delta_batch});
  build.modulated_parameters =
      add_bf16({batch, 6U * features}, true);
  add_operation(Opcode::Add,
                {build.modulation_input, modulation_delta_batch},
                {build.modulated_parameters});

  const auto modulation_chunk = [&](std::uint64_t chunk) {
    const auto sliced = add_bf16({batch, features});
    add_operation(Opcode::Slice, {build.modulated_parameters}, {sliced},
                  {Attribute::u64(AttrKey::Axis, 1U),
                   Attribute::u64(AttrKey::Start, chunk * features)});
    const auto expanded = add_bf16({batch, 1U, features});
    add_operation(Opcode::Reshape, {sliced}, {expanded});
    const auto broadcast = add_bf16(sequence_shape);
    add_operation(Opcode::BroadcastTo, {expanded}, {broadcast});
    return broadcast;
  };
  const auto prescale = modulation_chunk(0U);
  const auto preshift = modulation_chunk(1U);
  const auto pregate = modulation_chunk(2U);
  const auto postscale = modulation_chunk(3U);
  const auto postshift = modulation_chunk(4U);
  const auto postgate = modulation_chunk(5U);

  const auto ones = add_bf16(sequence_shape);
  add_operation(Opcode::Fill, {}, {ones},
                {Attribute::f64(AttrKey::Value, 1.0)});
  const auto prenorm_weight = add_checkpoint("prenorm.scale", {features});
  build.input_normalized = add_bf16(sequence_shape, true);
  add_operation(Opcode::RmsNorm,
                {build.sequence_input, prenorm_weight},
                {build.input_normalized},
                {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                 Attribute::f64(AttrKey::WeightOffset, 1.0),
                 Attribute::u64(AttrKey::BlockSize, 256U)});
  const auto prescale_plus_one = add_bf16(sequence_shape);
  add_operation(Opcode::Add, {ones, prescale}, {prescale_plus_one});
  const auto prenorm_scaled = add_bf16(sequence_shape);
  add_operation(Opcode::Multiply,
                {prescale_plus_one, build.input_normalized},
                {prenorm_scaled});
  build.attention_input = add_bf16(sequence_shape, true);
  add_operation(Opcode::Add, {prenorm_scaled, preshift},
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
  const auto norm_attributes =
      std::vector<Attribute>{Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                             Attribute::f64(AttrKey::WeightOffset, 1.0),
                             Attribute::u64(AttrKey::BlockSize, 128U)};
  add_operation(Opcode::RmsNorm, {q_shaped, qnorm}, {build.query},
                norm_attributes);
  add_operation(Opcode::RmsNorm, {k_shaped, knorm}, {build.key},
                norm_attributes);

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
                {post_normalized}, norm_attributes);
  const auto postscale_plus_one = add_bf16(sequence_shape);
  add_operation(Opcode::Add, {ones, postscale}, {postscale_plus_one});
  const auto postnorm_scaled = add_bf16(sequence_shape);
  add_operation(Opcode::Multiply, {postscale_plus_one, post_normalized},
                {postnorm_scaled});
  build.mlp_input = add_bf16(sequence_shape, true);
  add_operation(Opcode::Add, {postnorm_scaled, postshift}, {build.mlp_input});
  const auto mlp_input_flat = add_bf16({rows, features});
  add_operation(Opcode::Reshape, {build.mlp_input}, {mlp_input_flat});
  const auto mlp_gate_weight =
      add_checkpoint("mlp.gate.weight", {mlp, features});
  const auto mlp_up_weight =
      add_checkpoint("mlp.up.weight", {mlp, features});
  const auto mlp_down_weight =
      add_checkpoint("mlp.down.weight", {features, mlp});
  const auto mlp_gate = add_bf16({rows, mlp});
  const auto mlp_up = add_bf16({rows, mlp});
  add_operation(Opcode::Linear, {mlp_input_flat, mlp_gate_weight},
                {mlp_gate});
  add_operation(Opcode::Linear, {mlp_input_flat, mlp_up_weight}, {mlp_up});
  const auto mlp_gate_activated = add_bf16({rows, mlp});
  add_operation(Opcode::SiLU, {mlp_gate}, {mlp_gate_activated});
  build.mlp_activation = add_bf16({rows, mlp}, true);
  add_operation(Opcode::Multiply, {mlp_gate_activated, mlp_up},
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

Krea2TextFusionBuild make_krea2_text_fusion(bool capture_boundaries) {
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
                              std::uint32_t attention_bias) {
    const std::vector<std::uint64_t> shape{block_batch, block_tokens, hidden};
    const auto prenorm_weight =
        checkpoint(prefix + "prenorm.scale", {hidden});
    const auto prenorm = bf16(shape);
    operation(Opcode::RmsNorm, {input, prenorm_weight}, {prenorm},
              norm_attributes);
    const auto rows = block_batch * block_tokens;
    const auto prenorm_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {prenorm}, {prenorm_flat});
    const auto wq = checkpoint(prefix + "attn.wq.weight", {hidden, hidden});
    const auto wk = checkpoint(prefix + "attn.wk.weight", {hidden, hidden});
    const auto wv = checkpoint(prefix + "attn.wv.weight", {hidden, hidden});
    const auto q_flat = bf16({rows, hidden});
    const auto k_flat = bf16({rows, hidden});
    const auto v_flat = bf16({rows, hidden});
    operation(Opcode::Linear, {prenorm_flat, wq}, {q_flat});
    operation(Opcode::Linear, {prenorm_flat, wk}, {k_flat});
    operation(Opcode::Linear, {prenorm_flat, wv}, {v_flat});
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
    const auto qnorm = bf16({block_batch, block_tokens, heads, head_dim});
    const auto knorm = bf16({block_batch, block_tokens, heads, head_dim});
    operation(Opcode::RmsNorm, {q, qnorm_weight}, {qnorm}, norm_attributes);
    operation(Opcode::RmsNorm, {k, knorm_weight}, {knorm}, norm_attributes);
    const auto attended = bf16({block_batch, block_tokens, heads, head_dim});
    auto attention_inputs = std::vector<std::uint32_t>{qnorm, knorm, v};
    if (attention_bias != 0U)
      attention_inputs.push_back(attention_bias);
    operation(Opcode::Attention, std::move(attention_inputs), {attended},
              {Attribute::u64(AttrKey::KvHeads, heads),
               Attribute::u64(AttrKey::Implementation, 2U)});
    const auto attended_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {attended}, {attended_flat});
    const auto gate_weight =
        checkpoint(prefix + "attn.gate.weight", {hidden, hidden});
    const auto gate_logits = bf16({rows, hidden});
    operation(Opcode::Linear, {prenorm_flat, gate_weight}, {gate_logits});
    const auto gate = bf16({rows, hidden});
    operation(Opcode::Sigmoid, {gate_logits}, {gate});
    const auto gated = bf16({rows, hidden});
    operation(Opcode::Multiply, {attended_flat, gate}, {gated});
    const auto wo = checkpoint(prefix + "attn.wo.weight", {hidden, hidden});
    const auto projected_flat = bf16({rows, hidden});
    operation(Opcode::Linear, {gated, wo}, {projected_flat});
    const auto projected = bf16(shape);
    operation(Opcode::Reshape, {projected_flat}, {projected});
    const auto attention_residual = bf16(shape);
    operation(Opcode::Add, {input, projected}, {attention_residual});

    const auto postnorm_weight =
        checkpoint(prefix + "postnorm.scale", {hidden});
    const auto postnorm = bf16(shape);
    operation(Opcode::RmsNorm, {attention_residual, postnorm_weight},
              {postnorm}, norm_attributes);
    const auto postnorm_flat = bf16({rows, hidden});
    operation(Opcode::Reshape, {postnorm}, {postnorm_flat});
    const auto gate_mlp_weight =
        checkpoint(prefix + "mlp.gate.weight", {mlp, hidden});
    const auto up_mlp_weight =
        checkpoint(prefix + "mlp.up.weight", {mlp, hidden});
    const auto down_mlp_weight =
        checkpoint(prefix + "mlp.down.weight", {hidden, mlp});
    const auto gate_mlp = bf16({rows, mlp});
    const auto up_mlp = bf16({rows, mlp});
    operation(Opcode::Linear, {postnorm_flat, gate_mlp_weight}, {gate_mlp});
    operation(Opcode::Linear, {postnorm_flat, up_mlp_weight}, {up_mlp});
    const auto activated = bf16({rows, mlp});
    operation(Opcode::SiLU, {gate_mlp}, {activated});
    const auto multiplied = bf16({rows, mlp});
    operation(Opcode::Multiply, {activated, up_mlp}, {multiplied});
    const auto down_flat = bf16({rows, hidden});
    operation(Opcode::Linear, {multiplied, down_mlp_weight}, {down_flat});
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
        tokens, taps, 0U);

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
        batch, tokens, text_bias);

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
  const auto timestep_embedding = add_tensor(
      DType::F32, TensorRole::Internal,
      {batch, Krea2Config::kTimestepDim});
  operation(Opcode::SinusoidalTimestep, {timestep_f32}, {timestep_embedding},
            {Attribute::boolean(AttrKey::FlipSinToCos, true),
             Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
             Attribute::f64(AttrKey::Scale, 1000.0),
             Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  const auto timestep_embedding_bf16 =
      bf16({batch, Krea2Config::kTimestepDim});
  operation(Opcode::Cast, {timestep_embedding}, {timestep_embedding_bf16});
  const auto tmlp0_weight = checkpoint(
      "tmlp.0.weight", {features, Krea2Config::kTimestepDim});
  const auto tmlp0_bias = checkpoint("tmlp.0.bias", {features});
  const auto tmlp0 = bf16({batch, features});
  operation(Opcode::Linear,
            {timestep_embedding_bf16, tmlp0_weight, tmlp0_bias}, {tmlp0});
  const auto gelu_attributes = std::vector<Attribute>{Attribute::u64(
      AttrKey::Approximation,
      static_cast<std::uint64_t>(GeluApproximation::Tanh))};
  const auto tmlp0_activated = bf16({batch, features});
  operation(Opcode::Gelu, {tmlp0}, {tmlp0_activated}, gelu_attributes);
  const auto tmlp2_weight =
      checkpoint("tmlp.2.weight", {features, features});
  const auto tmlp2_bias = checkpoint("tmlp.2.bias", {features});
  build.timestep_output = bf16(
      {batch, features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Linear,
            {tmlp0_activated, tmlp2_weight, tmlp2_bias},
            {build.timestep_output});
  const auto tproj_activated = bf16({batch, features});
  operation(Opcode::Gelu, {build.timestep_output}, {tproj_activated},
            gelu_attributes);
  const auto tproj_weight =
      checkpoint("tproj.1.weight", {6U * features, features});
  const auto tproj_bias = checkpoint("tproj.1.bias", {6U * features});
  build.modulation_output = bf16(
      {batch, 6U * features},
      capture_block_outputs ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal));
  operation(Opcode::Linear, {tproj_activated, tproj_weight, tproj_bias},
            {build.modulation_output});

  auto combined = bf16(sequence_shape);
  operation(Opcode::Concat, {build.context_input, build.projected_image},
            {combined}, {Attribute::u64(AttrKey::Axis, 1U)});

  for (std::uint64_t layer = 0U; layer < Krea2Config::kLayers; ++layer) {
    const auto block = make_krea2_block(config, layer, false);
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

  const auto last_norm_weight = checkpoint("last.norm.scale", {features});
  const auto normalized = bf16(sequence_shape);
  operation(Opcode::RmsNorm, {combined, last_norm_weight}, {normalized},
            {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
             Attribute::f64(AttrKey::WeightOffset, 1.0),
             Attribute::u64(AttrKey::BlockSize, 256U)});
  const auto last_modulation =
      checkpoint("last.modulation.lin", {2U, features});
  const auto last_modulation_3d = bf16({1U, 2U, features});
  operation(Opcode::Reshape, {last_modulation}, {last_modulation_3d});
  const auto last_modulation_batch = bf16({batch, 2U, features});
  operation(Opcode::BroadcastTo, {last_modulation_3d},
            {last_modulation_batch});
  const auto timestep_3d = bf16({batch, 1U, features});
  operation(Opcode::Reshape, {build.timestep_output}, {timestep_3d});
  const auto timestep_two = bf16({batch, 2U, features});
  operation(Opcode::BroadcastTo, {timestep_3d}, {timestep_two});
  const auto last_parameters = bf16({batch, 2U, features});
  operation(Opcode::Add, {timestep_two, last_modulation_batch},
            {last_parameters});
  const auto scale = bf16({batch, 1U, features});
  const auto shift = bf16({batch, 1U, features});
  operation(Opcode::Slice, {last_parameters}, {scale},
            {Attribute::u64(AttrKey::Axis, 1U),
             Attribute::u64(AttrKey::Start, 0U)});
  operation(Opcode::Slice, {last_parameters}, {shift},
            {Attribute::u64(AttrKey::Axis, 1U),
             Attribute::u64(AttrKey::Start, 1U)});
  const auto scale_sequence = bf16(sequence_shape);
  const auto shift_sequence = bf16(sequence_shape);
  operation(Opcode::BroadcastTo, {scale}, {scale_sequence});
  operation(Opcode::BroadcastTo, {shift}, {shift_sequence});
  const auto ones = bf16(sequence_shape);
  operation(Opcode::Fill, {}, {ones},
            {Attribute::f64(AttrKey::Value, 1.0)});
  const auto scale_plus_one = bf16(sequence_shape);
  operation(Opcode::Add, {ones, scale_sequence}, {scale_plus_one});
  const auto scaled = bf16(sequence_shape);
  operation(Opcode::Multiply, {scale_plus_one, normalized}, {scaled});
  const auto modulated = bf16(sequence_shape);
  operation(Opcode::Add, {scaled, shift_sequence}, {modulated});
  const auto modulated_flat = bf16({batch * sequence, features});
  operation(Opcode::Reshape, {modulated}, {modulated_flat});
  const auto last_weight =
      checkpoint("last.linear.weight", {architecture.patch_output_dim, features});
  const auto last_bias =
      checkpoint("last.linear.bias", {architecture.patch_output_dim});
  const auto final_flat =
      bf16({batch * sequence, architecture.patch_output_dim});
  operation(Opcode::Linear, {modulated_flat, last_weight, last_bias},
            {final_flat});
  const auto final_sequence =
      bf16({batch, sequence, architecture.patch_output_dim});
  operation(Opcode::Reshape, {final_flat}, {final_sequence});
  build.velocity_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {batch, image_tokens, architecture.patch_output_dim});
  operation(Opcode::Slice, {final_sequence}, {build.velocity_output},
            {Attribute::u64(AttrKey::Axis, 1U),
             Attribute::u64(AttrKey::Start, config.text_tokens)});

  verify(program);
  return build;
}

} // namespace dif::frontend
