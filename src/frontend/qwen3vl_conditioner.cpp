#include "dif/frontend/qwen3vl_conditioner.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

using ir::AttrKey;
using ir::Attribute;
using ir::DType;
using ir::Opcode;
using ir::TensorRole;

runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                           const std::vector<float> &values) {
  runtime::Tensor tensor{DType::F32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

struct Builder {
  Qwen3VlConditionerBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t add_tensor(DType dtype, std::uint32_t roles,
                           std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }
  // Activations are BF16 end to end, matching the checkpoint's storage dtype
  // and the repo-wide contract (F32 lives inside kernels, never as a storage
  // round trip a caller can take by accident).
  std::uint32_t internal(std::vector<std::uint64_t> dims) {
    return add_tensor(DType::BF16,
                      static_cast<std::uint32_t>(TensorRole::Internal),
                      std::move(dims));
  }
  // Checkpoint weights are Streamed: the 50 layers are ~0.98 GiB each, so
  // they ride the existing plan-slot prefetcher instead of going resident.
  std::uint32_t streamed(std::string name, std::vector<std::uint64_t> dims) {
    const auto id =
        add_tensor(DType::BF16,
                   static_cast<std::uint32_t>(TensorRole::Constant) |
                       static_cast<std::uint32_t>(TensorRole::Streamed),
                   std::move(dims));
    build.bindings.push_back({id, std::move(name)});
    return id;
  }
  std::uint32_t generated(runtime::Tensor tensor) {
    const auto id = add_tensor(
        tensor.dtype, static_cast<std::uint32_t>(TensorRole::Constant),
        tensor.dims);
    build.generated_constants.emplace(id, std::move(tensor));
    return id;
  }
  void operation(Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  }
};

std::string layer_prefix(std::string_view checkpoint_prefix,
                         std::uint64_t layer) {
  return std::string(checkpoint_prefix) + "layers." +
         std::to_string(layer) + ".";
}

} // namespace

Qwen3VlConditionerBuild
build_qwen3vl_conditioner_program(std::uint64_t sequence_length,
                                  const Qwen3VlConditionerConfig &config) {
  if (sequence_length == 0U)
    fail("qwen3-vl conditioner requires a positive sequence length");
  if (config.executed_layers == 0U || config.attention_heads == 0U ||
      config.key_value_heads == 0U || config.head_dim == 0U ||
      config.hidden_size == 0U || config.intermediate_size == 0U ||
      config.vocabulary == 0U)
    fail("qwen3-vl conditioner configuration must be positive");
  if (config.attention_heads % config.key_value_heads != 0U)
    fail("qwen3-vl attention heads must be a multiple of key/value heads");
  // NOTE: heads * head_dim need NOT equal hidden_size. Qwen3-VL-32B runs a
  // 8192-wide attention projection over a 5120-wide residual stream, which is
  // exactly why q_proj is [8192,5120] and o_proj is [5120,8192].
  if ((config.head_dim % 2U) != 0U)
    fail("qwen3-vl rotary requires an even head dimension");
  if (config.attention_implementation != 1U &&
      config.attention_implementation != 2U)
    fail("qwen3-vl attention implementation must be 1 (generated) or 2 (cuDNN)");
  if (config.use_attention_mask && config.attention_implementation != 2U)
    fail("masked qwen3-vl attention requires the exact cuDNN implementation");
  if (config.dynamic_position_ids != config.use_attention_mask)
    fail("qwen3-vl dynamic position ids and attention masking must be enabled together");
  if (config.checkpoint_prefix.empty())
    fail("qwen3-vl checkpoint prefix must not be empty");
  for (const auto hidden_state : config.selected_hidden_states)
    if (hidden_state == 0U || hidden_state > config.executed_layers)
      fail("selected qwen3-vl hidden state is outside the executed layer range");
  if (!config.selected_hidden_states.empty()) {
    if (config.output_sequence_length == 0U ||
        config.output_slice_start > sequence_length ||
        config.output_sequence_length >
            sequence_length - config.output_slice_start)
      fail("qwen3-vl selected hidden-state slice is outside the input sequence");
    for (std::size_t index = 1; index < config.selected_hidden_states.size();
         ++index)
      if (config.selected_hidden_states[index - 1U] >=
          config.selected_hidden_states[index])
        fail("selected qwen3-vl hidden states must be strictly increasing");
  }
  if (config.vision_token_count != 0U) {
    if (config.deepstack_language_layers.size() != 3U ||
        !std::is_sorted(config.deepstack_language_layers.begin(),
                        config.deepstack_language_layers.end()) ||
        config.deepstack_language_layers.back() >= config.executed_layers)
      fail("qwen3-vl multimodal conditioning requires three ordered in-range deepstack layers");
  }

  const auto sequence = sequence_length;
  const auto hidden = config.hidden_size;
  const auto heads = config.attention_heads;
  const auto kv_heads = config.key_value_heads;
  const auto head_dim = config.head_dim;
  const auto intermediate = config.intermediate_size;
  const auto query_width = heads * head_dim;
  const auto kv_width = kv_heads * head_dim;
  const auto half_dim = head_dim / 2U;

  Builder builder;
  auto &build = builder.build;

  // ---- inputs -------------------------------------------------------------
  build.token_ids_input_id = builder.add_tensor(
      DType::I32, static_cast<std::uint32_t>(TensorRole::Input), {sequence});
  if (config.vision_token_count != 0U) {
    build.vision_embeddings_input_id = builder.add_tensor(
        DType::BF16, static_cast<std::uint32_t>(TensorRole::Input),
        {config.vision_token_count, hidden});
    build.vision_destination_map_input_id = builder.add_tensor(
        DType::I32, static_cast<std::uint32_t>(TensorRole::Input), {sequence});
    build.visual_positions_input_id = builder.add_tensor(
        DType::I32, static_cast<std::uint32_t>(TensorRole::Input),
        {config.vision_token_count});
    for (std::size_t index = 0U;
         index < config.deepstack_language_layers.size(); ++index)
      build.vision_deepstack_input_ids.push_back(builder.add_tensor(
          DType::BF16, static_cast<std::uint32_t>(TensorRole::Input),
          {config.vision_token_count, hidden}));
  }
  std::uint32_t attention_bias_id = 0U;
  if (config.use_attention_mask) {
    build.attention_mask_input_id = builder.add_tensor(
        DType::Bool, static_cast<std::uint32_t>(TensorRole::Input),
        {1U, sequence});
    attention_bias_id = builder.internal({1U, 1U, sequence, sequence});
    builder.operation(Opcode::BooleanMaskToBias,
                      {build.attention_mask_input_id}, {attention_bias_id},
                      {Attribute::boolean(AttrKey::MaskQueries,
                                          config.mask_padding_queries)});
  }

  // ---- rotary tables ------------------------------------------------------
  // Text-only prompts collapse Qwen3-VL's 3-axis MRoPE to ordinary 1-D RoPE:
  // every modality axis carries the same sequential positions, so one axis of
  // positions 0..S-1 reproduces it exactly (plan §1, settled by the depth-1
  // parity gate).
  std::vector<float> positions(sequence);
  for (std::uint64_t index = 0U; index < sequence; ++index)
    positions[index] = static_cast<float>(index);
  std::vector<float> inverse_frequency(half_dim);
  for (std::uint64_t index = 0U; index < half_dim; ++index)
    inverse_frequency[index] = static_cast<float>(
        1.0 / std::pow(config.rope_theta,
                       static_cast<double>(2U * index) /
                           static_cast<double>(head_dim)));
  const auto positions_id = config.dynamic_position_ids
                                ? (build.position_ids_input_id =
                                       builder.add_tensor(
                                           DType::F32,
                                           static_cast<std::uint32_t>(
                                               TensorRole::Input),
                                           {sequence, 1U}))
                                : builder.generated(
                                      f32_tensor({sequence, 1U}, positions));
  const auto inverse_frequency_id =
      builder.generated(f32_tensor({half_dim}, inverse_frequency));
  const auto cosine_id = builder.internal({sequence, head_dim});
  const auto sine_id = builder.internal({sequence, head_dim});
  builder.operation(Opcode::RotaryPosition, {positions_id, inverse_frequency_id},
                    {cosine_id, sine_id});
  std::uint32_t rope_cosine_id = 0U;
  std::uint32_t rope_sine_id = 0U;
  if (!config.qk_norm) {
    // Plain RoPE needs F32 per-pair tables [1,S,half_dim]; RotaryFrequency
    // produces them from the same positions with angle = pos * theta^(-2i/D),
    // the formula behind inverse_frequency above.
    const auto i32_tensor = [](std::vector<std::uint64_t> dims,
                               const std::vector<std::int32_t> &values) {
      std::vector<std::uint8_t> bytes(values.size() * 4U);
      std::memcpy(bytes.data(), values.data(), bytes.size());
      return runtime::Tensor{DType::I32, std::move(dims), std::move(bytes)};
    };
    std::vector<std::int32_t> pair_axes(half_dim, 0);
    std::vector<std::int32_t> pair_indices(half_dim);
    for (std::uint64_t index = 0U; index < half_dim; ++index)
      pair_indices[index] = static_cast<std::int32_t>(index);
    const auto pair_axes_id = builder.generated(i32_tensor({half_dim}, pair_axes));
    const auto pair_indices_id =
        builder.generated(i32_tensor({half_dim}, pair_indices));
    const auto axis_dims_id = builder.generated(
        i32_tensor({1U}, {static_cast<std::int32_t>(head_dim)}));
    const auto batched_positions = builder.add_tensor(
        DType::F32, static_cast<std::uint32_t>(TensorRole::Internal),
        {1U, sequence, 1U});
    builder.operation(Opcode::Reshape, {positions_id}, {batched_positions});
    rope_cosine_id = builder.add_tensor(
        DType::F32, static_cast<std::uint32_t>(TensorRole::Internal),
        {1U, sequence, half_dim});
    rope_sine_id = builder.add_tensor(
        DType::F32, static_cast<std::uint32_t>(TensorRole::Internal),
        {1U, sequence, half_dim});
    builder.operation(Opcode::RotaryFrequency,
                      {batched_positions, pair_axes_id, pair_indices_id,
                       axis_dims_id},
                      {rope_cosine_id, rope_sine_id},
                      {Attribute::f64(AttrKey::Theta, config.rope_theta)});
  }

  // ---- embedding ----------------------------------------------------------
  const auto embedding_id = builder.streamed(
      config.checkpoint_prefix + "embed_tokens.weight",
      {config.vocabulary, hidden});
  auto residual_id = builder.internal({sequence, hidden});
  builder.operation(Opcode::GatherRows, {embedding_id, build.token_ids_input_id},
                    {residual_id});
  if (config.vision_token_count != 0U) {
    const auto spliced = builder.internal({sequence, hidden});
    builder.operation(Opcode::IndexedUpdateRows,
                      {residual_id, build.vision_embeddings_input_id,
                       build.vision_destination_map_input_id},
                      {spliced});
    residual_id = spliced;
  }

  const auto capture = [&](std::uint64_t layer, std::string name,
                           std::uint32_t tensor_id) {
    if (layer != 0U || !config.capture_first_layer_boundaries)
      return;
    const auto tensor = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &candidate) { return candidate.id == tensor_id; });
    if (tensor == build.program.tensors.end())
      fail("qwen3-vl boundary capture lost its tensor");
    tensor->roles |= static_cast<std::uint32_t>(TensorRole::Output);
    build.first_layer_boundaries.emplace_back(std::move(name), tensor_id);
  };
  if (config.capture_first_layer_boundaries) {
    const auto tensor = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &candidate) { return candidate.id == residual_id; });
    if (tensor == build.program.tensors.end())
      fail("qwen3-vl embedding capture lost its tensor");
    tensor->roles |= static_cast<std::uint32_t>(TensorRole::Output);
    build.first_layer_boundaries.emplace_back("embedding", residual_id);
  }

  const auto rms_attributes = [&](void) {
    return std::vector<Attribute>{
        Attribute::f64(AttrKey::Epsilon, config.rms_norm_epsilon)};
  };

  // ---- layers 0 .. executed_layers-1 --------------------------------------
  for (std::uint64_t layer = 0U; layer < config.executed_layers; ++layer) {
    const auto prefix = layer_prefix(config.checkpoint_prefix, layer);

    // Attention block.
    const auto input_norm_weight_id =
        builder.streamed(prefix + "input_layernorm.weight", {hidden});
    const auto normed_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::RmsNorm, {residual_id, input_norm_weight_id},
                      {normed_id}, rms_attributes());
    capture(layer, "input_norm", normed_id);

    // Linear flattens trailing output dims, so the projections write the
    // head-split shapes their consumers need without any reshape opcode.
    const auto query_weight_id =
        builder.streamed(prefix + "self_attn.q_proj.weight", {query_width, hidden});
    const auto query_id = builder.internal({sequence, heads, head_dim});
    builder.operation(Opcode::Linear, {normed_id, query_weight_id}, {query_id});
    capture(layer, "q_proj", query_id);
    const auto key_weight_id =
        builder.streamed(prefix + "self_attn.k_proj.weight", {kv_width, hidden});
    const auto key_id = builder.internal({sequence, kv_heads, head_dim});
    builder.operation(Opcode::Linear, {normed_id, key_weight_id}, {key_id});
    capture(layer, "k_proj", key_id);
    const auto value_weight_id =
        builder.streamed(prefix + "self_attn.v_proj.weight", {kv_width, hidden});
    const auto value_id = builder.internal({sequence, kv_heads, head_dim});
    builder.operation(Opcode::Linear, {normed_id, value_weight_id}, {value_id});
    capture(layer, "v_proj", value_id);
    build.linear_operations += 3U;

    std::uint32_t rotated_query_id = 0U;
    std::uint32_t rotated_key_id = 0U;
    if (config.qk_norm) {
      // Qwen3 per-head QK RMSNorm followed by full-width rotate-half RoPE:
      // exactly QkNormPartialRope with RotaryDim == head_dim. Every shape
      // attribute is stamped so no consumer can resolve a different default.
      const auto query_norm_weight_id =
          builder.streamed(prefix + "self_attn.q_norm.weight", {head_dim});
      rotated_query_id = builder.internal({sequence, heads, head_dim});
      builder.operation(
          Opcode::QkNormPartialRope,
          {query_id, query_norm_weight_id, cosine_id, sine_id},
          {rotated_query_id},
          {Attribute::u64(AttrKey::Heads, heads),
           Attribute::u64(AttrKey::HeadDim, head_dim),
           Attribute::u64(AttrKey::RotaryDim, head_dim),
           Attribute::f64(AttrKey::Epsilon, config.rms_norm_epsilon)});
      capture(layer, "rotated_q", rotated_query_id);
      const auto key_norm_weight_id =
          builder.streamed(prefix + "self_attn.k_norm.weight", {head_dim});
      rotated_key_id = builder.internal({sequence, kv_heads, head_dim});
      builder.operation(
          Opcode::QkNormPartialRope,
          {key_id, key_norm_weight_id, cosine_id, sine_id}, {rotated_key_id},
          {Attribute::u64(AttrKey::Heads, kv_heads),
           Attribute::u64(AttrKey::HeadDim, head_dim),
           Attribute::u64(AttrKey::RotaryDim, head_dim),
           Attribute::f64(AttrKey::Epsilon, config.rms_norm_epsilon)});
      capture(layer, "rotated_k", rotated_key_id);
    } else {
      // No QK-norm (Mistral/Llama): plain full-width rotate-half RoPE. The
      // RotaryPosition table duplicates each angle over both halves, so its
      // first half_dim columns are exactly the per-pair cos/sin RotaryApply
      // wants; the 4D views alias the [S,H,D] projections.
      const auto rotate = [&](std::uint32_t input_id, std::uint64_t head_count,
                              const char *name) {
        const auto batched = builder.internal({1U, sequence, head_count, head_dim});
        builder.operation(Opcode::Reshape, {input_id}, {batched});
        const auto rotated = builder.internal({1U, sequence, head_count, head_dim});
        builder.operation(
            Opcode::RotaryApply, {batched, rope_cosine_id, rope_sine_id},
            {rotated},
            {Attribute::u64(AttrKey::RotaryLayout,
                            static_cast<std::uint64_t>(ir::RotaryLayout::HalfSplit))});
        const auto flat = builder.internal({sequence, head_count, head_dim});
        builder.operation(Opcode::Reshape, {rotated}, {flat});
        capture(layer, name, flat);
        return flat;
      };
      rotated_query_id = rotate(query_id, heads, "rotated_q");
      rotated_key_id = rotate(key_id, kv_heads, "rotated_k");
    }

    // Causal grouped-query attention: K/V keep their 8 heads and the KvHeads
    // attribute maps query head h to kv head h/(H/KvHeads) — no materialized
    // repeat of K/V.
    std::uint32_t attention_id = 0U;
    if (config.use_attention_mask) {
      const auto batched_query = builder.internal({1U, sequence, heads, head_dim});
      const auto batched_key =
          builder.internal({1U, sequence, kv_heads, head_dim});
      const auto batched_value =
          builder.internal({1U, sequence, kv_heads, head_dim});
      builder.operation(Opcode::Reshape, {rotated_query_id}, {batched_query});
      builder.operation(Opcode::Reshape, {rotated_key_id}, {batched_key});
      builder.operation(Opcode::Reshape, {value_id}, {batched_value});
      const auto batched_attention =
          builder.internal({1U, sequence, heads, head_dim});
      builder.operation(
          Opcode::Attention,
          {batched_query, batched_key, batched_value, attention_bias_id},
          {batched_attention},
          {Attribute::boolean(AttrKey::Causal, true),
           Attribute::u64(AttrKey::KvHeads, kv_heads),
           Attribute::u64(AttrKey::Implementation,
                          config.attention_implementation)});
      attention_id = builder.internal({sequence, heads, head_dim});
      builder.operation(Opcode::Reshape, {batched_attention}, {attention_id});
    } else {
      attention_id = builder.internal({sequence, heads, head_dim});
      builder.operation(Opcode::Attention,
                        {rotated_query_id, rotated_key_id, value_id},
                        {attention_id},
                        {Attribute::boolean(AttrKey::Causal, true),
                         Attribute::u64(AttrKey::KvHeads, kv_heads),
                         Attribute::u64(AttrKey::Implementation,
                                        config.attention_implementation)});
    }
    build.attention_operations += 1U;
    capture(layer, "attention", attention_id);

    const auto output_weight_id =
        builder.streamed(prefix + "self_attn.o_proj.weight", {hidden, query_width});
    const auto projected_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::Linear, {attention_id, output_weight_id},
                      {projected_id});
    capture(layer, "attention_projected", projected_id);
    build.linear_operations += 1U;
    const auto attention_residual_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::Add, {residual_id, projected_id},
                      {attention_residual_id});
    capture(layer, "attention_residual", attention_residual_id);
    residual_id = attention_residual_id;

    // Feed-forward block. gate_proj and up_proj are separate checkpoint
    // tensors, so the SwiGLU is expressed as SiLU * up over generic opcodes
    // rather than packing a 26 GiB derived [2I,H] weight set; the merged
    // elementwise region fuser can collapse the pair into one launch when a
    // candidate enables it.
    const auto post_norm_weight_id =
        builder.streamed(prefix + "post_attention_layernorm.weight", {hidden});
    const auto post_normed_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::RmsNorm, {residual_id, post_norm_weight_id},
                      {post_normed_id}, rms_attributes());
    capture(layer, "post_attention_norm", post_normed_id);
    const auto gate_weight_id =
        builder.streamed(prefix + "mlp.gate_proj.weight", {intermediate, hidden});
    const auto gate_id = builder.internal({sequence, intermediate});
    builder.operation(Opcode::Linear, {post_normed_id, gate_weight_id},
                      {gate_id});
    capture(layer, "mlp_gate", gate_id);
    const auto up_weight_id =
        builder.streamed(prefix + "mlp.up_proj.weight", {intermediate, hidden});
    const auto up_id = builder.internal({sequence, intermediate});
    builder.operation(Opcode::Linear, {post_normed_id, up_weight_id}, {up_id});
    capture(layer, "mlp_up", up_id);
    const auto activated_id = builder.internal({sequence, intermediate});
    builder.operation(Opcode::SiLU, {gate_id}, {activated_id});
    capture(layer, "mlp_gate_activated", activated_id);
    const auto gated_id = builder.internal({sequence, intermediate});
    builder.operation(Opcode::Multiply, {activated_id, up_id}, {gated_id});
    capture(layer, "mlp_gated", gated_id);
    const auto down_weight_id =
        builder.streamed(prefix + "mlp.down_proj.weight", {hidden, intermediate});
    const auto down_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::Linear, {gated_id, down_weight_id}, {down_id});
    capture(layer, "mlp_down", down_id);
    build.linear_operations += 3U;
    const auto layer_output_id = builder.internal({sequence, hidden});
    builder.operation(Opcode::Add, {residual_id, down_id}, {layer_output_id});
    capture(layer, "layer_output", layer_output_id);
    residual_id = layer_output_id;

    if (config.vision_token_count != 0U) {
      const auto deepstack = std::find(config.deepstack_language_layers.begin(),
                                       config.deepstack_language_layers.end(),
                                       layer);
      if (deepstack != config.deepstack_language_layers.end()) {
        const auto deepstack_index = static_cast<std::size_t>(std::distance(
            config.deepstack_language_layers.begin(), deepstack));
        const auto selected_visual_rows =
            builder.internal({config.vision_token_count, hidden});
        builder.operation(Opcode::GatherRows,
                          {residual_id, build.visual_positions_input_id},
                          {selected_visual_rows});
        const auto injected_visual_rows =
            builder.internal({config.vision_token_count, hidden});
        builder.operation(
            Opcode::Add,
            {selected_visual_rows,
             build.vision_deepstack_input_ids.at(deepstack_index)},
            {injected_visual_rows});
        const auto injected = builder.internal({sequence, hidden});
        builder.operation(Opcode::IndexedUpdateRows,
                          {residual_id, injected_visual_rows,
                           build.vision_destination_map_input_id},
                          {injected});
        residual_id = injected;
      }
    }

    const auto hidden_state_index = layer + 1U;
    if (std::find(config.selected_hidden_states.begin(),
                  config.selected_hidden_states.end(), hidden_state_index) !=
        config.selected_hidden_states.end()) {
      const auto selected = builder.add_tensor(
          DType::BF16, static_cast<std::uint32_t>(TensorRole::Output),
          {config.output_sequence_length, hidden});
      builder.operation(
          Opcode::Slice, {residual_id}, {selected},
          {Attribute::u64(AttrKey::Axis, 0U),
           Attribute::u64(AttrKey::Start, config.output_slice_start)});
      build.conditioning_output_ids.push_back(selected);
    }
  }

  // The conditioning contract is the RAW residual stream: model.norm is
  // deliberately NOT applied (extraction rule, plan §1).
  if (build.conditioning_output_ids.empty()) {
    for (auto &tensor : build.program.tensors)
      if (tensor.id == residual_id)
        tensor.roles = static_cast<std::uint32_t>(TensorRole::Output);
    build.conditioning_output_id = residual_id;
    build.conditioning_output_ids.push_back(residual_id);
  } else if (config.concatenate_selected_hidden_states) {
    const auto concatenated = builder.add_tensor(
        DType::BF16, static_cast<std::uint32_t>(TensorRole::Output),
        {config.output_sequence_length,
         hidden * build.conditioning_output_ids.size()});
    builder.operation(
        Opcode::Concat, build.conditioning_output_ids, {concatenated},
        {Attribute::u64(AttrKey::Axis, 1U)});
    build.conditioning_output_id = concatenated;
  } else {
    build.conditioning_output_id = build.conditioning_output_ids.back();
  }

  return build;
}

} // namespace dif::frontend
