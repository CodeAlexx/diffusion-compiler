#include "dif/frontend/flux2.hpp"

#include "dif/support/error.hpp"

#include "dif/ir/verify.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dif::frontend {

Qwen3VlConditionerConfig
make_flux2_klein_9b_conditioner_config(std::uint64_t executed_layers) {
  if (executed_layers == 0U || executed_layers > 36U)
    fail("FLUX.2 Qwen3-8B executed depth must be in [1,36]");
  Qwen3VlConditionerConfig config;
  config.hidden_size = 4096U;
  config.executed_layers = executed_layers;
  config.attention_heads = 32U;
  config.key_value_heads = 8U;
  config.head_dim = 128U;
  config.intermediate_size = 12288U;
  config.vocabulary = 151936U;
  config.rms_norm_epsilon = 1.0e-6;
  config.rope_theta = 1.0e6;
  config.attention_implementation = 2U;
  if (executed_layers >= 9U)
    config.selected_hidden_states.push_back(9U);
  if (executed_layers >= 18U)
    config.selected_hidden_states.push_back(18U);
  if (executed_layers >= 27U)
    config.selected_hidden_states.push_back(27U);
  if (config.selected_hidden_states.empty())
    config.selected_hidden_states.push_back(executed_layers);
  config.output_slice_start = 0U;
  config.output_sequence_length = 512U;
  config.use_attention_mask = true;
  config.dynamic_position_ids = true;
  config.mask_padding_queries = false;
  config.checkpoint_prefix = "model.";
  config.concatenate_selected_hidden_states =
      config.selected_hidden_states.size() > 1U;
  return config;
}

namespace {

runtime::Tensor i32_tensor(std::vector<std::int32_t> values) {
  runtime::Tensor result{ir::DType::I32,
                         {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  result.validate();
  return result;
}

} // namespace

Flux2KleinDoubleBlockBuild make_flux2_klein_9b_double_block(
    const Flux2KleinDoubleBlockConfig &config) {
  using namespace ir;
  constexpr std::uint64_t hidden = 4096U;
  constexpr std::uint64_t heads = 32U;
  constexpr std::uint64_t head_dim = 128U;
  constexpr std::uint64_t mlp = 12288U;
  constexpr std::uint64_t qkv = hidden * 3U;
  constexpr std::uint64_t packed_mlp = mlp * 2U;
  if (config.batch_size == 0U || config.image_tokens == 0U ||
      config.text_tokens == 0U ||
      config.block_index >= 8U || config.attention_implementation < 1U ||
      config.attention_implementation > 4U)
    fail("FLUX.2 double block requires nonzero batch/tokens and block index < 8");
  const auto total = config.text_tokens + config.image_tokens;
  const bool batched = config.batch_size != 1U;
  const auto token_axis = batched ? 1U : 0U;
  const auto feature_axis = batched ? 2U : 1U;
  const auto stream_shape = [&](std::uint64_t rows, std::uint64_t width) {
    return batched ? std::vector<std::uint64_t>{config.batch_size, rows, width}
                   : std::vector<std::uint64_t>{rows, width};
  };
  const auto head_shape = [&](std::uint64_t rows) {
    return batched
               ? std::vector<std::uint64_t>{config.batch_size, rows, heads,
                                            head_dim}
               : std::vector<std::uint64_t>{rows, heads, head_dim};
  };
  const auto modulation_shape = [&](std::uint64_t rows) {
    return batched
               ? std::vector<std::uint64_t>{config.batch_size, rows, hidden}
               : std::vector<std::uint64_t>{rows, hidden};
  };

  Flux2KleinDoubleBlockBuild build;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto bf16 = [&](std::vector<std::uint64_t> dims,
                        bool output = false) {
    return add_tensor(DType::BF16,
                      output ? static_cast<std::uint32_t>(TensorRole::Output)
                             : static_cast<std::uint32_t>(TensorRole::Internal),
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
  const auto linear_rows = [&](std::uint32_t input, std::uint32_t weight,
                               std::uint64_t rows, std::uint64_t inner,
                               std::uint64_t width) {
    auto matrix = input;
    if (batched) {
      matrix = bf16({config.batch_size * rows, inner});
      operation(Opcode::Reshape, {input}, {matrix});
    }
    const auto flat = bf16({config.batch_size * rows, width});
    operation(Opcode::Linear, {matrix, weight}, {flat});
    if (!batched)
      return flat;
    const auto result = bf16(stream_shape(rows, width));
    operation(Opcode::Reshape, {flat}, {result});
    return result;
  };
  const auto capture = [&](std::string name, std::uint32_t id) {
    if (!config.capture_boundaries)
      return;
    const auto tensor = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &candidate) { return candidate.id == id; });
    if (tensor == build.program.tensors.end())
      fail("FLUX.2 double-block capture lost its tensor");
    tensor->roles |= static_cast<std::uint32_t>(TensorRole::Output);
    build.boundaries.emplace_back(std::move(name), id);
  };

  build.image_input = add_tensor(DType::BF16, TensorRole::Input,
                                 stream_shape(config.image_tokens, hidden));
  build.text_input = add_tensor(DType::BF16, TensorRole::Input,
                                stream_shape(config.text_tokens, hidden));
  build.position_ids_input = add_tensor(DType::F32, TensorRole::Input,
                                        {config.batch_size, total, 4U});
  build.image_modulation_input =
      add_tensor(DType::BF16, TensorRole::Input, modulation_shape(6U));
  build.text_modulation_input =
      add_tensor(DType::BF16, TensorRole::Input, modulation_shape(6U));

  const auto ones = bf16({hidden});
  const auto zeros = bf16({hidden});
  const auto head_ones = bf16({head_dim});
  operation(Opcode::Fill, {}, {ones}, {Attribute::f64(AttrKey::Value, 1.0)});
  operation(Opcode::Fill, {}, {zeros}, {Attribute::f64(AttrKey::Value, 0.0)});
  operation(Opcode::Fill, {}, {head_ones},
            {Attribute::f64(AttrKey::Value, 1.0)});
  std::uint32_t ones_row = 0U;
  if (config.capture_boundaries) {
    ones_row = bf16(modulation_shape(1U));
    operation(Opcode::Fill, {}, {ones_row},
              {Attribute::f64(AttrKey::Value, 1.0)});
  }

  const auto modulation_row = [&](std::uint32_t packed, std::uint64_t row) {
    const auto result = bf16(modulation_shape(1U));
    operation(Opcode::Slice, {packed}, {result},
              {Attribute::u64(AttrKey::Axis, token_axis),
               Attribute::u64(AttrKey::Start, row)});
    return result;
  };
  std::vector<std::uint32_t> image_mods;
  std::vector<std::uint32_t> text_mods;
  for (std::uint64_t row = 0U; row < 6U; ++row) {
    image_mods.push_back(modulation_row(build.image_modulation_input, row));
    text_mods.push_back(modulation_row(build.text_modulation_input, row));
  }

  const auto modulate = [&](std::uint32_t input, std::uint64_t rows,
                            std::uint32_t shift, std::uint32_t scale,
                            std::string_view label) {
    const auto shape = stream_shape(rows, hidden);
    if (!config.capture_boundaries) {
      const auto result = bf16(shape);
      operation(Opcode::LayerNormModulate,
                {input, ones, zeros, scale, shift}, {result},
                {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
                 Attribute::u64(AttrKey::BlockSize, 128U)});
      return result;
    }
    const auto normalized = bf16(shape);
    operation(Opcode::LayerNorm, {input, ones, zeros}, {normalized},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::BlockSize, 128U)});
    capture(std::string(label) + "_norm", normalized);
    const auto one_plus_scale = bf16(modulation_shape(1U));
    operation(Opcode::Add, {ones_row, scale}, {one_plus_scale});
    const auto expanded_scale = bf16(shape);
    const auto expanded_shift = bf16(shape);
    operation(Opcode::BroadcastTo, {one_plus_scale}, {expanded_scale});
    operation(Opcode::BroadcastTo, {shift}, {expanded_shift});
    const auto scaled = bf16(shape);
    operation(Opcode::Multiply, {normalized, expanded_scale}, {scaled});
    const auto result = bf16(shape);
    operation(Opcode::Add, {scaled, expanded_shift}, {result});
    capture(std::string(label) + "_modulated", result);
    return result;
  };
  const auto gated_residual = [&](std::uint32_t residual,
                                  std::uint32_t branch, std::uint32_t gate,
                                  std::uint64_t rows, std::string name) {
    const auto shape = stream_shape(rows, hidden);
    const auto result = bf16(shape);
    operation(Opcode::ResidualGate, {residual, branch, gate}, {result});
    capture(std::move(name), result);
    return result;
  };

  const auto prefix = "double_blocks." + std::to_string(config.block_index) + ".";
  const auto image_attn_input =
      modulate(build.image_input, config.image_tokens, image_mods[0],
               image_mods[1], "image_attention");
  const auto text_attn_input =
      modulate(build.text_input, config.text_tokens, text_mods[0], text_mods[1],
               "text_attention");

  std::vector<std::int32_t> pair_axes;
  std::vector<std::int32_t> pair_indices;
  for (std::int32_t axis = 0; axis < 4; ++axis)
    for (std::int32_t pair = 0; pair < 16; ++pair) {
      pair_axes.push_back(axis);
      pair_indices.push_back(pair);
    }
  const auto pair_axes_id = add_tensor(DType::I32, TensorRole::Constant, {64U});
  const auto pair_indices_id =
      add_tensor(DType::I32, TensorRole::Constant, {64U});
  const auto axis_dims_id = add_tensor(DType::I32, TensorRole::Constant, {4U});
  build.generated_constants.emplace(pair_axes_id, i32_tensor(pair_axes));
  build.generated_constants.emplace(pair_indices_id, i32_tensor(pair_indices));
  build.generated_constants.emplace(axis_dims_id,
                                    i32_tensor({32, 32, 32, 32}));
  const auto cosine = add_tensor(DType::F32, TensorRole::Internal,
                                 {config.batch_size, total, 64U});
  const auto sine = add_tensor(DType::F32, TensorRole::Internal,
                               {config.batch_size, total, 64U});
  operation(Opcode::RotaryFrequency,
            {build.position_ids_input, pair_axes_id, pair_indices_id,
             axis_dims_id},
            {cosine, sine},
            {Attribute::f64(AttrKey::Theta, 2000.0),
             Attribute::f64(AttrKey::Ntk, 1.0)});

  struct Qkv {
    std::uint32_t q{};
    std::uint32_t k{};
    std::uint32_t v{};
  };
  const auto make_qkv = [&](std::uint32_t input, std::uint64_t rows,
                            std::string_view stream) {
    const auto packed_weight = checkpoint(
        prefix + std::string(stream) + "_attn.qkv.weight", {qkv, hidden});
    const auto packed = linear_rows(input, packed_weight, rows, hidden, qkv);
    capture(std::string(stream) + "_qkv", packed);
    Qkv result;
    const auto slice = [&](std::uint64_t start, std::string name) {
      const auto flat = bf16(stream_shape(rows, hidden));
      operation(Opcode::Slice, {packed}, {flat},
                {Attribute::u64(AttrKey::Axis, feature_axis),
                 Attribute::u64(AttrKey::Start, start)});
      const auto shaped = bf16(head_shape(rows));
      operation(Opcode::Reshape, {flat}, {shaped});
      capture(std::string(stream) + "_" + name, shaped);
      return shaped;
    };
    result.q = slice(0U, "q");
    result.k = slice(hidden, "k");
    result.v = slice(hidden * 2U, "v");
    const auto q_scale = checkpoint(
        prefix + std::string(stream) + "_attn.norm.query_norm.scale",
        {head_dim});
    const auto k_scale = checkpoint(
        prefix + std::string(stream) + "_attn.norm.key_norm.scale",
        {head_dim});
    if (!config.capture_boundaries) {
      const auto q_rotated = bf16(head_shape(rows));
      const auto k_rotated = bf16(head_shape(rows));
      const auto table_start =
          stream == "img" ? config.text_tokens : std::uint64_t{0U};
      const auto attributes = std::vector<Attribute>{
          Attribute::f64(AttrKey::Epsilon, 1.0e-6),
          Attribute::u64(AttrKey::Implementation, 2U),
          Attribute::u64(AttrKey::BlockSize, 128U),
          Attribute::u64(AttrKey::RotaryDim, head_dim),
          Attribute::u64(AttrKey::Start, table_start),
          Attribute::u64(
              AttrKey::RotaryLayout,
              static_cast<std::uint64_t>(RotaryLayout::Interleaved))};
      operation(Opcode::QkNormPartialRope,
                {result.q, q_scale, cosine, sine}, {q_rotated}, attributes);
      operation(Opcode::QkNormPartialRope,
                {result.k, k_scale, cosine, sine}, {k_rotated}, attributes);
      result.q = q_rotated;
      result.k = k_rotated;
      return result;
    }
    const auto q_normalized = bf16(head_shape(rows));
    const auto k_normalized = bf16(head_shape(rows));
    operation(Opcode::RmsNorm, {result.q, head_ones}, {q_normalized},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::Implementation, 2U),
               Attribute::u64(AttrKey::BlockSize, 128U)});
    operation(Opcode::RmsNorm, {result.k, head_ones}, {k_normalized},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::Implementation, 2U),
               Attribute::u64(AttrKey::BlockSize, 128U)});
    capture(std::string(stream) + "_q_rms", q_normalized);
    capture(std::string(stream) + "_k_rms", k_normalized);
    const auto q_scaled = bf16(head_shape(rows));
    const auto k_scaled = bf16(head_shape(rows));
    operation(Opcode::AffineLastDim, {q_normalized, q_scale}, {q_scaled});
    operation(Opcode::AffineLastDim, {k_normalized, k_scale}, {k_scaled});
    result.q = q_scaled;
    result.k = k_scaled;
    capture(std::string(stream) + "_q_norm", result.q);
    capture(std::string(stream) + "_k_norm", result.k);
    return result;
  };
  const auto image_qkv = make_qkv(image_attn_input, config.image_tokens, "img");
  const auto text_qkv = make_qkv(text_attn_input, config.text_tokens, "txt");

  const auto concat_heads = [&](std::uint32_t text, std::uint32_t image,
                                std::string name) {
    const auto combined = bf16(head_shape(total));
    operation(Opcode::Concat, {text, image}, {combined},
              {Attribute::u64(AttrKey::Axis, token_axis)});
    if (batched) {
      capture(std::move(name), combined);
      return combined;
    }
    const auto with_batch = bf16({1U, total, heads, head_dim});
    operation(Opcode::Reshape, {combined}, {with_batch});
    capture(std::move(name), with_batch);
    return with_batch;
  };
  const auto query = concat_heads(text_qkv.q, image_qkv.q, "query");
  const auto key = concat_heads(text_qkv.k, image_qkv.k, "key");
  const auto value = concat_heads(text_qkv.v, image_qkv.v, "value");

  auto rotated_query = query;
  auto rotated_key = key;
  if (config.capture_boundaries) {
    rotated_query = bf16({config.batch_size, total, heads, head_dim});
    rotated_key = bf16({config.batch_size, total, heads, head_dim});
    operation(Opcode::RotaryApply, {query, cosine, sine}, {rotated_query},
              {Attribute::u64(
                  AttrKey::RotaryLayout,
                  static_cast<std::uint64_t>(RotaryLayout::Interleaved))});
    operation(Opcode::RotaryApply, {key, cosine, sine}, {rotated_key},
              {Attribute::u64(
                  AttrKey::RotaryLayout,
                  static_cast<std::uint64_t>(RotaryLayout::Interleaved))});
  }
  capture("rotated_query", rotated_query);
  capture("rotated_key", rotated_key);
  const auto attention =
      bf16({config.batch_size, total, heads, head_dim});
  operation(Opcode::Attention, {rotated_query, rotated_key, value}, {attention},
            {Attribute::u64(AttrKey::KvHeads, heads),
             Attribute::u64(AttrKey::Implementation,
                            config.attention_implementation)});
  capture("attention", attention);
  const auto attention_flat = bf16(stream_shape(total, hidden));
  operation(Opcode::Reshape, {attention}, {attention_flat});
  const auto text_attention =
      bf16(stream_shape(config.text_tokens, hidden));
  const auto image_attention =
      bf16(stream_shape(config.image_tokens, hidden));
  operation(Opcode::Slice, {attention_flat}, {text_attention},
            {Attribute::u64(AttrKey::Axis, token_axis),
             Attribute::u64(AttrKey::Start, 0U)});
  operation(Opcode::Slice, {attention_flat}, {image_attention},
            {Attribute::u64(AttrKey::Axis, token_axis),
             Attribute::u64(AttrKey::Start, config.text_tokens)});

  const auto branch = [&](std::uint32_t residual, std::uint32_t attention_rows,
                          std::uint64_t rows,
                          const std::vector<std::uint32_t> &mods,
                          std::string_view stream) {
    const auto proj_weight = checkpoint(
        prefix + std::string(stream) + "_attn.proj.weight", {hidden, hidden});
    const auto projected =
        linear_rows(attention_rows, proj_weight, rows, hidden, hidden);
    capture(std::string(stream) + "_attention_projected", projected);
    const auto attention_residual = gated_residual(
        residual, projected, mods[2], rows,
        std::string(stream) + "_attention_residual");
    const auto mlp_input = modulate(
        attention_residual, rows, mods[3], mods[4],
        std::string(stream) + "_mlp");
    const auto first_weight = checkpoint(
        prefix + std::string(stream) + "_mlp.0.weight",
        {packed_mlp, hidden});
    const auto packed =
        linear_rows(mlp_input, first_weight, rows, hidden, packed_mlp);
    capture(std::string(stream) + "_mlp_packed", packed);
    const auto gated = bf16(stream_shape(rows, mlp));
    operation(Opcode::SwiGlu, {packed}, {gated},
              {Attribute::boolean(AttrKey::GateFirst, true)});
    capture(std::string(stream) + "_mlp_activation", gated);
    const auto second_weight = checkpoint(
        prefix + std::string(stream) + "_mlp.2.weight", {hidden, mlp});
    const auto down = linear_rows(gated, second_weight, rows, mlp, hidden);
    capture(std::string(stream) + "_mlp_down", down);
    return gated_residual(attention_residual, down, mods[5], rows,
                          std::string(stream) + "_output");
  };

  build.image_output = branch(build.image_input, image_attention,
                              config.image_tokens, image_mods, "img");
  build.text_output = branch(build.text_input, text_attention,
                             config.text_tokens, text_mods, "txt");
  for (auto &tensor : build.program.tensors)
    if (tensor.id == build.image_output || tensor.id == build.text_output)
      tensor.roles |= static_cast<std::uint32_t>(TensorRole::Output);
  verify(build.program);
  return build;
}

Flux2KleinSingleBlockBuild make_flux2_klein_9b_single_block(
    const Flux2KleinSingleBlockConfig &config) {
  using namespace ir;
  constexpr std::uint64_t hidden = 4096U;
  constexpr std::uint64_t heads = 32U;
  constexpr std::uint64_t head_dim = 128U;
  constexpr std::uint64_t mlp = 12288U;
  constexpr std::uint64_t qkv = hidden * 3U;
  constexpr std::uint64_t packed_mlp = mlp * 2U;
  if (config.batch_size == 0U || config.tokens == 0U ||
      config.block_index >= 24U || config.attention_implementation < 1U ||
      config.attention_implementation > 4U)
    fail("FLUX.2 single block requires nonzero batch/tokens and block index < 24");
  const bool batched = config.batch_size != 1U;
  const auto token_axis = batched ? 1U : 0U;
  const auto feature_axis = batched ? 2U : 1U;
  const auto stream_shape = [&](std::uint64_t rows, std::uint64_t width) {
    return batched ? std::vector<std::uint64_t>{config.batch_size, rows, width}
                   : std::vector<std::uint64_t>{rows, width};
  };
  const auto head_shape = [&](std::uint64_t rows) {
    return batched
               ? std::vector<std::uint64_t>{config.batch_size, rows, heads,
                                            head_dim}
               : std::vector<std::uint64_t>{rows, heads, head_dim};
  };
  const auto modulation_shape = [&](std::uint64_t rows) {
    return batched
               ? std::vector<std::uint64_t>{config.batch_size, rows, hidden}
               : std::vector<std::uint64_t>{rows, hidden};
  };

  Flux2KleinSingleBlockBuild build;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto bf16 = [&](std::vector<std::uint64_t> dims,
                        bool output = false) {
    return add_tensor(DType::BF16,
                      output ? static_cast<std::uint32_t>(TensorRole::Output)
                             : static_cast<std::uint32_t>(TensorRole::Internal),
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
  const auto linear_rows = [&](std::uint32_t input, std::uint32_t weight,
                               std::uint64_t inner,
                               std::uint64_t width) {
    auto matrix = input;
    if (batched) {
      matrix = bf16({config.batch_size * config.tokens, inner});
      operation(Opcode::Reshape, {input}, {matrix});
    }
    const auto flat =
        bf16({config.batch_size * config.tokens, width});
    operation(Opcode::Linear, {matrix, weight}, {flat});
    if (!batched)
      return flat;
    const auto result = bf16(stream_shape(config.tokens, width));
    operation(Opcode::Reshape, {flat}, {result});
    return result;
  };
  const auto capture = [&](std::string name, std::uint32_t id) {
    if (!config.capture_boundaries)
      return;
    const auto tensor = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &candidate) { return candidate.id == id; });
    if (tensor == build.program.tensors.end())
      fail("FLUX.2 single-block capture lost its tensor");
    tensor->roles |= static_cast<std::uint32_t>(TensorRole::Output);
    build.boundaries.emplace_back(std::move(name), id);
  };

  build.sequence_input = add_tensor(DType::BF16, TensorRole::Input,
                                    stream_shape(config.tokens, hidden));
  build.position_ids_input =
      add_tensor(DType::F32, TensorRole::Input,
                 {config.batch_size, config.tokens, 4U});
  build.modulation_input =
      add_tensor(DType::BF16, TensorRole::Input, modulation_shape(3U));

  const auto ones = bf16({hidden});
  const auto zeros = bf16({hidden});
  operation(Opcode::Fill, {}, {ones}, {Attribute::f64(AttrKey::Value, 1.0)});
  operation(Opcode::Fill, {}, {zeros}, {Attribute::f64(AttrKey::Value, 0.0)});
  std::uint32_t ones_row = 0U;
  if (config.capture_boundaries) {
    ones_row = bf16(modulation_shape(1U));
    operation(Opcode::Fill, {}, {ones_row},
              {Attribute::f64(AttrKey::Value, 1.0)});
  }
  const auto modulation_row = [&](std::uint64_t row) {
    const auto result = bf16(modulation_shape(1U));
    operation(Opcode::Slice, {build.modulation_input}, {result},
              {Attribute::u64(AttrKey::Axis, token_axis),
               Attribute::u64(AttrKey::Start, row)});
    return result;
  };
  const auto shift = modulation_row(0U);
  const auto scale = modulation_row(1U);
  const auto gate = modulation_row(2U);

  const auto modulated = bf16(stream_shape(config.tokens, hidden));
  if (!config.capture_boundaries) {
    operation(Opcode::LayerNormModulate,
              {build.sequence_input, ones, zeros, scale, shift}, {modulated},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::BlockSize, 128U)});
  } else {
    const auto normalized = bf16(stream_shape(config.tokens, hidden));
    operation(Opcode::LayerNorm, {build.sequence_input, ones, zeros},
              {normalized},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::BlockSize, 128U)});
    capture("pre_norm", normalized);
    const auto one_plus_scale = bf16(modulation_shape(1U));
    operation(Opcode::Add, {ones_row, scale}, {one_plus_scale});
    const auto expanded_scale = bf16(stream_shape(config.tokens, hidden));
    const auto expanded_shift = bf16(stream_shape(config.tokens, hidden));
    operation(Opcode::BroadcastTo, {one_plus_scale}, {expanded_scale});
    operation(Opcode::BroadcastTo, {shift}, {expanded_shift});
    const auto scaled = bf16(stream_shape(config.tokens, hidden));
    operation(Opcode::Multiply, {normalized, expanded_scale}, {scaled});
    operation(Opcode::Add, {scaled, expanded_shift}, {modulated});
    capture("modulated", modulated);
  }

  const auto prefix =
      "single_blocks." + std::to_string(config.block_index) + ".";
  const auto first_weight =
      checkpoint(prefix + "linear1.weight", {qkv + packed_mlp, hidden});
  const auto first =
      linear_rows(modulated, first_weight, hidden, qkv + packed_mlp);
  capture("linear1", first);
  const auto mlp_activation = bf16(stream_shape(config.tokens, mlp));
  operation(Opcode::SwiGlu, {first}, {mlp_activation},
            {Attribute::boolean(AttrKey::GateFirst, true),
             Attribute::u64(AttrKey::Start, qkv)});
  capture("mlp_activation", mlp_activation);
  const auto slice_head = [&](std::uint64_t start, std::string name) {
    const auto flat = bf16(stream_shape(config.tokens, hidden));
    operation(Opcode::Slice, {first}, {flat},
              {Attribute::u64(AttrKey::Axis, feature_axis),
               Attribute::u64(AttrKey::Start, start)});
    const auto shaped = bf16(head_shape(config.tokens));
    operation(Opcode::Reshape, {flat}, {shaped});
    capture(std::move(name), shaped);
    return shaped;
  };
  const auto query = slice_head(0U, "query");
  const auto key = slice_head(hidden, "key");
  const auto value = slice_head(hidden * 2U, "value");
  const auto query_scale =
      checkpoint(prefix + "norm.query_norm.scale", {head_dim});
  const auto key_scale =
      checkpoint(prefix + "norm.key_norm.scale", {head_dim});
  const auto query_batch =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  const auto key_batch =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  const auto value_batch =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  operation(Opcode::Reshape, {query}, {query_batch});
  operation(Opcode::Reshape, {key}, {key_batch});
  operation(Opcode::Reshape, {value}, {value_batch});
  std::vector<std::int32_t> pair_axes;
  std::vector<std::int32_t> pair_indices;
  for (std::int32_t axis = 0; axis < 4; ++axis)
    for (std::int32_t pair = 0; pair < 16; ++pair) {
      pair_axes.push_back(axis);
      pair_indices.push_back(pair);
    }
  const auto pair_axes_id = add_tensor(DType::I32, TensorRole::Constant, {64U});
  const auto pair_indices_id =
      add_tensor(DType::I32, TensorRole::Constant, {64U});
  const auto axis_dims_id = add_tensor(DType::I32, TensorRole::Constant, {4U});
  build.generated_constants.emplace(pair_axes_id, i32_tensor(pair_axes));
  build.generated_constants.emplace(pair_indices_id, i32_tensor(pair_indices));
  build.generated_constants.emplace(axis_dims_id,
                                    i32_tensor({32, 32, 32, 32}));
  const auto cosine = add_tensor(DType::F32, TensorRole::Internal,
                                 {config.batch_size, config.tokens, 64U});
  const auto sine = add_tensor(DType::F32, TensorRole::Internal,
                               {config.batch_size, config.tokens, 64U});
  operation(Opcode::RotaryFrequency,
            {build.position_ids_input, pair_axes_id, pair_indices_id,
             axis_dims_id},
            {cosine, sine},
            {Attribute::f64(AttrKey::Theta, 2000.0),
             Attribute::f64(AttrKey::Ntk, 1.0)});
  const auto rotated_query =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  const auto rotated_key =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  const auto fused_qk_attributes = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-6),
      Attribute::u64(AttrKey::Implementation, 2U),
      Attribute::u64(AttrKey::BlockSize, 128U),
      Attribute::u64(AttrKey::RotaryDim, head_dim),
      Attribute::u64(
          AttrKey::RotaryLayout,
          static_cast<std::uint64_t>(RotaryLayout::Interleaved))};
  operation(Opcode::QkNormPartialRope,
            {query_batch, query_scale, cosine, sine}, {rotated_query},
            fused_qk_attributes);
  operation(Opcode::QkNormPartialRope,
            {key_batch, key_scale, cosine, sine}, {rotated_key},
            fused_qk_attributes);
  capture("rotated_query", rotated_query);
  capture("rotated_key", rotated_key);
  const auto attention =
      bf16({config.batch_size, config.tokens, heads, head_dim});
  operation(Opcode::Attention, {rotated_query, rotated_key, value_batch},
            {attention},
            {Attribute::u64(AttrKey::KvHeads, heads),
             Attribute::u64(AttrKey::Implementation,
                            config.attention_implementation)});
  capture("attention", attention);
  const auto attention_flat = bf16(stream_shape(config.tokens, hidden));
  operation(Opcode::Reshape, {attention}, {attention_flat});

  const auto joined = bf16(stream_shape(config.tokens, hidden + mlp));
  operation(Opcode::Concat, {attention_flat, mlp_activation}, {joined},
            {Attribute::u64(AttrKey::Axis, feature_axis)});
  capture("linear2_input", joined);
  const auto second_weight =
      checkpoint(prefix + "linear2.weight", {hidden, hidden + mlp});
  const auto branch =
      linear_rows(joined, second_weight, hidden + mlp, hidden);
  capture("linear2", branch);
  build.sequence_output = bf16(stream_shape(config.tokens, hidden), true);
  operation(Opcode::ResidualGate, {build.sequence_input, branch, gate},
            {build.sequence_output});
  capture("output", build.sequence_output);
  verify(build.program);
  return build;
}

Flux2KleinTransformerBuild make_flux2_klein_9b_transformer(
    const Flux2KleinTransformerConfig &config) {
  using namespace ir;
  constexpr std::uint64_t latent_channels = 128U;
  constexpr std::uint64_t context_width = 12288U;
  constexpr std::uint64_t hidden = 4096U;
  constexpr std::uint64_t timestep_width = 256U;
  if (config.batch_size == 0U || config.image_tokens == 0U ||
      config.text_tokens == 0U ||
      config.double_depth > 8U || config.single_depth > 24U ||
      (config.double_depth < 8U && config.single_depth != 0U))
    fail("invalid FLUX.2 transformer parity depth or token geometry");

  Flux2KleinTransformerBuild build;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto total = config.text_tokens + config.image_tokens;
  const bool batched = config.batch_size != 1U;
  const auto token_axis = batched ? 1U : 0U;
  const auto stream_shape = [&](std::uint64_t rows, std::uint64_t width) {
    return batched ? std::vector<std::uint64_t>{config.batch_size, rows, width}
                   : std::vector<std::uint64_t>{rows, width};
  };
  const auto modulation_shape = [&](std::uint64_t rows) {
    return batched
               ? std::vector<std::uint64_t>{config.batch_size, rows, hidden}
               : std::vector<std::uint64_t>{rows, hidden};
  };
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto bf16 = [&](std::vector<std::uint64_t> dims) {
    return add_tensor(DType::BF16, TensorRole::Internal, std::move(dims));
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
    auto roles = static_cast<std::uint32_t>(TensorRole::Constant);
    if (config.streamed_constants)
      roles |= static_cast<std::uint32_t>(TensorRole::Streamed);
    const auto id = add_tensor(DType::BF16, roles, std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.push_back(std::move(name));
    return id;
  };
  const auto linear_rows = [&](std::uint32_t input, std::uint32_t weight,
                               std::uint64_t rows, std::uint64_t inner,
                               std::uint64_t width) {
    auto matrix = input;
    if (batched) {
      matrix = bf16({config.batch_size * rows, inner});
      operation(Opcode::Reshape, {input}, {matrix});
    }
    const auto flat = bf16({config.batch_size * rows, width});
    operation(Opcode::Linear, {matrix, weight}, {flat});
    if (!batched)
      return flat;
    const auto result = bf16(stream_shape(rows, width));
    operation(Opcode::Reshape, {flat}, {result});
    return result;
  };
  const auto capture = [&](std::string name, std::uint32_t id) {
    if (!config.capture_depth_boundaries)
      return;
    const auto tensor = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &candidate) { return candidate.id == id; });
    if (tensor == build.program.tensors.end())
      fail("FLUX.2 transformer capture lost its tensor");
    tensor->roles |= static_cast<std::uint32_t>(TensorRole::Output);
    build.boundaries.emplace_back(std::move(name), id);
  };

  build.latent_input = add_tensor(DType::BF16, TensorRole::Input,
                                  stream_shape(config.image_tokens,
                                               latent_channels));
  build.conditioning_input =
      add_tensor(DType::BF16, TensorRole::Input,
                 stream_shape(config.text_tokens, context_width));
  build.timestep_input =
      add_tensor(DType::BF16, TensorRole::Input, {config.batch_size});
  build.position_ids_input =
      add_tensor(DType::F32, TensorRole::Input,
                 {config.batch_size, total, 4U});

  const auto image_in_weight =
      checkpoint("img_in.weight", {hidden, latent_channels});
  const auto text_in_weight =
      checkpoint("txt_in.weight", {hidden, context_width});
  auto image = linear_rows(build.latent_input, image_in_weight,
                           config.image_tokens, latent_channels, hidden);
  auto text = linear_rows(build.conditioning_input, text_in_weight,
                          config.text_tokens, context_width, hidden);
  capture("image_projected", image);
  capture("text_projected", text);

  const auto thousand = bf16({config.batch_size});
  operation(Opcode::Fill, {}, {thousand},
            {Attribute::f64(AttrKey::Value, 1000.0)});
  const auto scaled_timestep_bf16 = bf16({config.batch_size});
  operation(Opcode::Multiply, {build.timestep_input, thousand},
            {scaled_timestep_bf16});
  const auto scaled_timestep_f32 =
      add_tensor(DType::F32, TensorRole::Internal, {config.batch_size});
  operation(Opcode::Cast, {scaled_timestep_bf16}, {scaled_timestep_f32});
  const auto timestep_embedding_f32 =
      add_tensor(DType::F32, TensorRole::Internal,
                 {config.batch_size, timestep_width});
  operation(Opcode::SinusoidalTimestep, {scaled_timestep_f32},
            {timestep_embedding_f32},
            {Attribute::boolean(AttrKey::FlipSinToCos, true),
             Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
             Attribute::f64(AttrKey::Scale, 1.0),
             Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  const auto timestep_embedding =
      bf16({config.batch_size, timestep_width});
  operation(Opcode::Cast, {timestep_embedding_f32}, {timestep_embedding});
  capture("timestep_embedding", timestep_embedding);
  const auto time_in_weight =
      checkpoint("time_in.in_layer.weight", {hidden, timestep_width});
  const auto time_out_weight =
      checkpoint("time_in.out_layer.weight", {hidden, hidden});
  const auto time_hidden = bf16({config.batch_size, hidden});
  operation(Opcode::Linear, {timestep_embedding, time_in_weight},
            {time_hidden});
  const auto time_activated = bf16({config.batch_size, hidden});
  operation(Opcode::SiLU, {time_hidden}, {time_activated});
  const auto vector = bf16({config.batch_size, hidden});
  operation(Opcode::Linear, {time_activated, time_out_weight}, {vector});
  capture("time_vector", vector);
  const auto modulation_input = bf16({config.batch_size, hidden});
  operation(Opcode::SiLU, {vector}, {modulation_input});
  const auto image_modulation_weight = checkpoint(
      "double_stream_modulation_img.lin.weight", {6U * hidden, hidden});
  const auto text_modulation_weight = checkpoint(
      "double_stream_modulation_txt.lin.weight", {6U * hidden, hidden});
  const auto single_modulation_weight = checkpoint(
      "single_stream_modulation.lin.weight", {3U * hidden, hidden});
  const auto image_modulation_flat =
      bf16({config.batch_size, 6U * hidden});
  const auto text_modulation_flat =
      bf16({config.batch_size, 6U * hidden});
  const auto single_modulation_flat =
      bf16({config.batch_size, 3U * hidden});
  operation(Opcode::Linear, {modulation_input, image_modulation_weight},
            {image_modulation_flat});
  operation(Opcode::Linear, {modulation_input, text_modulation_weight},
            {text_modulation_flat});
  operation(Opcode::Linear, {modulation_input, single_modulation_weight},
            {single_modulation_flat});
  const auto image_modulation = bf16(modulation_shape(6U));
  const auto text_modulation = bf16(modulation_shape(6U));
  const auto single_modulation = bf16(modulation_shape(3U));
  operation(Opcode::Reshape, {image_modulation_flat}, {image_modulation});
  operation(Opcode::Reshape, {text_modulation_flat}, {text_modulation});
  operation(Opcode::Reshape, {single_modulation_flat}, {single_modulation});
  capture("image_modulation", image_modulation);
  capture("text_modulation", text_modulation);
  capture("single_modulation", single_modulation);

  const auto inline_program =
      [&](const ir::Program &source,
          const runtime::TensorMap &source_generated,
          const std::vector<std::pair<std::uint32_t, std::uint32_t>> &inputs,
          const std::vector<std::uint32_t> &source_checkpoints,
          const std::vector<std::string> &source_names) {
        std::unordered_map<std::uint32_t, std::uint32_t> mapping;
        for (const auto &[source_id, destination_id] : inputs) {
          if (!source.tensor(source_id) || !build.program.tensor(destination_id))
            fail("FLUX.2 subprogram input mapping is invalid");
          mapping.emplace(source_id, destination_id);
        }
        const std::unordered_set<std::uint32_t> checkpoint_ids(
            source_checkpoints.begin(), source_checkpoints.end());
        for (const auto &source_tensor : source.tensors) {
          if (mapping.contains(source_tensor.id))
            continue;
          if (source_tensor.has_role(TensorRole::Input))
            fail("FLUX.2 subprogram has an unmapped input");
          auto destination = source_tensor;
          destination.id = next_tensor++;
          destination.roles &=
              ~static_cast<std::uint32_t>(TensorRole::Output);
          if (checkpoint_ids.contains(source_tensor.id) &&
              config.streamed_constants)
            destination.roles |=
                static_cast<std::uint32_t>(TensorRole::Streamed);
          mapping.emplace(source_tensor.id, destination.id);
          build.program.tensors.push_back(std::move(destination));
        }
        const auto remap = [&](std::uint32_t id) {
          const auto found = mapping.find(id);
          if (found == mapping.end())
            fail("FLUX.2 subprogram tensor remap is missing");
          return found->second;
        };
        for (const auto &source_operation : source.operations) {
          auto destination = source_operation;
          destination.id = next_operation++;
          for (auto &id : destination.inputs)
            id = remap(id);
          for (auto &id : destination.outputs)
            id = remap(id);
          build.program.operations.push_back(std::move(destination));
        }
        for (const auto &[source_id, tensor] : source_generated)
          build.generated_constants.emplace(remap(source_id), tensor);
        if (source_checkpoints.size() != source_names.size())
          fail("FLUX.2 subprogram checkpoint inventory is malformed");
        for (std::size_t index = 0; index < source_checkpoints.size(); ++index) {
          build.checkpoint_tensors.push_back(
              remap(source_checkpoints.at(index)));
          build.checkpoint_names.push_back(source_names.at(index));
        }
        return mapping;
      };

  for (std::uint64_t depth = 0U; depth < config.double_depth; ++depth) {
    Flux2KleinDoubleBlockConfig block_config;
    block_config.batch_size = config.batch_size;
    block_config.image_tokens = config.image_tokens;
    block_config.text_tokens = config.text_tokens;
    block_config.block_index = depth;
    block_config.attention_implementation =
        config.attention_implementation;
    block_config.capture_boundaries =
        config.capture_first_block_boundaries && depth == 0U;
    const auto block = make_flux2_klein_9b_double_block(block_config);
    const auto mapping = inline_program(
        block.program, block.generated_constants,
        {{block.image_input, image},
         {block.text_input, text},
         {block.position_ids_input, build.position_ids_input},
         {block.image_modulation_input, image_modulation},
         {block.text_modulation_input, text_modulation}},
        block.checkpoint_tensors, block.checkpoint_names);
    image = mapping.at(block.image_output);
    text = mapping.at(block.text_output);
    if (block_config.capture_boundaries)
      for (const auto &[name, id] : block.boundaries)
        capture("double_1_" + name, mapping.at(id));
    capture("double_" + std::to_string(depth + 1U) + "_image", image);
    capture("double_" + std::to_string(depth + 1U) + "_text", text);
  }

  if (config.double_depth < 8U) {
    build.prediction_output = image;
  } else {
    auto sequence = bf16(stream_shape(total, hidden));
    operation(Opcode::Concat, {text, image}, {sequence},
              {Attribute::u64(AttrKey::Axis, token_axis)});
    for (std::uint64_t depth = 0U; depth < config.single_depth; ++depth) {
      Flux2KleinSingleBlockConfig block_config;
      block_config.batch_size = config.batch_size;
      block_config.tokens = total;
      block_config.block_index = depth;
      block_config.attention_implementation =
          config.attention_implementation;
      block_config.capture_boundaries =
          config.capture_first_block_boundaries && depth == 0U;
      const auto block = make_flux2_klein_9b_single_block(block_config);
      const auto mapping = inline_program(
          block.program, block.generated_constants,
          {{block.sequence_input, sequence},
           {block.position_ids_input, build.position_ids_input},
           {block.modulation_input, single_modulation}},
          block.checkpoint_tensors, block.checkpoint_names);
      sequence = mapping.at(block.sequence_output);
      if (block_config.capture_boundaries)
        for (const auto &[name, id] : block.boundaries)
          capture("single_1_" + name, mapping.at(id));
      capture("single_" + std::to_string(depth + 1U), sequence);
    }
    if (config.single_depth < 24U) {
      build.prediction_output = sequence;
    } else {
      const auto image_hidden =
          bf16(stream_shape(config.image_tokens, hidden));
      operation(Opcode::Slice, {sequence}, {image_hidden},
                {Attribute::u64(AttrKey::Axis, token_axis),
                 Attribute::u64(AttrKey::Start, config.text_tokens)});
      const auto ones = bf16({hidden});
      const auto zeros = bf16({hidden});
      operation(Opcode::Fill, {}, {ones},
                {Attribute::f64(AttrKey::Value, 1.0)});
      operation(Opcode::Fill, {}, {zeros},
                {Attribute::f64(AttrKey::Value, 0.0)});
      const auto final_modulation_weight = checkpoint(
          "final_layer.adaLN_modulation.1.weight", {2U * hidden, hidden});
      const auto final_modulation_flat =
          bf16({config.batch_size, 2U * hidden});
      operation(Opcode::Linear, {modulation_input, final_modulation_weight},
                {final_modulation_flat});
      const auto shift_flat = bf16({config.batch_size, hidden});
      const auto scale_flat = bf16({config.batch_size, hidden});
      operation(Opcode::Slice, {final_modulation_flat}, {shift_flat},
                {Attribute::u64(AttrKey::Axis, 1U),
                 Attribute::u64(AttrKey::Start, 0U)});
      operation(Opcode::Slice, {final_modulation_flat}, {scale_flat},
                {Attribute::u64(AttrKey::Axis, 1U),
                 Attribute::u64(AttrKey::Start, hidden)});
      const auto shift = bf16(modulation_shape(1U));
      const auto scale = bf16(modulation_shape(1U));
      operation(Opcode::Reshape, {shift_flat}, {shift});
      operation(Opcode::Reshape, {scale_flat}, {scale});
      const auto modulated =
          bf16(stream_shape(config.image_tokens, hidden));
      operation(Opcode::LayerNormModulate,
                {image_hidden, ones, zeros, scale, shift}, {modulated},
                {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
                 Attribute::u64(AttrKey::BlockSize, 128U)});
      capture("final_modulated", modulated);
      const auto final_weight = checkpoint(
          "final_layer.linear.weight", {latent_channels, hidden});
      build.prediction_output =
          linear_rows(modulated, final_weight, config.image_tokens, hidden,
                      latent_channels);
    }
  }
  const auto prediction = std::find_if(
      build.program.tensors.begin(), build.program.tensors.end(),
      [&](const auto &candidate) {
        return candidate.id == build.prediction_output;
      });
  if (prediction == build.program.tensors.end())
    fail("FLUX.2 transformer has no prediction output");
  prediction->roles |= static_cast<std::uint32_t>(TensorRole::Output);
  verify(build.program);
  return build;
}

Flux2KleinCfgEulerBuild make_flux2_klein_base_cfg_euler_step(
    std::vector<std::uint64_t> sample_shape) {
  using namespace ir;
  if (sample_shape.empty() ||
      std::any_of(sample_shape.begin(), sample_shape.end(),
                  [](std::uint64_t value) { return value == 0U; }))
    fail("FLUX.2 CFG/Euler sample shape must be nonempty and positive");

  Flux2KleinCfgEulerBuild build;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto tensor = [&](DType dtype, std::uint32_t roles,
                          const std::vector<std::uint64_t> &dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, dims});
    return id;
  };
  const auto operation = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                             std::vector<std::uint32_t> outputs) {
    program.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs), {}});
  };

  build.sample_input = tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.conditional_velocity_input =
      tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.unconditional_velocity_input =
      tensor(DType::BF16, TensorRole::Input, sample_shape);
  build.guidance_input = tensor(DType::BF16, TensorRole::Input, {1U});
  build.current_timestep_input = tensor(DType::F32, TensorRole::Input, {1U});
  build.next_timestep_input = tensor(DType::F32, TensorRole::Input, {1U});
  build.negative_one_constant =
      tensor(DType::BF16, TensorRole::Constant, {1U});

  const auto negative_one = tensor(DType::BF16, TensorRole::Internal,
                                   sample_shape);
  operation(Opcode::BroadcastTo, {build.negative_one_constant},
            {negative_one});
  const auto negative_unconditional =
      tensor(DType::BF16, TensorRole::Internal, sample_shape);
  operation(Opcode::Multiply,
            {build.unconditional_velocity_input, negative_one},
            {negative_unconditional});
  const auto difference = tensor(DType::BF16, TensorRole::Internal,
                                 sample_shape);
  operation(Opcode::Add,
            {build.conditional_velocity_input, negative_unconditional},
            {difference});
  const auto broadcast_guidance =
      tensor(DType::BF16, TensorRole::Internal, sample_shape);
  operation(Opcode::BroadcastTo, {build.guidance_input},
            {broadcast_guidance});
  const auto guided_delta = tensor(DType::BF16, TensorRole::Internal,
                                   sample_shape);
  operation(Opcode::Multiply, {difference, broadcast_guidance},
            {guided_delta});
  build.guided_velocity_output =
      tensor(DType::BF16, TensorRole::Output, sample_shape);
  operation(Opcode::Add,
            {build.unconditional_velocity_input, guided_delta},
            {build.guided_velocity_output});
  build.sample_output = tensor(DType::BF16, TensorRole::Output, sample_shape);
  operation(Opcode::EulerVelocityStep,
            {build.sample_input, build.guided_velocity_output,
             build.current_timestep_input, build.next_timestep_input},
            {build.sample_output});
  verify(program);
  return build;
}

} // namespace dif::frontend
