#include "dif/frontend/h3.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

ir::TensorDesc tensor(std::uint32_t id, std::uint32_t roles,
                      std::vector<std::uint64_t> dims) {
  return {id, ir::DType::BF16, roles, std::move(dims)};
}

ir::TensorDesc typed_tensor(std::uint32_t id, ir::DType dtype,
                            std::uint32_t roles,
                            std::vector<std::uint64_t> dims) {
  return {id, dtype, roles, std::move(dims)};
}

} // namespace

ir::Program make_h3_denoiser(const H3DenoiserConfig &config) {
  using namespace ir;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (config.video_tokens == 0U || config.audio_tokens == 0U ||
      config.text_tokens == 0U || config.timestep_tables == 0U ||
      config.hidden == 0U || config.heads == 0U || config.head_dim == 0U ||
      config.ffn == 0U || config.rotary == 0U ||
      config.rotary > config.head_dim || (config.rotary % 2U) != 0U ||
      (config.rotary % 6U) != 0U ||
      config.layers == 0U || config.refiner_layers == 0U ||
      config.video_input_dim == 0U || config.audio_input_dim == 0U ||
      config.text_input_dim == 0U || config.time_input_dim == 0U ||
      config.time_hidden_dim == 0U || config.time_embed_dim == 0U ||
      config.heads > maximum / config.head_dim ||
      config.ffn > maximum / 2U || config.hidden > maximum / 18U ||
      config.video_tokens > maximum - config.audio_tokens ||
      config.video_tokens + config.audio_tokens >
          maximum - config.text_tokens ||
      (config.attention_implementation != 1U &&
       config.attention_implementation != 2U))
    fail("invalid mixed-precision H3 denoiser geometry");
  const auto sequence =
      config.video_tokens + config.audio_tokens + config.text_tokens;
  const auto inner = config.heads * config.head_dim;
  const auto adaln_width = 18U * config.hidden;
  const auto constants = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed
                                 : TensorRole::Internal));

  Program program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto add_tensor = [&](DType dtype, std::uint32_t roles,
                        std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back(
        typed_tensor(id, dtype, roles, std::move(dims)));
    return id;
  };
  auto add_input = [&](DType dtype, std::vector<std::uint64_t> dims) {
    return add_tensor(dtype, TensorRole::Input, std::move(dims));
  };
  auto add_constant = [&](DType dtype, std::vector<std::uint64_t> dims) {
    return add_tensor(dtype, constants, std::move(dims));
  };
  auto add_internal = [&](DType dtype, std::vector<std::uint64_t> dims) {
    return add_tensor(dtype, TensorRole::Internal, std::move(dims));
  };
  auto add_output = [&](DType dtype, std::vector<std::uint64_t> dims) {
    return add_tensor(dtype, TensorRole::Output, std::move(dims));
  };
  auto add_operation = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                           std::vector<std::uint32_t> outputs,
                           std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };

  const auto video =
      add_input(DType::F32, {config.video_tokens, config.video_input_dim});
  const auto audio =
      add_input(DType::F32, {config.audio_tokens, config.audio_input_dim});
  const auto text =
      add_input(DType::BF16, {config.text_tokens, config.text_input_dim});
  const auto timesteps = add_input(DType::F32, {config.timestep_tables});
  const auto text_map = add_input(DType::I32, {sequence});
  const auto video_map = add_input(DType::I32, {sequence});
  const auto audio_map = add_input(DType::I32, {sequence});
  const auto adaln_indices = add_input(DType::I32, {sequence});
  const auto timestep_indices = add_input(DType::I32, {sequence});
  const auto video_indices = add_input(DType::I32, {config.video_tokens});
  const auto audio_indices = add_input(DType::I32, {config.audio_tokens});
  const auto position_ids = add_input(DType::F32, {sequence, 3U});
  const auto rope_inv_freq =
      add_constant(DType::F32, {config.rotary / 6U});

  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, config.block_size),
      Attribute::u64(AttrKey::Implementation, 1U),
  };
  auto gate_first_attrs = linear_attrs;
  gate_first_attrs.push_back(Attribute::boolean(AttrKey::GateFirst, true));
  const auto rms_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, config.block_size),
  };
  const auto qkv_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::Heads, config.heads),
      Attribute::u64(AttrKey::HeadDim, config.head_dim),
      Attribute::u64(AttrKey::BlockSize, config.block_size),
  };
  const auto qk_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, config.heads),
      Attribute::u64(AttrKey::HeadDim, config.head_dim),
      Attribute::u64(AttrKey::RotaryDim, config.rotary),
      Attribute::u64(AttrKey::BlockSize, config.block_size),
  };
  const auto attention_attrs = std::vector<Attribute>{
      Attribute::f64(
          AttrKey::AttentionScale,
          1.0 / std::sqrt(static_cast<double>(config.head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::BlockSize, 64U),
      Attribute::u64(AttrKey::Implementation,
                     config.attention_implementation),
  };

  const auto video_weight =
      add_constant(DType::F32, {config.hidden, config.video_input_dim});
  const auto video_bias = add_constant(DType::F32, {config.hidden});
  const auto audio_weight =
      add_constant(DType::F32, {config.hidden, config.audio_input_dim});
  const auto audio_bias = add_constant(DType::F32, {config.hidden});
  const auto context_weight =
      add_constant(DType::BF16, {config.hidden, config.text_input_dim});
  const auto context_bias = add_constant(DType::BF16, {config.hidden});
  const auto time_in_weight = add_constant(
      DType::F32, {config.time_hidden_dim, config.time_input_dim});
  const auto time_in_bias =
      add_constant(DType::F32, {config.time_hidden_dim});
  const auto time_out_weight = add_constant(
      DType::F32, {config.time_embed_dim, config.time_hidden_dim});
  const auto time_out_bias =
      add_constant(DType::F32, {config.time_embed_dim});

  const auto video_f32 =
      add_internal(DType::F32, {config.video_tokens, config.hidden});
  const auto video_bf16 =
      add_internal(DType::BF16, {config.video_tokens, config.hidden});
  const auto audio_f32 =
      add_internal(DType::F32, {config.audio_tokens, config.hidden});
  const auto audio_bf16 =
      add_internal(DType::BF16, {config.audio_tokens, config.hidden});
  auto refined_text =
      add_internal(DType::BF16, {config.text_tokens, config.hidden});
  add_operation(Opcode::Linear, {video, video_weight, video_bias},
                {video_f32}, linear_attrs);
  add_operation(Opcode::Cast, {video_f32}, {video_bf16});
  add_operation(Opcode::Linear, {audio, audio_weight, audio_bias},
                {audio_f32}, linear_attrs);
  add_operation(Opcode::Cast, {audio_f32}, {audio_bf16});
  add_operation(Opcode::Linear, {text, context_weight, context_bias},
                {refined_text}, linear_attrs);

  for (std::uint64_t layer = 0; layer < config.refiner_layers; ++layer) {
    const auto qkv_weight =
        add_constant(DType::BF16, {3U * inner, config.hidden});
    const auto q_norm_weight =
        add_constant(DType::BF16, {config.head_dim});
    const auto k_norm_weight =
        add_constant(DType::BF16, {config.head_dim});
    const auto out_weight =
        add_constant(DType::BF16, {config.hidden, inner});
    const auto fc1_weight =
        add_constant(DType::BF16, {2U * config.ffn, config.hidden});
    const auto fc2_weight =
        add_constant(DType::BF16, {config.hidden, config.ffn});
    const auto norm1_weight = add_constant(DType::BF16, {config.hidden});
    const auto norm2_weight = add_constant(DType::BF16, {config.hidden});
    const auto attention_input =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    const auto q_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto k_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto v_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto q = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto k = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto v = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto normalized_q = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto normalized_k = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto attended = add_internal(
        DType::BF16, {config.text_tokens, config.heads, config.head_dim});
    const auto projected =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    const auto after_attention =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    const auto mlp_input =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    const auto fc1 = add_internal(
        DType::BF16, {config.text_tokens, 2U * config.ffn});
    const auto activated =
        add_internal(DType::BF16, {config.text_tokens, config.ffn});
    const auto mlp_output =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    const auto block_output =
        add_internal(DType::BF16, {config.text_tokens, config.hidden});
    add_operation(Opcode::RmsNorm, {refined_text, norm1_weight},
                  {attention_input}, rms_attrs);
    add_operation(Opcode::H3DeinterleaveQkvWeight, {qkv_weight},
                  {q_weight, k_weight, v_weight}, qkv_attrs);
    add_operation(Opcode::Linear, {attention_input, q_weight}, {q},
                  linear_attrs);
    add_operation(Opcode::Linear, {attention_input, k_weight}, {k},
                  linear_attrs);
    add_operation(Opcode::Linear, {attention_input, v_weight}, {v},
                  linear_attrs);
    add_operation(Opcode::RmsNorm, {q, q_norm_weight}, {normalized_q},
                  rms_attrs);
    add_operation(Opcode::RmsNorm, {k, k_norm_weight}, {normalized_k},
                  rms_attrs);
    add_operation(Opcode::Attention, {normalized_q, normalized_k, v},
                  {attended}, attention_attrs);
    add_operation(Opcode::Linear, {attended, out_weight}, {projected},
                  linear_attrs);
    add_operation(Opcode::Add, {refined_text, projected},
                  {after_attention});
    add_operation(Opcode::RmsNorm, {after_attention, norm2_weight},
                  {mlp_input}, rms_attrs);
    add_operation(Opcode::Linear, {mlp_input, fc1_weight}, {fc1},
                  linear_attrs);
    add_operation(Opcode::SwiGlu, {fc1}, {activated}, gate_first_attrs);
    add_operation(Opcode::Linear, {activated, fc2_weight}, {mlp_output},
                  linear_attrs);
    add_operation(Opcode::Add, {after_attention, mlp_output},
                  {block_output});
    refined_text = block_output;
  }
  const auto refiner_final_norm =
      add_constant(DType::BF16, {config.hidden});
  const auto final_text =
      add_internal(DType::BF16, {config.text_tokens, config.hidden});
  add_operation(Opcode::RmsNorm, {refined_text, refiner_final_norm},
                {final_text}, rms_attrs);

  const auto packed_zero =
      add_internal(DType::BF16, {sequence, config.hidden});
  const auto packed_text =
      add_internal(DType::BF16, {sequence, config.hidden});
  const auto packed_video =
      add_internal(DType::BF16, {sequence, config.hidden});
  auto residual = add_internal(DType::BF16, {sequence, config.hidden});
  add_operation(Opcode::Fill, {}, {packed_zero},
                {Attribute::f64(AttrKey::Value, 0.0)});
  add_operation(Opcode::IndexedUpdateRows,
                {packed_zero, final_text, text_map}, {packed_text});
  add_operation(Opcode::IndexedUpdateRows,
                {packed_text, video_bf16, video_map}, {packed_video});
  add_operation(Opcode::IndexedUpdateRows,
                {packed_video, audio_bf16, audio_map}, {residual});

  const auto time_features = add_internal(
      DType::F32, {config.timestep_tables, config.time_input_dim});
  const auto time_hidden = add_internal(
      DType::F32, {config.timestep_tables, config.time_hidden_dim});
  const auto time_activated = add_internal(
      DType::F32, {config.timestep_tables, config.time_hidden_dim});
  const auto temb = add_internal(
      DType::F32, {config.timestep_tables, config.time_embed_dim});
  const auto temb_silu = add_internal(
      DType::F32, {config.timestep_tables, config.time_embed_dim});
  const auto temb_bf16 = add_internal(
      DType::BF16, {config.timestep_tables, config.time_embed_dim});
  const auto cos_id =
      add_internal(DType::BF16, {sequence, config.rotary});
  const auto sin_id =
      add_internal(DType::BF16, {sequence, config.rotary});
  add_operation(
      Opcode::SinusoidalTimestep, {timesteps}, {time_features},
      {Attribute::boolean(AttrKey::FlipSinToCos, true),
       Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
       Attribute::f64(AttrKey::Scale, 1.0),
       Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  add_operation(Opcode::Linear,
                {time_features, time_in_weight, time_in_bias}, {time_hidden},
                linear_attrs);
  add_operation(Opcode::SiLU, {time_hidden}, {time_activated});
  add_operation(Opcode::Linear,
                {time_activated, time_out_weight, time_out_bias}, {temb},
                linear_attrs);
  add_operation(Opcode::SiLU, {temb}, {temb_silu});
  add_operation(Opcode::Cast, {temb_silu}, {temb_bf16});
  add_operation(Opcode::RotaryPosition, {position_ids, rope_inv_freq},
                {cos_id, sin_id});

  for (std::uint64_t layer = 0; layer < config.layers; ++layer) {
    const auto adaln_weight = add_constant(
        DType::BF16, {adaln_width, config.time_embed_dim});
    const auto adaln_bias = add_constant(DType::BF16, {adaln_width});
    const auto qkv_weight =
        add_constant(DType::BF16, {3U * inner, config.hidden});
    const auto q_norm = add_constant(DType::BF16, {config.head_dim});
    const auto k_norm = add_constant(DType::BF16, {config.head_dim});
    const auto out_weight =
        add_constant(DType::BF16, {config.hidden, inner});
    const auto fc1_weight =
        add_constant(DType::BF16, {2U * config.ffn, config.hidden});
    const auto fc2_weight =
        add_constant(DType::BF16, {config.hidden, config.ffn});
    const auto norm1 = add_constant(DType::BF16, {config.hidden});
    const auto norm2 = add_constant(DType::BF16, {config.hidden});
    const auto adaln_biased = add_internal(
        DType::BF16, {config.timestep_tables, adaln_width});
    const auto shift_msa =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto scale_msa =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto gate_msa =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto shift_mlp =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto scale_mlp =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto gate_mlp =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto attention_input =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto q_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto k_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto v_weight =
        add_internal(DType::BF16, {inner, config.hidden});
    const auto q = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto k = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto v = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto normalized_q = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto normalized_k = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto attended = add_internal(
        DType::BF16, {sequence, config.heads, config.head_dim});
    const auto projected =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto after_attention =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto mlp_input =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto fc1 =
        add_internal(DType::BF16, {sequence, 2U * config.ffn});
    const auto activated =
        add_internal(DType::BF16, {sequence, config.ffn});
    const auto mlp_output =
        add_internal(DType::BF16, {sequence, config.hidden});
    const auto block_output =
        add_internal(DType::BF16, {sequence, config.hidden});
    add_operation(Opcode::Linear,
                  {temb_bf16, adaln_weight, adaln_bias}, {adaln_biased},
                  linear_attrs);
    add_operation(Opcode::H3AdaLNSelect,
                  {adaln_biased, adaln_indices},
                  {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
                   gate_mlp},
                  linear_attrs);
    add_operation(Opcode::RmsNormModulate,
                  {residual, norm1, scale_msa, shift_msa}, {attention_input},
                  rms_attrs);
    add_operation(Opcode::H3DeinterleaveQkvWeight, {qkv_weight},
                  {q_weight, k_weight, v_weight}, qkv_attrs);
    add_operation(Opcode::Linear, {attention_input, q_weight}, {q},
                  linear_attrs);
    add_operation(Opcode::Linear, {attention_input, k_weight}, {k},
                  linear_attrs);
    add_operation(Opcode::Linear, {attention_input, v_weight}, {v},
                  linear_attrs);
    add_operation(Opcode::QkNormPartialRope,
                  {q, q_norm, cos_id, sin_id}, {normalized_q}, qk_attrs);
    add_operation(Opcode::QkNormPartialRope,
                  {k, k_norm, cos_id, sin_id}, {normalized_k}, qk_attrs);
    add_operation(Opcode::Attention,
                  {normalized_q, normalized_k, v}, {attended},
                  attention_attrs);
    add_operation(Opcode::Linear, {attended, out_weight}, {projected},
                  linear_attrs);
    add_operation(Opcode::ResidualGate,
                  {residual, projected, gate_msa}, {after_attention},
                  linear_attrs);
    add_operation(Opcode::RmsNormModulate,
                  {after_attention, norm2, scale_mlp, shift_mlp}, {mlp_input},
                  rms_attrs);
    add_operation(Opcode::Linear, {mlp_input, fc1_weight}, {fc1},
                  linear_attrs);
    add_operation(Opcode::SwiGlu, {fc1}, {activated}, gate_first_attrs);
    add_operation(Opcode::Linear, {activated, fc2_weight}, {mlp_output},
                  linear_attrs);
    add_operation(Opcode::ResidualGate,
                  {after_attention, mlp_output, gate_mlp}, {block_output},
                  linear_attrs);
    residual = block_output;
  }

  const auto final_adaln_weight = add_constant(
      DType::BF16, {2U * config.hidden, config.time_embed_dim});
  const auto final_adaln_bias =
      add_constant(DType::BF16, {2U * config.hidden});
  const auto final_norm_weight =
      add_constant(DType::BF16, {config.hidden});
  const auto final_modulation = add_internal(
      DType::BF16, {config.timestep_tables, 2U * config.hidden});
  const auto final_shift =
      add_internal(DType::BF16, {sequence, config.hidden});
  const auto final_scale =
      add_internal(DType::BF16, {sequence, config.hidden});
  const auto normalized =
      add_internal(DType::BF16, {sequence, config.hidden});
  const auto normalized_f32 =
      add_internal(DType::F32, {sequence, config.hidden});
  add_operation(Opcode::Linear,
                {temb_bf16, final_adaln_weight, final_adaln_bias},
                {final_modulation}, linear_attrs);
  add_operation(Opcode::SelectRowChunks,
                {final_modulation, timestep_indices},
                {final_shift, final_scale});
  add_operation(Opcode::RmsNormModulate,
                {residual, final_norm_weight, final_scale, final_shift},
                {normalized}, rms_attrs);
  add_operation(Opcode::Cast, {normalized}, {normalized_f32});

  const auto video_out_weight = add_constant(
      DType::F32, {config.video_input_dim, config.hidden});
  const auto video_out_bias =
      add_constant(DType::F32, {config.video_input_dim});
  const auto audio_out_weight = add_constant(
      DType::F32, {config.audio_input_dim, config.hidden});
  const auto audio_out_bias =
      add_constant(DType::F32, {config.audio_input_dim});
  const auto video_all =
      add_internal(DType::F32, {sequence, config.video_input_dim});
  const auto video_output =
      add_output(DType::F32, {config.video_tokens, config.video_input_dim});
  const auto audio_all =
      add_internal(DType::F32, {sequence, config.audio_input_dim});
  const auto audio_output =
      add_output(DType::F32, {config.audio_tokens, config.audio_input_dim});
  add_operation(Opcode::Linear,
                {normalized_f32, video_out_weight, video_out_bias},
                {video_all}, linear_attrs);
  add_operation(Opcode::GatherRows, {video_all, video_indices},
                {video_output});
  add_operation(Opcode::Linear,
                {normalized_f32, audio_out_weight, audio_out_bias},
                {audio_all}, linear_attrs);
  add_operation(Opcode::GatherRows, {audio_all, audio_indices},
                {audio_output});
  ir::verify(program);
  return program;
}

ir::Program make_h3_stack_bf16(std::uint64_t sequence,
                               std::uint64_t hidden,
                               std::uint64_t heads,
                               std::uint64_t head_dim,
                               std::uint64_t ffn,
                               std::uint64_t rotary,
                               std::uint64_t layers,
                               std::uint64_t block_size,
                               bool streamed_constants) {
  using namespace ir;
  if (sequence == 0U || hidden == 0U || heads == 0U || head_dim == 0U ||
      ffn == 0U || layers == 0U || rotary == 0U || rotary > head_dim ||
      (rotary % 2U) != 0U ||
      heads > std::numeric_limits<std::uint64_t>::max() / head_dim ||
      ffn > std::numeric_limits<std::uint64_t>::max() / 2U)
    fail("invalid H3 stack geometry");
  const auto inner = heads * head_dim;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (streamed_constants ? TensorRole::Streamed : TensorRole::Internal));
  Program program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto residual_input = next_tensor++;
  const auto cos_id = next_tensor++;
  const auto sin_id = next_tensor++;
  program.tensors.push_back(tensor(residual_input, TensorRole::Input,
                                   {sequence, hidden}));
  program.tensors.push_back(tensor(cos_id, constant_roles, {sequence, rotary}));
  program.tensors.push_back(tensor(sin_id, constant_roles, {sequence, rotary}));

  const auto rms_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto qk_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::RotaryDim, rotary),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, block_size),
      Attribute::u64(AttrKey::Implementation, 1U),
  };
  const auto attention_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::AttentionScale,
                     1.0 / std::sqrt(static_cast<double>(head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::BlockSize, 64U),
  };

  auto residual = residual_input;
  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    auto add_constant = [&](std::vector<std::uint64_t> dims) {
      const auto id = next_tensor++;
      program.tensors.push_back(tensor(id, constant_roles, std::move(dims)));
      return id;
    };
    auto add_internal = [&](std::vector<std::uint64_t> dims,
                            bool final_output = false) {
      const auto id = next_tensor++;
      program.tensors.push_back(tensor(
          id, final_output ? static_cast<std::uint32_t>(TensorRole::Output)
                           : static_cast<std::uint32_t>(TensorRole::Internal),
          std::move(dims)));
      return id;
    };

    const auto scale_msa = add_constant({sequence, hidden});
    const auto shift_msa = add_constant({sequence, hidden});
    const auto gate_msa = add_constant({sequence, hidden});
    const auto wq = add_constant({inner, hidden});
    const auto wk = add_constant({inner, hidden});
    const auto wv = add_constant({inner, hidden});
    const auto q_norm = add_constant({head_dim});
    const auto k_norm = add_constant({head_dim});
    const auto wout = add_constant({hidden, inner});
    const auto scale_mlp = add_constant({sequence, hidden});
    const auto shift_mlp = add_constant({sequence, hidden});
    const auto gate_mlp = add_constant({sequence, hidden});
    const auto wfc1 = add_constant({2U * ffn, hidden});
    const auto wfc2 = add_constant({hidden, ffn});
    const auto norm1 = add_constant({hidden});
    const auto norm2 = add_constant({hidden});

    const auto attn_in = add_internal({sequence, hidden});
    const auto q = add_internal({sequence, heads, head_dim});
    const auto k = add_internal({sequence, heads, head_dim});
    const auto v = add_internal({sequence, heads, head_dim});
    const auto qn = add_internal({sequence, heads, head_dim});
    const auto kn = add_internal({sequence, heads, head_dim});
    const auto attention = add_internal({sequence, heads, head_dim});
    const auto projected = add_internal({sequence, hidden});
    const auto x1 = add_internal({sequence, hidden});
    const auto mlp_in = add_internal({sequence, hidden});
    const auto fc1 = add_internal({sequence, 2U * ffn});
    const auto activation = add_internal({sequence, ffn});
    const auto mlp_out = add_internal({sequence, hidden});
    const auto output = add_internal({sequence, hidden}, layer + 1U == layers);

    program.operations.push_back(
        {next_operation++, Opcode::RmsNormModulate,
         {residual, norm1, scale_msa, shift_msa}, {attn_in}, rms_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {attn_in, wq}, {q}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {attn_in, wk}, {k}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {attn_in, wv}, {v}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::QkNormPartialRope,
         {q, q_norm, cos_id, sin_id}, {qn}, qk_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::QkNormPartialRope,
         {k, k_norm, cos_id, sin_id}, {kn}, qk_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Attention, {qn, kn, v}, {attention},
         attention_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {attention, wout}, {projected},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::ResidualGate,
         {residual, projected, gate_msa}, {x1}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::RmsNormModulate,
         {x1, norm2, scale_mlp, shift_mlp}, {mlp_in}, rms_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {mlp_in, wfc1}, {fc1}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::SwiGlu, {fc1}, {activation}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {activation, wfc2}, {mlp_out},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::ResidualGate,
         {x1, mlp_out, gate_mlp}, {output}, linear_attrs});
    residual = output;
  }
  return program;
}

ir::Program make_h3_transformer_bf16(
    std::uint64_t sequence, std::uint64_t hidden, std::uint64_t heads,
    std::uint64_t head_dim, std::uint64_t ffn, std::uint64_t rotary,
    std::uint64_t layers, std::uint64_t timestep_tables,
    std::uint64_t time_embed_dim, std::uint64_t block_size,
    bool streamed_constants, bool source_shaped_qkv,
    std::uint64_t attention_implementation) {
  using namespace ir;
  if (sequence == 0U || hidden == 0U || heads == 0U || head_dim == 0U ||
      ffn == 0U || layers == 0U || timestep_tables == 0U ||
      time_embed_dim == 0U || rotary == 0U || rotary > head_dim ||
      (rotary % 2U) != 0U ||
      (attention_implementation != 1U && attention_implementation != 2U) ||
      heads > std::numeric_limits<std::uint64_t>::max() / head_dim ||
      hidden > std::numeric_limits<std::uint64_t>::max() / 18U ||
      ffn > std::numeric_limits<std::uint64_t>::max() / 2U)
    fail("invalid released-layout H3 transformer geometry");
  const auto inner = heads * head_dim;
  const auto adaln_width = 18U * hidden;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (streamed_constants ? TensorRole::Streamed : TensorRole::Internal));
  Program program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto residual_input = next_tensor++;
  const auto temb_silu = next_tensor++;
  const auto adaln_indices = next_tensor++;
  const auto cos_id = next_tensor++;
  const auto sin_id = next_tensor++;
  program.tensors.push_back(
      tensor(residual_input, TensorRole::Input, {sequence, hidden}));
  program.tensors.push_back(
      tensor(temb_silu, TensorRole::Input, {timestep_tables, time_embed_dim}));
  program.tensors.push_back(
      {adaln_indices, DType::I32, TensorRole::Input, {sequence}});
  program.tensors.push_back(tensor(cos_id, constant_roles, {sequence, rotary}));
  program.tensors.push_back(tensor(sin_id, constant_roles, {sequence, rotary}));

  const auto rms_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto qk_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::RotaryDim, rotary),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, block_size),
      Attribute::u64(AttrKey::Implementation, 1U),
  };
  auto gate_first_attrs = linear_attrs;
  gate_first_attrs.push_back(Attribute::boolean(AttrKey::GateFirst, true));
  const auto attention_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::AttentionScale,
                     1.0 / std::sqrt(static_cast<double>(head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::BlockSize, 64U),
      Attribute::u64(AttrKey::Implementation, attention_implementation),
  };
  const auto qkv_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };

  auto residual = residual_input;
  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    auto add_constant = [&](std::vector<std::uint64_t> dims) {
      const auto id = next_tensor++;
      program.tensors.push_back(tensor(id, constant_roles, std::move(dims)));
      return id;
    };
    auto add_internal = [&](std::vector<std::uint64_t> dims,
                            bool final_output = false) {
      const auto id = next_tensor++;
      program.tensors.push_back(tensor(
          id, final_output ? static_cast<std::uint32_t>(TensorRole::Output)
                           : static_cast<std::uint32_t>(TensorRole::Internal),
          std::move(dims)));
      return id;
    };

    const auto adaln_weight = add_constant({adaln_width, time_embed_dim});
    const auto adaln_bias = add_constant({adaln_width});
    const auto qkv_weight = add_constant({3U * inner, hidden});
    const auto q_norm = add_constant({head_dim});
    const auto k_norm = add_constant({head_dim});
    const auto wout = add_constant({hidden, inner});
    const auto fc1_weight = add_constant({2U * ffn, hidden});
    const auto fc2_weight = add_constant({hidden, ffn});
    const auto norm1 = add_constant({hidden});
    const auto norm2 = add_constant({hidden});

    const auto adaln_biased = add_internal({timestep_tables, adaln_width});
    const auto shift_msa = add_internal({sequence, hidden});
    const auto scale_msa = add_internal({sequence, hidden});
    const auto gate_msa = add_internal({sequence, hidden});
    const auto shift_mlp = add_internal({sequence, hidden});
    const auto scale_mlp = add_internal({sequence, hidden});
    const auto gate_mlp = add_internal({sequence, hidden});
    const auto attn_in = add_internal({sequence, hidden});
    std::uint32_t packed_qkv = 0U;
    std::uint32_t q_weight = 0U;
    std::uint32_t k_weight = 0U;
    std::uint32_t v_weight = 0U;
    if (source_shaped_qkv) {
      q_weight = add_internal({inner, hidden});
      k_weight = add_internal({inner, hidden});
      v_weight = add_internal({inner, hidden});
    } else {
      packed_qkv = add_internal({sequence, 3U * inner});
    }
    const auto q = add_internal({sequence, heads, head_dim});
    const auto k = add_internal({sequence, heads, head_dim});
    const auto v = add_internal({sequence, heads, head_dim});
    const auto qn = add_internal({sequence, heads, head_dim});
    const auto kn = add_internal({sequence, heads, head_dim});
    const auto attention = add_internal({sequence, heads, head_dim});
    const auto projected = add_internal({sequence, hidden});
    const auto x1 = add_internal({sequence, hidden});
    const auto mlp_in = add_internal({sequence, hidden});
    const auto fc1 = add_internal({sequence, 2U * ffn});
    const auto activation = add_internal({sequence, ffn});
    const auto mlp_out = add_internal({sequence, hidden});
    const auto output = add_internal({sequence, hidden}, layer + 1U == layers);

    program.operations.push_back(
        {next_operation++, Opcode::Linear,
         {temb_silu, adaln_weight, adaln_bias},
         {adaln_biased}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::H3AdaLNSelect,
         {adaln_biased, adaln_indices},
         {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::RmsNormModulate,
         {residual, norm1, scale_msa, shift_msa}, {attn_in}, rms_attrs});
    if (source_shaped_qkv) {
      program.operations.push_back(
          {next_operation++, Opcode::H3DeinterleaveQkvWeight, {qkv_weight},
           {q_weight, k_weight, v_weight}, qkv_attrs});
      program.operations.push_back(
          {next_operation++, Opcode::Linear, {attn_in, q_weight}, {q},
           linear_attrs});
      program.operations.push_back(
          {next_operation++, Opcode::Linear, {attn_in, k_weight}, {k},
           linear_attrs});
      program.operations.push_back(
          {next_operation++, Opcode::Linear, {attn_in, v_weight}, {v},
           linear_attrs});
    } else {
      program.operations.push_back(
          {next_operation++, Opcode::Linear, {attn_in, qkv_weight}, {packed_qkv},
           linear_attrs});
      program.operations.push_back(
          {next_operation++, Opcode::H3DeinterleaveQkv, {packed_qkv}, {q, k, v},
           qkv_attrs});
    }
    program.operations.push_back(
        {next_operation++, Opcode::QkNormPartialRope,
         {q, q_norm, cos_id, sin_id}, {qn}, qk_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::QkNormPartialRope,
         {k, k_norm, cos_id, sin_id}, {kn}, qk_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Attention, {qn, kn, v}, {attention},
         attention_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {attention, wout}, {projected},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::ResidualGate,
         {residual, projected, gate_msa}, {x1}, linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::RmsNormModulate,
         {x1, norm2, scale_mlp, shift_mlp}, {mlp_in}, rms_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {mlp_in, fc1_weight}, {fc1},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::SwiGlu, {fc1}, {activation}, gate_first_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::Linear, {activation, fc2_weight}, {mlp_out},
         linear_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::ResidualGate,
         {x1, mlp_out, gate_mlp}, {output}, linear_attrs});
    residual = output;
  }
  return program;
}

ir::Program make_h3_token_refiner_bf16(
    std::uint64_t sequence, std::uint64_t hidden, std::uint64_t heads,
    std::uint64_t head_dim, std::uint64_t ffn, std::uint64_t layers,
    std::uint64_t block_size, bool streamed_constants) {
  using namespace ir;
  if (sequence == 0U || hidden == 0U || heads == 0U || head_dim == 0U ||
      ffn == 0U || layers == 0U ||
      heads > std::numeric_limits<std::uint64_t>::max() / head_dim ||
      ffn > std::numeric_limits<std::uint64_t>::max() / 2U ||
      heads * head_dim == 0U)
    fail("invalid H3 token-refiner geometry");
  const auto inner = heads * head_dim;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (streamed_constants ? TensorRole::Streamed : TensorRole::Internal));
  Program program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto residual = next_tensor++;
  program.tensors.push_back(
      tensor(residual, TensorRole::Input, {sequence, hidden}));

  const auto norm_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto qkv_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::BlockSize, block_size),
  };
  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, block_size),
      Attribute::u64(AttrKey::Implementation, 1U),
  };
  auto gate_first_attrs = linear_attrs;
  gate_first_attrs.push_back(Attribute::boolean(AttrKey::GateFirst, true));
  const auto attention_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::AttentionScale,
                     1.0 / std::sqrt(static_cast<double>(head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::BlockSize, 64U),
  };

  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    auto add_constant = [&](std::vector<std::uint64_t> dims) {
      const auto id = next_tensor++;
      program.tensors.push_back(tensor(id, constant_roles, std::move(dims)));
      return id;
    };
    auto add_internal = [&](std::vector<std::uint64_t> dims) {
      const auto id = next_tensor++;
      program.tensors.push_back(
          tensor(id, TensorRole::Internal, std::move(dims)));
      return id;
    };

    const auto qkv_weight = add_constant({3U * inner, hidden});
    const auto q_norm_weight = add_constant({head_dim});
    const auto k_norm_weight = add_constant({head_dim});
    const auto out_weight = add_constant({hidden, inner});
    const auto fc1_weight = add_constant({2U * ffn, hidden});
    const auto fc2_weight = add_constant({hidden, ffn});
    const auto norm1_weight = add_constant({hidden});
    const auto norm2_weight = add_constant({hidden});

    const auto attention_input = add_internal({sequence, hidden});
    const auto q_weight = add_internal({inner, hidden});
    const auto k_weight = add_internal({inner, hidden});
    const auto v_weight = add_internal({inner, hidden});
    const auto q = add_internal({sequence, heads, head_dim});
    const auto k = add_internal({sequence, heads, head_dim});
    const auto v = add_internal({sequence, heads, head_dim});
    const auto normalized_q = add_internal({sequence, heads, head_dim});
    const auto normalized_k = add_internal({sequence, heads, head_dim});
    const auto attended = add_internal({sequence, heads, head_dim});
    const auto projected = add_internal({sequence, hidden});
    const auto after_attention = add_internal({sequence, hidden});
    const auto mlp_input = add_internal({sequence, hidden});
    const auto fc1 = add_internal({sequence, 2U * ffn});
    const auto activated = add_internal({sequence, ffn});
    const auto mlp_output = add_internal({sequence, hidden});
    const auto block_output = add_internal({sequence, hidden});

    program.operations.push_back({next_operation++, Opcode::RmsNorm,
                                  {residual, norm1_weight}, {attention_input},
                                  norm_attrs});
    program.operations.push_back(
        {next_operation++, Opcode::H3DeinterleaveQkvWeight, {qkv_weight},
         {q_weight, k_weight, v_weight}, qkv_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {attention_input, q_weight}, {q},
                                  linear_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {attention_input, k_weight}, {k},
                                  linear_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {attention_input, v_weight}, {v},
                                  linear_attrs});
    program.operations.push_back({next_operation++, Opcode::RmsNorm,
                                  {q, q_norm_weight}, {normalized_q},
                                  norm_attrs});
    program.operations.push_back({next_operation++, Opcode::RmsNorm,
                                  {k, k_norm_weight}, {normalized_k},
                                  norm_attrs});
    program.operations.push_back({next_operation++, Opcode::Attention,
                                  {normalized_q, normalized_k, v}, {attended},
                                  attention_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {attended, out_weight}, {projected},
                                  linear_attrs});
    program.operations.push_back({next_operation++, Opcode::Add,
                                  {residual, projected}, {after_attention}, {}});
    program.operations.push_back({next_operation++, Opcode::RmsNorm,
                                  {after_attention, norm2_weight}, {mlp_input},
                                  norm_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {mlp_input, fc1_weight}, {fc1}, linear_attrs});
    program.operations.push_back({next_operation++, Opcode::SwiGlu, {fc1},
                                  {activated}, gate_first_attrs});
    program.operations.push_back({next_operation++, Opcode::Linear,
                                  {activated, fc2_weight}, {mlp_output},
                                  linear_attrs});
    program.operations.push_back({next_operation++, Opcode::Add,
                                  {after_attention, mlp_output}, {block_output},
                                  {}});
    residual = block_output;
  }

  const auto final_norm_weight = next_tensor++;
  const auto output = next_tensor++;
  program.tensors.push_back(
      tensor(final_norm_weight, constant_roles, {hidden}));
  program.tensors.push_back(
      tensor(output, TensorRole::Output, {sequence, hidden}));
  program.operations.push_back({next_operation++, Opcode::RmsNorm,
                                {residual, final_norm_weight}, {output},
                                norm_attrs});
  ir::verify(program);
  return program;
}

ir::Program make_h3_block_raw_bf16(std::uint64_t sequence,
                                   std::uint64_t hidden,
                                   std::uint64_t heads,
                                   std::uint64_t head_dim,
                                   std::uint64_t ffn,
                                   std::uint64_t rotary,
                                   std::uint64_t block_size,
                                   bool streamed_constants) {
  using namespace ir;
  if (sequence == 0U || hidden == 0U || heads == 0U || head_dim == 0U ||
      ffn == 0U || rotary == 0U || rotary > head_dim || (rotary % 2U) != 0U ||
      heads > std::numeric_limits<std::uint64_t>::max() / head_dim ||
      ffn > std::numeric_limits<std::uint64_t>::max() / 2U)
    fail("invalid raw-layout H3 block geometry");
  const auto inner = heads * head_dim;
  const auto constants = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (streamed_constants ? TensorRole::Streamed : TensorRole::Internal));
  Program program;
  program.tensors = {
      tensor(1, TensorRole::Input, {sequence, hidden}),
      tensor(2, TensorRole::Input, {sequence, hidden}),
      tensor(3, TensorRole::Input, {sequence, hidden}),
      tensor(4, TensorRole::Input, {sequence, hidden}),
      tensor(5, constants, {3U * inner, hidden}),
      tensor(6, constants, {head_dim}),
      tensor(7, constants, {head_dim}),
      tensor(8, constants, {sequence, rotary}),
      tensor(9, constants, {sequence, rotary}),
      tensor(10, constants, {hidden, inner}),
      tensor(11, TensorRole::Input, {sequence, hidden}),
      tensor(12, TensorRole::Input, {sequence, hidden}),
      tensor(13, TensorRole::Input, {sequence, hidden}),
      tensor(14, constants, {2U * ffn, hidden}),
      tensor(15, constants, {hidden, ffn}),
      tensor(16, constants, {hidden}),
      tensor(17, constants, {hidden}),
      tensor(18, TensorRole::Internal, {sequence, hidden}),
      tensor(19, TensorRole::Internal, {sequence, 3U * inner}),
      tensor(20, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(21, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(22, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(23, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(24, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(25, TensorRole::Internal, {sequence, heads, head_dim}),
      tensor(26, TensorRole::Internal, {sequence, hidden}),
      tensor(27, TensorRole::Internal, {sequence, hidden}),
      tensor(28, TensorRole::Internal, {sequence, hidden}),
      tensor(29, TensorRole::Internal, {sequence, 2U * ffn}),
      tensor(30, TensorRole::Internal, {sequence, ffn}),
      tensor(31, TensorRole::Internal, {sequence, hidden}),
      tensor(32, TensorRole::Output, {sequence, hidden}),
  };
  const auto rms = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, block_size)};
  const auto linear = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, block_size),
      Attribute::u64(AttrKey::Implementation, 1U)};
  auto gate_first = linear;
  gate_first.push_back(Attribute::boolean(AttrKey::GateFirst, true));
  const auto qkv = std::vector<Attribute>{
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::BlockSize, block_size)};
  const auto qk = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, head_dim),
      Attribute::u64(AttrKey::RotaryDim, rotary),
      Attribute::u64(AttrKey::BlockSize, block_size)};
  program.operations = {
      {1, Opcode::RmsNormModulate, {1, 16, 2, 3}, {18}, rms},
      {2, Opcode::Linear, {18, 5}, {19}, linear},
      {3, Opcode::H3DeinterleaveQkv, {19}, {20, 21, 22}, qkv},
      {4, Opcode::QkNormPartialRope, {20, 6, 8, 9}, {23}, qk},
      {5, Opcode::QkNormPartialRope, {21, 7, 8, 9}, {24}, qk},
      {6, Opcode::Attention,
       {23, 24, 22}, {25},
       {Attribute::f64(AttrKey::AttentionScale,
                       1.0 / std::sqrt(static_cast<double>(head_dim))),
        Attribute::boolean(AttrKey::Causal, false),
        Attribute::u64(AttrKey::BlockSize, 64U)}},
      {7, Opcode::Linear, {25, 10}, {26}, linear},
      {8, Opcode::ResidualGate, {1, 26, 4}, {27}, linear},
      {9, Opcode::RmsNormModulate, {27, 17, 11, 12}, {28}, rms},
      {10, Opcode::Linear, {28, 14}, {29}, linear},
      {11, Opcode::SwiGlu, {29}, {30}, gate_first},
      {12, Opcode::Linear, {30, 15}, {31}, linear},
      {13, Opcode::ResidualGate, {27, 31, 13}, {32}, linear},
  };
  return program;
}

} // namespace dif::frontend
