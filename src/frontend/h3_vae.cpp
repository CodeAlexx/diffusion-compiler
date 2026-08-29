#include "dif/frontend/h3_vae.hpp"

#include "dif/ir/verify.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right,
                          const char *label) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left)
    fail(std::string("H3 video VAE ") + label + " overflows");
  return left * right;
}

runtime::Tensor float_tensor(DType dtype, std::vector<std::uint64_t> dims,
                             const std::vector<float> &values) {
  runtime::Tensor tensor{dtype, std::move(dims), {}};
  tensor.bytes.resize(static_cast<std::size_t>(tensor.element_count() *
                                               ir::dtype_size(dtype)));
  tensor.validate();
  if (tensor.element_count() != values.size())
    fail("H3 video VAE generated constant shape mismatch");
  for (std::size_t index = 0; index < values.size(); ++index)
    runtime::store_float(tensor, index, values[index]);
  return tensor;
}

runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                           const std::vector<std::int32_t> &values) {
  runtime::Tensor tensor{DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  tensor.validate();
  if (tensor.element_count() != values.size())
    fail("H3 video VAE generated index shape mismatch");
  auto *output = reinterpret_cast<std::int32_t *>(tensor.bytes.data());
  for (std::size_t index = 0; index < values.size(); ++index)
    output[index] = values[index];
  return tensor;
}

constexpr std::array<float, 24> kLatentMean = {
    0.858090341091156F,   -0.9606591463088989F, 1.0661640167236328F,
    -0.5090325474739075F, -0.2727581858634949F, -1.3675414323806763F,
    -0.2553254961967468F, -0.26907554268836975F, -0.5376840829849243F,
    -0.0464097298681736F, 0.6657370328903198F,  0.19690127670764923F,
    -0.5460608005523682F, -0.4035342037677765F, -0.23683024942874908F,
    0.25928452610969543F, -0.30133944749832153F, 0.211341992020607F,
    -1.1206848621368408F, 0.3581933379173279F,  -0.04225143790245056F,
    0.2604829967021942F,  0.22864092886447906F, 0.7056031823158264F,
};

constexpr std::array<float, 24> kLatentStd = {
    1.2223774194717407F, 1.2767263650894165F, 1.6831774711608887F,
    1.7549455165863037F, 1.5636216402053833F, 2.194143533706665F,
    0.9653137922286987F, 1.0569885969161987F, 0.841948926448822F,
    0.7729952931404114F, 1.8955937623977661F, 0.946841835975647F,
    0.7996809482574463F, 0.44988900423049927F, 0.7197399735450745F,
    0.6936293244361877F, 2.961095094680786F,  2.7694199085235596F,
    3.0496184825897217F, 2.1088054180145264F, 3.276226282119751F,
    3.1627357006073F,    2.2816812992095947F, 2.6127843856811523F,
};

} // namespace

H3VideoVaeBuild make_h3_video_vae_decoder(const H3VideoVaeConfig &config) {
  if (config.latent_frames == 0U || config.latent_height == 0U ||
      config.latent_width == 0U || config.layers == 0U ||
      config.latent_channels != kLatentMean.size() ||
      config.output_channels != 3U || config.heads == 0U ||
      config.head_dim == 0U || config.ffn == 0U ||
      config.register_tokens == 0U || config.patch_t == 0U ||
      config.patch_h == 0U || config.patch_w == 0U ||
      config.rotary_dim == 0U || config.rotary_dim > config.head_dim ||
      config.rotary_dim % 6U != 0U || !(config.rope_theta > 0.0) ||
      (config.attention_implementation != 1U &&
       config.attention_implementation != 2U))
    fail("invalid released H3 video VAE decoder configuration");
  const auto hidden = checked_mul(config.heads, config.head_dim, "hidden");
  if (hidden != 2048U)
    fail("released H3 video VAE hidden size must be 2048");
  const auto patches = checked_mul(
      checked_mul(config.latent_frames, config.latent_height, "patch rows"),
      config.latent_width, "patch rows");
  const auto suffix = config.register_tokens + 1U;
  if (patches > std::numeric_limits<std::uint64_t>::max() - suffix)
    fail("H3 video VAE sequence length overflows");
  const auto sequence = patches + suffix;
  const auto patch_volume = checked_mul(
      checked_mul(config.patch_t, config.patch_h, "pixel patch"),
      config.patch_w, "pixel patch");
  const auto patch_dim =
      checked_mul(config.output_channels, patch_volume, "pixel patch");
  const auto frequencies = config.rotary_dim / 6U;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed : TensorRole::Internal));

  H3VideoVaeBuild build;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto add_tensor = [&](DType dtype, std::uint32_t roles,
                        std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  auto add_internal = [&](DType dtype, std::vector<std::uint64_t> dims,
                          bool output = false) {
    return add_tensor(dtype,
                      output ? static_cast<std::uint32_t>(TensorRole::Output)
                             : static_cast<std::uint32_t>(TensorRole::Internal),
                      std::move(dims));
  };
  auto add_constant = [&](std::string name, std::string source_name,
                          DType dtype, std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(dtype, constant_roles, std::move(dims));
    build.bindings.push_back(
        {id, std::move(name), std::move(source_name)});
    return id;
  };
  auto add_generated = [&](std::string name, runtime::Tensor tensor) {
    const auto dtype = tensor.dtype;
    const auto dims = tensor.dims;
    const auto id = add_constant(name, {}, dtype, dims);
    build.generated_constants.emplace(id, std::move(tensor));
    return id;
  };
  auto op = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                std::vector<std::uint32_t> outputs,
                std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };

  const auto latent = add_tensor(
      DType::F32, TensorRole::Input,
      {1U, config.latent_channels, config.latent_frames,
       config.latent_height, config.latent_width});
  const auto latent_std = add_generated(
      "dif.latents_std",
      float_tensor(DType::F32, {config.latent_channels},
                   std::vector<float>(kLatentStd.begin(), kLatentStd.end())));
  const auto latent_mean = add_generated(
      "dif.latents_mean",
      float_tensor(DType::F32, {config.latent_channels},
                   std::vector<float>(kLatentMean.begin(), kLatentMean.end())));

  std::vector<float> position_values;
  position_values.reserve(static_cast<std::size_t>(sequence * 3U));
  constexpr double two_pi = 6.283185307179586476925286766559;
  for (std::uint64_t frame = 0; frame < config.latent_frames; ++frame) {
    for (std::uint64_t y = 0; y < config.latent_height; ++y) {
      for (std::uint64_t x = 0; x < config.latent_width; ++x) {
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> axes{{
            {frame, config.latent_frames}, {y, config.latent_height},
            {x, config.latent_width}}};
        for (const auto &[index, size] : axes) {
          const auto normalized =
              2.0 * ((static_cast<double>(index) + 0.5) /
                     static_cast<double>(size)) -
              1.0;
          position_values.push_back(
              static_cast<float>(normalized * two_pi));
        }
      }
    }
  }
  position_values.insert(position_values.end(),
                         static_cast<std::size_t>(suffix * 3U), 0.0F);
  const auto positions = add_generated(
      "dif.position_ids_2pi",
      float_tensor(DType::F32, {sequence, 3U}, position_values));
  std::vector<float> inv_frequency(frequencies);
  for (std::uint64_t index = 0; index < frequencies; ++index) {
    const auto exponent =
        static_cast<double>(index) / static_cast<double>(frequencies);
    inv_frequency[index] = static_cast<float>(
        1.0 / std::pow(config.rope_theta, exponent));
  }
  const auto inv_freq = add_generated(
      "dif.rope_inv_freq",
      float_tensor(DType::F32, {frequencies}, inv_frequency));
  const auto qk_ones = add_generated(
      "dif.qk_norm_ones",
      float_tensor(DType::F16, {config.head_dim},
                   std::vector<float>(config.head_dim, 1.0F)));
  std::vector<std::int32_t> patch_map(sequence, -1);
  std::vector<std::int32_t> register_map(sequence, -1);
  std::vector<std::int32_t> patch_indices(patches);
  for (std::uint64_t index = 0; index < patches; ++index) {
    if (index > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()))
      fail("H3 video VAE patch index exceeds i32");
    patch_map[index] = static_cast<std::int32_t>(index);
    patch_indices[index] = static_cast<std::int32_t>(index);
  }
  for (std::uint64_t index = 0; index < config.register_tokens; ++index)
    register_map[patches + index] = static_cast<std::int32_t>(index);
  const auto patch_map_id = add_generated(
      "dif.patch_insert_map", i32_tensor({sequence}, patch_map));
  const auto register_map_id = add_generated(
      "dif.register_insert_map", i32_tensor({sequence}, register_map));
  const auto patch_indices_id = add_generated(
      "dif.patch_indices", i32_tensor({patches}, patch_indices));

  std::vector<float> pixel_scale(patch_dim);
  std::vector<float> pixel_bias(patch_dim);
  constexpr std::array<float, 3> imagenet_std = {0.229F, 0.224F, 0.225F};
  constexpr std::array<float, 3> imagenet_mean = {0.485F, 0.456F, 0.406F};
  for (std::uint64_t column = 0; column < patch_dim; ++column) {
    const auto channel = column / patch_volume;
    pixel_scale[column] = imagenet_std[channel];
    pixel_bias[column] = imagenet_mean[channel];
  }
  const auto pixel_scale_id = add_generated(
      "dif.imagenet_std", float_tensor(DType::F32, {patch_dim}, pixel_scale));
  const auto pixel_bias_id = add_generated(
      "dif.imagenet_mean", float_tensor(DType::F32, {patch_dim}, pixel_bias));

  const auto post_weight = add_constant(
      "post_quant_conv.weight", "post_quant_conv.weight", DType::F16,
      {config.latent_channels, config.latent_channels});
  const auto post_bias = add_constant(
      "post_quant_conv.bias", "post_quant_conv.bias", DType::F16,
      {config.latent_channels});
  const auto embed_weight = add_constant(
      "decoder.x_embedder.weight", "decoder.x_embedder.weight", DType::F32,
      {hidden, config.latent_channels});
  const auto embed_bias = add_constant(
      "decoder.x_embedder.bias", "decoder.x_embedder.bias", DType::F32,
      {hidden});
  const auto registers = add_constant(
      "decoder.register_tokens", "decoder.register_tokens", DType::F32,
      {config.register_tokens, hidden});

  const auto latent_rows = add_internal(DType::F32, {patches, config.latent_channels});
  const auto denormalized_rows =
      add_internal(DType::F32, {patches, config.latent_channels});
  const auto post_input = add_internal(DType::F16, {patches, config.latent_channels});
  const auto post_output = add_internal(DType::F16, {patches, config.latent_channels});
  const auto embed_input = add_internal(DType::F32, {patches, config.latent_channels});
  const auto patch_tokens = add_internal(DType::F32, {patches, hidden});
  const auto zero_sequence = add_internal(DType::F32, {sequence, hidden});
  const auto with_patches = add_internal(DType::F32, {sequence, hidden});
  auto residual = add_internal(DType::F32, {sequence, hidden});
  const auto cosine = add_internal(DType::F16, {sequence, config.rotary_dim});
  const auto sine = add_internal(DType::F16, {sequence, config.rotary_dim});

  op(Opcode::Patchify3D, {latent}, {latent_rows},
     {Attribute::u64(AttrKey::PatchT, 1U),
      Attribute::u64(AttrKey::PatchH, 1U),
      Attribute::u64(AttrKey::PatchW, 1U)});
  op(Opcode::AffineLastDim, {latent_rows, latent_std, latent_mean},
     {denormalized_rows});
  op(Opcode::Cast, {denormalized_rows}, {post_input});
  op(Opcode::Linear, {post_input, post_weight, post_bias}, {post_output});
  op(Opcode::Cast, {post_output}, {embed_input});
  op(Opcode::Linear, {embed_input, embed_weight, embed_bias}, {patch_tokens});
  op(Opcode::Fill, {}, {zero_sequence}, {Attribute::f64(AttrKey::Value, 0.0)});
  op(Opcode::IndexedUpdateRows, {zero_sequence, patch_tokens, patch_map_id},
     {with_patches});
  op(Opcode::IndexedUpdateRows, {with_patches, registers, register_map_id},
     {residual});
  op(Opcode::RotaryPosition, {positions, inv_freq}, {cosine, sine});

  const auto norm_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, config.block_size)};
  const auto qk_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, config.heads),
      Attribute::u64(AttrKey::HeadDim, config.head_dim),
      Attribute::u64(AttrKey::RotaryDim, config.rotary_dim),
      Attribute::u64(AttrKey::BlockSize, 64U)};
  const auto attention_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::AttentionScale,
                     1.0 / std::sqrt(static_cast<double>(config.head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::Implementation,
                     config.attention_implementation),
      Attribute::u64(AttrKey::BlockSize, 64U)};
  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::Implementation, 1U),
      Attribute::u64(AttrKey::BlockSize, config.block_size)};
  const auto swiglu_attrs = std::vector<Attribute>{
      Attribute::boolean(AttrKey::GateFirst, true)};

  for (std::uint64_t layer = 0; layer < config.layers; ++layer) {
    const auto prefix =
        "decoder.transformer_blocks." + std::to_string(layer) + ".";
    const auto norm1_weight = add_constant(prefix + "norm1.weight",
                                           prefix + "norm1.weight", DType::F32,
                                           {hidden});
    const auto qkv_weight = add_constant(prefix + "attn.to_qkv.weight",
                                         prefix + "attn.to_qkv.weight",
                                         DType::F16, {3U * hidden, hidden});
    const auto qkv_bias = add_constant(prefix + "attn.to_qkv.bias",
                                       prefix + "attn.to_qkv.bias", DType::F16,
                                       {3U * hidden});
    const auto out_weight = add_constant(prefix + "attn.to_out.weight",
                                         prefix + "attn.to_out.weight",
                                         DType::F16, {hidden, hidden});
    const auto out_bias = add_constant(prefix + "attn.to_out.bias",
                                       prefix + "attn.to_out.bias", DType::F16,
                                       {hidden});
    const auto scale1 = add_constant(prefix + "scale1", prefix + "scale1",
                                     DType::F32, {hidden});
    const auto norm2_weight = add_constant(prefix + "norm2.weight",
                                           prefix + "norm2.weight", DType::F32,
                                           {hidden});
    const auto w1_weight = add_constant(prefix + "ff.w1.weight",
                                        prefix + "ff.w1.weight", DType::F16,
                                        {2U * config.ffn, hidden});
    const auto w1_bias = add_constant(prefix + "ff.w1.bias",
                                      prefix + "ff.w1.bias", DType::F16,
                                      {2U * config.ffn});
    const auto w2_weight = add_constant(prefix + "ff.w2.weight",
                                        prefix + "ff.w2.weight", DType::F16,
                                        {hidden, config.ffn});
    const auto w2_bias = add_constant(prefix + "ff.w2.bias",
                                      prefix + "ff.w2.bias", DType::F16,
                                      {hidden});
    const auto scale2 = add_constant(prefix + "scale2", prefix + "scale2",
                                     DType::F32, {hidden});

    const auto norm1 = add_internal(DType::F32, {sequence, hidden});
    const auto qkv_input = add_internal(DType::F16, {sequence, hidden});
    const auto packed_qkv = add_internal(DType::F16, {sequence, 3U * hidden});
    const auto q = add_internal(DType::F16,
                                {sequence, config.heads, config.head_dim});
    const auto k = add_internal(DType::F16,
                                {sequence, config.heads, config.head_dim});
    const auto v = add_internal(DType::F16,
                                {sequence, config.heads, config.head_dim});
    const auto qn = add_internal(DType::F16,
                                 {sequence, config.heads, config.head_dim});
    const auto kn = add_internal(DType::F16,
                                 {sequence, config.heads, config.head_dim});
    const auto attended = add_internal(
        DType::F16, {sequence, config.heads, config.head_dim});
    const auto attention_output = add_internal(DType::F16, {sequence, hidden});
    const auto attention_f32 = add_internal(DType::F32, {sequence, hidden});
    const auto scaled_attention = add_internal(DType::F32, {sequence, hidden});
    const auto after_attention = add_internal(DType::F32, {sequence, hidden});
    const auto norm2 = add_internal(DType::F32, {sequence, hidden});
    const auto ff_input = add_internal(DType::F16, {sequence, hidden});
    const auto w1 = add_internal(DType::F16, {sequence, 2U * config.ffn});
    const auto activated = add_internal(DType::F16, {sequence, config.ffn});
    const auto ff_output = add_internal(DType::F16, {sequence, hidden});
    const auto ff_f32 = add_internal(DType::F32, {sequence, hidden});
    const auto scaled_ff = add_internal(DType::F32, {sequence, hidden});
    const auto block_output = add_internal(DType::F32, {sequence, hidden});

    op(Opcode::RmsNorm, {residual, norm1_weight}, {norm1}, norm_attrs);
    op(Opcode::Cast, {norm1}, {qkv_input});
    op(Opcode::Linear, {qkv_input, qkv_weight, qkv_bias}, {packed_qkv},
       linear_attrs);
    op(Opcode::H3DeinterleaveQkv, {packed_qkv}, {q, k, v},
       {Attribute::u64(AttrKey::Heads, config.heads),
        Attribute::u64(AttrKey::HeadDim, config.head_dim)});
    op(Opcode::QkNormPartialRope, {q, qk_ones, cosine, sine}, {qn}, qk_attrs);
    op(Opcode::QkNormPartialRope, {k, qk_ones, cosine, sine}, {kn}, qk_attrs);
    op(Opcode::Attention, {qn, kn, v}, {attended}, attention_attrs);
    op(Opcode::Linear, {attended, out_weight, out_bias}, {attention_output},
       linear_attrs);
    op(Opcode::Cast, {attention_output}, {attention_f32});
    op(Opcode::AffineLastDim, {attention_f32, scale1}, {scaled_attention});
    op(Opcode::Add, {residual, scaled_attention}, {after_attention});
    op(Opcode::RmsNorm, {after_attention, norm2_weight}, {norm2}, norm_attrs);
    op(Opcode::Cast, {norm2}, {ff_input});
    op(Opcode::Linear, {ff_input, w1_weight, w1_bias}, {w1}, linear_attrs);
    op(Opcode::SwiGlu, {w1}, {activated}, swiglu_attrs);
    op(Opcode::Linear, {activated, w2_weight, w2_bias}, {ff_output},
       linear_attrs);
    op(Opcode::Cast, {ff_output}, {ff_f32});
    op(Opcode::AffineLastDim, {ff_f32, scale2}, {scaled_ff});
    op(Opcode::Add, {after_attention, scaled_ff}, {block_output});
    residual = block_output;
  }

  const auto final_norm_weight = add_constant(
      "decoder.norm_out.weight", "decoder.norm_out.weight", DType::F32,
      {hidden});
  const auto final_norm_bias = add_constant(
      "decoder.norm_out.bias", "decoder.norm_out.bias", DType::F32,
      {hidden});
  const auto projection_weight = add_constant(
      "decoder.proj_out.weight", "decoder.proj_out.weight", DType::F32,
      {patch_dim, hidden});
  const auto projection_bias = add_constant(
      "decoder.proj_out.bias", "decoder.proj_out.bias", DType::F32,
      {patch_dim});
  const auto final_norm = add_internal(DType::F32, {sequence, hidden});
  const auto projected = add_internal(DType::F32, {sequence, patch_dim});
  const auto patch_pixels = add_internal(DType::F32, {patches, patch_dim});
  const auto output_dims = std::vector<std::uint64_t>{
      1U, config.output_channels,
      checked_mul(config.latent_frames, config.patch_t, "output frames"),
      checked_mul(config.latent_height, config.patch_h, "output height"),
      checked_mul(config.latent_width, config.patch_w, "output width")};
  const auto raw_pixels = add_internal(DType::F32, output_dims, true);
  const auto denormalized_patches =
      add_internal(DType::F32, {patches, patch_dim});
  const auto denormalized_pixels = add_internal(DType::F32, output_dims);
  const auto decoded_pixels = add_internal(DType::F32, output_dims, true);
  const auto unpack_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::PatchT, config.patch_t),
      Attribute::u64(AttrKey::PatchH, config.patch_h),
      Attribute::u64(AttrKey::PatchW, config.patch_w)};
  op(Opcode::LayerNorm, {residual, final_norm_weight, final_norm_bias},
     {final_norm}, norm_attrs);
  op(Opcode::Linear, {final_norm, projection_weight, projection_bias},
     {projected}, linear_attrs);
  op(Opcode::GatherRows, {projected, patch_indices_id}, {patch_pixels});
  op(Opcode::Unpatchify3D, {patch_pixels}, {raw_pixels}, unpack_attrs);
  op(Opcode::AffineLastDim,
     {patch_pixels, pixel_scale_id, pixel_bias_id}, {denormalized_patches});
  op(Opcode::Unpatchify3D, {denormalized_patches}, {denormalized_pixels},
     unpack_attrs);
  op(Opcode::Clamp, {denormalized_pixels}, {decoded_pixels},
     {Attribute::f64(AttrKey::Lower, 0.0),
      Attribute::f64(AttrKey::Upper, 1.0)});
  build.raw_output_id = raw_pixels;
  build.decoded_output_id = decoded_pixels;
  ir::verify(program);
  return build;
}

} // namespace dif::frontend
