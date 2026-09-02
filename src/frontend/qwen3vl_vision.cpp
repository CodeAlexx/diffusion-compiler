#include "dif/frontend/qwen3vl_vision.hpp"

#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

using ir::AttrKey;
using ir::Attribute;
using ir::DType;
using ir::GeluApproximation;
using ir::LinearBiasMode;
using ir::Opcode;
using ir::RotaryLayout;
using ir::TensorRole;

constexpr std::array<float, 18> kH3VisionInverseFrequency{
    1.0F,          0.5994842052459717F,  0.35938137769699097F,
    0.2154434472322464F,  0.1291549652814865F,  0.07742635905742645F,
    0.04641588404774666F, 0.027825593948364258F, 0.01668100617825985F,
    0.009999999776482582F, 0.005994841456413269F, 0.0035938138607889414F,
    0.002154434332624078F, 0.001291549764573574F, 0.0007742635789327323F,
    0.00046415894757956266F, 0.00027825593133457005F,
    0.00016681010311003774F};

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left)
    fail(std::string(label) + " overflows U64");
  return left * right;
}

float bf16_round(float value) {
  return runtime::bf16_to_float(runtime::float_to_bf16(value));
}

runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                           const std::vector<float> &values) {
  runtime::Tensor tensor{DType::F32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

struct Builder {
  Qwen3VlVisionBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t tensor(DType dtype, std::uint32_t roles,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }
  std::uint32_t input(std::vector<std::uint64_t> dims) {
    return tensor(DType::BF16, static_cast<std::uint32_t>(TensorRole::Input),
                  std::move(dims));
  }
  std::uint32_t internal(std::vector<std::uint64_t> dims) {
    return tensor(DType::BF16,
                  static_cast<std::uint32_t>(TensorRole::Internal),
                  std::move(dims));
  }
  std::uint32_t streamed(std::string name,
                         std::vector<std::uint64_t> dims) {
    const auto id = tensor(
        DType::BF16,
        static_cast<std::uint32_t>(TensorRole::Constant) |
            static_cast<std::uint32_t>(TensorRole::Streamed),
        std::move(dims));
    build.bindings.push_back({id, std::move(name)});
    return id;
  }
  std::uint32_t generated(runtime::Tensor value) {
    const auto id = tensor(DType::F32,
                           static_cast<std::uint32_t>(TensorRole::Constant),
                           value.dims);
    build.generated_constants.emplace(id, std::move(value));
    return id;
  }
  void output(std::uint32_t id) {
    const auto found = std::find_if(
        build.program.tensors.begin(), build.program.tensors.end(),
        [&](const auto &description) { return description.id == id; });
    if (found == build.program.tensors.end())
      fail("Qwen3-VL vision output references an unknown tensor");
    found->roles |= static_cast<std::uint32_t>(TensorRole::Output);
  }
  void operation(Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  }
};

std::vector<Attribute> linear_bias_attributes() {
  return {Attribute::u64(
      AttrKey::LinearBiasMode,
      static_cast<std::uint64_t>(LinearBiasMode::Addmm))};
}

std::vector<Attribute> layer_norm_attributes(double epsilon) {
  return {Attribute::f64(AttrKey::Epsilon, epsilon),
          Attribute::u64(AttrKey::BlockSize, 256U)};
}

std::vector<Attribute> gelu_attributes(GeluApproximation approximation) {
  return {Attribute::u64(AttrKey::Approximation,
                         static_cast<std::uint64_t>(approximation))};
}

std::vector<Attribute> permutation(std::initializer_list<std::uint64_t> axes) {
  std::vector<Attribute> attributes;
  attributes.reserve(axes.size());
  std::uint32_t key = static_cast<std::uint32_t>(AttrKey::Permutation0);
  for (const auto axis : axes)
    attributes.push_back(
        Attribute::u64(static_cast<AttrKey>(key++), axis));
  return attributes;
}

std::pair<runtime::Tensor, runtime::Tensor>
vision_rope(std::uint64_t grid_t, std::uint64_t grid_h,
            std::uint64_t grid_w) {
  const auto patches = checked_multiply(
      grid_t, checked_multiply(grid_h, grid_w, "vision patch grid"),
      "vision patch grid");
  constexpr std::uint64_t pairs = 36U;
  std::vector<float> cosine(patches * pairs);
  std::vector<float> sine(patches * pairs);
  std::array<float, kH3VisionInverseFrequency.size()> inverse{};
  for (std::size_t index = 0; index < inverse.size(); ++index)
    inverse[index] = bf16_round(kH3VisionInverseFrequency[index]);

  const auto merge = std::uint64_t{2U};
  std::uint64_t token = 0U;
  for (std::uint64_t temporal = 0U; temporal < grid_t; ++temporal) {
    (void)temporal;
    for (std::uint64_t block_h = 0U; block_h < grid_h / merge; ++block_h) {
      for (std::uint64_t block_w = 0U; block_w < grid_w / merge; ++block_w) {
        for (std::uint64_t inner_h = 0U; inner_h < merge; ++inner_h) {
          for (std::uint64_t inner_w = 0U; inner_w < merge; ++inner_w) {
            const auto row = block_h * merge + inner_h;
            const auto column = block_w * merge + inner_w;
            for (std::size_t frequency = 0U; frequency < inverse.size();
                 ++frequency) {
              const auto row_angle = bf16_round(
                  static_cast<float>(row) * inverse[frequency]);
              const auto column_angle = bf16_round(
                  static_cast<float>(column) * inverse[frequency]);
              cosine[token * pairs + frequency] =
                  bf16_round(std::cos(row_angle));
              sine[token * pairs + frequency] =
                  bf16_round(std::sin(row_angle));
              cosine[token * pairs + inverse.size() + frequency] =
                  bf16_round(std::cos(column_angle));
              sine[token * pairs + inverse.size() + frequency] =
                  bf16_round(std::sin(column_angle));
            }
            ++token;
          }
        }
      }
    }
  }
  if (token != patches)
    fail("Qwen3-VL vision rotary token count drift");
  return {f32_tensor({1U, patches, pairs}, cosine),
          f32_tensor({1U, patches, pairs}, sine)};
}

float torch_linspace_f32(std::uint64_t side, std::uint64_t count,
                         std::uint64_t index) {
  if (count <= 1U)
    return 0.0F;
  const auto end = static_cast<float>(side - 1U);
  const auto step = static_cast<float>(
      static_cast<double>(side - 1U) / static_cast<double>(count - 1U));
  if (index < count / 2U)
    return step * static_cast<float>(index);
  return end - step * static_cast<float>(count - 1U - index);
}

void validate_config(const Qwen3VlVisionConfig &config, std::uint64_t grid_t,
                     std::uint64_t grid_h, std::uint64_t grid_w) {
  if (config.depth == 0U || config.hidden_size == 0U ||
      config.attention_heads == 0U || config.intermediate_size == 0U ||
      config.output_hidden_size == 0U || config.patch_size == 0U ||
      config.temporal_patch_size == 0U || config.spatial_merge_size == 0U ||
      config.position_grid_side == 0U || grid_t == 0U || grid_h == 0U ||
      grid_w == 0U)
    fail("Qwen3-VL vision dimensions must be positive");
  if (config.hidden_size % config.attention_heads != 0U ||
      config.hidden_size / config.attention_heads != 72U)
    fail("native Qwen3-VL vision rotary currently requires head dimension 72");
  if (config.spatial_merge_size != 2U ||
      grid_h % config.spatial_merge_size != 0U ||
      grid_w % config.spatial_merge_size != 0U)
    fail("Qwen3-VL vision grid must divide the released spatial merge of 2");
  if (config.deepstack_tap_blocks.size() != 3U ||
      !std::is_sorted(config.deepstack_tap_blocks.begin(),
                      config.deepstack_tap_blocks.end()) ||
      config.deepstack_tap_blocks.back() >= config.depth)
    fail("Qwen3-VL vision requires three ordered in-range deepstack taps");
  if (config.attention_implementation != 1U &&
      config.attention_implementation != 2U)
    fail("Qwen3-VL vision attention implementation must be generated or cuDNN");
}

} // namespace

Qwen3VlVisionBuild build_qwen3vl_vision_program(
    std::uint64_t grid_t, std::uint64_t grid_h, std::uint64_t grid_w,
    const Qwen3VlVisionConfig &config) {
  validate_config(config, grid_t, grid_h, grid_w);
  const auto sequence = checked_multiply(
      grid_t, checked_multiply(grid_h, grid_w, "vision patch grid"),
      "vision patch grid");
  const auto merge_unit = checked_multiply(
      config.spatial_merge_size, config.spatial_merge_size,
      "vision merge unit");
  const auto tokens = sequence / merge_unit;
  const auto head_dim = config.hidden_size / config.attention_heads;
  const auto merged_width = config.hidden_size * merge_unit;
  const auto patch_width =
      3U * config.temporal_patch_size * config.patch_size * config.patch_size;

  Builder builder;
  auto &build = builder.build;
  build.pixel_patches_input_id = builder.input({sequence, patch_width});
  build.position_embeddings_input_id =
      builder.input({sequence, config.hidden_size});

  const auto patch_weight = builder.streamed(
      "model.visual.patch_embed.proj.weight",
      {config.hidden_size, 3U, config.temporal_patch_size, config.patch_size,
       config.patch_size});
  const auto patch_weight_flat =
      builder.internal({config.hidden_size, patch_width});
  builder.operation(Opcode::Reshape, {patch_weight}, {patch_weight_flat});
  const auto patch_bias = builder.streamed(
      "model.visual.patch_embed.proj.bias", {config.hidden_size});
  const auto patch_projection =
      builder.internal({sequence, config.hidden_size});
  builder.operation(Opcode::Linear,
                    {build.pixel_patches_input_id, patch_weight_flat,
                     patch_bias},
                    {patch_projection}, linear_bias_attributes());
  ++build.linear_operations;
  auto hidden = builder.internal({sequence, config.hidden_size});
  builder.operation(Opcode::Add,
                    {patch_projection, build.position_embeddings_input_id},
                    {hidden});
  if (config.trace_outputs) {
    builder.output(hidden);
    build.trace_output_ids.push_back(hidden);
  }

  auto rope = vision_rope(grid_t, grid_h, grid_w);
  const auto cosine = builder.generated(std::move(rope.first));
  const auto sine = builder.generated(std::move(rope.second));
  const auto rotary_attrs = std::vector<Attribute>{Attribute::u64(
      AttrKey::RotaryLayout,
      static_cast<std::uint64_t>(RotaryLayout::Interleaved))};

  const auto apply_half_split = [&](std::uint32_t input) {
    const auto paired = builder.internal(
        {1U, sequence, config.attention_heads, 2U, head_dim / 2U});
    builder.operation(Opcode::Reshape, {input}, {paired});
    const auto interleaved = builder.internal(
        {1U, sequence, config.attention_heads, head_dim / 2U, 2U});
    builder.operation(Opcode::Permute, {paired}, {interleaved},
                      permutation({0U, 1U, 2U, 4U, 3U}));
    const auto flattened = builder.internal(
        {1U, sequence, config.attention_heads, head_dim});
    builder.operation(Opcode::Reshape, {interleaved}, {flattened});
    const auto rotated_flat = builder.internal(
        {1U, sequence, config.attention_heads, head_dim});
    builder.operation(Opcode::RotaryApply, {flattened, cosine, sine},
                      {rotated_flat}, rotary_attrs);
    const auto rotated_interleaved = builder.internal(
        {1U, sequence, config.attention_heads, head_dim / 2U, 2U});
    builder.operation(Opcode::Reshape, {rotated_flat}, {rotated_interleaved});
    const auto rotated_paired = builder.internal(
        {1U, sequence, config.attention_heads, 2U, head_dim / 2U});
    builder.operation(Opcode::Permute, {rotated_interleaved}, {rotated_paired},
                      permutation({0U, 1U, 2U, 4U, 3U}));
    const auto output4 = builder.internal(
        {1U, sequence, config.attention_heads, head_dim});
    builder.operation(Opcode::Reshape, {rotated_paired}, {output4});
    if (config.attention_implementation == 2U)
      return output4;
    const auto output3 =
        builder.internal({sequence, config.attention_heads, head_dim});
    builder.operation(Opcode::Reshape, {output4}, {output3});
    return output3;
  };

  const auto merger = [&](std::uint32_t value, const std::string &prefix,
                          bool postshuffle) {
    auto current = value;
    if (postshuffle) {
      const auto shuffled = builder.internal({tokens, merged_width});
      builder.operation(Opcode::Reshape, {current}, {shuffled});
      current = shuffled;
    }
    const auto norm_width = postshuffle ? merged_width : config.hidden_size;
    const auto norm_weight =
        builder.streamed(prefix + "norm.weight", {norm_width});
    const auto norm_bias =
        builder.streamed(prefix + "norm.bias", {norm_width});
    const auto normed = builder.internal(
        postshuffle ? std::vector<std::uint64_t>{tokens, merged_width}
                    : std::vector<std::uint64_t>{sequence,
                                                 config.hidden_size});
    builder.operation(Opcode::LayerNorm,
                      {current, norm_weight, norm_bias}, {normed},
                      layer_norm_attributes(config.layer_norm_epsilon));
    current = normed;
    if (!postshuffle) {
      const auto shuffled = builder.internal({tokens, merged_width});
      builder.operation(Opcode::Reshape, {current}, {shuffled});
      current = shuffled;
    }
    const auto fc1_weight =
        builder.streamed(prefix + "linear_fc1.weight",
                         {merged_width, merged_width});
    const auto fc1_bias =
        builder.streamed(prefix + "linear_fc1.bias", {merged_width});
    const auto fc1 = builder.internal({tokens, merged_width});
    builder.operation(Opcode::Linear, {current, fc1_weight, fc1_bias}, {fc1},
                      linear_bias_attributes());
    const auto activated = builder.internal({tokens, merged_width});
    builder.operation(Opcode::Gelu, {fc1}, {activated},
                      gelu_attributes(GeluApproximation::ExactErf));
    const auto fc2_weight =
        builder.streamed(prefix + "linear_fc2.weight",
                         {config.output_hidden_size, merged_width});
    const auto fc2_bias = builder.streamed(prefix + "linear_fc2.bias",
                                           {config.output_hidden_size});
    const auto output =
        builder.internal({tokens, config.output_hidden_size});
    builder.operation(Opcode::Linear,
                      {activated, fc2_weight, fc2_bias}, {output},
                      linear_bias_attributes());
    build.linear_operations += 2U;
    return output;
  };

  for (std::uint64_t layer = 0U; layer < config.depth; ++layer) {
    const auto prefix =
        "model.visual.blocks." + std::to_string(layer) + ".";
    const auto norm1_weight =
        builder.streamed(prefix + "norm1.weight", {config.hidden_size});
    const auto norm1_bias =
        builder.streamed(prefix + "norm1.bias", {config.hidden_size});
    const auto norm1 = builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::LayerNorm,
                      {hidden, norm1_weight, norm1_bias}, {norm1},
                      layer_norm_attributes(config.layer_norm_epsilon));
    const auto qkv_weight = builder.streamed(
        prefix + "attn.qkv.weight", {3U * config.hidden_size,
                                     config.hidden_size});
    const auto qkv_bias = builder.streamed(prefix + "attn.qkv.bias",
                                           {3U * config.hidden_size});
    const auto qkv = builder.internal({sequence, 3U * config.hidden_size});
    builder.operation(Opcode::Linear, {norm1, qkv_weight, qkv_bias}, {qkv},
                      linear_bias_attributes());
    ++build.linear_operations;
    std::array<std::uint32_t, 3> components{};
    for (std::uint64_t component = 0U; component < components.size();
         ++component) {
      components[component] =
          builder.internal({sequence, config.hidden_size});
      builder.operation(
          Opcode::Slice, {qkv}, {components[component]},
          {Attribute::u64(AttrKey::Axis, 1U),
           Attribute::u64(AttrKey::Start, component * config.hidden_size)});
    }
    const auto query = apply_half_split(components[0]);
    const auto key = apply_half_split(components[1]);
    const auto attention_dims = config.attention_implementation == 1U
                                    ? std::vector<std::uint64_t>{
                                          sequence, config.attention_heads,
                                          head_dim}
                                    : std::vector<std::uint64_t>{
                                          1U, sequence,
                                          config.attention_heads, head_dim};
    const auto value = builder.internal(attention_dims);
    builder.operation(Opcode::Reshape, {components[2]}, {value});
    const auto attended = builder.internal(attention_dims);
    builder.operation(
        Opcode::Attention, {query, key, value}, {attended},
        {Attribute::boolean(AttrKey::Causal, false),
         Attribute::u64(AttrKey::KvHeads, config.attention_heads),
         Attribute::u64(AttrKey::Implementation,
                        config.attention_implementation),
         Attribute::f64(AttrKey::AttentionScale,
                        1.0 / std::sqrt(static_cast<double>(head_dim)))});
    ++build.attention_operations;
    const auto attended_flat =
        builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::Reshape, {attended}, {attended_flat});
    const auto projection_weight = builder.streamed(
        prefix + "attn.proj.weight", {config.hidden_size,
                                      config.hidden_size});
    const auto projection_bias =
        builder.streamed(prefix + "attn.proj.bias", {config.hidden_size});
    const auto projection =
        builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::Linear,
                      {attended_flat, projection_weight, projection_bias},
                      {projection}, linear_bias_attributes());
    ++build.linear_operations;
    const auto attention_residual =
        builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::Add, {hidden, projection},
                      {attention_residual});

    const auto norm2_weight =
        builder.streamed(prefix + "norm2.weight", {config.hidden_size});
    const auto norm2_bias =
        builder.streamed(prefix + "norm2.bias", {config.hidden_size});
    const auto norm2 = builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::LayerNorm,
                      {attention_residual, norm2_weight, norm2_bias}, {norm2},
                      layer_norm_attributes(config.layer_norm_epsilon));
    const auto fc1_weight = builder.streamed(
        prefix + "mlp.linear_fc1.weight",
        {config.intermediate_size, config.hidden_size});
    const auto fc1_bias = builder.streamed(prefix + "mlp.linear_fc1.bias",
                                           {config.intermediate_size});
    const auto fc1 = builder.internal({sequence, config.intermediate_size});
    builder.operation(Opcode::Linear, {norm2, fc1_weight, fc1_bias}, {fc1},
                      linear_bias_attributes());
    const auto activated =
        builder.internal({sequence, config.intermediate_size});
    builder.operation(Opcode::Gelu, {fc1}, {activated},
                      gelu_attributes(GeluApproximation::Tanh));
    const auto fc2_weight = builder.streamed(
        prefix + "mlp.linear_fc2.weight",
        {config.hidden_size, config.intermediate_size});
    const auto fc2_bias =
        builder.streamed(prefix + "mlp.linear_fc2.bias", {config.hidden_size});
    const auto fc2 = builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::Linear, {activated, fc2_weight, fc2_bias},
                      {fc2}, linear_bias_attributes());
    build.linear_operations += 2U;
    const auto layer_output =
        builder.internal({sequence, config.hidden_size});
    builder.operation(Opcode::Add, {attention_residual, fc2}, {layer_output});
    hidden = layer_output;

    if (config.trace_outputs &&
        (layer == 0U || layer == 8U || layer == 16U || layer == 24U ||
         layer + 1U == config.depth)) {
      builder.output(hidden);
      build.trace_output_ids.push_back(hidden);
    }
    const auto tap = std::find(config.deepstack_tap_blocks.begin(),
                               config.deepstack_tap_blocks.end(), layer);
    if (tap != config.deepstack_tap_blocks.end()) {
      const auto tap_index = static_cast<std::size_t>(
          std::distance(config.deepstack_tap_blocks.begin(), tap));
      const auto output = merger(
          hidden, "model.visual.deepstack_merger_list." +
                      std::to_string(tap_index) + ".",
          true);
      builder.output(output);
      build.deepstack_output_ids.push_back(output);
    }
  }

  build.embeds_output_id = merger(hidden, "model.visual.merger.", false);
  builder.output(build.embeds_output_id);
  return build;
}

runtime::Tensor qwen3vl_vision_image_patch_rows(
    const RgbImage &image, const Qwen3VlVisionConfig &config) {
  validate_config(config, 1U, image.height / config.patch_size,
                  image.width / config.patch_size);
  constexpr std::uint64_t minimum_pixels = 65536U;
  constexpr std::uint64_t maximum_pixels = 16777216U;
  const auto pixels = checked_multiply(image.width, image.height,
                                       "Qwen3-VL image pixels");
  const auto factor = config.patch_size * config.spatial_merge_size;
  if (image.width == 0U || image.height == 0U ||
      image.pixels.size() != pixels * 3U || image.width % factor != 0U ||
      image.height % factor != 0U || pixels < minimum_pixels ||
      pixels > maximum_pixels ||
      std::max(image.width, image.height) >
          200U * std::min(image.width, image.height))
    fail("Qwen3-VL image must already be a smart-resize identity RGB canvas");

  const auto grid_h = image.height / config.patch_size;
  const auto grid_w = image.width / config.patch_size;
  const auto patch_width =
      3U * config.temporal_patch_size * config.patch_size * config.patch_size;
  runtime::Tensor output{DType::BF16, {grid_h * grid_w, patch_width}, {}};
  output.bytes.resize(static_cast<std::size_t>(output.element_count()) *
                      sizeof(std::uint16_t));
  auto *destination =
      reinterpret_cast<std::uint16_t *>(output.mutable_data());
  std::uint64_t row = 0U;
  const auto merge = config.spatial_merge_size;
  for (std::uint64_t block_h = 0U; block_h < grid_h / merge; ++block_h) {
    for (std::uint64_t block_w = 0U; block_w < grid_w / merge; ++block_w) {
      for (std::uint64_t inner_h = 0U; inner_h < merge; ++inner_h) {
        for (std::uint64_t inner_w = 0U; inner_w < merge; ++inner_w) {
          const auto patch_y =
              (block_h * merge + inner_h) * config.patch_size;
          const auto patch_x =
              (block_w * merge + inner_w) * config.patch_size;
          for (std::uint64_t channel = 0U; channel < 3U; ++channel) {
            for (std::uint64_t temporal = 0U;
                 temporal < config.temporal_patch_size; ++temporal) {
              (void)temporal;
              for (std::uint64_t y = 0U; y < config.patch_size; ++y) {
                for (std::uint64_t x = 0U; x < config.patch_size; ++x) {
                  const auto source =
                      ((patch_y + y) * image.width + patch_x + x) * 3U +
                      channel;
                  const auto at =
                      row * patch_width +
                      ((channel * config.temporal_patch_size + temporal) *
                               config.patch_size +
                           y) *
                          config.patch_size +
                      x;
                  const auto normalized =
                      (static_cast<float>(image.pixels[source]) - 127.5F) /
                      127.5F;
                  destination[at] = runtime::float_to_bf16(normalized);
                }
              }
            }
          }
          ++row;
        }
      }
    }
  }
  output.validate();
  return output;
}

runtime::Tensor qwen3vl_vision_position_embeddings(
    const runtime::Tensor &position_table, std::uint64_t grid_t,
    std::uint64_t grid_h, std::uint64_t grid_w,
    const Qwen3VlVisionConfig &config) {
  validate_config(config, grid_t, grid_h, grid_w);
  if (position_table.dtype != DType::BF16 ||
      position_table.dims !=
          std::vector<std::uint64_t>{config.position_grid_side *
                                         config.position_grid_side,
                                     config.hidden_size})
    fail("Qwen3-VL learned position table has the wrong dtype or shape");
  const auto sequence = grid_t * grid_h * grid_w;
  runtime::Tensor output{DType::BF16, {sequence, config.hidden_size}, {}};
  output.bytes.resize(static_cast<std::size_t>(output.element_count()) *
                      sizeof(std::uint16_t));
  const auto *source =
      reinterpret_cast<const std::uint16_t *>(position_table.data());
  auto *destination =
      reinterpret_cast<std::uint16_t *>(output.mutable_data());
  const auto merge = config.spatial_merge_size;
  std::uint64_t token = 0U;
  for (std::uint64_t temporal = 0U; temporal < grid_t; ++temporal) {
    (void)temporal;
    for (std::uint64_t block_h = 0U; block_h < grid_h / merge; ++block_h) {
      for (std::uint64_t block_w = 0U; block_w < grid_w / merge; ++block_w) {
        for (std::uint64_t inner_h = 0U; inner_h < merge; ++inner_h) {
          for (std::uint64_t inner_w = 0U; inner_w < merge; ++inner_w) {
            const auto y = block_h * merge + inner_h;
            const auto x = block_w * merge + inner_w;
            const auto hy = torch_linspace_f32(
                config.position_grid_side, grid_h, y);
            const auto wx = torch_linspace_f32(
                config.position_grid_side, grid_w, x);
            const auto y0 = static_cast<std::uint64_t>(hy);
            const auto x0 = static_cast<std::uint64_t>(wx);
            const auto y1 = std::min(y0 + 1U, config.position_grid_side - 1U);
            const auto x1 = std::min(x0 + 1U, config.position_grid_side - 1U);
            const auto dy = static_cast<double>(hy - static_cast<float>(y0));
            const auto dx = static_cast<double>(wx - static_cast<float>(x0));
            const std::array<std::uint64_t, 4> indices{
                y0 * config.position_grid_side + x0,
                y0 * config.position_grid_side + x1,
                y1 * config.position_grid_side + x0,
                y1 * config.position_grid_side + x1};
            const std::array<double, 4> weights{
                (1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx,
                dy * (1.0 - dx), dy * dx};
            for (std::uint64_t channel = 0U; channel < config.hidden_size;
                 ++channel) {
              float value = 0.0F;
              for (std::size_t corner = 0U; corner < indices.size(); ++corner)
                value += static_cast<float>(weights[corner]) *
                         runtime::bf16_to_float(
                             source[indices[corner] * config.hidden_size +
                                    channel]);
              destination[token * config.hidden_size + channel] =
                  runtime::float_to_bf16(value);
            }
            ++token;
          }
        }
      }
    }
  }
  if (token != sequence)
    fail("Qwen3-VL position interpolation token count drift");
  output.validate();
  return output;
}

} // namespace dif::frontend
