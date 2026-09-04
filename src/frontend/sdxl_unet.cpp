#include "dif/frontend/sdxl_unet.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

using ir::Attribute;
using ir::AttrKey;
using ir::DType;
using ir::Opcode;
using ir::TensorRole;

// The reference's UNetModel for SDXL (comfy/ldm/modules/diffusionmodules/
// openaimodel.py with comfy/supported_models.py SDXL.unet_config):
// model_channels 320, channel_mult (1,2,4), num_res_blocks 2,
// transformer_depth (0,0,2,2,10,10), middle depth 10, num_head_channels 64,
// context_dim 2048, adm_in_channels 2816, use_linear_in_transformer.
constexpr std::uint64_t kLatentChannels = 4U;
constexpr std::uint64_t kModelChannels = 320U;
constexpr std::uint64_t kTimeEmbed = 1280U;
constexpr std::uint64_t kContextDim = 2048U;
constexpr std::uint64_t kAdmChannels = 2816U;
constexpr std::uint64_t kHeadDim = 64U;
constexpr std::uint64_t kGroups = 32U;
constexpr double kResNormEpsilon = 1.0e-5;
constexpr double kTransformerNormEpsilon = 1.0e-6;
constexpr double kLayerNormEpsilon = 1.0e-5;
constexpr std::uint64_t kConvWorkspace = 64ULL * 1024ULL * 1024ULL;

class Builder {
public:
  explicit Builder(SdxlUnetConfig value_config)
      : config(std::move(value_config)) {
    if (config.batch == 0U || config.latent_height % 4U != 0U ||
        config.latent_width % 4U != 0U || config.latent_height == 0U ||
        config.latent_width == 0U || config.context_tokens == 0U)
      fail("SDXL UNet requires nonzero batch, latent sides divisible by 4, "
           "and a nonzero context length");
    if (config.dtype != DType::F16 && config.dtype != DType::BF16)
      fail("SDXL UNet admits f16 or bf16");
    build.config = config;
  }

  SdxlUnetBuild finish() {
    const auto batch = config.batch;
    build.latent_input =
        tensor(config.dtype, TensorRole::Input,
               {batch, kLatentChannels, config.latent_height,
                config.latent_width});
    build.timestep_input = tensor(DType::F32, TensorRole::Input, {batch});
    build.context_input =
        tensor(config.dtype, TensorRole::Input,
               {batch, config.context_tokens, kContextDim});
    build.vector_input =
        tensor(config.dtype, TensorRole::Input, {batch, kAdmChannels});

    // Cross-attention keys/values read the flattened context rows.
    context_rows = tensor(config.dtype, TensorRole::Internal,
                          {batch * config.context_tokens, kContextDim});
    operation(Opcode::Reshape, {build.context_input}, {context_rows});

    emb = embeddings();
    capture("emb", emb);
    // emb_layers apply SiLU to the shared embedding in every res block;
    // one shared activation is the same value.
    emb_activated = same(emb);
    operation(Opcode::SiLU, {emb}, {emb_activated});

    std::vector<std::uint32_t> skips;
    auto h = conv(build.latent_input, kModelChannels, 3U, 1U, 1U,
                  "input_blocks.0.0");
    capture("input_block_0", h);
    skips.push_back(h);
    const std::array<std::uint64_t, 3> level_channels{
        kModelChannels, kModelChannels * 2U, kModelChannels * 4U};
    const std::array<std::uint64_t, 3> level_depth{0U, 2U, 10U};
    std::uint64_t block_index = 1U;
    for (std::uint64_t level = 0U; level < 3U; ++level) {
      for (std::uint64_t repeat = 0U; repeat < 2U; ++repeat) {
        const auto prefix = "input_blocks." + std::to_string(block_index);
        h = res_block(h, level_channels[level], prefix + ".0");
        if (level_depth[level] != 0U)
          h = spatial_transformer(h, level_depth[level], prefix + ".1");
        capture("input_block_" + std::to_string(block_index), h);
        skips.push_back(h);
        ++block_index;
      }
      if (level + 1U != 3U) {
        h = conv(h, level_channels[level], 3U, 2U, 1U,
                 "input_blocks." + std::to_string(block_index) + ".0.op");
        capture("input_block_" + std::to_string(block_index), h);
        skips.push_back(h);
        ++block_index;
      }
    }

    h = res_block(h, level_channels[2], "middle_block.0");
    h = spatial_transformer(h, 10U, "middle_block.1");
    h = res_block(h, level_channels[2], "middle_block.2");
    capture("middle_block", h);

    block_index = 0U;
    for (std::uint64_t level = 3U; level-- > 0U;) {
      for (std::uint64_t repeat = 0U; repeat < 3U; ++repeat) {
        const auto prefix = "output_blocks." + std::to_string(block_index);
        const auto skip = skips.back();
        skips.pop_back();
        h = concat_channels(h, skip);
        h = res_block(h, level_channels[level], prefix + ".0");
        if (level_depth[level] != 0U)
          h = spatial_transformer(h, level_depth[level], prefix + ".1");
        if (repeat == 2U && level != 0U)
          h = upsample(h, prefix + (level_depth[level] != 0U ? ".2.conv"
                                                              : ".1.conv"));
        capture("output_block_" + std::to_string(block_index), h);
        ++block_index;
      }
    }
    if (!skips.empty())
      fail("SDXL UNet skip stack is unbalanced");

    h = group_norm(h, "out.0", kResNormEpsilon);
    auto activated = same(h);
    operation(Opcode::SiLU, {h}, {activated});
    build.output = conv(activated, kLatentChannels, 3U, 1U, 1U, "out.2");
    mark_output(build.output);
    build.boundaries.emplace_back("output", build.output);
    ir::verify(build.program);
    return std::move(build);
  }

private:
  SdxlUnetConfig config;
  SdxlUnetBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};
  std::uint32_t context_rows{};
  std::uint32_t emb{};
  std::uint32_t emb_activated{};

  std::uint32_t tensor(DType dtype, std::uint32_t roles,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }

  const ir::TensorDesc &description(std::uint32_t id) const {
    const auto *value = build.program.tensor(id);
    if (!value)
      fail("SDXL UNet builder lost a tensor descriptor");
    return *value;
  }

  std::uint32_t same(std::uint32_t source) {
    const auto dims = description(source).dims;
    return tensor(config.dtype, TensorRole::Internal, dims);
  }

  void operation(Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  }

  std::vector<Attribute>
  permutation(std::initializer_list<std::uint64_t> axes) const {
    constexpr std::array<AttrKey, 8> keys{
        AttrKey::Permutation0, AttrKey::Permutation1, AttrKey::Permutation2,
        AttrKey::Permutation3, AttrKey::Permutation4, AttrKey::Permutation5,
        AttrKey::Permutation6, AttrKey::Permutation7};
    std::vector<Attribute> result;
    std::size_t index = 0U;
    for (const auto axis : axes)
      result.push_back(Attribute::u64(keys[index++], axis));
    return result;
  }

  std::uint32_t checkpoint(const std::string &name,
                           std::vector<std::uint64_t> dims) {
    const auto roles = static_cast<std::uint32_t>(
        TensorRole::Constant |
        (config.streamed_constants ? TensorRole::Streamed
                                   : TensorRole::Internal));
    const auto id = tensor(config.dtype, roles, std::move(dims));
    build.weights.push_back({id, config.checkpoint_prefix + name});
    return id;
  }

  std::uint32_t conv(std::uint32_t input, std::uint64_t out_channels,
                     std::uint64_t kernel, std::uint64_t stride,
                     std::uint64_t padding, const std::string &prefix) {
    const auto input_desc = description(input);
    const auto weight = checkpoint(
        prefix + ".weight",
        {out_channels, input_desc.dims[1], kernel, kernel});
    const auto bias = checkpoint(prefix + ".bias", {out_channels});
    const auto extent = [&](std::uint64_t size) {
      return (size + 2U * padding - kernel) / stride + 1U;
    };
    const auto output = tensor(
        config.dtype, TensorRole::Internal,
        {input_desc.dims[0], out_channels, extent(input_desc.dims[2]),
         extent(input_desc.dims[3])});
    operation(Opcode::Conv2d, {input, weight, bias}, {output},
              {Attribute::u64(AttrKey::StrideH, stride),
               Attribute::u64(AttrKey::StrideW, stride),
               Attribute::u64(AttrKey::DilationH, 1U),
               Attribute::u64(AttrKey::DilationW, 1U),
               Attribute::u64(AttrKey::PadTop, padding),
               Attribute::u64(AttrKey::PadBottom, padding),
               Attribute::u64(AttrKey::PadWest, padding),
               Attribute::u64(AttrKey::PadEast, padding),
               Attribute::u64(AttrKey::Groups, 1U),
               Attribute::u64(AttrKey::WorkspaceLimitBytes, kConvWorkspace)});
    return output;
  }

  std::uint32_t linear(std::uint32_t input, const std::string &prefix,
                       std::uint64_t out_features, bool biased) {
    const auto input_desc = description(input);
    if (input_desc.dims.size() != 2U)
      fail("SDXL UNet linear expects [rows, features]");
    const auto weight =
        checkpoint(prefix + ".weight", {out_features, input_desc.dims[1]});
    const auto output = tensor(config.dtype, TensorRole::Internal,
                               {input_desc.dims[0], out_features});
    if (biased) {
      const auto bias = checkpoint(prefix + ".bias", {out_features});
      operation(Opcode::Linear, {input, weight, bias}, {output});
    } else {
      operation(Opcode::Linear, {input, weight}, {output});
    }
    return output;
  }

  std::uint32_t group_norm(std::uint32_t input, const std::string &prefix,
                           double epsilon) {
    const auto channels = description(input).dims[1];
    const auto weight = checkpoint(prefix + ".weight", {channels});
    const auto bias = checkpoint(prefix + ".bias", {channels});
    const auto output = same(input);
    operation(Opcode::GroupNorm, {input, weight, bias}, {output},
              {Attribute::u64(AttrKey::Groups, kGroups),
               Attribute::f64(AttrKey::Epsilon, epsilon),
               Attribute::u64(AttrKey::BlockSize, 256U)});
    return output;
  }

  std::uint32_t layer_norm(std::uint32_t input, const std::string &prefix) {
    const auto width = description(input).dims.back();
    const auto weight = checkpoint(prefix + ".weight", {width});
    const auto bias = checkpoint(prefix + ".bias", {width});
    const auto output = same(input);
    operation(Opcode::LayerNorm, {input, weight, bias}, {output},
              {Attribute::f64(AttrKey::Epsilon, kLayerNormEpsilon)});
    return output;
  }

  // timestep_embedding(t, 320) is cos-then-sin with exponent
  // -ln(10000) * i / half (no downscale shift); time_embed and label_emb
  // are Linear -> SiLU -> Linear; emb = time + label.
  std::uint32_t embeddings() {
    const auto batch = config.batch;
    const auto sinusoid =
        tensor(DType::F32, TensorRole::Internal, {batch, kModelChannels});
    operation(Opcode::SinusoidalTimestep, {build.timestep_input}, {sinusoid},
              {Attribute::boolean(AttrKey::FlipSinToCos, true),
               Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
               Attribute::f64(AttrKey::Scale, 1.0),
               Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
    const auto sinusoid_typed =
        tensor(config.dtype, TensorRole::Internal, {batch, kModelChannels});
    operation(Opcode::Cast, {sinusoid}, {sinusoid_typed});
    capture("time_embedding", sinusoid_typed);
    auto time = linear(sinusoid_typed, "time_embed.0", kTimeEmbed, true);
    auto activated = same(time);
    operation(Opcode::SiLU, {time}, {activated});
    time = linear(activated, "time_embed.2", kTimeEmbed, true);
    auto label = linear(build.vector_input, "label_emb.0.0", kTimeEmbed, true);
    activated = same(label);
    operation(Opcode::SiLU, {label}, {activated});
    label = linear(activated, "label_emb.0.2", kTimeEmbed, true);
    const auto sum = same(time);
    operation(Opcode::Add, {time, label}, {sum});
    return sum;
  }

  // ResBlock: in_layers (GroupNorm, SiLU, conv) + emb_layers(SiLU(emb))
  // broadcast over H,W, out_layers (GroupNorm, SiLU, conv), skip 1x1 conv
  // when the channel count changes.
  std::uint32_t res_block(std::uint32_t x, std::uint64_t out_channels,
                          const std::string &prefix) {
    const auto x_desc = description(x);
    const auto batch = x_desc.dims[0];
    const auto height = x_desc.dims[2];
    const auto width = x_desc.dims[3];
    auto h = group_norm(x, prefix + ".in_layers.0", kResNormEpsilon);
    auto activated = same(h);
    operation(Opcode::SiLU, {h}, {activated});
    h = conv(activated, out_channels, 3U, 1U, 1U, prefix + ".in_layers.2");
    const auto emb_out =
        linear(emb_activated, prefix + ".emb_layers.1", out_channels, true);
    const auto emb_column = tensor(config.dtype, TensorRole::Internal,
                                   {batch, out_channels, 1U, 1U});
    operation(Opcode::Reshape, {emb_out}, {emb_column});
    const auto emb_plane = tensor(config.dtype, TensorRole::Internal,
                                  {batch, out_channels, height, width});
    operation(Opcode::BroadcastTo, {emb_column}, {emb_plane});
    auto shifted = same(h);
    operation(Opcode::Add, {h, emb_plane}, {shifted});
    h = group_norm(shifted, prefix + ".out_layers.0", kResNormEpsilon);
    activated = same(h);
    operation(Opcode::SiLU, {h}, {activated});
    h = conv(activated, out_channels, 3U, 1U, 1U, prefix + ".out_layers.3");
    auto skip = x;
    if (x_desc.dims[1] != out_channels)
      skip = conv(x, out_channels, 1U, 1U, 0U, prefix + ".skip_connection");
    const auto output = same(h);
    operation(Opcode::Add, {skip, h}, {output});
    return output;
  }

  // SpatialTransformer (use_linear): GroupNorm(eps 1e-6) -> NHWC rows ->
  // proj_in Linear -> depth x BasicTransformerBlock -> proj_out Linear ->
  // NCHW -> + x.
  std::uint32_t spatial_transformer(std::uint32_t x, std::uint64_t depth,
                                    const std::string &prefix) {
    const auto x_desc = description(x);
    const auto batch = x_desc.dims[0];
    const auto channels = x_desc.dims[1];
    const auto height = x_desc.dims[2];
    const auto width = x_desc.dims[3];
    const auto normalized =
        group_norm(x, prefix + ".norm", kTransformerNormEpsilon);
    const auto nhwc = tensor(config.dtype, TensorRole::Internal,
                             {batch, height, width, channels});
    operation(Opcode::Permute, {normalized}, {nhwc}, permutation({0, 2, 3, 1}));
    auto rows = tensor(config.dtype, TensorRole::Internal,
                       {batch * height * width, channels});
    operation(Opcode::Reshape, {nhwc}, {rows});
    rows = linear(rows, prefix + ".proj_in", channels, true);
    for (std::uint64_t block = 0U; block < depth; ++block)
      rows = transformer_block(rows, batch, height * width,
                               prefix + ".transformer_blocks." +
                                   std::to_string(block));
    rows = linear(rows, prefix + ".proj_out", channels, true);
    const auto rows_nhwc = tensor(config.dtype, TensorRole::Internal,
                                  {batch, height, width, channels});
    operation(Opcode::Reshape, {rows}, {rows_nhwc});
    const auto nchw = same(x);
    operation(Opcode::Permute, {rows_nhwc}, {nchw}, permutation({0, 3, 1, 2}));
    const auto output = same(x);
    operation(Opcode::Add, {nchw, x}, {output});
    return output;
  }

  std::uint32_t attention(std::uint32_t query_rows, std::uint32_t key_rows,
                          std::uint32_t value_rows, std::uint64_t batch,
                          std::uint64_t query_tokens,
                          std::uint64_t key_tokens, std::uint64_t channels) {
    const auto heads = channels / kHeadDim;
    auto view = [&](std::uint32_t source, std::uint64_t tokens) {
      const auto shaped = tensor(config.dtype, TensorRole::Internal,
                                 {batch, tokens, heads, kHeadDim});
      operation(Opcode::Reshape, {source}, {shaped});
      return shaped;
    };
    const auto attended = tensor(config.dtype, TensorRole::Internal,
                                 {batch, query_tokens, heads, kHeadDim});
    operation(Opcode::Attention,
              {view(query_rows, query_tokens), view(key_rows, key_tokens),
               view(value_rows, key_tokens)},
              {attended},
              {Attribute::u64(AttrKey::Heads, heads),
               Attribute::u64(AttrKey::KvHeads, heads),
               Attribute::u64(AttrKey::HeadDim, kHeadDim),
               Attribute::f64(AttrKey::AttentionScale,
                              1.0 / std::sqrt(static_cast<double>(kHeadDim))),
               Attribute::boolean(AttrKey::Causal, false),
               Attribute::u64(AttrKey::Implementation,
                              config.attention_implementation)});
    const auto rows = tensor(config.dtype, TensorRole::Internal,
                             {batch * query_tokens, channels});
    operation(Opcode::Reshape, {attended}, {rows});
    return rows;
  }

  // BasicTransformerBlock: self-attention, cross-attention on the context,
  // GEGLU feed-forward; each with its LayerNorm and residual.
  std::uint32_t transformer_block(std::uint32_t rows, std::uint64_t batch,
                                  std::uint64_t tokens,
                                  const std::string &prefix) {
    const auto channels = description(rows).dims[1];
    auto normalized = layer_norm(rows, prefix + ".norm1");
    auto q = linear(normalized, prefix + ".attn1.to_q", channels, false);
    auto k = linear(normalized, prefix + ".attn1.to_k", channels, false);
    auto v = linear(normalized, prefix + ".attn1.to_v", channels, false);
    auto attended = attention(q, k, v, batch, tokens, tokens, channels);
    attended = linear(attended, prefix + ".attn1.to_out.0", channels, true);
    auto residual = same(rows);
    operation(Opcode::Add, {rows, attended}, {residual});

    normalized = layer_norm(residual, prefix + ".norm2");
    q = linear(normalized, prefix + ".attn2.to_q", channels, false);
    k = linear(context_rows, prefix + ".attn2.to_k", channels, false);
    v = linear(context_rows, prefix + ".attn2.to_v", channels, false);
    attended = attention(q, k, v, batch, tokens, config.context_tokens,
                         channels);
    attended = linear(attended, prefix + ".attn2.to_out.0", channels, true);
    auto residual2 = same(rows);
    operation(Opcode::Add, {residual, attended}, {residual2});

    normalized = layer_norm(residual2, prefix + ".norm3");
    const auto inner = channels * 4U;
    const auto projected =
        linear(normalized, prefix + ".ff.net.0.proj", inner * 2U, true);
    const auto value = tensor(config.dtype, TensorRole::Internal,
                              {batch * tokens, inner});
    operation(Opcode::Slice, {projected}, {value},
              {Attribute::u64(AttrKey::Axis, 1U),
               Attribute::u64(AttrKey::Start, 0U)});
    const auto gate = tensor(config.dtype, TensorRole::Internal,
                             {batch * tokens, inner});
    operation(Opcode::Slice, {projected}, {gate},
              {Attribute::u64(AttrKey::Axis, 1U),
               Attribute::u64(AttrKey::Start, inner)});
    const auto gate_activated = same(gate);
    operation(Opcode::Gelu, {gate}, {gate_activated},
              {Attribute::u64(AttrKey::Approximation,
                              static_cast<std::uint64_t>(
                                  ir::GeluApproximation::ExactErf))});
    const auto gated = same(value);
    operation(Opcode::Multiply, {value, gate_activated}, {gated});
    const auto contracted =
        linear(gated, prefix + ".ff.net.2", channels, true);
    const auto output = same(rows);
    operation(Opcode::Add, {residual2, contracted}, {output});
    return output;
  }

  std::uint32_t concat_channels(std::uint32_t h, std::uint32_t skip) {
    const auto h_desc = description(h);
    const auto skip_desc = description(skip);
    if (h_desc.dims[0] != skip_desc.dims[0] ||
        h_desc.dims[2] != skip_desc.dims[2] ||
        h_desc.dims[3] != skip_desc.dims[3])
      fail("SDXL UNet skip geometry disagrees with the decoder path");
    const auto output = tensor(config.dtype, TensorRole::Internal,
                               {h_desc.dims[0],
                                h_desc.dims[1] + skip_desc.dims[1],
                                h_desc.dims[2], h_desc.dims[3]});
    operation(Opcode::Concat, {h, skip}, {output},
              {Attribute::u64(AttrKey::Axis, 1U)});
    return output;
  }

  std::uint32_t upsample(std::uint32_t input, const std::string &conv_prefix) {
    const auto input_desc = description(input);
    const auto expanded = tensor(
        config.dtype, TensorRole::Internal,
        {input_desc.dims[0], input_desc.dims[1], 2U * input_desc.dims[2],
         2U * input_desc.dims[3]});
    operation(Opcode::UpsampleNearest2d, {input}, {expanded},
              {Attribute::u64(AttrKey::ScaleH, 2U),
               Attribute::u64(AttrKey::ScaleW, 2U)});
    return conv(expanded, input_desc.dims[1], 3U, 1U, 1U, conv_prefix);
  }

  void capture(const std::string &name, std::uint32_t value) {
    if (!config.capture_boundaries)
      return;
    mark_output(value);
    build.boundaries.emplace_back(name, value);
  }

  void mark_output(std::uint32_t value) {
    for (auto &desc : build.program.tensors) {
      if (desc.id != value)
        continue;
      desc.roles |= TensorRole::Output;
      return;
    }
    fail("SDXL UNet output lost its tensor");
  }
};

} // namespace

SdxlUnetBuild make_sdxl_unet(const SdxlUnetConfig &config) {
  return Builder(config).finish();
}

} // namespace dif::frontend
