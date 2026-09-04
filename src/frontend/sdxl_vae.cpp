#include "dif/frontend/sdxl_vae.hpp"

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

// The reference decoder (comfy/ldm/modules/diffusionmodules/model.py:
// Decoder, ResnetBlock, AttnBlock, Upsample, Normalize) for the SDXL
// configuration ch=128, ch_mult=(1,2,4,4), num_res_blocks=2, z_channels=4,
// out_ch=3, no attention outside the mid block.
constexpr std::uint64_t kLatentChannels = 4U;
constexpr std::uint64_t kBaseChannels = 128U;
constexpr std::uint64_t kGroups = 32U;
constexpr double kNormEpsilon = 1.0e-6;
constexpr std::uint64_t kConvWorkspace = 64ULL * 1024ULL * 1024ULL;

class Builder {
public:
  explicit Builder(SdxlVaeConfig value_config) : config(std::move(value_config)) {
    if (config.batch == 0U || config.latent_height == 0U ||
        config.latent_width == 0U)
      fail("SDXL VAE decoder requires nonzero batch and latent geometry");
    if (config.dtype != ir::DType::BF16 && config.dtype != ir::DType::F16 &&
        config.dtype != ir::DType::F32)
      fail("SDXL VAE decoder admits bf16, f16, or f32");
    build.config = config;
    build.latent_input = tensor(ir::TensorRole::Input,
                                {config.batch, kLatentChannels,
                                 config.latent_height, config.latent_width});
  }

  SdxlVaeBuild finish() {
    using namespace ir;
    auto value = conv(build.latent_input, kLatentChannels, 1U, 0U,
                      "post_quant_conv");
    capture("post_quant_conv", value);
    const auto top = kBaseChannels * 4U;
    value = conv(value, top, 3U, 1U, "decoder.conv_in");
    capture("conv_in", value);

    value = residual(value, top, "decoder.mid.block_1");
    capture("mid_block_1", value);
    value = attention(value, "decoder.mid.attn_1");
    capture("mid_attn_1", value);
    value = residual(value, top, "decoder.mid.block_2");
    capture("mid_block_2", value);

    // up levels execute 3, 2, 1, 0 (the reference iterates reversed); the
    // checkpoint keys keep the construction index.
    const std::array<std::uint64_t, 4> level_channels{
        kBaseChannels, kBaseChannels * 2U, kBaseChannels * 4U,
        kBaseChannels * 4U};
    for (std::uint64_t level = level_channels.size(); level-- > 0U;) {
      const auto prefix = "decoder.up." + std::to_string(level);
      for (std::uint64_t block = 0U; block < 3U; ++block)
        value = residual(value, level_channels[level],
                         prefix + ".block." + std::to_string(block));
      capture("up_" + std::to_string(level) + "_blocks", value);
      if (level != 0U) {
        value = upsample(value, prefix + ".upsample.conv");
        capture("up_" + std::to_string(level) + "_upsample", value);
      }
    }

    value = group_norm(value, "decoder.norm_out");
    capture("norm_out", value);
    const auto activated = same(value);
    operation(Opcode::SiLU, {value}, {activated});
    value = conv(activated, 3U, 3U, 1U, "decoder.conv_out");
    build.raw_output = value;
    mark_output(value, "raw_output");
    build.boundaries.emplace_back("raw_output", value);
    verify(build.program);
    return std::move(build);
  }

private:
  SdxlVaeConfig config;
  SdxlVaeBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t tensor(std::uint32_t roles, std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, config.dtype, roles, std::move(dims)});
    return id;
  }

  const ir::TensorDesc &description(std::uint32_t id) const {
    const auto *value = build.program.tensor(id);
    if (!value)
      fail("SDXL VAE builder lost a tensor descriptor");
    return *value;
  }

  std::uint32_t same(std::uint32_t source) {
    // Copy the dims: helpers append descriptors and may reallocate.
    const auto dims = description(source).dims;
    return tensor(ir::TensorRole::Internal, dims);
  }

  void operation(ir::Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<ir::Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  }

  std::vector<ir::Attribute>
  permutation(std::initializer_list<std::uint64_t> axes) const {
    constexpr std::array<ir::AttrKey, 8> keys{
        ir::AttrKey::Permutation0, ir::AttrKey::Permutation1,
        ir::AttrKey::Permutation2, ir::AttrKey::Permutation3,
        ir::AttrKey::Permutation4, ir::AttrKey::Permutation5,
        ir::AttrKey::Permutation6, ir::AttrKey::Permutation7};
    std::vector<ir::Attribute> result;
    std::size_t index = 0U;
    for (const auto axis : axes)
      result.push_back(ir::Attribute::u64(keys[index++], axis));
    return result;
  }

  std::uint32_t checkpoint(const std::string &name,
                           std::vector<std::uint64_t> dims) {
    const auto roles = static_cast<std::uint32_t>(
        ir::TensorRole::Constant |
        (config.streamed_constants ? ir::TensorRole::Streamed
                                   : ir::TensorRole::Internal));
    const auto id = tensor(roles, std::move(dims));
    build.weights.push_back({id, config.checkpoint_prefix + name});
    return id;
  }

  std::uint32_t conv(std::uint32_t input, std::uint64_t out_channels,
                     std::uint64_t kernel, std::uint64_t padding,
                     const std::string &prefix) {
    using namespace ir;
    const auto input_desc = description(input);
    const auto weight = checkpoint(
        prefix + ".weight",
        {out_channels, input_desc.dims[1], kernel, kernel});
    const auto bias = checkpoint(prefix + ".bias", {out_channels});
    const auto output = tensor(
        TensorRole::Internal,
        {input_desc.dims[0], out_channels,
         input_desc.dims[2] + 2U * padding - kernel + 1U,
         input_desc.dims[3] + 2U * padding - kernel + 1U});
    operation(Opcode::Conv2d, {input, weight, bias}, {output},
              {Attribute::u64(AttrKey::StrideH, 1U),
               Attribute::u64(AttrKey::StrideW, 1U),
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

  std::uint32_t group_norm(std::uint32_t input, const std::string &prefix) {
    using namespace ir;
    const auto channels = description(input).dims[1];
    const auto weight = checkpoint(prefix + ".weight", {channels});
    const auto bias = checkpoint(prefix + ".bias", {channels});
    const auto output = same(input);
    operation(Opcode::GroupNorm, {input, weight, bias}, {output},
              {Attribute::u64(AttrKey::Groups, kGroups),
               Attribute::f64(AttrKey::Epsilon, kNormEpsilon),
               Attribute::u64(AttrKey::BlockSize, 256U)});
    return output;
  }

  // ResnetBlock: norm1 -> swish -> conv1 -> norm2 -> swish -> conv2, plus the
  // 1x1 nin_shortcut when the channel count changes; output = shortcut + h.
  std::uint32_t residual(std::uint32_t input, std::uint64_t out_channels,
                         const std::string &prefix) {
    auto shortcut = input;
    if (description(input).dims[1] != out_channels)
      shortcut = conv(input, out_channels, 1U, 0U, prefix + ".nin_shortcut");
    auto value = group_norm(input, prefix + ".norm1");
    auto activated = same(value);
    operation(ir::Opcode::SiLU, {value}, {activated});
    value = conv(activated, out_channels, 3U, 1U, prefix + ".conv1");
    value = group_norm(value, prefix + ".norm2");
    activated = same(value);
    operation(ir::Opcode::SiLU, {value}, {activated});
    value = conv(activated, out_channels, 3U, 1U, prefix + ".conv2");
    const auto output = same(value);
    operation(ir::Opcode::Add, {shortcut, value}, {output});
    return output;
  }

  // AttnBlock: norm -> q/k/v 1x1 convs -> single-head attention over the
  // H*W positions (scale 1/sqrt(C)) -> proj_out 1x1 -> x + h.
  std::uint32_t attention(std::uint32_t input, const std::string &prefix) {
    using namespace ir;
    const auto input_desc = description(input);
    const auto batch = input_desc.dims[0];
    const auto channels = input_desc.dims[1];
    const auto height = input_desc.dims[2];
    const auto width = input_desc.dims[3];
    const auto normalized = group_norm(input, prefix + ".norm");
    // Batch one keeps the unbatched [S,1,C] form every attention
    // implementation admits (the F32 parity form needs it); larger batches
    // use the batched cuDNN form.
    const std::vector<std::uint64_t> rows =
        batch == 1U ? std::vector<std::uint64_t>{height * width, 1U, channels}
                    : std::vector<std::uint64_t>{batch, height * width, 1U,
                                                 channels};
    if (batch != 1U && config.dtype == DType::F32)
      fail("SDXL VAE F32 attention parity form requires batch one");
    std::array<std::uint32_t, 3> qkv{};
    const std::array<const char *, 3> names{".q", ".k", ".v"};
    for (std::size_t index = 0U; index < 3U; ++index) {
      const auto projected =
          conv(normalized, channels, 1U, 0U, prefix + names[index]);
      const auto channels_last =
          tensor(TensorRole::Internal, {batch, height, width, channels});
      operation(Opcode::Permute, {projected}, {channels_last},
                permutation({0, 2, 3, 1}));
      qkv[index] = tensor(TensorRole::Internal, rows);
      operation(Opcode::Reshape, {channels_last}, {qkv[index]});
    }
    const auto attended = tensor(TensorRole::Internal, rows);
    operation(Opcode::Attention, {qkv[0], qkv[1], qkv[2]}, {attended},
              {Attribute::u64(AttrKey::Heads, 1U),
               Attribute::u64(AttrKey::KvHeads, 1U),
               Attribute::u64(AttrKey::HeadDim, channels),
               Attribute::f64(AttrKey::AttentionScale,
                              1.0 / std::sqrt(static_cast<double>(channels))),
               Attribute::boolean(AttrKey::Causal, false),
               Attribute::u64(AttrKey::Implementation,
                              config.dtype == DType::F32 ? 3U : 2U)});
    const auto attended_nhwc =
        tensor(TensorRole::Internal, {batch, height, width, channels});
    operation(Opcode::Reshape, {attended}, {attended_nhwc});
    auto value = same(input);
    operation(Opcode::Permute, {attended_nhwc}, {value},
              permutation({0, 3, 1, 2}));
    value = conv(value, channels, 1U, 0U, prefix + ".proj_out");
    const auto output = same(input);
    operation(Opcode::Add, {input, value}, {output});
    return output;
  }

  std::uint32_t upsample(std::uint32_t input, const std::string &conv_prefix) {
    using namespace ir;
    const auto input_desc = description(input);
    const auto expanded = tensor(
        TensorRole::Internal,
        {input_desc.dims[0], input_desc.dims[1], 2U * input_desc.dims[2],
         2U * input_desc.dims[3]});
    operation(Opcode::UpsampleNearest2d, {input}, {expanded},
              {Attribute::u64(AttrKey::ScaleH, 2U),
               Attribute::u64(AttrKey::ScaleW, 2U)});
    return conv(expanded, input_desc.dims[1], 3U, 1U, conv_prefix);
  }

  void capture(const std::string &name, std::uint32_t value) {
    if (!config.capture_boundaries)
      return;
    mark_output(value, "capture");
    build.boundaries.emplace_back(name, value);
  }

  void mark_output(std::uint32_t value, const char *kind) {
    for (auto &desc : build.program.tensors) {
      if (desc.id != value)
        continue;
      desc.roles |= ir::TensorRole::Output;
      return;
    }
    fail(std::string("SDXL VAE ") + kind + " lost its tensor");
  }
};

} // namespace

SdxlVaeBuild make_sdxl_vae_decoder(const SdxlVaeConfig &config) {
  return Builder(config).finish();
}

} // namespace dif::frontend
