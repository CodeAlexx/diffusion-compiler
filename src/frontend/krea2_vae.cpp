#include "dif/frontend/krea2_vae.hpp"

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

class Builder {
public:
  explicit Builder(Krea2VaeConfig value_config) : config(std::move(value_config)) {
    if (config.batch != 1U || config.latent_height == 0U ||
        config.latent_width == 0U)
      fail("Krea 2 Qwen Image VAE currently requires batch 1 and nonzero tile geometry");
    build.config = config;
    build.latent_input = tensor(ir::DType::BF16, ir::TensorRole::Input,
                                {config.batch, 16U, config.latent_height,
                                 config.latent_width});
    build.latent_std = tensor(ir::DType::BF16, ir::TensorRole::Constant, {16U});
    build.latent_mean = tensor(ir::DType::BF16, ir::TensorRole::Constant, {16U});
  }

  Krea2VaeBuild finish() {
    using namespace ir;
    auto value = build.latent_input;
    const auto nhwc = tensor(DType::BF16, TensorRole::Internal,
                             {config.batch, config.latent_height,
                              config.latent_width, 16U});
    operation(Opcode::Permute, {value}, {nhwc}, permutation({0, 2, 3, 1}));
    const auto denormalized_nhwc = tensor(
        DType::BF16, TensorRole::Internal,
        {config.batch, config.latent_height, config.latent_width, 16U});
    operation(Opcode::AffineLastDim,
              {nhwc, build.latent_std, build.latent_mean},
              {denormalized_nhwc});
    value = tensor(DType::BF16, TensorRole::Internal,
                   {config.batch, 16U, config.latent_height,
                    config.latent_width});
    operation(Opcode::Permute, {denormalized_nhwc}, {value},
              permutation({0, 3, 1, 2}));
    value = conv(value, 16U, 1U, 0U, "post_quant_conv",
                 Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    capture("post_quant_conv", value);
    value = conv(value, 384U, 3U, 1U, "decoder.conv_in",
                 Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    capture("decoder_conv_in", value);

    value = residual(value, 384U, "decoder.mid_block.resnets.0");
    capture("mid_resnet_0", value);
    value = attention(value, "decoder.mid_block.attentions.0");
    capture("mid_attention", value);
    value = residual(value, 384U, "decoder.mid_block.resnets.1");
    capture("mid_resnet_1", value);

    const std::array<std::uint64_t, 4> outputs{384U, 384U, 192U, 96U};
    for (std::size_t block = 0U; block < outputs.size(); ++block) {
      for (std::size_t residual_index = 0U; residual_index < 3U;
           ++residual_index)
        value = residual(value, outputs[block],
                         "decoder.up_blocks." + std::to_string(block) +
                             ".resnets." + std::to_string(residual_index));
      capture("up_block_" + std::to_string(block) + "_residual", value);
      if (block + 1U != outputs.size()) {
        value = upsample(value,
                         "decoder.up_blocks." + std::to_string(block) +
                             ".upsamplers.0.resample.1");
        capture("up_block_" + std::to_string(block) + "_upsample", value);
      }
    }

    value = norm(value, "decoder.norm_out.gamma");
    capture("decoder_norm_out", value);
    const auto activated = same(value);
    operation(Opcode::SiLU, {value}, {activated});
    value = conv(activated, 3U, 3U, 1U, "decoder.conv_out",
                 Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    build.raw_output = output_alias(value, "raw_output");
    build.clamped_output = tensor(DType::BF16, TensorRole::Output,
                                  description(value).dims);
    operation(Opcode::Clamp, {value}, {build.clamped_output},
              {Attribute::f64(AttrKey::Lower, -1.0),
               Attribute::f64(AttrKey::Upper, 1.0)});
    build.boundaries.emplace_back("clamped_output", build.clamped_output);
    verify(build.program);
    return std::move(build);
  }

private:
  Krea2VaeConfig config;
  Krea2VaeBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t tensor(ir::DType dtype, std::uint32_t roles,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }

  const ir::TensorDesc &description(std::uint32_t id) const {
    const auto *value = build.program.tensor(id);
    if (!value)
      fail("Krea 2 VAE builder lost a tensor descriptor");
    return *value;
  }

  std::uint32_t same(std::uint32_t source,
                     std::uint32_t roles = ir::TensorRole::Internal) {
    const auto &desc = description(source);
    return tensor(desc.dtype, roles, desc.dims);
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

  std::uint32_t checkpoint(std::string name,
                           std::vector<std::uint64_t> dims,
                           Krea2VaeWeightTransform transform) {
    const auto roles = static_cast<std::uint32_t>(
        ir::TensorRole::Constant |
        (config.streamed_constants ? ir::TensorRole::Streamed
                                   : ir::TensorRole::Internal));
    const auto id = tensor(ir::DType::BF16, roles, std::move(dims));
    build.weights.push_back({id, std::move(name), transform});
    return id;
  }

  std::uint32_t conv(std::uint32_t input, std::uint64_t out_channels,
                     std::uint64_t kernel, std::uint64_t padding,
                     const std::string &prefix,
                     Krea2VaeWeightTransform transform) {
    using namespace ir;
    const auto &input_desc = description(input);
    if (transform == Krea2VaeWeightTransform::Conv3dLastTemporalSlice) {
      const auto weight = checkpoint(
          prefix + ".weight",
          {out_channels, input_desc.dims[1], kernel, kernel, kernel},
          Krea2VaeWeightTransform::Direct);
      const auto bias = checkpoint(prefix + ".bias", {out_channels},
                                   Krea2VaeWeightTransform::Direct);
      auto volume = tensor(DType::BF16, TensorRole::Internal,
                           {input_desc.dims[0], input_desc.dims[1], 1U,
                            input_desc.dims[2], input_desc.dims[3]});
      operation(Opcode::Reshape, {input}, {volume});
      if (padding != 0U) {
        const auto padded = tensor(
            DType::BF16, TensorRole::Internal,
            {input_desc.dims[0], input_desc.dims[1], 1U + 2U * padding,
             input_desc.dims[2] + 2U * padding,
             input_desc.dims[3] + 2U * padding});
        operation(Opcode::PadConstant, {volume}, {padded},
                  {Attribute::u64(AttrKey::PadFront, 2U * padding),
                   Attribute::u64(AttrKey::PadBack, 0U),
                   Attribute::u64(AttrKey::PadTop, padding),
                   Attribute::u64(AttrKey::PadBottom, padding),
                   Attribute::u64(AttrKey::PadWest, padding),
                   Attribute::u64(AttrKey::PadEast, padding),
                   Attribute::f64(AttrKey::Value, 0.0)});
        volume = padded;
      }
      const auto output_volume = tensor(
          DType::BF16, TensorRole::Internal,
          {input_desc.dims[0], out_channels, 1U, input_desc.dims[2],
           input_desc.dims[3]});
      operation(Opcode::Conv3d, {volume, weight, bias}, {output_volume},
                {Attribute::u64(AttrKey::StrideT, 1U),
                 Attribute::u64(AttrKey::StrideH, 1U),
                 Attribute::u64(AttrKey::StrideW, 1U),
                 Attribute::u64(AttrKey::DilationT, 1U),
                 Attribute::u64(AttrKey::DilationH, 1U),
                 Attribute::u64(AttrKey::DilationW, 1U),
                 Attribute::u64(AttrKey::Groups, 1U),
                 Attribute::u64(AttrKey::WorkspaceLimitBytes,
                                64ULL * 1024ULL * 1024ULL)});
      const auto output = tensor(
          DType::BF16, TensorRole::Internal,
          {input_desc.dims[0], out_channels, input_desc.dims[2],
           input_desc.dims[3]});
      operation(Opcode::Reshape, {output_volume}, {output});
      return output;
    }
    const auto weight = checkpoint(prefix + ".weight",
                                   {out_channels, input_desc.dims[1], kernel,
                                    kernel},
                                   transform);
    const auto bias = checkpoint(prefix + ".bias", {out_channels},
                                 Krea2VaeWeightTransform::Direct);
    const auto output_h = input_desc.dims[2] + 2U * padding - kernel + 1U;
    const auto output_w = input_desc.dims[3] + 2U * padding - kernel + 1U;
    const auto output = tensor(DType::BF16, TensorRole::Internal,
                               {input_desc.dims[0], out_channels, output_h,
                                output_w});
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
               Attribute::u64(AttrKey::WorkspaceLimitBytes,
                              64ULL * 1024ULL * 1024ULL)});
    return output;
  }

  std::uint32_t norm(std::uint32_t input, const std::string &gamma_name) {
    using namespace ir;
    const auto channels = description(input).dims[1];
    const auto gamma = checkpoint(gamma_name, {channels},
                                  Krea2VaeWeightTransform::FlattenSingletonDimensions);
    const auto output = same(input);
    auto block = std::uint64_t{32U};
    while (block < channels)
      block *= 2U;
    operation(Opcode::ChannelRmsNorm, {input, gamma}, {output},
              {Attribute::u64(AttrKey::Axis, 1U),
               Attribute::u64(AttrKey::BlockSize, block),
               Attribute::f64(AttrKey::Epsilon, 1.0e-12)});
    return output;
  }

  std::uint32_t residual(std::uint32_t input, std::uint64_t out_channels,
                         const std::string &prefix) {
    const auto detailed = prefix == "decoder.mid_block.resnets.0";
    auto shortcut = input;
    if (description(input).dims[1] != out_channels)
      shortcut = conv(input, out_channels, 1U, 0U, prefix + ".conv_shortcut",
                      Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    auto value = norm(input, prefix + ".norm1.gamma");
    if (detailed)
      capture("mid_resnet_0_norm1", value);
    auto activated = same(value);
    operation(ir::Opcode::SiLU, {value}, {activated});
    if (detailed)
      capture("mid_resnet_0_activation1", activated);
    value = conv(activated, out_channels, 3U, 1U, prefix + ".conv1",
                 Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    if (detailed)
      capture("mid_resnet_0_conv1", value);
    value = norm(value, prefix + ".norm2.gamma");
    if (detailed)
      capture("mid_resnet_0_norm2", value);
    activated = same(value);
    operation(ir::Opcode::SiLU, {value}, {activated});
    if (detailed)
      capture("mid_resnet_0_activation2", activated);
    value = conv(activated, out_channels, 3U, 1U, prefix + ".conv2",
                 Krea2VaeWeightTransform::Conv3dLastTemporalSlice);
    if (detailed)
      capture("mid_resnet_0_conv2", value);
    const auto output = same(value);
    operation(ir::Opcode::Add, {value, shortcut}, {output});
    return output;
  }

  std::uint32_t attention(std::uint32_t input, const std::string &prefix) {
    using namespace ir;
    const auto &input_desc = description(input);
    const auto channels = input_desc.dims[1];
    const auto height = input_desc.dims[2];
    const auto width = input_desc.dims[3];
    auto value = norm(input, prefix + ".norm.gamma");
    capture("mid_attention_norm", value);
    value = conv(value, 3U * channels, 1U, 0U, prefix + ".to_qkv",
                 Krea2VaeWeightTransform::Direct);
    capture("mid_attention_qkv", value);
    const auto channels_last = tensor(
        DType::BF16, TensorRole::Internal,
        {input_desc.dims[0], height, width, 3U * channels});
    operation(Opcode::Permute, {value}, {channels_last},
              permutation({0, 2, 3, 1}));
    const auto flat = tensor(DType::BF16, TensorRole::Internal,
                             {input_desc.dims[0] * height * width,
                              3U * channels});
    operation(Opcode::Reshape, {channels_last}, {flat});
    std::array<std::uint32_t, 3> qkv{};
    for (std::uint64_t index = 0U; index < 3U; ++index) {
      const auto sliced = tensor(DType::BF16, TensorRole::Internal,
                                 {input_desc.dims[0] * height * width,
                                  channels});
      operation(Opcode::Slice, {flat}, {sliced},
                {Attribute::u64(AttrKey::Axis, 1U),
                 Attribute::u64(AttrKey::Start, index * channels)});
      qkv[index] = tensor(DType::BF16, TensorRole::Internal,
                          {input_desc.dims[0] * height * width, 1U, channels});
      operation(Opcode::Reshape, {sliced}, {qkv[index]});
    }
    const auto attended = tensor(
        DType::BF16, TensorRole::Internal,
        {input_desc.dims[0] * height * width, 1U, channels});
    operation(Opcode::Attention, {qkv[0], qkv[1], qkv[2]}, {attended},
              {Attribute::u64(AttrKey::Heads, 1U),
               Attribute::u64(AttrKey::KvHeads, 1U),
               Attribute::u64(AttrKey::HeadDim, channels),
               Attribute::f64(AttrKey::AttentionScale,
                              1.0 / std::sqrt(static_cast<double>(channels))),
               Attribute::boolean(AttrKey::Causal, false),
               Attribute::u64(AttrKey::Implementation, 2U)});
    const auto attended_nhwc = tensor(DType::BF16, TensorRole::Internal,
                                      {input_desc.dims[0], height, width,
                                       channels});
    operation(Opcode::Reshape, {attended}, {attended_nhwc});
    value = same(input);
    operation(Opcode::Permute, {attended_nhwc}, {value},
              permutation({0, 3, 1, 2}));
    capture("mid_attention_sdpa", value);
    value = conv(value, channels, 1U, 0U, prefix + ".proj",
                 Krea2VaeWeightTransform::Direct);
    capture("mid_attention_proj", value);
    const auto output = same(input);
    operation(Opcode::Add, {value, input}, {output});
    return output;
  }

  std::uint32_t upsample(std::uint32_t input,
                         const std::string &conv_prefix) {
    using namespace ir;
    const auto &input_desc = description(input);
    const auto expanded = tensor(
        DType::BF16, TensorRole::Internal,
        {input_desc.dims[0], input_desc.dims[1], 2U * input_desc.dims[2],
         2U * input_desc.dims[3]});
    operation(Opcode::UpsampleNearest2d, {input}, {expanded},
              {Attribute::u64(AttrKey::ScaleH, 2U),
               Attribute::u64(AttrKey::ScaleW, 2U)});
    return conv(expanded, input_desc.dims[1] / 2U, 3U, 1U, conv_prefix,
                Krea2VaeWeightTransform::Direct);
  }

  void capture(const std::string &name, std::uint32_t value) {
    if (!config.capture_boundaries)
      return;
    mark_output(value, "capture");
    build.boundaries.emplace_back(name, value);
  }

  std::uint32_t output_alias(std::uint32_t value, const std::string &name) {
    mark_output(value, "output");
    build.boundaries.emplace_back(name, value);
    return value;
  }

  void mark_output(std::uint32_t value, const char *kind) {
    for (auto &desc : build.program.tensors) {
      if (desc.id != value)
        continue;
      desc.roles |= ir::TensorRole::Output;
      return;
    }
    fail(std::string("Krea 2 VAE ") + kind + " lost its tensor");
  }
};

} // namespace

Krea2VaeBuild make_krea2_qwen_image_vae(const Krea2VaeConfig &config) {
  return Builder(config).finish();
}

} // namespace dif::frontend
