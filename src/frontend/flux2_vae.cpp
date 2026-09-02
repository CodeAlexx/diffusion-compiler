#include "dif/frontend/flux2_vae.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

class Builder {
public:
  explicit Builder(Flux2VaeConfig value_config)
      : config(std::move(value_config)) {
    if (config.latent_height == 0U || config.latent_width == 0U)
      fail("FLUX.2 VAE requires nonzero latent geometry");
    build.config = config;
    build.latent_tokens_input = tensor(
        ir::DType::BF16, ir::TensorRole::Input,
        {config.latent_height * config.latent_width, 128U});
  }

  Flux2VaeBuild finish() {
    using namespace ir;
    auto value = tensor(DType::BF16, TensorRole::Internal,
                        {config.latent_height, config.latent_width, 128U});
    operation(Opcode::Reshape, {build.latent_tokens_input}, {value});
    auto channels_first = tensor(
        DType::BF16, TensorRole::Internal,
        {128U, config.latent_height, config.latent_width});
    operation(Opcode::Permute, {value}, {channels_first},
              permutation({2, 0, 1}));
    value = tensor(DType::BF16, TensorRole::Internal,
                   {1U, 128U, config.latent_height, config.latent_width});
    operation(Opcode::Reshape, {channels_first}, {value});
    auto value_f32 = tensor(DType::F32, TensorRole::Internal,
                            description(value).dims);
    operation(Opcode::Cast, {value}, {value_f32});

    const auto bn_std = checkpoint(
        "bn.running_var", {128U},
        Flux2VaeWeightTransform::BatchNormStandardDeviation);
    const auto bn_mean = checkpoint("bn.running_mean", {128U});
    auto nhwc = tensor(DType::F32, TensorRole::Internal,
                       {1U, config.latent_height, config.latent_width, 128U});
    operation(Opcode::Permute, {value_f32}, {nhwc},
              permutation({0, 2, 3, 1}));
    auto normalized_nhwc = same(nhwc);
    operation(Opcode::AffineLastDim, {nhwc, bn_std, bn_mean},
              {normalized_nhwc});
    value_f32 = same(value_f32);
    operation(Opcode::Permute, {normalized_nhwc}, {value_f32},
              permutation({0, 3, 1, 2}));
    capture("inv_normalize", value_f32);

    const auto shuffled = tensor(
        DType::F32, TensorRole::Internal,
        {1U, 32U, 2U, 2U, config.latent_height, config.latent_width});
    operation(Opcode::Reshape, {value_f32}, {shuffled});
    const auto ordered = tensor(
        DType::F32, TensorRole::Internal,
        {1U, 32U, config.latent_height, 2U, config.latent_width, 2U});
    operation(Opcode::Permute, {shuffled}, {ordered},
              permutation({0, 1, 4, 2, 5, 3}));
    value = tensor(DType::F32, TensorRole::Internal,
                   {1U, 32U, 2U * config.latent_height,
                    2U * config.latent_width});
    operation(Opcode::Reshape, {ordered}, {value});
    capture("pixel_shuffle", value);

    value = conv(value, 32U, 1U, 0U, "decoder.post_quant_conv");
    capture("post_quant_conv", value);
    value = conv(value, 512U, 3U, 1U, "decoder.conv_in");
    capture("decoder_conv_in", value);
    value = residual(value, 512U, "decoder.mid.block_1");
    capture("mid_block_1", value);
    value = attention(value, "decoder.mid.attn_1");
    capture("mid_attention", value);
    value = residual(value, 512U, "decoder.mid.block_2");
    capture("mid_block_2", value);

    constexpr std::array<std::uint64_t, 4> channels{512U, 512U, 256U,
                                                    128U};
    for (std::uint64_t stage = 0U; stage < channels.size(); ++stage) {
      const auto level = 3U - stage;
      for (std::uint64_t block_index = 0U; block_index < 3U; ++block_index)
        value = residual(
            value, channels[stage],
            "decoder.up." + std::to_string(level) + ".block." +
                std::to_string(block_index));
      capture("up_" + std::to_string(level) + "_blocks", value);
      if (level != 0U) {
        value = upsample(value, "decoder.up." + std::to_string(level) +
                                    ".upsample.conv");
        capture("up_" + std::to_string(level) + "_upsample", value);
      }
    }

    value = group_norm(value, "decoder.norm_out");
    capture("decoder_norm_out", value);
    value = swish(value);
    value = conv(value, 3U, 3U, 1U, "decoder.conv_out");
    build.raw_output = output(value, "raw_output");
    build.clamped_output = tensor(DType::F32, TensorRole::Output,
                                  description(value).dims);
    operation(Opcode::Clamp, {value}, {build.clamped_output},
              {Attribute::f64(AttrKey::Lower, -1.0),
               Attribute::f64(AttrKey::Upper, 1.0)});
    build.boundaries.emplace_back("clamped_output", build.clamped_output);
    verify(build.program);
    return std::move(build);
  }

private:
  Flux2VaeConfig config;
  Flux2VaeBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};
  std::map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> affine_free;

  std::uint32_t tensor(ir::DType dtype, std::uint32_t roles,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }

  const ir::TensorDesc &description(std::uint32_t id) const {
    const auto *result = build.program.tensor(id);
    if (!result)
      fail("FLUX.2 VAE builder lost a tensor");
    return *result;
  }

  std::uint32_t same(std::uint32_t source,
                     std::uint32_t roles = ir::TensorRole::Internal) {
    const auto desc = description(source);
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
      result.push_back(ir::Attribute::u64(keys.at(index++), axis));
    return result;
  }

  std::uint32_t checkpoint(
      std::string name, std::vector<std::uint64_t> dims,
      Flux2VaeWeightTransform transform = Flux2VaeWeightTransform::Direct) {
    auto roles = static_cast<std::uint32_t>(ir::TensorRole::Constant);
    if (config.streamed_constants)
      roles |= static_cast<std::uint32_t>(ir::TensorRole::Streamed);
    const auto id = tensor(ir::DType::F32, roles, std::move(dims));
    build.weights.push_back({id, std::move(name), transform});
    return id;
  }

  std::uint32_t conv(std::uint32_t input, std::uint64_t out_channels,
                     std::uint64_t kernel, std::uint64_t padding,
                     const std::string &prefix) {
    using namespace ir;
    const auto desc = description(input);
    const auto weight = checkpoint(prefix + ".weight",
                                   {out_channels, desc.dims[1], kernel, kernel});
    const auto bias = checkpoint(prefix + ".bias", {out_channels});
    const auto output_h = desc.dims[2] + 2U * padding - kernel + 1U;
    const auto output_w = desc.dims[3] + 2U * padding - kernel + 1U;
    const auto result = tensor(desc.dtype, TensorRole::Internal,
                               {desc.dims[0], out_channels, output_h, output_w});
    operation(Opcode::Conv2d, {input, weight, bias}, {result},
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
    return result;
  }

  std::pair<std::uint32_t, std::uint32_t>
  affine_free_vectors(std::uint64_t width) {
    const auto found = affine_free.find(width);
    if (found != affine_free.end())
      return found->second;
    const auto ones = tensor(ir::DType::F32, ir::TensorRole::Internal, {width});
    const auto zeros = tensor(ir::DType::F32, ir::TensorRole::Internal, {width});
    operation(ir::Opcode::Fill, {}, {ones},
              {ir::Attribute::f64(ir::AttrKey::Value, 1.0)});
    operation(ir::Opcode::Fill, {}, {zeros},
              {ir::Attribute::f64(ir::AttrKey::Value, 0.0)});
    affine_free.emplace(width, std::pair{ones, zeros});
    return {ones, zeros};
  }

  std::uint32_t group_norm(std::uint32_t input, const std::string &prefix) {
    using namespace ir;
    const auto desc = description(input);
    const auto channels = desc.dims[1];
    if (channels % 32U != 0U)
      fail("FLUX.2 VAE GroupNorm channel count is not divisible by 32");
    const auto width = (channels / 32U) * desc.dims[2] * desc.dims[3];
    const auto grouped = tensor(DType::F32, TensorRole::Internal, {32U, width});
    operation(Opcode::Reshape, {input}, {grouped});
    const auto [ones, zeros] = affine_free_vectors(width);
    const auto normalized = same(grouped);
    operation(Opcode::LayerNorm, {grouped, ones, zeros}, {normalized},
              {Attribute::f64(AttrKey::Epsilon, 1.0e-6),
               Attribute::u64(AttrKey::BlockSize, 1024U)});
    const auto nchw = same(input);
    operation(Opcode::Reshape, {normalized}, {nchw});
    const auto nhwc = tensor(
        DType::F32, TensorRole::Internal,
        {desc.dims[0], desc.dims[2], desc.dims[3], channels});
    operation(Opcode::Permute, {nchw}, {nhwc}, permutation({0, 2, 3, 1}));
    const auto weight = checkpoint(prefix + ".weight", {channels});
    const auto bias = checkpoint(prefix + ".bias", {channels});
    const auto affine = same(nhwc);
    operation(Opcode::AffineLastDim, {nhwc, weight, bias}, {affine});
    const auto result = same(input);
    operation(Opcode::Permute, {affine}, {result},
              permutation({0, 3, 1, 2}));
    return result;
  }

  std::uint32_t swish(std::uint32_t input) {
    const auto sigmoid = same(input);
    operation(ir::Opcode::Sigmoid, {input}, {sigmoid});
    const auto result = same(input);
    operation(ir::Opcode::Multiply, {input, sigmoid}, {result});
    return result;
  }

  std::uint32_t residual(std::uint32_t input, std::uint64_t out_channels,
                         const std::string &prefix) {
    auto shortcut = input;
    if (description(input).dims[1] != out_channels)
      shortcut = conv(input, out_channels, 1U, 0U, prefix + ".nin_shortcut");
    auto value = group_norm(input, prefix + ".norm1");
    value = swish(value);
    value = conv(value, out_channels, 3U, 1U, prefix + ".conv1");
    value = group_norm(value, prefix + ".norm2");
    value = swish(value);
    value = conv(value, out_channels, 3U, 1U, prefix + ".conv2");
    const auto result = same(value);
    operation(ir::Opcode::Add, {shortcut, value}, {result});
    return result;
  }

  std::uint32_t attention(std::uint32_t input, const std::string &prefix) {
    using namespace ir;
    const auto desc = description(input);
    const auto channels = desc.dims[1];
    const auto height = desc.dims[2];
    const auto width = desc.dims[3];
    const auto normalized = group_norm(input, prefix + ".norm");
    std::array<std::uint32_t, 3> heads{};
    for (std::size_t index = 0U; index < heads.size(); ++index) {
      const auto name = index == 0U ? "q" : index == 1U ? "k" : "v";
      const auto projected = conv(normalized, channels, 1U, 0U,
                                  prefix + "." + name);
      const auto nhwc = tensor(DType::F32, TensorRole::Internal,
                               {1U, height, width, channels});
      operation(Opcode::Permute, {projected}, {nhwc},
                permutation({0, 2, 3, 1}));
      heads[index] = tensor(DType::F32, TensorRole::Internal,
                            {height * width, 1U, channels});
      operation(Opcode::Reshape, {nhwc}, {heads[index]});
    }
    const auto attended = tensor(DType::F32, TensorRole::Internal,
                                 {height * width, 1U, channels});
    operation(Opcode::Attention, {heads[0], heads[1], heads[2]}, {attended},
              {Attribute::u64(AttrKey::KvHeads, 1U),
               Attribute::f64(AttrKey::AttentionScale,
                              1.0 / std::sqrt(static_cast<double>(channels))),
               Attribute::u64(AttrKey::Implementation, 3U)});
    const auto nhwc = tensor(DType::F32, TensorRole::Internal,
                             {1U, height, width, channels});
    operation(Opcode::Reshape, {attended}, {nhwc});
    auto value = same(input);
    operation(Opcode::Permute, {nhwc}, {value}, permutation({0, 3, 1, 2}));
    value = conv(value, channels, 1U, 0U, prefix + ".proj_out");
    const auto result = same(input);
    operation(Opcode::Add, {input, value}, {result});
    return result;
  }

  std::uint32_t upsample(std::uint32_t input,
                         const std::string &conv_prefix) {
    using namespace ir;
    const auto desc = description(input);
    const auto expanded = tensor(
        desc.dtype, TensorRole::Internal,
        {desc.dims[0], desc.dims[1], 2U * desc.dims[2], 2U * desc.dims[3]});
    operation(Opcode::UpsampleNearest2d, {input}, {expanded},
              {Attribute::u64(AttrKey::ScaleH, 2U),
               Attribute::u64(AttrKey::ScaleW, 2U)});
    return conv(expanded, desc.dims[1], 3U, 1U, conv_prefix);
  }

  void mark_output(std::uint32_t id) {
    for (auto &desc : build.program.tensors)
      if (desc.id == id) {
        desc.roles |= ir::TensorRole::Output;
        return;
      }
    fail("FLUX.2 VAE output tensor is missing");
  }

  void capture(std::string name, std::uint32_t id) {
    if (!config.capture_boundaries)
      return;
    mark_output(id);
    build.boundaries.emplace_back(std::move(name), id);
  }

  std::uint32_t output(std::uint32_t id, std::string name) {
    mark_output(id);
    build.boundaries.emplace_back(std::move(name), id);
    return id;
  }
};

} // namespace

Flux2VaeBuild make_flux2_vae_decoder(const Flux2VaeConfig &config) {
  return Builder(config).finish();
}

} // namespace dif::frontend
