#include "dif/frontend/h3_video_encoder.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

class Builder {
public:
  explicit Builder(H3VideoEncoderConfig value_config)
      : config(std::move(value_config)) {
    if (config.batch != 1U || config.frames == 0U || config.height == 0U ||
        config.width == 0U)
      fail("H3 video encoder requires batch one and nonzero NCTHW geometry");
    build.config = config;
    build.pixels_input =
        tensor(ir::DType::F32, ir::TensorRole::Input,
               {config.batch, 3U, config.frames, config.height, config.width});
  }

  H3VideoEncoderBuild finish() {
    using namespace ir;
    auto value = causal_conv(build.pixels_input, 128U, 3U, 1U, 2U, 1U, 1U,
                             "encoder.conv_in");
    capture("conv_in", value);
    constexpr std::array<std::uint64_t, 6> channels{
        128U, 256U, 256U, 512U, 512U, 1024U};
    constexpr std::array<std::uint64_t, 6> spatial_stride{
        2U, 2U, 2U, 2U, 1U, 1U};
    constexpr std::array<std::uint64_t, 6> temporal_stride{
        1U, 2U, 2U, 1U, 1U, 1U};
    for (std::size_t level = 0U; level < channels.size(); ++level) {
      for (std::size_t block_index = 0U; block_index < 2U; ++block_index)
        value = residual(value, channels[level],
                         "encoder.down." + std::to_string(level) +
                             ".block." + std::to_string(block_index));
      capture("down_" + std::to_string(level) + "_residual", value);
      if (spatial_stride[level] * temporal_stride[level] > 1U) {
        if (spatial_stride[level] == 2U)
          value = reflect_pad(value, 0U, 0U, 0U, 1U, 0U, 1U);
        value = causal_conv(
            value, channels[level], 3U, 0U, 2U, temporal_stride[level],
            spatial_stride[level],
            "encoder.down." + std::to_string(level) + ".downsample.conv");
        capture("down_" + std::to_string(level) + "_sample", value);
      }
    }
    value = group_norm_per_frame(value, "encoder.norm_out");
    auto activated = same(value);
    operation(Opcode::SiLU, {value}, {activated});
    value = causal_conv(activated, 48U, 3U, 1U, 2U, 1U, 1U,
                        "encoder.conv_out");
    capture("encoder_conv_out", value);
    value = causal_conv(value, 48U, 1U, 0U, 0U, 1U, 1U, "quant_conv");
    build.moments_output = output(value, "moments");
    ir::verify(build.program);
    return std::move(build);
  }

private:
  H3VideoEncoderConfig config;
  H3VideoEncoderBuild build;
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
      fail("H3 video encoder builder lost a tensor descriptor");
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
                           std::vector<std::uint64_t> dims) {
    const auto roles = static_cast<std::uint32_t>(
        ir::TensorRole::Constant |
        (config.streamed_constants ? ir::TensorRole::Streamed
                                   : ir::TensorRole::Internal));
    const auto id = tensor(ir::DType::F32, roles, std::move(dims));
    build.weights.push_back({id, std::move(name)});
    return id;
  }

  std::uint32_t reflect_pad(std::uint32_t input, std::uint64_t front,
                            std::uint64_t back, std::uint64_t top,
                            std::uint64_t bottom, std::uint64_t west,
                            std::uint64_t east) {
    using namespace ir;
    const auto &desc = description(input);
    auto dims = desc.dims;
    dims[2] += front + back;
    dims[3] += top + bottom;
    dims[4] += west + east;
    const auto padded = tensor(desc.dtype, TensorRole::Internal, dims);
    operation(Opcode::PadReflect, {input}, {padded},
              {Attribute::u64(AttrKey::PadFront, front),
               Attribute::u64(AttrKey::PadBack, back),
               Attribute::u64(AttrKey::PadTop, top),
               Attribute::u64(AttrKey::PadBottom, bottom),
               Attribute::u64(AttrKey::PadWest, west),
               Attribute::u64(AttrKey::PadEast, east)});
    return padded;
  }

  std::uint32_t constant_pad(std::uint32_t input, std::uint64_t front) {
    using namespace ir;
    const auto &desc = description(input);
    auto dims = desc.dims;
    dims[2] += front;
    const auto padded = tensor(desc.dtype, TensorRole::Internal, dims);
    operation(Opcode::PadConstant, {input}, {padded},
              {Attribute::u64(AttrKey::PadFront, front),
               Attribute::f64(AttrKey::Value, 0.0)});
    return padded;
  }

  std::uint32_t causal_conv(std::uint32_t input,
                            std::uint64_t out_channels,
                            std::uint64_t kernel,
                            std::uint64_t spatial_padding,
                            std::uint64_t temporal_padding,
                            std::uint64_t stride_t,
                            std::uint64_t stride_s,
                            const std::string &prefix) {
    using namespace ir;
    auto value = input;
    if (spatial_padding != 0U)
      value = reflect_pad(value, 0U, 0U, spatial_padding, spatial_padding,
                          spatial_padding, spatial_padding);
    if (temporal_padding != 0U)
      value = constant_pad(value, temporal_padding);
    const auto &desc = description(value);
    const auto weight = checkpoint(
        prefix + ".weight",
        {out_channels, desc.dims[1], kernel, kernel, kernel});
    const auto bias = checkpoint(prefix + ".bias", {out_channels});
    const std::vector<std::uint64_t> output_dims{
        desc.dims[0], out_channels,
        (desc.dims[2] - kernel) / stride_t + 1U,
        (desc.dims[3] - kernel) / stride_s + 1U,
        (desc.dims[4] - kernel) / stride_s + 1U};
    const auto result = tensor(DType::F32, TensorRole::Internal, output_dims);
    operation(Opcode::Conv3d, {value, weight, bias}, {result},
              {Attribute::u64(AttrKey::StrideT, stride_t),
               Attribute::u64(AttrKey::StrideH, stride_s),
               Attribute::u64(AttrKey::StrideW, stride_s),
               Attribute::u64(AttrKey::DilationT, 1U),
               Attribute::u64(AttrKey::DilationH, 1U),
               Attribute::u64(AttrKey::DilationW, 1U),
               Attribute::u64(AttrKey::Groups, 1U),
               Attribute::u64(AttrKey::WorkspaceLimitBytes,
                              64ULL * 1024ULL * 1024ULL)});
    return result;
  }

  std::uint32_t group_norm_per_frame(std::uint32_t input,
                                     const std::string &prefix) {
    using namespace ir;
    const auto &desc = description(input);
    const auto channels = desc.dims[1];
    const auto weight = checkpoint(prefix + ".weight", {channels});
    const auto bias = checkpoint(prefix + ".bias", {channels});
    const auto ntchw = tensor(DType::F32, TensorRole::Internal,
                              {desc.dims[0], desc.dims[2], channels,
                               desc.dims[3], desc.dims[4]});
    operation(Opcode::Permute, {input}, {ntchw},
              permutation({0U, 2U, 1U, 3U, 4U}));
    const auto frames = tensor(
        DType::F32, TensorRole::Internal,
        {desc.dims[0] * desc.dims[2], channels, desc.dims[3], desc.dims[4]});
    operation(Opcode::Reshape, {ntchw}, {frames});
    const auto normalized = same(frames);
    operation(Opcode::GroupNorm, {frames, weight, bias}, {normalized},
              {Attribute::u64(AttrKey::Groups, 32U),
               Attribute::u64(AttrKey::BlockSize, 256U),
               Attribute::f64(AttrKey::Epsilon, 1.0e-6)});
    const auto restored = same(ntchw);
    operation(Opcode::Reshape, {normalized}, {restored});
    const auto result = same(input);
    operation(Opcode::Permute, {restored}, {result},
              permutation({0U, 2U, 1U, 3U, 4U}));
    return result;
  }

  std::uint32_t residual(std::uint32_t input,
                         std::uint64_t out_channels,
                         const std::string &prefix) {
    using namespace ir;
    const auto in_channels = description(input).dims[1];
    auto value = group_norm_per_frame(input, prefix + ".norm1");
    auto activated = same(value);
    operation(Opcode::SiLU, {value}, {activated});
    value = causal_conv(activated, out_channels, 3U, 1U, 2U, 1U, 1U,
                        prefix + ".conv1");
    value = group_norm_per_frame(value, prefix + ".norm2");
    activated = same(value);
    operation(Opcode::SiLU, {value}, {activated});
    value = causal_conv(activated, out_channels, 3U, 1U, 2U, 1U, 1U,
                        prefix + ".conv2");
    auto shortcut = input;
    if (in_channels != out_channels)
      shortcut = causal_conv(input, out_channels, 1U, 0U, 0U, 1U, 1U,
                             prefix + ".nin_shortcut");
    const auto result = same(value);
    operation(Opcode::Add, {shortcut, value}, {result});
    return result;
  }

  void capture(const std::string &name, std::uint32_t value) {
    if (!config.capture_boundaries)
      return;
    mark_output(value);
    build.boundaries.emplace_back(name, value);
  }

  std::uint32_t output(std::uint32_t value, const std::string &name) {
    mark_output(value);
    build.boundaries.emplace_back(name, value);
    return value;
  }

  void mark_output(std::uint32_t value) {
    for (auto &desc : build.program.tensors) {
      if (desc.id != value)
        continue;
      desc.roles |= ir::TensorRole::Output;
      return;
    }
    fail("H3 video encoder lost an output tensor");
  }
};

} // namespace

H3VideoEncoderBuild
make_h3_video_encoder(const H3VideoEncoderConfig &config) {
  return Builder(config).finish();
}

} // namespace dif::frontend
