#include "dif/frontend/h3_audio_vae.hpp"

#include "dif/support/error.hpp"

#include <array>
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
using ir::Opcode;
using ir::TensorRole;

// audio_vae/config.json latents_mean / latents_std — the canonical copy the
// plan names; baked so the program is self-contained (integrator decision 4).
constexpr std::array<float, 32> kLatentMean{
    -0.020211687488382354F, 0.3876466479950502F,   -0.04398279799186767F,
    -0.28591514936373F,     0.08179686214561671F,  -0.35782641352446604F,
    0.040623809960919084F,  -0.01552534501956604F, -0.223362481667332F,
    0.1821006842509091F,    0.2941778783780663F,   -0.07901167601970885F,
    -0.056815072777201F,    -0.3699028221860095F,  -0.31616315591624855F,
    0.5905951377425391F,    -0.052139568068853864F, 0.013673160263486295F,
    -0.03691647864630577F,  0.09732660653298163F,  -0.3394662328788498F,
    -0.30685677538541667F,  -0.24504598907458763F, -0.034698524462007344F,
    0.02868032184767538F,   -0.21217779266454084F, -0.1678263169941987F,
    0.3221287889040614F,    -0.1223055851554907F,  0.4356604928128464F,
    -0.0502599202236253F,   0.3979258376211797F};
constexpr std::array<float, 32> kLatentStd{
    1.6895524230479284F, 2.76263727217653F,   1.7945344281264435F,
    1.6801681847309828F, 1.6390226546605453F, 2.7788298348882177F,
    1.7659090095747236F, 1.6199757612137327F, 2.6336525640336896F,
    1.8539356672817833F, 2.5056497896915633F, 1.811019237886178F,
    1.9579657790720237F, 1.6685498243529284F, 1.4922469314453364F,
    3.298670198067373F,  1.9491804496832168F, 1.8720003270431442F,
    1.8334080103291832F, 1.6488070416529093F, 1.6176957696319716F,
    1.9131449234774398F, 1.5695245398428617F, 1.6943659940415912F,
    1.8318420762504692F, 1.5540637421583379F, 1.9344930328968526F,
    1.599198216109855F,  1.718045989838149F,  1.6307219190837705F,
    1.8661226051202384F, 1.5613768203168363F};

runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                           const std::vector<float> &values) {
  runtime::Tensor tensor{DType::F32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

struct Builder {
  AudioBigVganBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t add_tensor(DType dtype, std::uint32_t roles,
                           std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }
  std::uint32_t internal(std::vector<std::uint64_t> dims) {
    return add_tensor(DType::F32,
                      static_cast<std::uint32_t>(TensorRole::Internal),
                      std::move(dims));
  }
  std::uint32_t constant(std::string name, std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(
        DType::F32, static_cast<std::uint32_t>(TensorRole::Constant),
        std::move(dims));
    build.bindings.push_back({id, name, name});
    return id;
  }
  std::uint32_t generated(std::string name, runtime::Tensor tensor) {
    const auto id = add_tensor(
        DType::F32, static_cast<std::uint32_t>(TensorRole::Constant),
        tensor.dims);
    build.bindings.push_back({id, std::move(name), {}});
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

struct ConvGeometry {
  std::uint64_t stride{1}, dilation{1}, groups{1};
  std::uint64_t pad_left{0}, pad_right{0}, pad_mode{0};
  bool transposed{false};
  std::uint64_t trim_left{0}, trim_right{0};
};

std::vector<Attribute> conv_attributes(const ConvGeometry &geometry) {
  // Every attribute stamped explicitly — forward/backward/emitter can never
  // resolve different defaults (the DiT RotaryDim lesson).
  return {Attribute::u64(AttrKey::Stride, geometry.stride),
          Attribute::u64(AttrKey::Dilation, geometry.dilation),
          Attribute::u64(AttrKey::Groups, geometry.groups),
          Attribute::u64(AttrKey::PadLeft, geometry.pad_left),
          Attribute::u64(AttrKey::PadRight, geometry.pad_right),
          Attribute::u64(AttrKey::PadMode, geometry.pad_mode),
          Attribute::boolean(AttrKey::Transposed, geometry.transposed),
          Attribute::u64(AttrKey::TrimLeft, geometry.trim_left),
          Attribute::u64(AttrKey::TrimRight, geometry.trim_right)};
}

} // namespace

AudioBigVganBuild build_audio_bigvgan_program(
    std::uint64_t batch, std::uint64_t latent_frames, std::uint64_t stages,
    const AudioBigVganConfig &config) {
  if (batch == 0U || latent_frames == 0U)
    fail("audio BigVGAN program requires positive batch and frames");
  if (config.upsample_rates.size() != config.upsample_kernels.size() ||
      config.upsample_rates.empty() || config.resblock_kernels.empty() ||
      config.resblock_dilations.empty())
    fail("audio BigVGAN configuration is inconsistent");
  const auto num_stages = config.upsample_rates.size();
  const auto num_kernels = config.resblock_kernels.size();

  Builder builder;
  auto &counted_conv = builder.build.conv1d_operations;
  auto &counted_snake = builder.build.snake_beta_operations;

  // Emit one convolution: binds <name>.weight (+ optional .bias), computes
  // the output geometry with the verifier's exact formulas.
  const auto conv = [&](std::uint32_t input, std::uint64_t in_channels,
                        std::uint64_t length, const std::string &name,
                        std::uint64_t out_channels, std::uint64_t kernel,
                        const ConvGeometry &geometry, bool bias,
                        std::uint32_t filter_id = 0U) {
    std::uint32_t weight = filter_id;
    if (weight == 0U) {
      const auto weight_dims =
          geometry.transposed
              ? std::vector<std::uint64_t>{in_channels,
                                           out_channels / geometry.groups,
                                           kernel}
              : std::vector<std::uint64_t>{out_channels,
                                           in_channels / geometry.groups,
                                           kernel};
      weight = builder.constant(name + ".weight", weight_dims);
    }
    std::vector<std::uint32_t> inputs{input, weight};
    if (bias)
      inputs.push_back(builder.constant(name + ".bias", {out_channels}));
    const auto padded = length + geometry.pad_left + geometry.pad_right;
    const auto out_length =
        geometry.transposed
            ? (padded - 1U) * geometry.stride + kernel - geometry.trim_left -
                  geometry.trim_right
            : (padded - (geometry.dilation * (kernel - 1U) + 1U)) /
                      geometry.stride +
                  1U;
    const auto output =
        builder.internal({batch, out_channels, out_length});
    builder.operation(Opcode::Conv1d, std::move(inputs), {output},
                      conv_attributes(geometry));
    ++counted_conv;
    return std::pair{output, out_length};
  };

  const auto snake = [&](std::uint32_t input, std::uint64_t channels,
                         std::uint64_t length, const std::string &prefix) {
    const auto alpha = builder.constant(prefix + "act.alpha", {channels});
    const auto beta = builder.constant(prefix + "act.beta", {channels});
    const auto output = builder.internal({batch, channels, length});
    builder.operation(Opcode::SnakeBeta, {input, alpha, beta}, {output},
                      {Attribute::f64(AttrKey::Epsilon, 1.0e-9)});
    ++counted_snake;
    return output;
  };

  // Alias-free SnakeBeta (reference :158-226): replicate-padded depthwise
  // 2x transposed upsample with the ratio-folded Kaiser filter, SnakeBeta,
  // then the asymmetric-replicate-padded strided depthwise downsample.
  // Length is preserved.
  const auto alias_free_snake = [&](std::uint32_t input,
                                    std::uint64_t channels,
                                    std::uint64_t length,
                                    const std::string &prefix) {
    const auto kernel = config.resample_kernel;
    const auto ratio = config.resample_ratio;
    const auto pad = kernel / ratio - 1U;
    ConvGeometry up_geometry;
    up_geometry.stride = ratio;
    up_geometry.groups = channels;
    up_geometry.pad_left = up_geometry.pad_right = pad;
    up_geometry.pad_mode = 1U;
    up_geometry.transposed = true;
    up_geometry.trim_left = pad * ratio + (kernel - ratio) / 2U;
    up_geometry.trim_right = pad * ratio + (kernel - ratio + 1U) / 2U;
    const auto up_filter =
        builder.constant(prefix + "upsample.filter", {channels, 1U, kernel});
    const auto [upsampled, up_length] =
        conv(input, channels, length, prefix + "upsample", channels, kernel,
             up_geometry, false, up_filter);
    const auto activated = snake(upsampled, channels, up_length, prefix);
    ConvGeometry down_geometry;
    down_geometry.stride = ratio;
    down_geometry.groups = channels;
    down_geometry.pad_left = kernel / 2U - 1U;  // asymmetric 5/6 (even K)
    down_geometry.pad_right = kernel / 2U;
    down_geometry.pad_mode = 1U;
    const auto down_filter = builder.constant(
        prefix + "downsample.lowpass.filter", {channels, 1U, kernel});
    const auto [downsampled, down_length] =
        conv(activated, channels, up_length, prefix + "downsample", channels,
             kernel, down_geometry, false, down_filter);
    if (down_length != length)
      fail("alias-free resampler does not preserve length");
    return downsampled;
  };

  // ── the program ─────────────────────────────────────────────────────────
  const auto latent = builder.add_tensor(
      DType::F32, static_cast<std::uint32_t>(TensorRole::Input),
      {batch, config.latent_channels, latent_frames});
  builder.build.latent_input_id = latent;

  // In-program latent denormalization: depthwise K=1 conv, weight = std,
  // bias = mean (x*std + mean).
  const auto std_id = builder.generated(
      "dif.audio_latents_std",
      f32_tensor({config.latent_channels, 1U, 1U},
                 {kLatentStd.begin(), kLatentStd.end()}));
  const auto mean_id = builder.generated(
      "dif.audio_latents_mean",
      f32_tensor({config.latent_channels},
                 {kLatentMean.begin(), kLatentMean.end()}));
  const auto denormalized =
      builder.internal({batch, config.latent_channels, latent_frames});
  ConvGeometry denorm_geometry;
  denorm_geometry.groups = config.latent_channels;
  builder.operation(Opcode::Conv1d, {latent, std_id, mean_id}, {denormalized},
                    conv_attributes(denorm_geometry));
  ++counted_conv;

  auto [hidden, length] = conv(denormalized, config.latent_channels,
                               latent_frames, "dec_in_proj",
                               config.latent_dim, 1U, {}, true);
  std::uint64_t channels = config.decoder_dim;
  ConvGeometry pre_geometry;
  pre_geometry.pad_left = pre_geometry.pad_right = 3U;
  std::tie(hidden, length) = conv(hidden, config.latent_dim, length,
                                  "decoder.conv_pre", channels, 7U,
                                  pre_geometry, true);

  const auto emitted_stages = stages < num_stages ? stages : num_stages;
  for (std::uint64_t stage = 0; stage < emitted_stages; ++stage) {
    const auto rate = config.upsample_rates[stage];
    const auto kernel = config.upsample_kernels[stage];
    const auto out_channels = channels / 2U;
    ConvGeometry up_geometry;
    up_geometry.stride = rate;
    up_geometry.transposed = true;
    up_geometry.trim_left = up_geometry.trim_right = (kernel - rate) / 2U;
    std::tie(hidden, length) =
        conv(hidden, channels, length,
             "decoder.ups." + std::to_string(stage) + ".0", out_channels,
             kernel, up_geometry, true);
    channels = out_channels;

    std::uint32_t block_sum = 0U;
    for (std::uint64_t j = 0; j < num_kernels; ++j) {
      const auto block_index = stage * num_kernels + j;
      const auto block_kernel = config.resblock_kernels[j];
      const auto block_prefix =
          "decoder.resblocks." + std::to_string(block_index) + ".";
      auto residual_stream = hidden;
      for (std::uint64_t d = 0; d < config.resblock_dilations.size(); ++d) {
        const auto dilation = config.resblock_dilations[d];
        const auto act1_prefix =
            block_prefix + "activations." + std::to_string(2U * d) + ".";
        const auto act2_prefix =
            block_prefix + "activations." + std::to_string(2U * d + 1U) + ".";
        const auto activated1 =
            alias_free_snake(residual_stream, channels, length, act1_prefix);
        ConvGeometry conv1_geometry;
        conv1_geometry.dilation = dilation;
        conv1_geometry.pad_left = conv1_geometry.pad_right =
            (block_kernel * dilation - dilation) / 2U;
        const auto [convolved1, length1] =
            conv(activated1, channels, length,
                 block_prefix + "convs1." + std::to_string(d), channels,
                 block_kernel, conv1_geometry, true);
        const auto activated2 =
            alias_free_snake(convolved1, channels, length1, act2_prefix);
        ConvGeometry conv2_geometry;
        conv2_geometry.pad_left = conv2_geometry.pad_right =
            (block_kernel - 1U) / 2U;
        const auto [convolved2, length2] =
            conv(activated2, channels, length1,
                 block_prefix + "convs2." + std::to_string(d), channels,
                 block_kernel, conv2_geometry, true);
        if (length2 != length)
          fail("AMP block convolution changed the sequence length");
        const auto added = builder.internal({batch, channels, length});
        builder.operation(Opcode::Add, {residual_stream, convolved2}, {added});
        residual_stream = added;
      }
      if (block_sum == 0U) {
        block_sum = residual_stream;
      } else {
        const auto accumulated = builder.internal({batch, channels, length});
        builder.operation(Opcode::Add, {block_sum, residual_stream},
                          {accumulated});
        block_sum = accumulated;
      }
    }
    // Average of the num_kernels block outputs: AffineLastDim with a
    // generated [L] 1/num_kernels vector (the plan's Fill+Multiply pair,
    // realized without a broadcast op; the divide-by-3 trap stays explicit).
    const auto third = builder.generated(
        "dif.audio_block_average_stage" + std::to_string(stage),
        f32_tensor({length},
                   std::vector<float>(
                       length, 1.0F / static_cast<float>(num_kernels))));
    const auto averaged = builder.internal({batch, channels, length});
    builder.operation(Opcode::AffineLastDim, {block_sum, third}, {averaged});
    hidden = averaged;
  }

  std::uint32_t boundary = hidden;
  if (stages > num_stages) {
    hidden = alias_free_snake(hidden, channels, length,
                              "decoder.activation_post.");
    ConvGeometry post_geometry;
    post_geometry.pad_left = post_geometry.pad_right = 3U;
    std::tie(hidden, length) = conv(hidden, channels, length,
                                    "decoder.conv_post", 1U, 7U,
                                    post_geometry, false);
    const auto clamped = builder.internal({batch, 1U, length});
    builder.operation(Opcode::Clamp, {hidden}, {clamped},
                      {Attribute::f64(AttrKey::Lower, -1.0),
                       Attribute::f64(AttrKey::Upper, 1.0)});
    boundary = clamped;
  }
  // Mark the boundary tensor as the program output.
  for (auto &tensor : builder.build.program.tensors) {
    if (tensor.id == boundary) {
      tensor.roles = static_cast<std::uint32_t>(TensorRole::Output);
      break;
    }
  }
  builder.build.waveform_output_id = boundary;
  return std::move(builder.build);
}

} // namespace dif::frontend
