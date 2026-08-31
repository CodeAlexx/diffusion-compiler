#include "dif/frontend/krea2.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <limits>
#include <utility>

namespace dif::frontend {

Krea2Architecture inspect_krea2_architecture(const Krea2Config &config) {
  constexpr auto alignment =
      Krea2Config::kVaeCompression * Krea2Config::kPatch;
  if (config.batch == 0U || config.width == 0U || config.height == 0U ||
      config.text_tokens == 0U || config.text_tokens > 512U ||
      config.width % alignment != 0U || config.height % alignment != 0U)
    fail("Krea 2 requires a positive batch, 1..512 text tokens, and image "
         "dimensions divisible by VAE compression * patch (16)");

  Krea2Architecture result;
  result.latent_height = config.height / Krea2Config::kVaeCompression;
  result.latent_width = config.width / Krea2Config::kVaeCompression;
  result.image_grid_height = result.latent_height / Krea2Config::kPatch;
  result.image_grid_width = result.latent_width / Krea2Config::kPatch;
  if (result.image_grid_height >
      std::numeric_limits<std::uint64_t>::max() / result.image_grid_width)
    fail("Krea 2 image-token geometry overflows");
  result.image_tokens = result.image_grid_height * result.image_grid_width;
  if (result.image_tokens >
      std::numeric_limits<std::uint64_t>::max() - config.text_tokens)
    fail("Krea 2 combined sequence overflows");
  result.combined_tokens = result.image_tokens + config.text_tokens;
  if (result.combined_tokens >
      std::numeric_limits<std::uint64_t>::max() -
          (Krea2Config::kSequenceAlignment - 1U))
    fail("Krea 2 padded sequence overflows");
  result.padded_tokens =
      ((result.combined_tokens + Krea2Config::kSequenceAlignment - 1U) /
       Krea2Config::kSequenceAlignment) *
      Krea2Config::kSequenceAlignment;
  result.patch_input_dim = Krea2Config::kLatentChannels *
                           Krea2Config::kPatch * Krea2Config::kPatch;
  result.patch_output_dim = result.patch_input_dim;
  return result;
}

Krea2TimeConditioningBuild
make_krea2_time_conditioning(const Krea2Config &config) {
  (void)inspect_krea2_architecture(config);
  using namespace ir;
  Krea2TimeConditioningBuild build;
  build.config = config;
  auto &program = build.program;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  const auto constant_roles = static_cast<std::uint32_t>(
      TensorRole::Constant |
      (config.streamed_constants ? TensorRole::Streamed
                                 : TensorRole::Internal));
  const auto add_tensor = [&](DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_operation = [&](Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<Attribute> attributes = {}) {
    program.operations.push_back({next_operation++, opcode, std::move(inputs),
                                  std::move(outputs), std::move(attributes)});
  };
  const auto add_checkpoint = [&](const char *name,
                                  std::vector<std::uint64_t> dims) {
    const auto id = add_tensor(DType::BF16, constant_roles, std::move(dims));
    build.checkpoint_tensors.push_back(id);
    build.checkpoint_names.emplace_back(name);
    return id;
  };

  build.timestep_input =
      add_tensor(DType::BF16, TensorRole::Input, {config.batch});
  const auto timestep_f32 =
      add_tensor(DType::F32, TensorRole::Internal, {config.batch});
  add_operation(Opcode::Cast, {build.timestep_input}, {timestep_f32});
  build.timestep_embedding = add_tensor(
      DType::F32, TensorRole::Internal,
      {config.batch, Krea2Config::kTimestepDim});
  add_operation(
      Opcode::SinusoidalTimestep, {timestep_f32},
      {build.timestep_embedding},
      {Attribute::boolean(AttrKey::FlipSinToCos, true),
       Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
       Attribute::f64(AttrKey::Scale, 1000.0),
       Attribute::f64(AttrKey::MaxPeriod, 10000.0)});
  const auto embedding_bf16 = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kTimestepDim});
  add_operation(Opcode::Cast, {build.timestep_embedding}, {embedding_bf16});

  const auto tmlp0_weight = add_checkpoint(
      "tmlp.0.weight",
      {Krea2Config::kFeatures, Krea2Config::kTimestepDim});
  const auto tmlp0_bias =
      add_checkpoint("tmlp.0.bias", {Krea2Config::kFeatures});
  const auto tmlp0 = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {embedding_bf16, tmlp0_weight, tmlp0_bias}, {tmlp0});
  const auto tmlp0_activated = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  const auto gelu_attrs = std::vector<Attribute>{Attribute::u64(
      AttrKey::Approximation,
      static_cast<std::uint64_t>(GeluApproximation::Tanh))};
  add_operation(Opcode::Gelu, {tmlp0}, {tmlp0_activated}, gelu_attrs);

  const auto tmlp2_weight = add_checkpoint(
      "tmlp.2.weight", {Krea2Config::kFeatures, Krea2Config::kFeatures});
  const auto tmlp2_bias =
      add_checkpoint("tmlp.2.bias", {Krea2Config::kFeatures});
  build.timestep_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {tmlp0_activated, tmlp2_weight, tmlp2_bias},
                {build.timestep_output});

  const auto tproj_activated = add_tensor(
      DType::BF16, TensorRole::Internal,
      {config.batch, Krea2Config::kFeatures});
  add_operation(Opcode::Gelu, {build.timestep_output}, {tproj_activated},
                gelu_attrs);
  const auto tproj1_weight = add_checkpoint(
      "tproj.1.weight",
      {6U * Krea2Config::kFeatures, Krea2Config::kFeatures});
  const auto tproj1_bias =
      add_checkpoint("tproj.1.bias", {6U * Krea2Config::kFeatures});
  build.modulation_output = add_tensor(
      DType::BF16, TensorRole::Output,
      {config.batch, 6U * Krea2Config::kFeatures});
  add_operation(Opcode::Linear,
                {tproj_activated, tproj1_weight, tproj1_bias},
                {build.modulation_output});

  verify(program);
  return build;
}

} // namespace dif::frontend
