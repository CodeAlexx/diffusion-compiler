#include "dif/frontend/training.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"

#include <algorithm>
#include <array>

namespace dif::frontend {

MlpTrainingBuild make_mlp_training(const MlpTrainingConfig &config) {
  if (config.rows == 0U || config.input_width == 0U ||
      config.hidden_width == 0U || config.output_width == 0U)
    fail("MLP training dimensions must be positive");
  if (config.compute_dtype != ir::DType::F32 &&
      config.compute_dtype != ir::DType::BF16)
    fail("MLP training compute dtype must be F32 or BF16");
  MlpTrainingBuild build;
  auto &forward = build.program;
  constexpr auto input = ir::TensorRole::Input;
  constexpr auto output = ir::TensorRole::Output;
  constexpr auto parameter = ir::TensorRole::Input | ir::TensorRole::Parameter;
  build.features_input = 1U;
  build.target_input = 2U;
  const std::array<std::uint32_t, 4> parameters = {3U, 4U, 5U, 6U};
  const auto compute = config.compute_dtype;
  const bool mixed = compute != ir::DType::F32;
  forward.tensors = {
      {1U, compute, input, {config.rows, config.input_width}},
      {2U, ir::DType::F32, input, {config.rows, config.output_width}},
      {3U, compute, parameter,
       {config.hidden_width, config.input_width}},
      {4U, compute, parameter, {config.hidden_width}},
      {5U, compute, parameter,
       {config.output_width, config.hidden_width}},
      {6U, compute, parameter, {config.output_width}},
      {7U, compute, ir::TensorRole::Internal,
       {config.rows, config.hidden_width}},
      {8U, compute, ir::TensorRole::Internal,
       {config.rows, config.hidden_width}},
      {9U, compute, ir::TensorRole::Internal,
       {config.rows, config.hidden_width}},
      {10U, compute, ir::TensorRole::Internal,
       {config.rows, config.output_width}},
      {11U, compute, output, {config.rows, config.output_width}},
  };
  forward.operations = {
      {1U, ir::Opcode::Linear, {1U, 3U}, {7U}, {}},
      {2U, ir::Opcode::BiasAdd, {7U, 4U}, {8U}, {}},
      {3U, ir::Opcode::SiLU, {8U}, {9U}, {}},
      {4U, ir::Opcode::Linear, {9U, 5U}, {10U}, {}},
      {5U, ir::Opcode::BiasAdd, {10U, 6U}, {11U}, {}},
  };
  build.prediction_output = 11U;
  if (mixed) {
    // The BF16 prediction crosses an explicit Cast boundary into the F32
    // loss; autodiff mirrors the boundary with a Cast back to BF16.
    forward.tensors.push_back({12U, ir::DType::F32, ir::TensorRole::Internal,
                               {config.rows, config.output_width}});
    forward.tensors.push_back({13U, ir::DType::F32, output, {1U}});
    forward.operations.push_back({6U, ir::Opcode::Cast, {11U}, {12U}, {}});
    forward.operations.push_back(
        {7U, ir::Opcode::MseLoss, {12U, 2U}, {13U}, {}});
    build.loss_output = 13U;
  } else {
    forward.tensors.push_back({12U, ir::DType::F32, output, {1U}});
    forward.operations.push_back(
        {6U, ir::Opcode::MseLoss, {11U, 2U}, {12U}, {}});
    build.loss_output = 12U;
  }
  const auto differentiated =
      training::differentiate(forward, build.loss_output, parameters);
  build.program = differentiated.program;

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : build.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : build.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);
  build.step_input = next_tensor++;
  build.program.tensors.push_back(
      {build.step_input, ir::DType::I32,
       ir::TensorRole::Input | ir::TensorRole::OptimizerState, {1U}});

  for (const auto parameter_id : parameters) {
    const auto *description = build.program.tensor(parameter_id);
    const auto parameter_dims = description->dims;
    OptimizerBinding binding;
    binding.parameter_input = parameter_id;
    binding.gradient_output = differentiated.gradients.at(parameter_id);
    binding.first_moment_input = next_tensor++;
    binding.second_moment_input = next_tensor++;
    binding.parameter_output = next_tensor++;
    binding.first_moment_output = next_tensor++;
    binding.second_moment_output = next_tensor++;
    build.program.tensors.push_back(
        {binding.first_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.second_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.parameter_output, description->dtype,
         ir::TensorRole::Output | ir::TensorRole::Parameter,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.first_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.second_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.operations.push_back(
        {next_operation++,
         ir::Opcode::AdamWUpdate,
         {binding.parameter_input, binding.gradient_output,
          binding.first_moment_input, binding.second_moment_input,
          build.step_input},
         {binding.parameter_output, binding.first_moment_output,
          binding.second_moment_output},
         {ir::Attribute::f64(ir::AttrKey::LearningRate,
                             config.learning_rate),
          ir::Attribute::f64(ir::AttrKey::Beta1, config.beta1),
          ir::Attribute::f64(ir::AttrKey::Beta2, config.beta2),
          ir::Attribute::f64(ir::AttrKey::Epsilon, config.epsilon),
          ir::Attribute::f64(ir::AttrKey::WeightDecay,
                             config.weight_decay)}});
    build.optimizer_bindings.push_back(binding);
  }
  ir::verify(build.program);
  return build;
}

RectifiedFlowTrainingBuild
make_rectified_flow_training(const RectifiedFlowTrainingConfig &config) {
  if (config.rows == 0U || config.latent_width == 0U ||
      config.timestep_width == 0U || config.hidden_width == 0U ||
      config.accumulation_steps == 0U || config.accumulation_steps > 64U)
    fail("rectified-flow training dimensions and accumulation must be positive");

  RectifiedFlowTrainingBuild build;
  auto &forward = build.program;
  constexpr auto input = ir::TensorRole::Input;
  constexpr auto output = ir::TensorRole::Output;
  constexpr auto internal = ir::TensorRole::Internal;
  constexpr auto parameter = ir::TensorRole::Input | ir::TensorRole::Parameter;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto add_tensor = [&](std::uint32_t role,
                        std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    forward.tensors.push_back({id, ir::DType::F32, role, std::move(dims)});
    return id;
  };
  auto add_operation = [&](ir::Opcode opcode,
                           std::vector<std::uint32_t> inputs,
                           std::vector<std::uint32_t> outputs,
                           std::vector<ir::Attribute> attributes = {}) {
    forward.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs),
         std::move(attributes)});
  };

  build.microbatches.reserve(
      static_cast<std::size_t>(config.accumulation_steps));
  for (std::uint64_t microbatch = 0U;
       microbatch < config.accumulation_steps; ++microbatch) {
    RectifiedFlowMicrobatch binding;
    binding.clean_input =
        add_tensor(input, {config.rows, config.latent_width});
    binding.noise_input =
        add_tensor(input, {config.rows, config.latent_width});
    binding.clean_scale_input =
        add_tensor(input, {config.rows, config.latent_width});
    binding.noise_scale_input =
        add_tensor(input, {config.rows, config.latent_width});
    binding.timestep_features_input =
        add_tensor(input, {config.rows, config.timestep_width});
    binding.target_velocity_input =
        add_tensor(input, {config.rows, config.latent_width});
    build.microbatches.push_back(binding);
  }
  build.loss_scale_tensor = add_tensor(internal, {1U});
  const std::array<std::uint32_t, 5> parameters = {
      add_tensor(parameter, {config.hidden_width, config.latent_width}),
      add_tensor(parameter, {config.hidden_width}),
      add_tensor(parameter, {config.hidden_width, config.timestep_width}),
      add_tensor(parameter, {config.latent_width, config.hidden_width}),
      add_tensor(parameter, {config.latent_width}),
  };
  add_operation(
      ir::Opcode::Fill, {}, {build.loss_scale_tensor},
      {ir::Attribute::f64(
          ir::AttrKey::Value,
          1.0 / static_cast<double>(config.accumulation_steps))});

  for (auto &microbatch : build.microbatches) {
    const auto scaled_clean =
        add_tensor(internal, {config.rows, config.latent_width});
    const auto scaled_noise =
        add_tensor(internal, {config.rows, config.latent_width});
    const auto noised =
        add_tensor(internal, {config.rows, config.latent_width});
    const auto latent_hidden =
        add_tensor(internal, {config.rows, config.hidden_width});
    const auto biased_hidden =
        add_tensor(internal, {config.rows, config.hidden_width});
    const auto timestep_hidden =
        add_tensor(internal, {config.rows, config.hidden_width});
    const auto conditioned_hidden =
        add_tensor(internal, {config.rows, config.hidden_width});
    const auto activated =
        add_tensor(internal, {config.rows, config.hidden_width});
    const auto projected =
        add_tensor(internal, {config.rows, config.latent_width});
    microbatch.prediction_output =
        add_tensor(output, {config.rows, config.latent_width});
    microbatch.loss_tensor = add_tensor(internal, {1U});

    add_operation(ir::Opcode::Multiply,
                  {microbatch.clean_input, microbatch.clean_scale_input},
                  {scaled_clean});
    add_operation(ir::Opcode::Multiply,
                  {microbatch.noise_input, microbatch.noise_scale_input},
                  {scaled_noise});
    add_operation(ir::Opcode::Add, {scaled_clean, scaled_noise}, {noised});
    add_operation(ir::Opcode::Linear, {noised, parameters[0]},
                  {latent_hidden});
    add_operation(ir::Opcode::BiasAdd, {latent_hidden, parameters[1]},
                  {biased_hidden});
    add_operation(ir::Opcode::Linear,
                  {microbatch.timestep_features_input, parameters[2]},
                  {timestep_hidden});
    add_operation(ir::Opcode::Add, {biased_hidden, timestep_hidden},
                  {conditioned_hidden});
    add_operation(ir::Opcode::SiLU, {conditioned_hidden}, {activated});
    add_operation(ir::Opcode::Linear, {activated, parameters[3]},
                  {projected});
    add_operation(ir::Opcode::BiasAdd, {projected, parameters[4]},
                  {microbatch.prediction_output});
    add_operation(ir::Opcode::MseLoss,
                  {microbatch.prediction_output,
                   microbatch.target_velocity_input},
                  {microbatch.loss_tensor});
  }

  auto loss_sum = build.microbatches.front().loss_tensor;
  for (std::size_t index = 1U; index < build.microbatches.size(); ++index) {
    const auto next_sum = add_tensor(internal, {1U});
    add_operation(ir::Opcode::Add,
                  {loss_sum, build.microbatches[index].loss_tensor},
                  {next_sum});
    loss_sum = next_sum;
  }
  build.loss_output = add_tensor(output, {1U});
  add_operation(ir::Opcode::Multiply, {loss_sum, build.loss_scale_tensor},
                {build.loss_output});

  const auto differentiated =
      training::differentiate(forward, build.loss_output, parameters);
  build.program = differentiated.program;
  next_tensor = 1U;
  next_operation = 1U;
  for (const auto &tensor : build.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : build.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);
  build.step_input = next_tensor++;
  build.program.tensors.push_back(
      {build.step_input, ir::DType::I32,
       ir::TensorRole::Input | ir::TensorRole::OptimizerState, {1U}});

  for (std::size_t index = 0U; index < parameters.size(); ++index) {
    const auto parameter_id = parameters[index];
    const auto parameter_dims = build.program.tensor(parameter_id)->dims;
    OptimizerBinding binding;
    binding.parameter_input = parameter_id;
    binding.gradient_output = differentiated.gradients.at(parameter_id);
    binding.first_moment_input = next_tensor++;
    binding.second_moment_input = next_tensor++;
    binding.parameter_output = next_tensor++;
    binding.first_moment_output = next_tensor++;
    binding.second_moment_output = next_tensor++;
    build.program.tensors.push_back(
        {binding.first_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.second_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.parameter_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::Parameter,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.first_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState,
         parameter_dims});
    build.program.tensors.push_back(
        {binding.second_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState,
         parameter_dims});
    const auto is_bias = index == 1U || index == 4U;
    build.program.operations.push_back(
        {next_operation++,
         ir::Opcode::AdamWUpdate,
         {binding.parameter_input, binding.gradient_output,
          binding.first_moment_input, binding.second_moment_input,
          build.step_input},
         {binding.parameter_output, binding.first_moment_output,
          binding.second_moment_output},
         {ir::Attribute::f64(ir::AttrKey::LearningRate,
                             config.learning_rate),
          ir::Attribute::f64(ir::AttrKey::Beta1, config.beta1),
          ir::Attribute::f64(ir::AttrKey::Beta2, config.beta2),
          ir::Attribute::f64(ir::AttrKey::Epsilon, config.epsilon),
          ir::Attribute::f64(ir::AttrKey::WeightDecay,
                             is_bias ? 0.0 : config.weight_decay)}});
    build.optimizer_bindings.push_back(binding);
  }
  ir::verify(build.program);
  return build;
}

} // namespace dif::frontend
