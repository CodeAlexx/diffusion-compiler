#include "dif/training/step.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"

#include <algorithm>

namespace dif::training {

TrainingStep build_training_step(
    const ir::Program &forward, std::uint32_t loss_tensor,
    std::span<const std::uint32_t> parameters,
    const OptimizerHyperparameters &hyperparameters,
    const std::function<double(std::size_t, std::uint32_t)> &decay_for) {
  if (parameters.empty())
    fail("a training step needs at least one parameter");

  TrainingStep step;
  auto differentiated = differentiate(forward, loss_tensor, parameters);
  step.program = std::move(differentiated.program);
  step.gradients = std::move(differentiated.gradients);

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : step.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : step.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  // One step counter for the whole update, so bias correction cannot drift
  // between parameters.
  step.step_input = next_tensor++;
  step.program.tensors.push_back(
      {step.step_input, ir::DType::I32,
       ir::TensorRole::Input | ir::TensorRole::OptimizerState, {1U}});

  for (std::size_t index = 0U; index < parameters.size(); ++index) {
    const auto parameter_id = parameters[index];
    const auto *description = step.program.tensor(parameter_id);
    if (!description)
      fail("a training step parameter is not a tensor of the program");
    const auto gradient = step.gradients.find(parameter_id);
    if (gradient == step.gradients.end())
      fail("a training step parameter has no gradient");
    const auto dims = description->dims;

    ParameterBinding binding;
    binding.parameter_input = parameter_id;
    binding.gradient_output = gradient->second;
    binding.first_moment_input = next_tensor++;
    binding.second_moment_input = next_tensor++;
    binding.parameter_output = next_tensor++;
    binding.first_moment_output = next_tensor++;
    binding.second_moment_output = next_tensor++;
    step.program.tensors.push_back(
        {binding.first_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState, dims});
    step.program.tensors.push_back(
        {binding.second_moment_input, ir::DType::F32,
         ir::TensorRole::Input | ir::TensorRole::OptimizerState, dims});
    // The updated parameter keeps the parameter's storage dtype: a BF16
    // parameter stays BF16 across the step, an F32 one stays F32.
    step.program.tensors.push_back(
        {binding.parameter_output, description->dtype,
         ir::TensorRole::Output | ir::TensorRole::Parameter, dims});
    step.program.tensors.push_back(
        {binding.first_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState, dims});
    step.program.tensors.push_back(
        {binding.second_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState, dims});

    const auto weight_decay = decay_for ? decay_for(index, parameter_id)
                                        : hyperparameters.weight_decay;
    step.program.operations.push_back(
        {next_operation++,
         ir::Opcode::AdamWUpdate,
         {binding.parameter_input, binding.gradient_output,
          binding.first_moment_input, binding.second_moment_input,
          step.step_input},
         {binding.parameter_output, binding.first_moment_output,
          binding.second_moment_output},
         {ir::Attribute::f64(ir::AttrKey::LearningRate,
                             hyperparameters.learning_rate),
          ir::Attribute::f64(ir::AttrKey::Beta1, hyperparameters.beta1),
          ir::Attribute::f64(ir::AttrKey::Beta2, hyperparameters.beta2),
          ir::Attribute::f64(ir::AttrKey::Epsilon, hyperparameters.epsilon),
          ir::Attribute::f64(ir::AttrKey::WeightDecay, weight_decay)}});
    step.bindings.push_back(binding);
  }

  ir::verify(step.program);
  return step;
}

} // namespace dif::training
