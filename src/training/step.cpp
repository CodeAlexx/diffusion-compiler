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

    // A parameter trains through an F32 master when it is asked for, and
    // always when its storage is F16: AdamW accepts F32 and BF16 storage, so
    // an F16 checkpoint has no other way to train at all.
    const auto storage = description->dtype;
    const bool needs_master =
        storage == ir::DType::F16 ||
        (hyperparameters.master_weights && storage != ir::DType::F32);

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
        {binding.parameter_output, storage,
         ir::TensorRole::Output | ir::TensorRole::Parameter, dims});
    step.program.tensors.push_back(
        {binding.first_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState, dims});
    step.program.tensors.push_back(
        {binding.second_moment_output, ir::DType::F32,
         ir::TensorRole::Output | ir::TensorRole::OptimizerState, dims});

    // What the optimizer actually reads and writes.  Without a master that is
    // the parameter itself; with one it is the F32 copy, and the forward
    // pass's parameter is the rounded-down result.
    auto optimizer_parameter = binding.parameter_input;
    auto optimizer_output = binding.parameter_output;
    auto optimizer_gradient = binding.gradient_output;
    if (needs_master) {
      binding.master_input = next_tensor++;
      binding.master_output = next_tensor++;
      step.program.tensors.push_back(
          {binding.master_input, ir::DType::F32,
           ir::TensorRole::Input | ir::TensorRole::Parameter |
               ir::TensorRole::OptimizerState,
           dims});
      step.program.tensors.push_back(
          {binding.master_output, ir::DType::F32,
           ir::TensorRole::Output | ir::TensorRole::Parameter |
               ir::TensorRole::OptimizerState,
           dims});
      optimizer_parameter = binding.master_input;
      optimizer_output = binding.master_output;
      // The gradient arrives in the parameter's storage dtype. AdamW takes
      // F32 or BF16 gradients, so an F16 gradient crosses an explicit cast
      // rather than being reinterpreted.
      const auto *gradient_description =
          step.program.tensor(binding.gradient_output);
      if (gradient_description->dtype == ir::DType::F16) {
        const auto cast_gradient = next_tensor++;
        step.program.tensors.push_back(
            {cast_gradient, ir::DType::F32, ir::TensorRole::Internal, dims});
        step.program.operations.push_back({next_operation++, ir::Opcode::Cast,
                                           {binding.gradient_output},
                                           {cast_gradient},
                                           {}});
        optimizer_gradient = cast_gradient;
      }
    }

    const auto weight_decay = decay_for ? decay_for(index, parameter_id)
                                        : hyperparameters.weight_decay;
    step.program.operations.push_back(
        {next_operation++,
         ir::Opcode::AdamWUpdate,
         {optimizer_parameter, optimizer_gradient, binding.first_moment_input,
          binding.second_moment_input, step.step_input},
         {optimizer_output, binding.first_moment_output,
          binding.second_moment_output},
         {ir::Attribute::f64(ir::AttrKey::LearningRate,
                             hyperparameters.learning_rate),
          ir::Attribute::f64(ir::AttrKey::Beta1, hyperparameters.beta1),
          ir::Attribute::f64(ir::AttrKey::Beta2, hyperparameters.beta2),
          ir::Attribute::f64(ir::AttrKey::Epsilon, hyperparameters.epsilon),
          ir::Attribute::f64(ir::AttrKey::WeightDecay, weight_decay)}});
    // Round the updated master back down for the next forward pass.
    if (needs_master)
      step.program.operations.push_back({next_operation++, ir::Opcode::Cast,
                                         {binding.master_output},
                                         {binding.parameter_output},
                                         {}});
    step.bindings.push_back(binding);
  }

  ir::verify(step.program);
  return step;
}

} // namespace dif::training
