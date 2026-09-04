#include "dif/training/accumulate.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"

#include <algorithm>
#include <unordered_set>

namespace dif::training {
namespace {

std::uint32_t next_free_tensor(const ir::Program &program) {
  std::uint32_t next = 1U;
  for (const auto &tensor : program.tensors)
    next = std::max(next, tensor.id + 1U);
  return next;
}

std::uint32_t next_free_operation(const ir::Program &program) {
  std::uint32_t next = 1U;
  for (const auto &operation : program.operations)
    next = std::max(next, operation.id + 1U);
  return next;
}

} // namespace

AccumulatingStep build_accumulating_step(
    const ir::Program &forward, std::uint32_t loss_tensor,
    std::span<const std::uint32_t> parameters, std::uint64_t micro_batches,
    const OptimizerHyperparameters &hyperparameters,
    const std::function<double(std::size_t, std::uint32_t)> &decay_for) {
  if (micro_batches == 0U)
    fail("a training step needs at least one micro-batch");
  if (parameters.empty())
    fail("a training step needs at least one parameter");

  AccumulatingStep step;
  step.micro_batches = micro_batches;

  // Scale the loss by 1/N before differentiating, so every gradient arrives
  // already averaged over the micro-batches.  One scalar multiply, against a
  // full-size constant per parameter if the scaling were done later.
  ir::Program scaled = forward;
  auto next_tensor = next_free_tensor(scaled);
  auto next_operation = next_free_operation(scaled);
  auto scaled_loss = loss_tensor;
  if (micro_batches > 1U) {
    const auto *loss = scaled.tensor(loss_tensor);
    if (!loss)
      fail("the loss tensor is not a tensor of the forward program");
    // The original loss stops being the program's output; the averaged one
    // takes its place, so nothing downstream can read the unaveraged value by
    // accident.
    for (auto &tensor : scaled.tensors)
      if (tensor.id == loss_tensor)
        tensor.roles = static_cast<std::uint32_t>(ir::TensorRole::Internal);
    const auto reciprocal = next_tensor++;
    scaled_loss = next_tensor++;
    scaled.tensors.push_back(
        {reciprocal, ir::DType::F32, ir::TensorRole::Internal, {1U}});
    scaled.tensors.push_back(
        {scaled_loss, ir::DType::F32, ir::TensorRole::Output, {1U}});
    scaled.operations.push_back(
        {next_operation++, ir::Opcode::Fill, {}, {reciprocal},
         {ir::Attribute::f64(ir::AttrKey::Value,
                             1.0 / static_cast<double>(micro_batches))}});
    scaled.operations.push_back({next_operation++, ir::Opcode::Multiply,
                                 {loss_tensor, reciprocal},
                                 {scaled_loss},
                                 {}});
    ir::verify(scaled);
  }

  // The whole step -- forward, backward, and the optimizer -- built once, so
  // the accumulate and update programs are two views of ONE namespace rather
  // than two programs that happen to agree about tensor ids.
  const auto whole = build_training_step(scaled, scaled_loss, parameters,
                                         hyperparameters, decay_for);
  step.step_input = whole.step_input;

  const auto optimizer_begin = whole.optimizer_operations;

  // What each optimizer update actually reads as its gradient.  With master
  // weights that is a cast of the raw gradient rather than the gradient
  // itself, and the cast belongs to the MICRO-BATCH: it happens once per
  // micro-batch, and it is what makes the accumulator F32.
  std::unordered_set<std::uint32_t> optimizer_gradients;
  for (std::size_t index = optimizer_begin;
       index < whole.program.operations.size(); ++index) {
    const auto &operation = whole.program.operations[index];
    if (operation.opcode == ir::Opcode::AdamWUpdate)
      optimizer_gradients.insert(operation.inputs[1]);
  }

  next_tensor = next_free_tensor(whole.program);
  next_operation = next_free_operation(whole.program);

  step.accumulate.tensors = whole.program.tensors;
  step.update.tensors = whole.program.tensors;
  for (std::size_t index = 0U; index < optimizer_begin; ++index)
    step.accumulate.operations.push_back(whole.program.operations[index]);
  std::unordered_set<std::size_t> moved;
  for (std::size_t index = optimizer_begin;
       index < whole.program.operations.size(); ++index) {
    const auto &operation = whole.program.operations[index];
    if (operation.opcode == ir::Opcode::Cast &&
        optimizer_gradients.contains(operation.outputs[0])) {
      step.accumulate.operations.push_back(operation);
      moved.insert(index);
    }
  }

  // One accumulator per parameter, in F32 whatever the gradient's storage is:
  // summing N contributions in half precision loses the small ones, which is
  // the whole reason to accumulate.
  for (std::size_t index = 0U; index < parameters.size(); ++index) {
    const auto &source = whole.bindings[index];
    AccumulatorBinding binding;
    binding.parameter_input = source.parameter_input;
    binding.gradient_output = source.gradient_output;
    binding.update = source;
    // The optimizer's own view of this parameter's gradient: the raw one, or
    // the cast the master-weight path put in front of it.
    const auto optimizer_parameter = source.master_input != 0U
                                         ? source.master_input
                                         : source.parameter_input;
    auto optimizer_gradient = source.gradient_output;
    for (std::size_t op = optimizer_begin;
         op < whole.program.operations.size(); ++op) {
      const auto &operation = whole.program.operations[op];
      if (operation.opcode == ir::Opcode::AdamWUpdate &&
          operation.inputs[0] == optimizer_parameter)
        optimizer_gradient = operation.inputs[1];
    }
    binding.gradient_output = optimizer_gradient;
    const auto *gradient = whole.program.tensor(optimizer_gradient);
    if (!gradient)
      fail("a parameter gradient is not a tensor of the step");
    const auto dims = gradient->dims;

    auto contribution = optimizer_gradient;
    if (gradient->dtype != ir::DType::F32) {
      contribution = next_tensor++;
      step.accumulate.tensors.push_back(
          {contribution, ir::DType::F32, ir::TensorRole::Internal, dims});
      step.update.tensors.push_back(
          {contribution, ir::DType::F32, ir::TensorRole::Internal, dims});
      step.accumulate.operations.push_back({next_operation++, ir::Opcode::Cast,
                                            {optimizer_gradient},
                                            {contribution},
                                            {}});
    }
    binding.accumulator_input = next_tensor++;
    binding.accumulator_output = next_tensor++;
    const ir::TensorDesc accumulator_in{
        binding.accumulator_input, ir::DType::F32,
        ir::TensorRole::Input | ir::TensorRole::OptimizerState, dims};
    const ir::TensorDesc accumulator_out{
        binding.accumulator_output, ir::DType::F32,
        ir::TensorRole::Output | ir::TensorRole::OptimizerState, dims};
    step.accumulate.tensors.push_back(accumulator_in);
    step.accumulate.tensors.push_back(accumulator_out);
    step.update.tensors.push_back(accumulator_in);
    step.update.tensors.push_back(accumulator_out);
    step.accumulate.operations.push_back({next_operation++, ir::Opcode::Add,
                                          {binding.accumulator_input,
                                           contribution},
                                          {binding.accumulator_output},
                                          {}});
    step.bindings.push_back(binding);
  }

  // The update program reads the accumulated sum where the single-step
  // program read this micro-batch's gradient.
  for (std::size_t index = optimizer_begin;
       index < whole.program.operations.size(); ++index) {
    if (moved.contains(index))
      continue;
    auto operation = whole.program.operations[index];
    for (auto &input : operation.inputs)
      for (const auto &binding : step.bindings)
        if (input == binding.gradient_output) {
          input = binding.accumulator_output;
          break;
        }
    step.update.operations.push_back(std::move(operation));
  }

  // The update's own view: the accumulators are what it reads, so they are
  // inputs to it; the micro-batch's gradients are not its business.
  for (auto &tensor : step.update.tensors) {
    const bool is_accumulator_output =
        std::any_of(step.bindings.begin(), step.bindings.end(),
                    [&](const AccumulatorBinding &binding) {
                      return binding.accumulator_output == tensor.id;
                    });
    if (is_accumulator_output)
      tensor.roles = ir::TensorRole::Input | ir::TensorRole::OptimizerState;
  }

  // Each program keeps only the tensors its own operations touch.  Without
  // this the micro-batch would declare the optimizer's outputs, which nothing
  // in it produces, and the update would declare the whole model.
  const auto keep_used = [](ir::Program &program) {
    std::unordered_set<std::uint32_t> used;
    for (const auto &operation : program.operations) {
      for (const auto tensor : operation.inputs)
        used.insert(tensor);
      for (const auto tensor : operation.outputs)
        used.insert(tensor);
    }
    std::vector<ir::TensorDesc> kept;
    for (const auto &tensor : program.tensors)
      if (used.contains(tensor.id))
        kept.push_back(tensor);
    program.tensors = std::move(kept);
  };
  keep_used(step.update);
  keep_used(step.accumulate);

  // The planner's view: the micro-batch followed by the update, so the
  // accumulators are placed knowing they outlive every micro-batch.
  step.planned = step.accumulate;
  for (const auto &tensor : step.update.tensors)
    if (!step.planned.tensor(tensor.id))
      step.planned.tensors.push_back(tensor);
  for (auto &tensor : step.planned.tensors) {
    const bool is_accumulator =
        std::any_of(step.bindings.begin(), step.bindings.end(),
                    [&](const AccumulatorBinding &binding) {
                      return binding.accumulator_output == tensor.id;
                    });
    if (is_accumulator)
      tensor.roles = static_cast<std::uint32_t>(ir::TensorRole::Internal);
  }
  for (const auto &operation : step.update.operations)
    step.planned.operations.push_back(operation);

  ir::verify(step.accumulate);
  ir::verify(step.update);
  ir::verify(step.planned);
  return step;
}

} // namespace dif::training
