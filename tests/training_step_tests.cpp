// The training step as one program.
//
// build_training_step composes forward, backward and the optimizer update
// into a single program with one tensor namespace. What has to hold: the
// composed program verifies, every parameter gets exactly one update in the
// order it was asked for, all updates share one step counter so bias
// correction cannot drift between parameters, the updated parameter keeps its
// storage dtype, the per-parameter decay hook reaches the right operation,
// and two builds of the same request are bit-identical.

#include "dif/ir/codec.hpp"
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/step.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

using dif::ir::AttrKey;
using dif::ir::Attribute;
using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

// x @ w1 + b1 -> silu -> @ w2, against a target, as an MSE loss. Four
// parameters of two different ranks, one of them a bias.
Program two_layer_forward(DType parameter_dtype) {
  Program program;
  const auto compute = parameter_dtype;
  program.tensors = {
      {1U, compute, TensorRole::Input, {4U, 6U}},
      {2U, parameter_dtype, TensorRole::Input | TensorRole::Parameter,
       {8U, 6U}},
      {3U, parameter_dtype, TensorRole::Input | TensorRole::Parameter, {8U}},
      {4U, parameter_dtype, TensorRole::Input | TensorRole::Parameter,
       {5U, 8U}},
      {5U, compute, TensorRole::Internal, {4U, 8U}},
      {6U, compute, TensorRole::Internal, {4U, 8U}},
      {7U, compute, TensorRole::Internal, {4U, 5U}},
      {8U, compute, TensorRole::Input, {4U, 5U}},
      {9U, DType::F32, TensorRole::Output, {1U}}};
  program.operations = {{1U, Opcode::Linear, {1U, 2U, 3U}, {5U}, {}},
                        {2U, Opcode::SiLU, {5U}, {6U}, {}},
                        {3U, Opcode::Linear, {6U, 4U}, {7U}, {}},
                        {4U, Opcode::MseLoss, {7U, 8U}, {9U}, {}}};
  dif::ir::verify(program);
  return program;
}

const dif::ir::Operation *update_for(const Program &program,
                                     std::uint32_t parameter) {
  for (const auto &operation : program.operations)
    if (operation.opcode == Opcode::AdamWUpdate &&
        operation.inputs[0] == parameter)
      return &operation;
  return nullptr;
}

void composes_one_program() {
  const auto forward = two_layer_forward(DType::F32);
  const std::vector<std::uint32_t> parameters{2U, 3U, 4U};
  dif::training::OptimizerHyperparameters hyperparameters;
  hyperparameters.learning_rate = 3.0e-4;
  hyperparameters.weight_decay = 0.01;
  const auto step = dif::training::build_training_step(
      forward, 9U, parameters, hyperparameters);

  // The forward program survives inside the composed one: every original
  // operation is still there, in order, before anything the pass added.
  for (std::size_t index = 0U; index < forward.operations.size(); ++index)
    expect(step.program.operations[index].id == forward.operations[index].id &&
               step.program.operations[index].opcode ==
                   forward.operations[index].opcode,
           "the forward program is a prefix of the training step");

  expect(step.bindings.size() == parameters.size(),
         "one binding per parameter");
  std::size_t updates = 0U;
  for (const auto &operation : step.program.operations)
    if (operation.opcode == Opcode::AdamWUpdate)
      ++updates;
  expect(updates == parameters.size(), "one update per parameter");

  for (std::size_t index = 0U; index < parameters.size(); ++index) {
    const auto &binding = step.bindings[index];
    expect(binding.parameter_input == parameters[index],
           "bindings keep the requested parameter order");
    const auto *update = update_for(step.program, parameters[index]);
    expect(update != nullptr, "every parameter has an update");
    if (!update)
      continue;
    expect(update->inputs[4] == step.step_input,
           "every update reads the same step counter");
    expect(update->inputs[1] == binding.gradient_output,
           "the update consumes the gradient the backward pass produced");
    expect(update->f64(AttrKey::LearningRate, 0.0) == 3.0e-4,
           "hyperparameters reach the update");
    expect(update->f64(AttrKey::WeightDecay, -1.0) == 0.01,
           "the default weight decay reaches the update");
    const auto *updated = step.program.tensor(binding.parameter_output);
    const auto *original = step.program.tensor(binding.parameter_input);
    expect(updated != nullptr && original != nullptr &&
               updated->dtype == original->dtype &&
               updated->dims == original->dims,
           "the updated parameter keeps its dtype and shape");
    const auto *first = step.program.tensor(binding.first_moment_output);
    expect(first != nullptr && first->dtype == DType::F32,
           "moments are F32 whatever the parameter storage is");
  }
}

void bf16_parameters_stay_bf16() {
  const auto forward = two_layer_forward(DType::BF16);
  const std::vector<std::uint32_t> parameters{2U, 4U};
  const auto step = dif::training::build_training_step(forward, 9U, parameters,
                                                       {});
  for (const auto &binding : step.bindings) {
    const auto *updated = step.program.tensor(binding.parameter_output);
    expect(updated != nullptr && updated->dtype == DType::BF16,
           "a BF16 parameter stays BF16 across the step");
    const auto *second = step.program.tensor(binding.second_moment_input);
    expect(second != nullptr && second->dtype == DType::F32,
           "a BF16 parameter still keeps F32 moments");
  }
}

void the_decay_hook_reaches_its_parameter() {
  const auto forward = two_layer_forward(DType::F32);
  const std::vector<std::uint32_t> parameters{2U, 3U, 4U};
  dif::training::OptimizerHyperparameters hyperparameters;
  hyperparameters.weight_decay = 0.05;
  // Parameter 3 is the bias: no decay, the way a trainer excludes biases and
  // norms without the engine knowing what a bias is.
  const auto step = dif::training::build_training_step(
      forward, 9U, parameters, hyperparameters,
      [](std::size_t index, std::uint32_t) {
        return index == 1U ? 0.0 : 0.05;
      });
  expect(update_for(step.program, 2U)->f64(AttrKey::WeightDecay, -1.0) == 0.05,
         "a decayed parameter keeps the configured decay");
  expect(update_for(step.program, 3U)->f64(AttrKey::WeightDecay, -1.0) == 0.0,
         "the hook clears the decay on the parameter it names");
  expect(update_for(step.program, 4U)->f64(AttrKey::WeightDecay, -1.0) == 0.05,
         "the hook leaves the other parameters alone");
}

void the_composition_is_deterministic() {
  const auto forward = two_layer_forward(DType::F32);
  const std::vector<std::uint32_t> parameters{2U, 3U, 4U};
  const auto first = dif::training::build_training_step(forward, 9U,
                                                        parameters, {});
  const auto second = dif::training::build_training_step(forward, 9U,
                                                         parameters, {});
  expect(dif::ir::fingerprint(first.program) ==
             dif::ir::fingerprint(second.program),
         "two builds of the same step are bit-identical");

  // Parameter order is part of the request, so a different order is a
  // different program -- and must still be a valid one.
  const std::vector<std::uint32_t> reordered{4U, 3U, 2U};
  const auto swapped = dif::training::build_training_step(forward, 9U,
                                                          reordered, {});
  expect(dif::ir::fingerprint(swapped.program) !=
             dif::ir::fingerprint(first.program),
         "parameter order changes the program");
  expect(swapped.bindings[0].parameter_input == 4U,
         "bindings follow the order asked for");
}

void bad_requests_are_refused() {
  const auto forward = two_layer_forward(DType::F32);
  const std::vector<std::uint32_t> none;
  bool refused = false;
  try {
    dif::training::build_training_step(forward, 9U, none, {});
  } catch (const dif::Error &) {
    refused = true;
  }
  expect(refused, "a step with no parameters is refused");

  refused = false;
  try {
    // Tensor 1 is the input, not a parameter: it has no gradient path to an
    // optimizer state, but it IS differentiable, so the refusal has to come
    // from somewhere honest -- it is simply not a parameter of the model.
    const std::vector<std::uint32_t> missing{999U};
    dif::training::build_training_step(forward, 9U, missing, {});
  } catch (const dif::Error &) {
    refused = true;
  }
  expect(refused, "a parameter that is not a tensor of the program is refused");
}

} // namespace

int main() {
  composes_one_program();
  bf16_parameters_stay_bf16();
  the_decay_hook_reaches_its_parameter();
  the_composition_is_deterministic();
  bad_requests_are_refused();
  if (failures != 0) {
    std::cerr << failures << " training step failure(s)\n";
    return 1;
  }
  std::cout << "training step tests passed\n";
  return 0;
}
