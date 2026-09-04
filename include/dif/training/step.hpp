#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

namespace dif::training {

// One training step as one program.
//
// The compiler holds the whole forward graph as verified DiffIR before
// anything runs, so the three phases a trainer needs -- forward, backward and
// the optimizer update -- do not have to be three independently planned
// executions.  build_training_step composes them into a single program
// sharing one tensor namespace, which is what lets the memory planner see the
// activations, the gradients and the optimizer state at the same time.
//
// This is the engine shape the Mojo autograd v2 contract specifies, reached
// the way a compiler reaches it: the eager runtime records a tape and walks it
// with a dependency counter, while here the graph is already static, so the
// engine is a composition pass and the dependency counting is the memory
// planner's liveness.

struct OptimizerHyperparameters {
  double learning_rate{1.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon{1.0e-8};
  double weight_decay{0.0};
};

// The tensors that tie one parameter to its gradient and its optimizer state
// across the step.  Moments are F32 always; the updated parameter keeps the
// parameter's storage dtype (the flame AdamW kernel matrix).
struct ParameterBinding {
  std::uint32_t parameter_input{};
  std::uint32_t gradient_output{};
  std::uint32_t first_moment_input{};
  std::uint32_t second_moment_input{};
  std::uint32_t parameter_output{};
  std::uint32_t first_moment_output{};
  std::uint32_t second_moment_output{};
};

struct TrainingStep {
  ir::Program program;
  std::uint32_t step_input{};
  std::vector<ParameterBinding> bindings;
  std::unordered_map<std::uint32_t, std::uint32_t> gradients;
};

// Differentiates `forward` with respect to `parameters` and appends one AdamW
// update per parameter, in the order given.  `decay_for`, when set, overrides
// the weight decay per parameter (index into `parameters`, then its tensor
// id) -- the hook a trainer needs to exclude biases and norms from decay
// without the engine knowing what a bias is.
TrainingStep build_training_step(
    const ir::Program &forward, std::uint32_t loss_tensor,
    std::span<const std::uint32_t> parameters,
    const OptimizerHyperparameters &hyperparameters,
    const std::function<double(std::size_t, std::uint32_t)> &decay_for = {});

} // namespace dif::training
