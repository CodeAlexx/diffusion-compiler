#pragma once

#include "dif/ir/ir.hpp"
#include "dif/training/step.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace dif::training {

// Gradient accumulation, expressed in the program rather than folded by the
// caller.
//
// A model that will not fit a useful batch on one card trains on N smaller
// ones and updates once.  Written by hand that is a loop the compiler cannot
// see: N executions whose gradient buffers are allocated, summed and thrown
// away outside the plan.  Written here it is two programs over ONE tensor
// namespace -- an accumulate program run N times and an update program run
// once -- plus their concatenation, which is what the memory planner reads so
// the accumulators are placed knowing they outlive every micro-batch.
//
// The averaging happens where it is cheapest and least surprising: the loss
// is scaled by 1/N before differentiation, one multiply of a scalar, so every
// gradient arrives already divided.  Scaling the accumulated gradients
// instead would need a full-size tensor of constants per parameter.

struct AccumulatorBinding {
  std::uint32_t parameter_input{};
  // The gradient this micro-batch produced, and the running sum it joins.
  std::uint32_t gradient_output{};
  std::uint32_t accumulator_input{};
  std::uint32_t accumulator_output{};
  // The optimizer's view, filled in by the update program.
  ParameterBinding update;
};

struct AccumulatingStep {
  // Run this micro_batches times, feeding each run's accumulator outputs back
  // in as the next run's accumulator inputs.
  ir::Program accumulate;
  // Then run this once.
  ir::Program update;
  // Both, in order: the program to hand the memory planner so the
  // accumulators are planned across the whole of it.
  ir::Program planned;
  std::uint32_t step_input{};
  std::uint64_t micro_batches{};
  std::vector<AccumulatorBinding> bindings;
};

// `micro_batches` must be at least one; one micro-batch is an ordinary step
// with an accumulator in front of the optimizer, which is worth keeping
// rather than special-casing -- the shape of the run should not change with
// the batch count.
AccumulatingStep build_accumulating_step(
    const ir::Program &forward, std::uint32_t loss_tensor,
    std::span<const std::uint32_t> parameters, std::uint64_t micro_batches,
    const OptimizerHyperparameters &hyperparameters,
    const std::function<double(std::size_t, std::uint32_t)> &decay_for = {});

} // namespace dif::training
