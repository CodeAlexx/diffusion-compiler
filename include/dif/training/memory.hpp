#pragma once

#include "dif/compiler/memory_plan.hpp"
#include "dif/target/profile.hpp"
#include "dif/training/recompute.hpp"
#include "dif/training/session.hpp"

#include <cstdint>

namespace dif::training {

// What a training step's memory is actually spent on.
//
// The memory planner sees one flat program and a lifetime per tensor. That is
// enough to place them well, and it already does -- but it cannot say WHY the
// step costs what it costs, and a recompute decision needs to know. A
// TrainingPlan records where its forward pass ends and its optimizer begins,
// so the classification below is read off the graph rather than guessed at
// from names.
//
// The one class that matters most is `saved_activations`: values the forward
// pass produced that the backward pass still needs. Those are the only bytes
// recompute can trade for arithmetic. Everything else -- weights, gradients,
// optimizer state -- is there whatever the schedule does.
struct TrainingMemoryReport {
  std::uint64_t frozen_weights{};
  std::uint64_t trainable_weights{};
  std::uint64_t optimizer_state{};
  std::uint64_t gradients{};
  // Produced and consumed entirely inside the forward pass.
  std::uint64_t transient_activations{};
  // Produced by the forward pass and still needed by the backward pass. These
  // are the recompute candidates.
  std::uint64_t saved_activations{};
  // Backward-region intermediates that are not a parameter's gradient.
  std::uint64_t backward_temporaries{};
  std::uint64_t other{};

  // What the planner actually allocates, and the largest total of bytes
  // simultaneously live -- the floor no scheduling can go under. Both come
  // from the same aligned assignment, so the floor never exceeds the plan.
  //
  // Neither is comparable to total(): the classification sums RAW tensor
  // bytes, while the plan aligns every buffer to 256 and shares one buffer
  // between tensors whose lifetimes do not overlap. On a large graph sharing
  // wins and the plan is smaller than the classification; on a small one
  // padding wins and it is larger. They answer different questions and are
  // both reported for that reason.
  std::uint64_t planned_bytes{};
  std::uint64_t live_peak_bytes{};

  std::uint64_t total() const {
    return frozen_weights + trainable_weights + optimizer_state + gradients +
           transient_activations + saved_activations + backward_temporaries +
           other;
  }
};

TrainingMemoryReport analyze_memory(const TrainingPlan &plan);

// How much device memory a training plan may use, and on what hardware.
//
// Recompute stops being a caller's guess at a byte count and becomes a
// decision the compiler makes from the same target facts inference already
// plans against.
struct RecomputePolicy {
  target::RuntimeBudget budget;
  // The fraction of the usable budget a plan may occupy. The remainder
  // absorbs what the planner cannot see: library workspaces, allocator
  // rounding, fragmentation.
  double headroom{0.9};

  std::uint64_t device_bytes() const;
};

// Chooses a segmentation for `plan` that fits the policy. Returns what it
// chose and why, and writes the resulting program to `chosen` when given one.
RecomputeChoice choose_recompute(const TrainingPlan &plan,
                                 const RecomputePolicy &policy,
                                 ir::Program *chosen = nullptr);

} // namespace dif::training
