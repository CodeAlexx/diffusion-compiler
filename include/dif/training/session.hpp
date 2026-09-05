#pragma once

#include "dif/ir/codec.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/sha256.hpp"
#include "dif/training/accumulate.hpp"
#include "dif/training/checkpoint.hpp"
#include "dif/training/memory.hpp"
#include "dif/training/report.hpp"
#include "dif/training/step.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dif::training {

// A training step, and the thing that runs it.
//
// The split is the point of this file. A TrainingPlan describes WHAT a step
// is -- the composed program, which tensors carry state, where the phases
// begin. A TrainingSession is HOW it executes -- it owns the device-resident
// state, advances the step counter, and is the only thing that touches the
// runtime. A model frontend builds a forward graph and names its trainable
// parameters; it writes none of this.
//
// Before this existed, every caller hand-wrote the loop, the state carry and
// the checkpoint plumbing. Two tools in this repository did it differently.

struct TrainingPlan {
  ir::Program program;
  std::uint32_t step_input{};
  std::uint32_t loss_tensor{};
  std::vector<ParameterBinding> bindings;
  // Where the forward pass ends and the optimizer begins, as indices into
  // `program.operations`.
  std::size_t forward_operations{};
  std::size_t optimizer_operations{};

  // The persistent-state declaration this plan implies: every tensor the step
  // advances, paired with where its next value comes from. DERIVED, never
  // authored by a caller -- a hand-written list is a list that can disagree
  // with the program it describes.
  std::vector<runtime::PersistentStateBinding> persistent_state() const;

  // Every tensor a checkpoint has to hold for a resume to be exact.
  std::vector<std::uint32_t> checkpoint_tensors() const;

  Sha256Digest fingerprint() const { return ir::fingerprint(program); }
};

// Wraps a step a model frontend already composed. Frontends build their
// training graph through build_training_step and hand back its pieces; this
// is how those become a plan a session can run, without the caller
// reassembling the persistent-state declaration by hand.
TrainingPlan plan_from_composed(ir::Program program, std::uint32_t step_input,
                                std::uint32_t loss_tensor,
                                std::vector<ParameterBinding> bindings);

// Composes forward, backward and the optimizer update into one plan.
TrainingPlan compile(const ir::Program &forward, std::uint32_t loss_tensor,
                     std::span<const std::uint32_t> parameters,
                     const OptimizerHyperparameters &hyperparameters,
                     const std::function<double(std::size_t, std::uint32_t)>
                         &decay_for = {});

struct TrainingStepResult {
  float loss{};
  double step_milliseconds{};
  std::uint64_t completed_steps{};
  // Which micro-batch of an accumulation group this was, and whether the
  // optimizer ran at the end of it. A plain training step applies its
  // optimizer every time, so these are 0 and true until accumulation is
  // driven through the session.
  std::uint64_t accumulation_index{};
  bool optimizer_applied{true};
  // Absent unless something computed it. The session does not: a gradient
  // norm costs a reduction and a readback, and a step that reports one it did
  // not measure is worse than a step that admits it has none.
  std::optional<double> gradient_norm;
  // Nonfinite values SEEN. The session checks its loss and nothing else,
  // because the gradients stay on the device where nobody reads them -- so a
  // zero here means "the loss was finite", not "the step was clean".
  std::uint64_t nonfinite_count{};
  // Zero on an ordinary step. Reported rather than assumed.
  std::uint64_t persistent_state_host_to_device_bytes{};
  std::uint64_t persistent_state_device_to_host_bytes{};
  // Whatever the program produced besides the carried state -- gradients, a
  // prediction. The state itself is NOT here: it stays on the device.
  runtime::TensorMap outputs;
  // What the runtime actually submitted for this step: launches, library
  // dispatches, every copy, every host-blocking synchronization. A step time
  // without these is a number nobody can act on.
  runtime::LaunchTelemetry telemetry;
  // Present only when the run was traced. An untraced step reports no phase
  // split rather than a split of zeros.
  std::optional<PhaseTimes> phases;
  // The raw events, when the run was traced. Empty otherwise, and empty is
  // the normal case: collecting these costs something, so a step that was
  // not asked to profile does not carry them.
  std::vector<runtime::TraceEvent> trace_events;
  // Device-side timing, when the run asked for it. Trace events record when
  // the HOST submitted work, which for an asynchronous launch is not when the
  // GPU did it; this is measured on the device timeline.
  runtime::PipelineProfile profile;
  // Per-operation DEVICE time, when the run asked to be profiled. This is
  // what a kernel actually cost, as opposed to when the host got round to
  // submitting it.
  std::vector<runtime::OperationTiming> operation_timings;
};

class TrainingSession {
public:
  // `initial` seeds everything the program needs once: constants, the
  // starting parameters, and zeroed optimizer state. After this the session
  // owns the state and a step supplies only its batch.
  // `reads` names the outputs a step actually looks at, beyond the loss.
  // Leaving it empty is the point: a step reads its loss and nothing else,
  // and every gradient stays on the device where the optimizer consumes it.
  // Naming a gradient here is what a gate does, and it costs a copy.
  // `initial` is taken BY VALUE so a caller with a large one can move it in.
  // At Krea's scale the seed map is eleven gigabytes; copying it here and
  // again in the caller is how a run that fits on paper dies to the host
  // memory limit before its first step.
  TrainingSession(TrainingPlan plan, runtime::Executor &executor,
                  runtime::TensorMap initial, runtime::RunOptions options,
                  std::vector<std::uint32_t> reads = {});

  // Runs one step. `batch` carries only what genuinely changes.
  TrainingStepResult step(const runtime::TensorMap &batch);

  // Device to host, on request only.
  Checkpoint capture() const;
  // Host to device, on request only. Refuses a checkpoint built for a
  // different program.
  void restore(const Checkpoint &checkpoint);

  // A machine-readable record of the step just run. The session fills in
  // everything it owns -- identity, plan fingerprint, what the runtime
  // submitted, what it holds resident -- and leaves the rest for the driver,
  // which is the only thing that knows the model's name or where its
  // checkpoint went. Phase times appear only if the run was traced.
  TrainingStepReport report(const TrainingStepResult &result) const;

  // The plan's static memory analysis, computed once. Every step reports the
  // same numbers because the plan does not change between them.
  const TrainingMemoryReport &memory() const { return memory_; }

  std::uint64_t completed_steps() const { return completed_steps_; }
  const TrainingPlan &plan() const { return plan_; }
  std::uint64_t persistent_state_bytes() const { return state_bytes_; }

private:
  TrainingPlan plan_;
  runtime::RunOptions options_;
  std::unique_ptr<runtime::PreparedExecution> prepared_;
  TrainingMemoryReport memory_;
  // What the runtime said it ran on, taken from the first step rather than
  // guessed at by a caller.
  std::string device_name_;
  std::uint64_t completed_steps_{};
  std::uint64_t state_bytes_{};
};

} // namespace dif::training
