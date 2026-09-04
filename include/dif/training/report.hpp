#pragma once

#include "dif/runtime/executor.hpp"
#include "dif/target/profile.hpp"
#include "dif/telemetry/document.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace dif::training {

// One training step, described so a machine can read it.
//
// The brief asks for a JSON record per step. The Mojo trainer already emits
// one and its shape is worth keeping, with one difference: its `phases` block
// reports every field as 0.0 except save_seconds, so it names the split
// without measuring it. Here a phase time is an OPTIONAL -- absent when
// nothing measured it, present when something did -- because a zero that
// means "not measured" is worse than no field at all.
//
// The step boundary is the one the brief defines. This record covers the
// MODEL STEP only: a prepared batch through forward, loss, backward,
// accumulation and the optimizer when it is due. Data preparation and
// sampling are the driver's to time, and mixing them in is how a step time
// stops meaning anything.
struct TrainingStepReport {
  // Identity: enough to know what was run without reading a log.
  std::string model;
  std::string backend;
  std::string device;
  std::string target_fingerprint;
  std::string runtime_budget_class;
  std::string plan_fingerprint;
  std::string checkpoint;

  // What is being trained.
  std::uint64_t trainable_tensors{};
  std::uint64_t trainable_parameters{};
  std::string parameter_policy;
  std::string physical_formats;

  // Where the step sits.
  std::uint64_t completed_steps{};
  std::uint64_t accumulation_index{};
  bool optimizer_applied{};

  // What it produced.
  double loss{};
  std::optional<double> gradient_norm;
  std::uint64_t nonfinite_count{};

  // What it cost. Phase times are present only when a profiling run
  // attributed them.
  double step_milliseconds{};
  std::optional<double> forward_milliseconds;
  std::optional<double> backward_milliseconds;
  std::optional<double> optimizer_milliseconds;
  std::optional<double> recompute_milliseconds;
  double transfer_milliseconds{};

  // What the runtime submitted, and what it holds.
  runtime::LaunchTelemetry telemetry;
  std::uint64_t persistent_state_bytes{};
  std::uint64_t planned_device_bytes{};
  std::uint64_t peak_device_bytes{};

  telemetry::Value document() const;
  std::string json() const;
};

// Attributes the wall time of a traced run to the three phases a training
// plan has, by the operation each trace event belongs to. Returns nothing
// when the run carried no trace events, rather than reporting zeros.
struct PhaseTimes {
  double forward_milliseconds{};
  double backward_milliseconds{};
  double optimizer_milliseconds{};
  double transfer_milliseconds{};
};
std::optional<PhaseTimes>
attribute_phases(const std::vector<runtime::TraceEvent> &events,
                 const ir::Program &program, std::size_t forward_operations,
                 std::size_t optimizer_operations);

} // namespace dif::training
