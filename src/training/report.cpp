#include "dif/training/report.hpp"

#include "dif/ir/ir.hpp"

#include <unordered_map>

namespace dif::training {

std::optional<PhaseTimes>
attribute_phases(const std::vector<runtime::TraceEvent> &events,
                 const ir::Program &program, std::size_t forward_operations,
                 std::size_t optimizer_operations) {
  if (events.empty())
    return std::nullopt;
  // A plan that does not know where its phases begin cannot have them
  // attributed. Splitting on boundaries of zero would put every operation in
  // the last phase and report it with total confidence, which is worse than
  // reporting nothing: a wrong number gets acted on.
  if (forward_operations == 0U || optimizer_operations <= forward_operations ||
      optimizer_operations > program.operations.size())
    return std::nullopt;
  // Where each operation sits in the program, so an event can be placed in a
  // phase by the operation it belongs to rather than by its name.
  std::unordered_map<std::uint32_t, std::size_t> position;
  for (std::size_t index = 0U; index < program.operations.size(); ++index)
    position.emplace(program.operations[index].id, index);

  PhaseTimes times;
  for (const auto &event : events) {
    const auto elapsed = event.host_end_ms - event.host_start_ms;
    if (elapsed <= 0.0)
      continue;
    if (event.operation_id == 0U) {
      // Input uploads and output readbacks belong to no operation. They are
      // the transfer cost, and they are exactly what a step should be trying
      // to drive to zero.
      times.transfer_milliseconds += elapsed;
      continue;
    }
    const auto found = position.find(event.operation_id);
    if (found == position.end())
      continue;
    if (found->second < forward_operations)
      times.forward_milliseconds += elapsed;
    else if (found->second < optimizer_operations)
      times.backward_milliseconds += elapsed;
    else
      times.optimizer_milliseconds += elapsed;
  }
  return times;
}

telemetry::Value TrainingStepReport::document() const {
  telemetry::Object root;
  // An identity nobody supplied is left out rather than written as "". The
  // rule is the same one the phase times follow: an empty field a reader has
  // to interpret is worse than a field that is not there.
  const auto identity = [&root](const char *key, const std::string &value) {
    if (!value.empty())
      root.set(key, value);
  };
  identity("model", model);
  identity("backend", backend);
  identity("device", device);
  identity("target", target_fingerprint);
  identity("runtime_budget", runtime_budget_class);
  identity("plan_fingerprint", plan_fingerprint);
  identity("checkpoint", checkpoint);
  root.set("trainable_tensors", trainable_tensors);
  root.set("trainable_parameters", trainable_parameters);
  if (!parameter_policy.empty())
    root.set("parameter_policy", parameter_policy);
  if (!physical_formats.empty())
    root.set("physical_formats", physical_formats);
  root.set("completed_steps", completed_steps);
  root.set("accumulation_index", accumulation_index);
  root.set("optimizer_applied", optimizer_applied);
  root.set("loss", loss);
  if (gradient_norm)
    root.set("grad_norm", *gradient_norm);
  root.set("nonfinites", nonfinite_count);
  root.set("step_milliseconds", step_milliseconds);

  // Only the phases something actually measured. A missing key says "not
  // measured"; a zero would say "took no time", which is a different claim.
  telemetry::Object phases;
  if (forward_milliseconds)
    phases.set("forward_milliseconds", *forward_milliseconds);
  if (backward_milliseconds)
    phases.set("backward_milliseconds", *backward_milliseconds);
  if (optimizer_milliseconds)
    phases.set("optimizer_milliseconds", *optimizer_milliseconds);
  if (recompute_milliseconds)
    phases.set("recompute_milliseconds", *recompute_milliseconds);
  phases.set("transfer_milliseconds", transfer_milliseconds);
  root.set("phases", telemetry::Value(std::move(phases)));

  telemetry::Object submitted;
  submitted.set("kernel_launches", telemetry.kernel_launches);
  submitted.set("gemms",
                telemetry.cublaslt_matmuls + telemetry.cublas_gemms);
  submitted.set("attention_dispatches",
                telemetry.cudnn_attention_dispatches +
                    telemetry.ck_attention_dispatches);
  submitted.set("convolution_dispatches",
                telemetry.cudnn_convolution_dispatches);
  submitted.set("h2d_copies", telemetry.h2d_copies);
  submitted.set("h2d_bytes", telemetry.h2d_bytes);
  submitted.set("d2h_copies", telemetry.d2h_copies);
  submitted.set("d2h_bytes", telemetry.d2h_bytes);
  submitted.set("host_synchronizations", telemetry.host_stream_synchronizes);
  submitted.set("device_allocations", telemetry.device_mem_allocs);
  root.set("submitted", telemetry::Value(std::move(submitted)));

  telemetry::Object memory;
  memory.set("persistent_state_bytes", persistent_state_bytes);
  memory.set("planned_device_bytes", planned_device_bytes);
  memory.set("peak_device_bytes", peak_device_bytes);
  root.set("memory", telemetry::Value(std::move(memory)));

  return telemetry::Value(std::move(root));
}

std::string TrainingStepReport::json() const {
  return telemetry::serialize_compact(document());
}

} // namespace dif::training
