#include "dif/training/memory.hpp"

#include "dif/training/session.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dif::training {

std::uint64_t RecomputePolicy::device_bytes() const {
  if (!(headroom > 0.0) || headroom > 1.0)
    fail("recompute headroom must be a fraction in (0,1]");
  const auto usable = budget.usable_device_memory_bytes != 0U
                          ? budget.usable_device_memory_bytes
                          : budget.free_device_memory_bytes;
  if (usable == 0U)
    fail("recompute policy has no device memory budget to plan against");
  return static_cast<std::uint64_t>(static_cast<double>(usable) * headroom);
}

TrainingMemoryReport analyze_memory(const TrainingPlan &plan) {
  TrainingMemoryReport report;
  const auto &program = plan.program;
  const auto forward_end = plan.forward_operations;

  // Which operation produced each tensor, and the last one that reads it.
  std::unordered_map<std::uint32_t, std::size_t> producer;
  std::unordered_map<std::uint32_t, std::size_t> last_consumer;
  for (std::size_t index = 0U; index < program.operations.size(); ++index) {
    for (const auto tensor : program.operations[index].outputs)
      producer.emplace(tensor, index);
    for (const auto tensor : program.operations[index].inputs)
      last_consumer[tensor] = index;
  }

  // The tensors the bindings name, so a parameter is not counted as a frozen
  // weight and a gradient is not counted as a backward temporary.
  std::unordered_set<std::uint32_t> trainable;
  std::unordered_set<std::uint32_t> gradients;
  for (const auto &binding : plan.bindings) {
    trainable.insert(binding.parameter_input);
    trainable.insert(binding.parameter_output);
    if (binding.master_input != 0U) {
      trainable.insert(binding.master_input);
      trainable.insert(binding.master_output);
    }
    gradients.insert(binding.gradient_output);
  }

  for (const auto &tensor : program.tensors) {
    const auto bytes = tensor.byte_count();
    if (tensor.has_role(ir::TensorRole::OptimizerState)) {
      report.optimizer_state += bytes;
      continue;
    }
    if (trainable.contains(tensor.id)) {
      report.trainable_weights += bytes;
      continue;
    }
    if (gradients.contains(tensor.id)) {
      report.gradients += bytes;
      continue;
    }
    if (tensor.has_role(ir::TensorRole::Constant)) {
      report.frozen_weights += bytes;
      continue;
    }
    const auto produced = producer.find(tensor.id);
    if (produced == producer.end()) {
      // An input the caller supplies each step: the batch, the step counter.
      report.other += bytes;
      continue;
    }
    const auto consumed = last_consumer.find(tensor.id);
    const auto last = consumed == last_consumer.end()
                          ? produced->second
                          : consumed->second;
    if (produced->second >= forward_end) {
      report.backward_temporaries += bytes;
    } else if (last >= forward_end) {
      // Produced by the forward pass, still wanted after it: the bytes a
      // training step pays that an inference pass does not.
      report.saved_activations += bytes;
    } else {
      report.transient_activations += bytes;
    }
  }

  const auto memory = compiler::plan_memory(program);
  report.planned_bytes = memory.total_bytes;

  // The floor: the largest total of bytes simultaneously live. Protected
  // tensors are live throughout, so they are added rather than swept.
  std::vector<std::int64_t> delta(program.operations.size() + 2U, 0);
  std::uint64_t protected_bytes = 0U;
  for (const auto &assignment : memory.assignments) {
    const auto *tensor = program.tensor(assignment.tensor_id);
    if (tensor->has_role(ir::TensorRole::Input) ||
        tensor->has_role(ir::TensorRole::Constant) ||
        tensor->has_role(ir::TensorRole::Output)) {
      protected_bytes += assignment.bytes;
      continue;
    }
    delta[assignment.first_operation] +=
        static_cast<std::int64_t>(assignment.bytes);
    delta[assignment.last_operation + 1U] -=
        static_cast<std::int64_t>(assignment.bytes);
  }
  std::int64_t live = 0, peak = 0;
  for (const auto step : delta) {
    live += step;
    peak = std::max(peak, live);
  }
  report.live_peak_bytes = protected_bytes + static_cast<std::uint64_t>(peak);
  return report;
}

RecomputeChoice choose_recompute(const TrainingPlan &plan,
                                 const RecomputePolicy &policy,
                                 ir::Program *chosen) {
  return choose_recompute(plan.program, plan.forward_operations,
                          policy.device_bytes(), chosen);
}

} // namespace dif::training
