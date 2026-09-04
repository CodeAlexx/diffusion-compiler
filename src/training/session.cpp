#include "dif/training/session.hpp"

#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <chrono>
#include <cmath>
#include <cstring>

namespace dif::training {

std::vector<runtime::PersistentStateBinding>
TrainingPlan::persistent_state() const {
  std::vector<runtime::PersistentStateBinding> state;
  state.reserve(bindings.size() * 4U);
  for (const auto &binding : bindings) {
    // A parameter trained through a master advances the MASTER; the
    // half-precision copy is recomputed from it every step and therefore
    // carries nothing.
    if (binding.master_input != 0U)
      state.push_back({binding.master_input, binding.master_output});
    else
      state.push_back({binding.parameter_input, binding.parameter_output});
    state.push_back({binding.first_moment_input, binding.first_moment_output});
    state.push_back(
        {binding.second_moment_input, binding.second_moment_output});
  }
  return state;
}

std::vector<std::uint32_t> TrainingPlan::checkpoint_tensors() const {
  std::vector<std::uint32_t> tensors;
  for (const auto &binding : persistent_state())
    tensors.push_back(binding.input);
  return tensors;
}

TrainingPlan plan_from_composed(ir::Program program, std::uint32_t step_input,
                                std::uint32_t loss_tensor,
                                std::vector<ParameterBinding> bindings) {
  TrainingPlan plan;
  plan.program = std::move(program);
  plan.step_input = step_input;
  plan.loss_tensor = loss_tensor;
  plan.bindings = std::move(bindings);
  return plan;
}

TrainingPlan compile(const ir::Program &forward, std::uint32_t loss_tensor,
                     std::span<const std::uint32_t> parameters,
                     const OptimizerHyperparameters &hyperparameters,
                     const std::function<double(std::size_t, std::uint32_t)>
                         &decay_for) {
  auto step = build_training_step(forward, loss_tensor, parameters,
                                  hyperparameters, decay_for);
  TrainingPlan plan;
  plan.program = std::move(step.program);
  plan.step_input = step.step_input;
  plan.loss_tensor = loss_tensor;
  plan.bindings = std::move(step.bindings);
  plan.forward_operations = step.forward_operations;
  plan.optimizer_operations = step.optimizer_operations;
  return plan;
}

namespace {

runtime::Tensor step_counter(std::uint64_t value) {
  runtime::Tensor tensor{ir::DType::I32, {1U}, {}};
  tensor.bytes.resize(sizeof(std::int32_t));
  const auto narrowed = static_cast<std::int32_t>(value);
  std::memcpy(tensor.bytes.data(), &narrowed, sizeof(narrowed));
  tensor.validate();
  return tensor;
}

} // namespace

TrainingSession::TrainingSession(TrainingPlan plan,
                                 runtime::Executor &executor,
                                 const runtime::TensorMap &initial,
                                 runtime::RunOptions options)
    : plan_(std::move(plan)), options_(std::move(options)) {
  // The declaration comes from the plan, so it cannot disagree with the
  // program it describes.
  options_.persistent_state = plan_.persistent_state();
  auto seed = initial;
  // The step counter is an ordinary input the session advances; a caller
  // never sets it.
  seed.insert_or_assign(plan_.step_input, step_counter(0U));
  prepared_ = executor.prepare(plan_.program, seed, options_);
  for (const auto &binding : options_.persistent_state) {
    const auto *desc = plan_.program.tensor(binding.input);
    state_bytes_ += desc->byte_count();
  }
}

TrainingStepResult TrainingSession::step(const runtime::TensorMap &batch) {
  auto inputs = batch;
  // The counter means "steps already completed": the AdamW kernel takes
  // `completed_steps` and forms the one-based step itself. Passing the
  // post-increment value here would shift every bias correction by one.
  inputs.insert_or_assign(plan_.step_input, step_counter(completed_steps_));
  const auto started = std::chrono::steady_clock::now();
  auto result = prepared_->run(inputs, options_);
  const auto stopped = std::chrono::steady_clock::now();

  TrainingStepResult step_result;
  const auto loss = result.outputs.find(plan_.loss_tensor);
  if (loss == result.outputs.end())
    fail("the training program did not produce its loss");
  step_result.loss = runtime::load_float(loss->second, 0U);
  if (!std::isfinite(step_result.loss))
    fail("training produced a non-finite loss at step " +
         std::to_string(completed_steps_ + 1U));
  step_result.step_milliseconds =
      std::chrono::duration<double, std::milli>(stopped - started).count();
  step_result.persistent_state_host_to_device_bytes =
      result.persistent_state_host_to_device_bytes;
  step_result.persistent_state_device_to_host_bytes =
      result.persistent_state_device_to_host_bytes;
  step_result.outputs = std::move(result.outputs);
  ++completed_steps_;
  step_result.completed_steps = completed_steps_;
  return step_result;
}

Checkpoint TrainingSession::capture() const {
  Checkpoint checkpoint;
  checkpoint.program_fingerprint = plan_.fingerprint();
  checkpoint.completed_steps = completed_steps_;
  checkpoint.state = prepared_->capture_persistent_state();
  return checkpoint;
}

void TrainingSession::restore(const Checkpoint &checkpoint) {
  if (checkpoint.program_fingerprint != plan_.fingerprint())
    fail("resume checkpoint targets a different program fingerprint");
  prepared_->restore_persistent_state(checkpoint.state);
  completed_steps_ = checkpoint.completed_steps;
}

} // namespace dif::training
