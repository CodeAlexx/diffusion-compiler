// Persistent state: a device value that survives from one execution to the
// next.
//
// The claim is narrow and checkable: running a program with persistent state
// declared must produce EXACTLY what running it with the value carried by hand
// through host memory produces -- same losses, same parameters, same moments,
// byte for byte -- while moving zero bytes across the host boundary per step.
//
// Nothing here mentions an optimizer. The program happens to contain one
// because that is the dataflow the mechanism was built for, but the runtime is
// only ever told "this output becomes that input".

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/accumulate.hpp"
#include "dif/training/step.hpp"

#include <cstring>
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
using dif::runtime::PersistentStateBinding;
using dif::runtime::TensorMap;

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                std::uint64_t seed) {
  std::uint64_t count = 1U;
  for (const auto dim : dims)
    count *= dim;
  dif::runtime::Tensor tensor{DType::F32, std::move(dims), {}};
  tensor.bytes.resize(static_cast<std::size_t>(count) * sizeof(float));
  tensor.validate();
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (std::uint64_t index = 0U; index < count; ++index) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto unit = static_cast<double>((state >> 33U) & 0x7fffffffU) /
                      static_cast<double>(0x7fffffffU);
    dif::runtime::store_float(tensor, index, static_cast<float>(unit * 2.0 - 1.0));
  }
  return tensor;
}

dif::runtime::Tensor i32_scalar(std::int32_t value) {
  dif::runtime::Tensor tensor{DType::I32, {1U}, {}};
  tensor.bytes.resize(sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), &value, sizeof(value));
  tensor.validate();
  return tensor;
}

struct Fixture {
  dif::training::TrainingStep step;
  TensorMap initial;
  std::uint32_t data{};
  std::uint32_t target{};
  std::uint32_t loss{};
  std::vector<PersistentStateBinding> state;
};

// A two-layer network with a loss and an AdamW update: the smallest thing
// whose state actually has to survive between runs.
Fixture make_fixture() {
  Fixture fixture;
  const std::uint64_t rows = 4U;
  const std::uint64_t width = 6U;
  Program forward;
  forward.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, width}},
      {2U, DType::F32, TensorRole::Input | TensorRole::Parameter, {width, width}},
      {3U, DType::F32, TensorRole::Input | TensorRole::Parameter, {width}},
      {4U, DType::F32, TensorRole::Internal, {rows, width}},
      {5U, DType::F32, TensorRole::Internal, {rows, width}},
      {6U, DType::F32, TensorRole::Input, {rows, width}},
      {7U, DType::F32, TensorRole::Output, {1U}}};
  forward.operations = {{1U, Opcode::Linear, {1U, 2U, 3U}, {4U}, {}},
                        {2U, Opcode::SiLU, {4U}, {5U}, {}},
                        {3U, Opcode::MseLoss, {5U, 6U}, {7U}, {}}};
  dif::ir::verify(forward);

  dif::training::OptimizerHyperparameters hyperparameters;
  hyperparameters.learning_rate = 1.0e-2;
  const std::vector<std::uint32_t> parameters{2U, 3U};
  fixture.step = dif::training::build_training_step(forward, 7U, parameters,
                                                    hyperparameters);
  fixture.data = 1U;
  fixture.target = 6U;
  fixture.loss = 7U;

  fixture.initial.emplace(1U, f32_tensor({rows, width}, 11U));
  fixture.initial.emplace(2U, f32_tensor({width, width}, 13U));
  fixture.initial.emplace(3U, f32_tensor({width}, 17U));
  fixture.initial.emplace(6U, f32_tensor({rows, width}, 19U));
  fixture.initial.emplace(fixture.step.step_input, i32_scalar(1));
  for (const auto &binding : fixture.step.bindings) {
    const auto *first = fixture.step.program.tensor(binding.first_moment_input);
    const auto *second =
        fixture.step.program.tensor(binding.second_moment_input);
    auto zero = [](const dif::ir::TensorDesc &desc) {
      dif::runtime::Tensor tensor{desc.dtype, desc.dims, {}};
      tensor.bytes.assign(static_cast<std::size_t>(desc.byte_count()), 0);
      tensor.validate();
      return tensor;
    };
    fixture.initial.emplace(binding.first_moment_input, zero(*first));
    fixture.initial.emplace(binding.second_moment_input, zero(*second));
    // The declaration the runtime is given: three pairs per parameter, each
    // saying "the value written here is what is read there next time".
    fixture.state.push_back({binding.parameter_input, binding.parameter_output});
    fixture.state.push_back(
        {binding.first_moment_input, binding.first_moment_output});
    fixture.state.push_back(
        {binding.second_moment_input, binding.second_moment_output});
  }
  return fixture;
}

dif::runtime::RunOptions base_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return options;
}

struct Trace {
  std::vector<float> losses;
  TensorMap final_state;
  std::uint64_t host_to_device{};
  std::uint64_t device_to_host{};
};

// The way training had to be written before persistent state existed: every
// parameter and moment comes back to the host and goes out again.
Trace run_by_hand(const Fixture &fixture, dif::runtime::Executor &executor,
                  std::uint64_t steps) {
  Trace trace;
  auto inputs = fixture.initial;
  auto options = base_options();
  auto prepared = executor.prepare(fixture.step.program, inputs, options);
  for (std::uint64_t step = 0U; step < steps; ++step) {
    inputs.insert_or_assign(fixture.step.step_input,
                            i32_scalar(static_cast<std::int32_t>(step + 1U)));
    auto result = prepared->run(inputs, options);
    trace.losses.push_back(
        dif::runtime::load_float(result.outputs.at(fixture.loss), 0U));
    for (const auto &binding : fixture.step.bindings) {
      inputs.insert_or_assign(binding.parameter_input,
                              result.outputs.at(binding.parameter_output));
      inputs.insert_or_assign(binding.first_moment_input,
                              result.outputs.at(binding.first_moment_output));
      inputs.insert_or_assign(binding.second_moment_input,
                              result.outputs.at(binding.second_moment_output));
    }
  }
  for (const auto &binding : fixture.state)
    trace.final_state.emplace(binding.input, inputs.at(binding.input));
  return trace;
}

// The same training, with the runtime keeping the state.
Trace run_persistent(const Fixture &fixture, dif::runtime::Executor &executor,
                     std::uint64_t steps,
                     dif::runtime::PreparedExecution **keep = nullptr,
                     std::unique_ptr<dif::runtime::PreparedExecution> *own =
                         nullptr) {
  Trace trace;
  auto options = base_options();
  options.persistent_state = fixture.state;
  auto prepared =
      executor.prepare(fixture.step.program, fixture.initial, options);
  // Only the things that genuinely change each step are handed over.
  TensorMap inputs;
  inputs.emplace(fixture.data, fixture.initial.at(fixture.data));
  inputs.emplace(fixture.target, fixture.initial.at(fixture.target));
  for (std::uint64_t step = 0U; step < steps; ++step) {
    inputs.insert_or_assign(fixture.step.step_input,
                            i32_scalar(static_cast<std::int32_t>(step + 1U)));
    auto result = prepared->run(inputs, options);
    trace.losses.push_back(
        dif::runtime::load_float(result.outputs.at(fixture.loss), 0U));
    trace.host_to_device += result.persistent_state_host_to_device_bytes;
    trace.device_to_host += result.persistent_state_device_to_host_bytes;
  }
  trace.final_state = prepared->capture_persistent_state();
  if (keep && own) {
    *own = std::move(prepared);
    *keep = own->get();
  }
  return trace;
}

bool identical(const TensorMap &left, const TensorMap &right) {
  if (left.size() != right.size())
    return false;
  for (const auto &[id, tensor] : left) {
    const auto found = right.find(id);
    if (found == right.end() || found->second.bytes != tensor.bytes)
      return false;
  }
  return true;
}

void matches_the_hand_written_carry(dif::runtime::Executor &executor,
                                    const std::string &backend) {
  const auto fixture = make_fixture();
  const std::uint64_t steps = 25U;
  const auto by_hand = run_by_hand(fixture, executor, steps);
  const auto persistent = run_persistent(fixture, executor, steps);

  expect(by_hand.losses.size() == steps && persistent.losses.size() == steps,
         backend + ": both paths ran every step");
  bool losses_identical = by_hand.losses.size() == persistent.losses.size();
  for (std::size_t index = 0U;
       losses_identical && index < by_hand.losses.size(); ++index)
    losses_identical = std::memcmp(&by_hand.losses[index],
                                   &persistent.losses[index],
                                   sizeof(float)) == 0;
  expect(losses_identical,
         backend + ": every loss is bit-identical to the hand-carried run");
  expect(identical(by_hand.final_state, persistent.final_state),
         backend + ": final parameters and both moments are bit-identical");
  // The property the mechanism exists for.
  expect(persistent.host_to_device == 0U,
         backend + ": zero persistent-state bytes to the device across " +
             std::to_string(steps) + " steps, saw " +
             std::to_string(persistent.host_to_device));
  expect(persistent.device_to_host == 0U,
         backend + ": zero persistent-state bytes from the device across " +
             std::to_string(steps) + " steps, saw " +
             std::to_string(persistent.device_to_host));
  std::cout << "  " << backend << ": " << steps
            << " steps, state traffic " << persistent.host_to_device << " up / "
            << persistent.device_to_host << " down\n";
}

void capture_and_restore_resume_exactly(dif::runtime::Executor &executor,
                                        const std::string &backend) {
  const auto fixture = make_fixture();
  const std::uint64_t first = 10U;
  const std::uint64_t second = 7U;

  const auto uninterrupted = run_persistent(fixture, executor, first + second);

  // Run the first leg, capture, then restore into a fresh session and finish.
  auto options = base_options();
  options.persistent_state = fixture.state;
  auto prepared =
      executor.prepare(fixture.step.program, fixture.initial, options);
  TensorMap inputs;
  inputs.emplace(fixture.data, fixture.initial.at(fixture.data));
  inputs.emplace(fixture.target, fixture.initial.at(fixture.target));
  std::vector<float> losses;
  for (std::uint64_t step = 0U; step < first; ++step) {
    inputs.insert_or_assign(fixture.step.step_input,
                            i32_scalar(static_cast<std::int32_t>(step + 1U)));
    auto result = prepared->run(inputs, options);
    losses.push_back(
        dif::runtime::load_float(result.outputs.at(fixture.loss), 0U));
  }
  const auto snapshot = prepared->capture_persistent_state();

  auto resumed =
      executor.prepare(fixture.step.program, fixture.initial, options);
  resumed->restore_persistent_state(snapshot);
  for (std::uint64_t step = 0U; step < second; ++step) {
    inputs.insert_or_assign(
        fixture.step.step_input,
        i32_scalar(static_cast<std::int32_t>(first + step + 1U)));
    auto result = resumed->run(inputs, options);
    losses.push_back(
        dif::runtime::load_float(result.outputs.at(fixture.loss), 0U));
  }

  bool losses_identical = losses.size() == uninterrupted.losses.size();
  for (std::size_t index = 0U;
       losses_identical && index < losses.size(); ++index)
    losses_identical = std::memcmp(&losses[index], &uninterrupted.losses[index],
                                   sizeof(float)) == 0;
  expect(losses_identical,
         backend + ": capture and restore reproduce the uninterrupted losses");
  expect(identical(resumed->capture_persistent_state(),
                   uninterrupted.final_state),
         backend + ": capture and restore reproduce the final state");
}

void a_caller_cannot_override_the_state(dif::runtime::Executor &executor,
                                        const std::string &backend) {
  const auto fixture = make_fixture();
  auto options = base_options();
  options.persistent_state = fixture.state;
  auto prepared =
      executor.prepare(fixture.step.program, fixture.initial, options);
  TensorMap inputs;
  inputs.emplace(fixture.data, fixture.initial.at(fixture.data));
  inputs.emplace(fixture.target, fixture.initial.at(fixture.target));
  inputs.emplace(fixture.step.step_input, i32_scalar(1));
  const auto first = prepared->run(inputs, options);
  const auto after_one =
      dif::runtime::load_float(first.outputs.at(fixture.loss), 0U);

  // Hand back the ORIGINAL parameters as if resuming from step zero. The
  // session owns the value; a stale host copy must not reach the program.
  for (const auto &binding : fixture.state)
    inputs.insert_or_assign(binding.input,
                            fixture.initial.count(binding.input)
                                ? fixture.initial.at(binding.input)
                                : dif::runtime::Tensor{});
  inputs.insert_or_assign(fixture.step.step_input, i32_scalar(2));
  const auto second = prepared->run(inputs, options);
  const auto after_two =
      dif::runtime::load_float(second.outputs.at(fixture.loss), 0U);
  expect(after_two != after_one,
         backend + ": a stale host copy does not resurrect an earlier state");
}

void bad_declarations_are_refused() {
  const auto fixture = make_fixture();
  const auto &program = fixture.step.program;
  const auto refused = [&](std::vector<PersistentStateBinding> state,
                           const std::string &why) {
    bool threw = false;
    try {
      dif::runtime::validate_persistent_state(program, state);
    } catch (const dif::Error &) {
      threw = true;
    }
    expect(threw, "refused: " + why);
  };
  const auto &first = fixture.step.bindings.front();
  refused({{999999U, first.parameter_output}}, "an unknown tensor");
  refused({{first.parameter_output, first.parameter_output}},
          "a source that is not a program input");
  refused({{first.parameter_input, first.parameter_input}},
          "a destination that is not a program output");
  // The two parameters have different shapes ([w,w] and [w]), so crossing
  // them is a real disagreement rather than a coincidentally matching one.
  refused({{first.parameter_input,
            fixture.step.bindings.back().parameter_output}},
          "a pair that disagrees on shape");
  refused({{first.parameter_input, first.parameter_output},
           {first.parameter_input, first.parameter_output}},
          "a tensor named twice");
  refused({{fixture.step.step_input, first.parameter_output}},
          "a pair that disagrees on dtype");
  // And the honest declaration is accepted.
  bool accepted = true;
  try {
    dif::runtime::validate_persistent_state(program, fixture.state);
  } catch (const dif::Error &) {
    accepted = false;
  }
  expect(accepted, "the honest declaration is accepted");
}

void the_declaration_is_fixed_at_prepare(dif::runtime::Executor &executor,
                                         const std::string &backend) {
  const auto fixture = make_fixture();
  auto options = base_options();
  options.persistent_state = fixture.state;
  auto prepared =
      executor.prepare(fixture.step.program, fixture.initial, options);
  auto changed = options;
  changed.persistent_state.pop_back();
  TensorMap inputs;
  inputs.emplace(fixture.data, fixture.initial.at(fixture.data));
  inputs.emplace(fixture.target, fixture.initial.at(fixture.target));
  inputs.emplace(fixture.step.step_input, i32_scalar(1));
  bool threw = false;
  try {
    prepared->run(inputs, changed);
  } catch (const dif::Error &) {
    threw = true;
  }
  expect(threw, backend + ": changing the declaration after prepare is refused");
}

// Gradient accumulation is a different dataflow -- a running sum rather than
// an optimizer transition -- and it must work through the same mechanism with
// no special case, because the runtime is only ever told "this output is that
// input next time".
void accumulators_carry_too(dif::runtime::Executor &executor,
                            const std::string &backend) {
  const std::uint64_t rows = 4U;
  const std::uint64_t width = 6U;
  Program forward;
  forward.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, width}},
      {2U, DType::F32, TensorRole::Input | TensorRole::Parameter, {width, width}},
      {3U, DType::F32, TensorRole::Internal, {rows, width}},
      {4U, DType::F32, TensorRole::Input, {rows, width}},
      {5U, DType::F32, TensorRole::Output, {1U}}};
  forward.operations = {{1U, Opcode::Linear, {1U, 2U}, {3U}, {}},
                        {2U, Opcode::MseLoss, {3U, 4U}, {5U}, {}}};
  dif::ir::verify(forward);
  const std::vector<std::uint32_t> parameters{2U};
  const std::uint64_t micro_batches = 4U;
  const auto step = dif::training::build_accumulating_step(
      forward, 5U, parameters, micro_batches, {});

  TensorMap seed;
  seed.emplace(1U, f32_tensor({rows, width}, 23U));
  seed.emplace(2U, f32_tensor({width, width}, 29U));
  seed.emplace(4U, f32_tensor({rows, width}, 31U));
  std::vector<PersistentStateBinding> state;
  for (const auto &binding : step.bindings) {
    const auto *desc = step.accumulate.tensor(binding.accumulator_input);
    dif::runtime::Tensor zero{desc->dtype, desc->dims, {}};
    zero.bytes.assign(static_cast<std::size_t>(desc->byte_count()), 0);
    zero.validate();
    seed.emplace(binding.accumulator_input, std::move(zero));
    state.push_back({binding.accumulator_input, binding.accumulator_output});
  }

  auto options = base_options();
  options.persistent_state = state;
  auto prepared = executor.prepare(step.accumulate, seed, options);
  TensorMap inputs;
  inputs.emplace(1U, seed.at(1U));
  inputs.emplace(2U, seed.at(2U));
  inputs.emplace(4U, seed.at(4U));
  std::uint64_t traffic = 0U;
  for (std::uint64_t micro = 0U; micro < micro_batches; ++micro) {
    const auto result = prepared->run(inputs, options);
    traffic += result.persistent_state_host_to_device_bytes +
               result.persistent_state_device_to_host_bytes;
  }
  const auto carried = prepared->capture_persistent_state();

  // The same four micro-batches, summed by hand.
  auto by_hand = executor.prepare(step.accumulate, seed, base_options());
  auto hand_inputs = seed;
  TensorMap running;
  for (std::uint64_t micro = 0U; micro < micro_batches; ++micro) {
    const auto result = by_hand->run(hand_inputs, base_options());
    for (const auto &binding : step.bindings) {
      hand_inputs.insert_or_assign(binding.accumulator_input,
                                   result.outputs.at(binding.accumulator_output));
      running.insert_or_assign(binding.accumulator_input,
                               result.outputs.at(binding.accumulator_output));
    }
  }
  expect(identical(carried, running),
         backend + ": accumulators carry bit-identically over " +
             std::to_string(micro_batches) + " micro-batches");
  expect(traffic == 0U,
         backend + ": accumulation moved zero persistent-state bytes");
}

void exercise(dif::runtime::Executor &executor, const std::string &backend) {
  accumulators_carry_too(executor, backend);
  matches_the_hand_written_carry(executor, backend);
  capture_and_restore_resume_exactly(executor, backend);
  a_caller_cannot_override_the_state(executor, backend);
  the_declaration_is_fixed_at_prepare(executor, backend);
}

} // namespace

int main() {
  bad_declarations_are_refused();
  auto cpu = dif::runtime::make_cpu_executor();
  exercise(*cpu, "cpu");
  if (dif::runtime::cuda_available()) {
    auto cuda = dif::runtime::make_cuda_executor();
    exercise(*cuda, "cuda");
  } else {
    std::cout << "CUDA unavailable; persistent state CUDA gates skipped\n";
  }
  if (failures != 0) {
    std::cerr << failures << " persistent state failure(s)\n";
    return 1;
  }
  std::cout << "persistent state tests passed\n";
  return 0;
}
