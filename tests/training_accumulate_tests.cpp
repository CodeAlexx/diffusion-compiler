// Gradient accumulation has to be arithmetic, not approximation.
//
// The claim the pass makes is that N micro-batches of size B train the same
// model as one batch of N*B. So that is what is checked: build both, run
// both, and compare the parameters and moments they land on. The two differ
// only in summation order, so they agree to F32 rounding rather than
// bit-identically, and the tolerance says so explicitly.

#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/accumulate.hpp"

#include <cmath>
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

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

const std::uint64_t kWidth = 8U;

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

struct Model {
  Program forward;
  std::uint32_t loss{};
  std::uint32_t input{};
  std::uint32_t target{};
  std::vector<std::uint32_t> parameters;
};

// Two linear layers with a nonlinearity, over `rows` batch rows.
Model model(std::uint64_t rows) {
  Model built;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto tensor = [&](std::uint32_t roles, std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    built.forward.tensors.push_back({id, DType::F32, roles, std::move(dims)});
    return id;
  };
  built.input = tensor(TensorRole::Input, {rows, kWidth});
  const auto weight1 =
      tensor(TensorRole::Input | TensorRole::Parameter, {kWidth, kWidth});
  const auto bias1 = tensor(TensorRole::Input | TensorRole::Parameter, {kWidth});
  const auto weight2 =
      tensor(TensorRole::Input | TensorRole::Parameter, {kWidth, kWidth});
  built.parameters = {weight1, bias1, weight2};
  const auto hidden = tensor(TensorRole::Internal, {rows, kWidth});
  built.forward.operations.push_back({next_operation++, Opcode::Linear,
                                      {built.input, weight1, bias1},
                                      {hidden},
                                      {}});
  const auto activated = tensor(TensorRole::Internal, {rows, kWidth});
  built.forward.operations.push_back(
      {next_operation++, Opcode::SiLU, {hidden}, {activated}, {}});
  const auto output = tensor(TensorRole::Internal, {rows, kWidth});
  built.forward.operations.push_back({next_operation++, Opcode::Linear,
                                      {activated, weight2},
                                      {output},
                                      {}});
  built.target = tensor(TensorRole::Input, {rows, kWidth});
  built.loss = tensor(TensorRole::Output, {1U});
  built.forward.operations.push_back({next_operation++, Opcode::MseLoss,
                                      {output, built.target},
                                      {built.loss},
                                      {}});
  dif::ir::verify(built.forward);
  return built;
}

dif::runtime::TensorMap run(const Program &program,
                            dif::runtime::TensorMap state) {
  for (const auto &tensor : program.tensors) {
    if (state.contains(tensor.id) || !tensor.has_role(TensorRole::Input))
      continue;
    dif::runtime::Tensor value{tensor.dtype, tensor.dims, {}};
    value.bytes.assign(static_cast<std::size_t>(tensor.byte_count()), 0);
    if (tensor.dtype == DType::I32) {
      const std::int32_t one = 1;
      std::memcpy(value.bytes.data(), &one, sizeof(one));
    }
    value.validate();
    state.emplace(tensor.id, std::move(value));
  }
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return dif::runtime::make_cpu_executor()->run(program, state, options).outputs;
}

// Rows of the full batch, sliced for a micro-batch.
dif::runtime::Tensor slice_rows(const dif::runtime::Tensor &whole,
                                std::uint64_t first, std::uint64_t rows) {
  dif::runtime::Tensor part{whole.dtype, {rows, kWidth}, {}};
  part.bytes.resize(static_cast<std::size_t>(rows * kWidth) * sizeof(float));
  part.validate();
  for (std::uint64_t index = 0U; index < rows * kWidth; ++index)
    dif::runtime::store_float(
        part, index, dif::runtime::load_float(whole, first * kWidth + index));
  return part;
}

double worst_relative_difference(const dif::runtime::Tensor &a,
                                 const dif::runtime::Tensor &b) {
  double worst = 0.0;
  const auto count = a.element_count();
  for (std::uint64_t index = 0U; index < count; ++index) {
    const auto left = dif::runtime::load_float(a, index);
    const auto right = dif::runtime::load_float(b, index);
    const auto scale = std::max({std::abs(left), std::abs(right), 1.0e-6f});
    worst = std::max(worst, static_cast<double>(std::abs(left - right) / scale));
  }
  return worst;
}

void micro_batches_match_one_batch() {
  const std::uint64_t micro_batches = 4U;
  const std::uint64_t micro_rows = 3U;
  const std::uint64_t rows = micro_batches * micro_rows;

  dif::training::OptimizerHyperparameters hyperparameters;
  hyperparameters.learning_rate = 1.0e-2;
  hyperparameters.weight_decay = 0.01;

  // The same data and the same initial parameters, seen two ways.
  const auto whole_input = f32_tensor({rows, kWidth}, 5U);
  const auto whole_target = f32_tensor({rows, kWidth}, 13U);
  std::vector<dif::runtime::Tensor> initial;
  for (std::uint64_t index = 0U; index < 3U; ++index)
    initial.push_back(index == 1U ? f32_tensor({kWidth}, 23U)
                                  : f32_tensor({kWidth, kWidth}, 17U + index));

  // One batch of twelve.
  const auto full = model(rows);
  const auto full_step = dif::training::build_accumulating_step(
      full.forward, full.loss, full.parameters, 1U, hyperparameters);
  dif::runtime::TensorMap full_state;
  full_state.emplace(full.input, whole_input);
  full_state.emplace(full.target, whole_target);
  for (std::uint64_t index = 0U; index < full.parameters.size(); ++index)
    full_state.emplace(full.parameters[index], initial[index]);
  auto accumulated = run(full_step.accumulate, full_state);
  dif::runtime::TensorMap full_update;
  for (std::uint64_t index = 0U; index < full.parameters.size(); ++index)
    full_update.emplace(full.parameters[index], initial[index]);
  for (const auto &binding : full_step.bindings)
    full_update.emplace(binding.accumulator_output,
                        accumulated.at(binding.accumulator_output));
  const auto full_result = run(full_step.update, full_update);

  // Four micro-batches of three.
  const auto small = model(micro_rows);
  const auto small_step = dif::training::build_accumulating_step(
      small.forward, small.loss, small.parameters, micro_batches,
      hyperparameters);
  expect(small_step.micro_batches == micro_batches,
         "the step remembers how many micro-batches it averages over");

  dif::runtime::TensorMap carry;
  for (std::uint64_t index = 0U; index < small.parameters.size(); ++index)
    carry.emplace(small.parameters[index], initial[index]);
  for (std::uint64_t batch = 0U; batch < micro_batches; ++batch) {
    auto state = carry;
    state.insert_or_assign(
        small.input, slice_rows(whole_input, batch * micro_rows, micro_rows));
    state.insert_or_assign(
        small.target, slice_rows(whole_target, batch * micro_rows, micro_rows));
    const auto produced = run(small_step.accumulate, state);
    // Feed each micro-batch's running sum into the next.
    for (const auto &binding : small_step.bindings)
      carry.insert_or_assign(binding.accumulator_input,
                             produced.at(binding.accumulator_output));
  }
  dif::runtime::TensorMap small_update;
  for (std::uint64_t index = 0U; index < small.parameters.size(); ++index)
    small_update.emplace(small.parameters[index], initial[index]);
  for (const auto &binding : small_step.bindings)
    small_update.emplace(binding.accumulator_output,
                         carry.at(binding.accumulator_input));
  const auto small_result = run(small_step.update, small_update);

  // The two paths differ only in the order the row contributions are summed,
  // so they agree to F32 rounding. The bar is set where a real defect -- a
  // missing average, a dropped micro-batch, a wrong slice -- lands orders of
  // magnitude above it.
  const double bar = 1.0e-5;
  double worst = 0.0;
  for (std::uint64_t index = 0U; index < full.parameters.size(); ++index) {
    const auto &a = full_step.bindings[index];
    const auto &b = small_step.bindings[index];
    // The accumulated gradient is an output of the micro-batch, so it is
    // compared where it is produced, not where the optimizer reads it.
    {
      const auto left = accumulated.find(a.accumulator_output);
      const auto right = carry.find(b.accumulator_input);
      if (left == accumulated.end() || right == carry.end())
        expect(false, "both paths produced an accumulated gradient");
      else
        worst = std::max(worst,
                         worst_relative_difference(left->second, right->second));
    }
    for (const auto pair :
         {std::pair{a.update.parameter_output, b.update.parameter_output},
          std::pair{a.update.first_moment_output, b.update.first_moment_output},
          std::pair{a.update.second_moment_output,
                    b.update.second_moment_output}}) {
      const auto left = full_result.find(pair.first);
      const auto right = small_result.find(pair.second);
      if (left == full_result.end() || right == small_result.end()) {
        expect(false, "both paths produced the same set of outputs");
        continue;
      }
      worst = std::max(worst,
                       worst_relative_difference(left->second, right->second));
    }
  }
  std::cout << "  four micro-batches vs one batch: worst relative difference "
            << worst << " (bar " << bar << ")\n";
  expect(worst < bar,
         "four micro-batches of three train the same as one batch of twelve");
}

void the_accumulators_outlive_the_micro_batch() {
  const auto built = model(4U);
  const auto step = dif::training::build_accumulating_step(
      built.forward, built.loss, built.parameters, 8U, {});
  for (const auto &binding : step.bindings) {
    expect(binding.accumulator_input != 0U && binding.accumulator_output != 0U,
           "every parameter gets an accumulator");
    const auto *accumulator = step.accumulate.tensor(binding.accumulator_input);
    expect(accumulator != nullptr && accumulator->dtype == DType::F32,
           "accumulators are F32 whatever the gradient storage is");
  }
  // The planner's view runs the micro-batch and the update together, so an
  // accumulator's slot cannot be handed to an activation.
  const auto plan = dif::compiler::plan_memory(step.planned);
  for (const auto &binding : step.bindings) {
    const auto *assignment = plan.assignment(binding.accumulator_output);
    expect(assignment != nullptr, "the accumulator is in the plan");
    if (!assignment)
      continue;
    bool reaches_the_update = false;
    for (std::size_t index = 0U; index < step.planned.operations.size();
         ++index) {
      const auto &operation = step.planned.operations[index];
      if (operation.opcode != Opcode::AdamWUpdate)
        continue;
      if (assignment->last_operation >= index)
        reaches_the_update = true;
    }
    expect(reaches_the_update,
           "an accumulator stays live until the optimizer reads it");
  }
}

void one_micro_batch_is_still_a_step() {
  const auto built = model(4U);
  const auto step = dif::training::build_accumulating_step(
      built.forward, built.loss, built.parameters, 1U, {});
  // With one micro-batch there is nothing to average, so the loss is not
  // multiplied by anything and stays the program's own loss.
  bool scaled = false;
  for (const auto &operation : step.accumulate.operations)
    if (operation.opcode == Opcode::Multiply &&
        (operation.inputs[0] == built.loss || operation.inputs[1] == built.loss))
      scaled = true;
  expect(!scaled, "a single micro-batch is not scaled by one");

  // Four of them are, and by the reciprocal of the count.
  const auto averaged = dif::training::build_accumulating_step(
      built.forward, built.loss, built.parameters, 4U, {});
  bool averaged_correctly = false;
  for (const auto &operation : averaged.accumulate.operations) {
    if (operation.opcode != Opcode::Multiply ||
        (operation.inputs[0] != built.loss && operation.inputs[1] != built.loss))
      continue;
    const auto other =
        operation.inputs[0] == built.loss ? operation.inputs[1] : operation.inputs[0];
    for (const auto &fill : averaged.accumulate.operations)
      if (fill.opcode == Opcode::Fill && fill.outputs[0] == other &&
          fill.f64(dif::ir::AttrKey::Value, 0.0) == 0.25)
        averaged_correctly = true;
  }
  expect(averaged_correctly,
         "four micro-batches scale the loss by exactly one quarter");
  expect(step.bindings.size() == built.parameters.size(),
         "a single micro-batch still runs through the accumulators");

  bool refused = false;
  try {
    dif::training::build_accumulating_step(built.forward, built.loss,
                                           built.parameters, 0U, {});
  } catch (const dif::Error &) {
    refused = true;
  }
  expect(refused, "zero micro-batches is refused");
}

} // namespace

int main() {
  micro_batches_match_one_batch();
  the_accumulators_outlive_the_micro_batch();
  one_micro_batch_is_still_a_step();
  if (failures != 0) {
    std::cerr << failures << " accumulation failure(s)\n";
    return 1;
  }
  std::cout << "training accumulation tests passed\n";
  return 0;
}
