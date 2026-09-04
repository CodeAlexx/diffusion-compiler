// Recompute has to be free of consequence except for memory.
//
// The replay is the same operations on the same inputs in the same order, so
// the gradients must come out BIT-IDENTICAL -- not close, identical. If they
// are not, the pass rewired something it should not have. The other half of
// the claim is the point of the pass at all: the memory plan has to get
// smaller, and by more than noise.

#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/training/recompute.hpp"
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

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

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

// A deep chain: many layers, so there is real interior activation to throw
// away and rebuild.
struct Built {
  Program forward;
  std::uint32_t loss{};
  std::vector<std::uint32_t> parameters;
  dif::runtime::TensorMap inputs;
  std::size_t forward_operations{};
};

Built deep_chain(std::size_t layers) {
  Built built;
  const std::uint64_t rows = 6U;
  const std::uint64_t width = 12U;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  auto tensor = [&](std::uint32_t roles, std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    built.forward.tensors.push_back({id, DType::F32, roles, std::move(dims)});
    return id;
  };
  const auto input = tensor(TensorRole::Input, {rows, width});
  built.inputs.emplace(input, f32_tensor({rows, width}, 3U));
  auto activation = input;
  for (std::size_t layer = 0U; layer < layers; ++layer) {
    const auto weight =
        tensor(TensorRole::Input | TensorRole::Parameter, {width, width});
    built.inputs.emplace(weight, f32_tensor({width, width}, 11U + layer));
    built.parameters.push_back(weight);
    const auto linear = tensor(TensorRole::Internal, {rows, width});
    built.forward.operations.push_back(
        {next_operation++, Opcode::Linear, {activation, weight}, {linear}, {}});
    const auto activated = tensor(TensorRole::Internal, {rows, width});
    built.forward.operations.push_back(
        {next_operation++, Opcode::SiLU, {linear}, {activated}, {}});
    activation = activated;
  }
  const auto target = tensor(TensorRole::Input, {rows, width});
  built.inputs.emplace(target, f32_tensor({rows, width}, 97U));
  built.loss = tensor(TensorRole::Output, {1U});
  built.forward.operations.push_back({next_operation++, Opcode::MseLoss,
                                      {activation, target}, {built.loss}, {}});
  built.forward_operations = built.forward.operations.size();
  dif::ir::verify(built.forward);
  return built;
}

dif::runtime::TensorMap run(const Program &program,
                            const dif::runtime::TensorMap &inputs) {
  auto state = inputs;
  // Optimizer state the update reads: zero moments and a step counter of one.
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

void recompute_changes_nothing_but_memory() {
  const auto built = deep_chain(8U);
  const auto step = dif::training::build_training_step(
      built.forward, built.loss, built.parameters, {});
  // The composed program's forward region is the original forward pass.
  const auto plain = run(step.program, built.inputs);
  const auto baseline = dif::compiler::plan_memory(step.program).total_bytes;

  bool any_smaller = false;
  for (const std::size_t segments : {2U, 4U, 8U}) {
    const auto rebuilt = dif::training::plan_recompute(
        step.program, built.forward_operations, segments);
    expect(rebuilt.operations.size() > step.program.operations.size(),
           "recompute adds the replay operations it promised");
    const auto planned = dif::compiler::plan_memory(rebuilt).total_bytes;
    if (planned < baseline)
      any_smaller = true;

    const auto recomputed = run(rebuilt, built.inputs);
    for (const auto &binding : step.bindings) {
      for (const auto tensor :
           {binding.gradient_output, binding.parameter_output,
            binding.first_moment_output, binding.second_moment_output}) {
        const auto expected = plain.find(tensor);
        const auto actual = recomputed.find(tensor);
        if (expected == plain.end() || actual == recomputed.end()) {
          expect(false, "recompute kept every gradient and update output");
          continue;
        }
        expect(expected->second.bytes == actual->second.bytes,
               "recompute leaves gradients and updates BIT-identical at " +
                   std::to_string(segments) + " segments");
      }
    }
  }
  expect(any_smaller, "recompute makes the memory plan smaller");
}

void one_segment_is_the_identity() {
  const auto built = deep_chain(4U);
  const auto step = dif::training::build_training_step(
      built.forward, built.loss, built.parameters, {});
  for (const std::size_t segments : {0U, 1U}) {
    const auto same = dif::training::plan_recompute(
        step.program, built.forward_operations, segments);
    expect(same.operations.size() == step.program.operations.size() &&
               same.tensors.size() == step.program.tensors.size(),
           "a single segment leaves the program alone");
  }
}

void the_budget_search_reports_honestly() {
  const auto built = deep_chain(10U);
  const auto step = dif::training::build_training_step(
      built.forward, built.loss, built.parameters, {});
  const auto unplanned =
      dif::compiler::plan_memory(step.program).total_bytes;

  // A budget that is already met asks for no recompute at all.
  auto generous = dif::training::choose_recompute(
      step.program, built.forward_operations, unplanned);
  expect(generous.segments == 1U && generous.within_budget,
         "a budget that already fits buys no recompute");

  // A budget nothing can meet comes back as the closest, marked honestly,
  // rather than as a silent success.
  auto impossible = dif::training::choose_recompute(
      step.program, built.forward_operations, 1U);
  expect(!impossible.within_budget,
         "an impossible budget is reported as not met");
  expect(impossible.planned_bytes <= unplanned,
         "the closest attempt is no worse than doing nothing");

  // A budget between the two picks a real segmentation and produces a program
  // that still verifies.
  Program chosen;
  auto tight = dif::training::choose_recompute(
      step.program, built.forward_operations, unplanned * 9U / 10U, &chosen);
  dif::ir::verify(chosen);
  expect(tight.segments >= 1U, "the search returns a segmentation");
  if (tight.segments > 1U)
    expect(tight.planned_bytes < unplanned,
           "a chosen segmentation actually saves memory");
}

} // namespace

int main() {
  recompute_changes_nothing_but_memory();
  one_segment_is_the_identity();
  the_budget_search_reports_honestly();
  if (failures != 0) {
    std::cerr << failures << " recompute failure(s)\n";
    return 1;
  }
  std::cout << "training recompute tests passed\n";
  return 0;
}
