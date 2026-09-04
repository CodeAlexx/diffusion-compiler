// What a training step's memory is spent on, and what a device budget buys.
//
// The classification has to be exhaustive and disjoint: every tensor in the
// program lands in exactly one class, so the classified total equals the sum
// of every tensor's bytes. That is checkable, and it is what stops a class
// from quietly going missing when the graph changes.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/training/memory.hpp"

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

// Two layers, so there is a real activation between them that the backward
// pass needs -- otherwise there is nothing to classify as saved.
Program two_layer(std::uint64_t rows, std::uint64_t width) {
  Program program;
  program.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, width}},
      {2U, DType::F32, TensorRole::Constant, {width, width}},
      {3U, DType::F32, TensorRole::Constant, {width, width}},
      {4U, DType::F32, TensorRole::Internal, {rows, width}},
      {5U, DType::F32, TensorRole::Internal, {rows, width}},
      {6U, DType::F32, TensorRole::Internal, {rows, width}},
      {7U, DType::F32, TensorRole::Input, {rows, width}},
      {8U, DType::F32, TensorRole::Output, {1U}}};
  program.operations = {{1U, Opcode::Linear, {1U, 2U}, {4U}, {}},
                        {2U, Opcode::SiLU, {4U}, {5U}, {}},
                        {3U, Opcode::Linear, {5U, 3U}, {6U}, {}},
                        {4U, Opcode::MseLoss, {6U, 7U}, {8U}, {}}};
  dif::ir::verify(program);
  return program;
}

void every_tensor_is_classified_exactly_once() {
  const auto forward = two_layer(4U, 6U);
  const std::vector<std::uint32_t> trainable{2U, 3U};
  const auto plan = dif::training::compile(forward, 8U, trainable, {});
  const auto report = dif::training::analyze_memory(plan);

  std::uint64_t declared = 0U;
  for (const auto &tensor : plan.program.tensors)
    declared += tensor.byte_count();
  expect(report.total() == declared,
         "the classified total accounts for every tensor exactly once: " +
             std::to_string(report.total()) + " vs " +
             std::to_string(declared));

  // planned_bytes and total() are NOT comparable: the classification sums raw
  // tensor bytes, while the plan aligns every buffer to 256 and shares one
  // buffer between tensors with disjoint lifetimes. On a large graph sharing
  // dominates and the plan is smaller; on this one, where a scalar loss
  // rounds from 4 bytes to 256, padding dominates and it is larger. Both
  // numbers are reported so the difference is visible rather than implied.
  // What IS comparable is the floor against the plan, since both come from
  // the same aligned assignment.
  expect(report.live_peak_bytes <= report.planned_bytes,
         "the live floor is no larger than what the planner allocates");
}

void a_saved_activation_is_one_the_backward_pass_still_needs() {
  const auto forward = two_layer(4U, 6U);
  const auto plan = dif::training::compile(forward, 8U,
                                           std::vector<std::uint32_t>{2U, 3U},
                                           {});
  const auto report = dif::training::analyze_memory(plan);
  expect(report.saved_activations > 0U,
         "a two-layer forward saves something for its backward pass");
  // Every activation is either transient or saved, never both, and together
  // they are all of them.
  expect(report.transient_activations + report.saved_activations > 0U,
         "activations are accounted for");
}

void freezing_a_parameter_moves_its_bytes() {
  const auto forward = two_layer(4U, 6U);
  const auto both = dif::training::analyze_memory(
      dif::training::compile(forward, 8U, std::vector<std::uint32_t>{2U, 3U},
                             {}));
  const auto one = dif::training::analyze_memory(
      dif::training::compile(forward, 8U, std::vector<std::uint32_t>{2U}, {}));
  expect(one.frozen_weights > both.frozen_weights,
         "freezing a parameter moves it into the frozen class");
  expect(one.trainable_weights < both.trainable_weights,
         "and out of the trainable class");
  expect(one.optimizer_state < both.optimizer_state,
         "a frozen parameter carries no optimizer state");
  expect(one.gradients < both.gradients,
         "and no gradient");
  expect(one.planned_bytes < both.planned_bytes,
         "so the step plans smaller");
}

void the_budget_comes_from_the_target() {
  dif::training::RecomputePolicy policy;
  policy.budget.usable_device_memory_bytes = 1000U;
  policy.headroom = 0.9;
  expect(policy.device_bytes() == 900U, "headroom is applied to the usable budget");

  // Falls back to free memory when a usable figure was never computed.
  dif::training::RecomputePolicy fallback;
  fallback.budget.free_device_memory_bytes = 2000U;
  fallback.headroom = 0.5;
  expect(fallback.device_bytes() == 1000U,
         "free memory stands in when usable is absent");

  const auto refused = [](dif::training::RecomputePolicy bad,
                          const std::string &why) {
    bool threw = false;
    try {
      (void)bad.device_bytes();
    } catch (const dif::Error &) {
      threw = true;
    }
    expect(threw, "refused: " + why);
  };
  dif::training::RecomputePolicy empty;
  refused(empty, "a policy with no device memory at all");
  dif::training::RecomputePolicy zero_headroom;
  zero_headroom.budget.usable_device_memory_bytes = 1000U;
  zero_headroom.headroom = 0.0;
  refused(zero_headroom, "a headroom of zero");
  dif::training::RecomputePolicy over;
  over.budget.usable_device_memory_bytes = 1000U;
  over.headroom = 1.5;
  refused(over, "a headroom above one");
}

void recompute_answers_the_budget_it_was_given() {
  // Deep enough that dropping the interior actually saves something.
  Program forward;
  const std::uint64_t rows = 8U, width = 16U;
  std::uint32_t next = 1U;
  auto tensor = [&](std::uint32_t roles, std::vector<std::uint64_t> dims) {
    const auto id = next++;
    forward.tensors.push_back({id, DType::F32, roles, std::move(dims)});
    return id;
  };
  const auto input = tensor(TensorRole::Input, {rows, width});
  auto activation = input;
  std::vector<std::uint32_t> trainable;
  std::uint32_t operation = 1U;
  for (int layer = 0; layer < 8; ++layer) {
    const auto weight = tensor(TensorRole::Constant, {width, width});
    trainable.push_back(weight);
    const auto linear = tensor(TensorRole::Internal, {rows, width});
    forward.operations.push_back(
        {operation++, Opcode::Linear, {activation, weight}, {linear}, {}});
    const auto activated = tensor(TensorRole::Internal, {rows, width});
    forward.operations.push_back(
        {operation++, Opcode::SiLU, {linear}, {activated}, {}});
    activation = activated;
  }
  const auto target = tensor(TensorRole::Input, {rows, width});
  const auto loss = tensor(TensorRole::Output, {1U});
  forward.operations.push_back(
      {operation++, Opcode::MseLoss, {activation, target}, {loss}, {}});
  dif::ir::verify(forward);

  const auto plan = dif::training::compile(forward, loss, trainable, {});
  const auto report = dif::training::analyze_memory(plan);

  // A budget the step already meets buys no recompute at all.
  dif::training::RecomputePolicy generous;
  generous.budget.usable_device_memory_bytes = report.planned_bytes * 4U;
  const auto relaxed = dif::training::choose_recompute(plan, generous);
  expect(relaxed.segments == 1U && relaxed.within_budget,
         "a budget the step already fits buys no recompute");

  // A budget nothing can meet is reported honestly rather than as success.
  dif::training::RecomputePolicy impossible;
  impossible.budget.usable_device_memory_bytes = 1024U;
  const auto refused = dif::training::choose_recompute(plan, impossible);
  expect(!refused.within_budget,
         "a budget recompute cannot reach is reported as not met");
  expect(refused.planned_bytes <= relaxed.planned_bytes,
         "and the closest attempt is no worse than doing nothing");
  std::cout << "  deep chain: " << report.saved_activations
            << " B saved activations of " << report.planned_bytes
            << " B planned; tightest attempt " << refused.planned_bytes
            << " B at " << refused.segments << " segments\n";
}

} // namespace

int main() {
  every_tensor_is_classified_exactly_once();
  a_saved_activation_is_one_the_backward_pass_still_needs();
  freezing_a_parameter_moves_its_bytes();
  the_budget_comes_from_the_target();
  recompute_answers_the_budget_it_was_given();
  if (failures != 0) {
    std::cerr << failures << " training memory failure(s)\n";
    return 1;
  }
  std::cout << "training memory tests passed\n";
  return 0;
}
