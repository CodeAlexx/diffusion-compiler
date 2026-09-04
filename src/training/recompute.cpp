#include "dif/training/recompute.hpp"

#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace dif::training {
namespace {

// A tensor the backward pass reads but the forward pass produced: the ones
// that make a training step cost more than an inference pass.
struct Crossing {
  std::uint32_t tensor{};
  std::size_t producer{};
  std::size_t first_backward_reader{};
};

} // namespace

ir::Program plan_recompute(const ir::Program &program,
                           std::size_t forward_operation_count,
                           std::size_t segments) {
  if (forward_operation_count > program.operations.size())
    fail("recompute forward operation count exceeds the program");
  if (segments <= 1U || forward_operation_count == 0U)
    return program;

  const auto operation_count = program.operations.size();

  // Which forward operation produced each tensor, and the earliest backward
  // operation that reads it.
  std::unordered_map<std::uint32_t, std::size_t> producer;
  for (std::size_t index = 0U; index < forward_operation_count; ++index)
    for (const auto tensor : program.operations[index].outputs)
      producer.emplace(tensor, index);

  std::unordered_map<std::uint32_t, std::size_t> first_backward_reader;
  for (std::size_t index = forward_operation_count; index < operation_count;
       ++index)
    for (const auto tensor : program.operations[index].inputs)
      first_backward_reader.try_emplace(tensor, index);

  std::vector<Crossing> crossings;
  for (const auto &[tensor, produced_at] : producer) {
    const auto reader = first_backward_reader.find(tensor);
    if (reader == first_backward_reader.end())
      continue;
    const auto *description = program.tensor(tensor);
    // A program output has to survive on its own terms; only pure internal
    // activations are candidates for being thrown away and rebuilt.
    if (!description ||
        description->roles != static_cast<std::uint32_t>(ir::TensorRole::Internal))
      continue;
    crossings.push_back({tensor, produced_at, reader->second});
  }
  if (crossings.empty())
    return program;

  // Contiguous segments of the forward region, by operation count.  Equal
  // counts rather than equal bytes: the replay cost is what is being bounded,
  // and it is the operations that cost.
  const auto piece = (forward_operation_count + segments - 1U) / segments;
  const auto segment_of = [&](std::size_t operation) {
    return std::min(operation / piece, segments - 1U);
  };

  // Everything a segment must rebuild, and when the backward pass first asks
  // for it.
  std::vector<std::unordered_set<std::uint32_t>> dropped(segments);
  std::vector<std::size_t> replay_before(segments, operation_count);
  for (const auto &crossing : crossings) {
    const auto segment = segment_of(crossing.producer);
    dropped[segment].insert(crossing.tensor);
    replay_before[segment] =
        std::min(replay_before[segment], crossing.first_backward_reader);
  }

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  ir::Program result;
  result.tensors = program.tensors;

  // Per segment: the operations that actually have to run to rebuild what was
  // dropped, found by walking the segment backwards from the dropped tensors.
  // Everything else in the segment is already dead and stays that way.
  std::vector<std::vector<ir::Operation>> replay(segments);
  // The rewiring the backward pass sees: dropped tensor -> its rebuilt copy.
  std::unordered_map<std::uint32_t, std::uint32_t> rebuilt;

  for (std::size_t segment = 0U; segment < segments; ++segment) {
    if (dropped[segment].empty())
      continue;
    const auto begin = segment * piece;
    const auto end = std::min(begin + piece, forward_operation_count);
    auto needed = dropped[segment];
    std::vector<std::size_t> required;
    for (std::size_t offset = end; offset > begin; --offset) {
      const auto index = offset - 1U;
      const auto &operation = program.operations[index];
      const bool produces_needed =
          std::any_of(operation.outputs.begin(), operation.outputs.end(),
                      [&](std::uint32_t tensor) { return needed.contains(tensor); });
      if (!produces_needed)
        continue;
      required.push_back(index);
      for (const auto tensor : operation.inputs) {
        const auto produced_at = producer.find(tensor);
        // Only inputs made inside this segment are rebuilt; anything older is
        // a segment boundary and is kept alive instead.
        if (produced_at != producer.end() && produced_at->second >= begin &&
            produced_at->second < end)
          needed.insert(tensor);
      }
    }
    std::reverse(required.begin(), required.end());

    std::unordered_map<std::uint32_t, std::uint32_t> copy;
    for (const auto index : required) {
      const auto &original = program.operations[index];
      ir::Operation clone;
      clone.id = next_operation++;
      clone.opcode = original.opcode;
      clone.attributes = original.attributes;
      for (const auto tensor : original.inputs) {
        const auto replaced = copy.find(tensor);
        clone.inputs.push_back(replaced == copy.end() ? tensor
                                                      : replaced->second);
      }
      for (const auto tensor : original.outputs) {
        const auto *description = program.tensor(tensor);
        if (!description)
          fail("recompute found an operation output that is not a tensor");
        const auto fresh = next_tensor++;
        result.tensors.push_back({fresh, description->dtype,
                                  static_cast<std::uint32_t>(
                                      ir::TensorRole::Internal),
                                  description->dims});
        copy.emplace(tensor, fresh);
        clone.outputs.push_back(fresh);
      }
      replay[segment].push_back(std::move(clone));
    }
    for (const auto tensor : dropped[segment]) {
      const auto replaced = copy.find(tensor);
      if (replaced != copy.end())
        rebuilt.emplace(tensor, replaced->second);
    }
  }

  // Emit: the forward region unchanged, then the backward region with each
  // segment's replay spliced in just before the operation that needs it, and
  // every backward read of a dropped activation pointed at its rebuilt copy.
  result.operations.reserve(operation_count);
  for (std::size_t index = 0U; index < forward_operation_count; ++index)
    result.operations.push_back(program.operations[index]);
  for (std::size_t index = forward_operation_count; index < operation_count;
       ++index) {
    for (std::size_t segment = 0U; segment < segments; ++segment)
      if (replay_before[segment] == index)
        for (auto &operation : replay[segment])
          result.operations.push_back(operation);
    auto operation = program.operations[index];
    for (auto &tensor : operation.inputs) {
      const auto replaced = rebuilt.find(tensor);
      if (replaced != rebuilt.end())
        tensor = replaced->second;
    }
    result.operations.push_back(std::move(operation));
  }

  ir::verify(result);
  return result;
}

RecomputeChoice choose_recompute(const ir::Program &program,
                                 std::size_t forward_operation_count,
                                 std::uint64_t budget_bytes,
                                 ir::Program *chosen) {
  RecomputeChoice choice;
  choice.bytes_without_recompute = compiler::plan_memory(program).total_bytes;
  choice.planned_bytes = choice.bytes_without_recompute;
  choice.segments = 1U;
  choice.within_budget = choice.bytes_without_recompute <= budget_bytes;
  if (choice.within_budget) {
    if (chosen)
      *chosen = program;
    return choice;
  }

  // Fewer segments replay less arithmetic, so stop at the first that fits.
  // The ladder is coarse on purpose: each rung costs a full memory plan, and
  // the returns flatten once the boundaries dominate.
  ir::Program best;
  for (const std::size_t segments : {2U, 3U, 4U, 6U, 8U, 12U, 16U, 24U, 32U}) {
    if (segments > forward_operation_count)
      break;
    auto candidate =
        plan_recompute(program, forward_operation_count, segments);
    const auto bytes = compiler::plan_memory(candidate).total_bytes;
    const auto replayed = candidate.operations.size() - program.operations.size();
    if (bytes < choice.planned_bytes || choice.segments == 1U) {
      choice.segments = segments;
      choice.planned_bytes = bytes;
      choice.replayed_operations = replayed;
      best = std::move(candidate);
    }
    if (bytes <= budget_bytes) {
      choice.within_budget = true;
      break;
    }
  }
  if (chosen)
    *chosen = choice.segments == 1U ? program : std::move(best);
  return choice;
}

} // namespace dif::training
