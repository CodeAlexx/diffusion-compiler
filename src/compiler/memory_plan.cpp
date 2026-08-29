#include "dif/compiler/memory_plan.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

namespace dif::compiler {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0U || (alignment & (alignment - 1U)) != 0U)
    fail("memory-plan alignment must be a nonzero power of two");
  if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U))
    fail("memory-plan alignment overflow");
  return (value + alignment - 1U) & ~(alignment - 1U);
}

struct Interval {
  const ir::TensorDesc *tensor{};
  std::uint64_t first{};
  std::uint64_t last{};
  bool dedicated{};
};

} // namespace

const BufferAssignment *MemoryPlan::assignment(std::uint32_t tensor_id) const {
  const auto found = std::find_if(
      assignments.begin(), assignments.end(),
      [&](const BufferAssignment &value) { return value.tensor_id == tensor_id; });
  return found == assignments.end() ? nullptr : &*found;
}

MemoryPlan plan_memory(const ir::Program &program, std::uint64_t alignment,
                       std::uint64_t stream_prefetch_distance,
                       const std::unordered_set<std::uint32_t>
                           &excluded_internal_tensors) {
  ir::verify(program);
  (void)align_up(0U, alignment);
  const auto end = static_cast<std::uint64_t>(program.operations.size());
  std::unordered_map<std::uint32_t, std::uint64_t> producer;
  std::unordered_map<std::uint32_t, std::uint64_t> first_consumer;
  std::unordered_map<std::uint32_t, std::uint64_t> last_consumer;
  for (std::uint64_t index = 0; index < end; ++index) {
    const auto &operation = program.operations[static_cast<std::size_t>(index)];
    for (const auto tensor : operation.outputs)
      producer.emplace(tensor, index);
    for (const auto tensor : operation.inputs) {
      first_consumer.try_emplace(tensor, index);
      last_consumer[tensor] = index;
    }
  }

  std::vector<Interval> intervals;
  intervals.reserve(program.tensors.size());
  MemoryPlan plan;
  for (const auto &tensor : program.tensors) {
    if (excluded_internal_tensors.contains(tensor.id)) {
      if (tensor.has_role(ir::TensorRole::Input) ||
          tensor.has_role(ir::TensorRole::Constant) ||
          tensor.has_role(ir::TensorRole::Output))
        fail("memory plan may exclude only internal lowering intermediates");
      continue;
    }
    const auto streamed = tensor.has_role(ir::TensorRole::Streamed);
    const auto protected_role = tensor.has_role(ir::TensorRole::Input) ||
                                (tensor.has_role(ir::TensorRole::Constant) &&
                                 !streamed) ||
                                tensor.has_role(ir::TensorRole::Output);
    const auto produced = producer.find(tensor.id);
    const auto consumed = last_consumer.find(tensor.id);
    const auto first_used = first_consumer.find(tensor.id);
    const auto streamed_first = first_used == first_consumer.end()
                                    ? 0U
                                    : first_used->second > stream_prefetch_distance
                                          ? first_used->second -
                                                stream_prefetch_distance
                                          : 0U;
    const auto first = streamed
                           ? streamed_first
                           : produced == producer.end() ? 0U : produced->second;
    const auto last = tensor.has_role(ir::TensorRole::Output)
                          ? end
                          : consumed == last_consumer.end() ? first : consumed->second;
    intervals.push_back({&tensor, first, last, protected_role});
    const auto bytes = align_up(tensor.byte_count(), alignment);
    if (plan.naive_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
      fail("naive memory-plan byte total overflow");
    plan.naive_bytes += bytes;
  }
  std::sort(intervals.begin(), intervals.end(), [](const Interval &a,
                                                    const Interval &b) {
    if (a.dedicated != b.dedicated)
      return a.dedicated > b.dedicated;
    if (a.first != b.first)
      return a.first < b.first;
    return a.tensor->id < b.tensor->id;
  });

  std::vector<std::uint64_t> available_after;
  for (const auto &interval : intervals) {
    const auto bytes = align_up(interval.tensor->byte_count(), alignment);
    std::uint32_t slot_id = static_cast<std::uint32_t>(plan.slots.size());
    if (!interval.dedicated) {
      std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
      for (std::uint32_t candidate = 0; candidate < plan.slots.size(); ++candidate) {
        if (available_after[candidate] < interval.first &&
            plan.slots[candidate].bytes >= bytes &&
            plan.slots[candidate].bytes < best_bytes) {
          slot_id = candidate;
          best_bytes = plan.slots[candidate].bytes;
        }
      }
    }
    if (slot_id == plan.slots.size()) {
      plan.slots.push_back({slot_id, bytes, alignment});
      available_after.push_back(interval.dedicated
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : interval.last);
      if (plan.total_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
        fail("memory-plan byte total overflow");
      plan.total_bytes += bytes;
    } else {
      available_after[slot_id] = interval.last;
    }
    plan.assignments.push_back({interval.tensor->id, slot_id, bytes,
                                interval.first, interval.last});
  }
  std::sort(plan.assignments.begin(), plan.assignments.end(),
            [](const BufferAssignment &a, const BufferAssignment &b) {
              return a.tensor_id < b.tensor_id;
            });
  return plan;
}

} // namespace dif::compiler
