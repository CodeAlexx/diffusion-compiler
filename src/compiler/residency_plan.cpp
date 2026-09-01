#include "dif/compiler/residency_plan.hpp"

#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dif::compiler {
namespace {

std::uint64_t align_256(std::uint64_t bytes) {
  if (bytes > std::numeric_limits<std::uint64_t>::max() - 255U)
    fail("streamed residency byte alignment overflow");
  return (bytes + 255U) & ~std::uint64_t{255U};
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *message) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right)
    fail(message);
  return left + right;
}

} // namespace

StreamedResidencyPlan plan_streamed_residency(
    const ir::Program &program, std::uint64_t maximum_total_bytes,
    std::uint64_t fixed_runtime_bytes,
    std::uint64_t stream_prefetch_distance,
    const std::unordered_map<std::uint32_t, std::uint32_t> &tensor_aliases,
    StreamedResidencyOrder order) {
  ir::verify(program);
  std::unordered_map<std::uint32_t, std::uint64_t> first_consumer;
  for (std::uint64_t index = 0U; index < program.operations.size(); ++index) {
    for (const auto id : program.operations[index].inputs)
      first_consumer.try_emplace(id, index);
  }

  std::vector<std::pair<std::uint64_t, std::uint32_t>> candidates;
  std::uint64_t total_streamed_bytes = 0U;
  for (const auto &tensor : program.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant) ||
        !tensor.has_role(ir::TensorRole::Streamed))
      continue;
    total_streamed_bytes = checked_add(
        total_streamed_bytes, tensor.byte_count(),
        "streamed residency total constant bytes overflow");
    const auto found = first_consumer.find(tensor.id);
    const auto first = found == first_consumer.end()
                           ? std::numeric_limits<std::uint64_t>::max()
                           : found->second;
    candidates.emplace_back(first, tensor.id);
  }
  std::sort(candidates.begin(), candidates.end(),
            [&](const auto &left, const auto &right) {
              if (order == StreamedResidencyOrder::LargestFirst) {
                const auto left_bytes = program.tensor(left.second)->byte_count();
                const auto right_bytes = program.tensor(right.second)->byte_count();
                if (left_bytes != right_bytes)
                  return left_bytes > right_bytes;
              }
              return left < right;
            });

  std::unordered_set<std::uint32_t> selected;
  std::uint64_t resident_bytes = 0U;
  auto memory = plan_memory(program, 256U, stream_prefetch_distance, {}, {},
                            tensor_aliases, false);
  auto required = checked_add(memory.total_bytes, fixed_runtime_bytes,
                              "streamed residency baseline bytes overflow");
  if (required > maximum_total_bytes)
    fail("streamed residency budget is below the unmodified execution plan");

  for (const auto &[first, id] : candidates) {
    (void)first;
    const auto *tensor = program.tensor(id);
    if (!tensor)
      fail("streamed residency candidate references a missing tensor");
    auto trial = selected;
    trial.insert(id);
    const auto trial_resident = checked_add(
        resident_bytes, align_256(tensor->byte_count()),
        "streamed residency selected constant bytes overflow");
    const auto trial_memory = plan_memory(
        program, 256U, stream_prefetch_distance, {}, trial, tensor_aliases,
        false);
    auto trial_required = checked_add(
        trial_memory.total_bytes, fixed_runtime_bytes,
        "streamed residency execution bytes overflow");
    trial_required = checked_add(
        trial_required, trial_resident,
        "streamed residency complete required bytes overflow");
    if (trial_required > maximum_total_bytes)
      continue;
    selected = std::move(trial);
    resident_bytes = trial_resident;
    memory = trial_memory;
    required = trial_required;
  }

  StreamedResidencyPlan result;
  result.maximum_total_bytes = maximum_total_bytes;
  result.fixed_runtime_bytes = fixed_runtime_bytes;
  result.memory_plan_bytes = memory.total_bytes;
  result.resident_constant_bytes = resident_bytes;
  result.streamed_constant_bytes = total_streamed_bytes;
  for (const auto &[first, id] : candidates) {
    (void)first;
    if (!selected.contains(id))
      continue;
    result.resident_tensor_ids.push_back(id);
    result.streamed_constant_bytes -= program.tensor(id)->byte_count();
  }
  result.required_bytes = required;
  return result;
}

} // namespace dif::compiler
