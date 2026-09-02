#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dif::compiler {

enum class StreamedResidencyOrder : std::uint8_t {
  FirstConsumer = 0,
  LargestFirst = 1,
};

// Explicit compiler policy for keeping reusable streamed constants resident.
// The runtime receives only resident_tensor_ids and implements the storage and
// upload mechanism; it does not choose what should stay resident.
// Why one streamed constant was admitted to residency or left streamed. The
// planner records this as it decides, so a plan can be explained from the
// real admission arithmetic rather than reconstructed later.
struct StreamedResidencyDecision {
  std::uint32_t tensor_id{};
  std::uint64_t bytes{};
  std::uint64_t first_consumer_operation{};
  bool resident{};
  // Complete bytes the execution would need with this tensor admitted, at the
  // moment it was considered, against the plan's ceiling.
  std::uint64_t required_bytes_if_resident{};
  std::uint64_t maximum_total_bytes{};
  std::string reason;
};

struct StreamedResidencyPlan {
  std::string selection_order;
  std::uint64_t maximum_total_bytes{};
  std::uint64_t fixed_runtime_bytes{};
  std::uint64_t memory_plan_bytes{};
  std::uint64_t resident_constant_bytes{};
  std::uint64_t streamed_constant_bytes{};
  std::uint64_t required_bytes{};
  std::vector<std::uint32_t> resident_tensor_ids;
  // One entry per streamed constant, in the order the planner considered
  // them.
  std::vector<StreamedResidencyDecision> decisions;
};

// Select streamed constants in the requested deterministic order, admitting
// each one only when the complete memory plan plus fixed backend workspace
// remains within maximum_total_bytes. Recomputing the memory plan captures the
// slot shrinkage caused by removing a streamed weight from the paging arena.
StreamedResidencyPlan plan_streamed_residency(
    const ir::Program &program, std::uint64_t maximum_total_bytes,
    std::uint64_t fixed_runtime_bytes = 0U,
    std::uint64_t stream_prefetch_distance = 1U,
    const std::unordered_map<std::uint32_t, std::uint32_t>
        &tensor_aliases = {},
    StreamedResidencyOrder order = StreamedResidencyOrder::FirstConsumer);

} // namespace dif::compiler
