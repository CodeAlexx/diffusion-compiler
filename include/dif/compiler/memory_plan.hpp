#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dif::compiler {

struct BufferSlot {
  std::uint32_t id{};
  std::uint64_t bytes{};
  std::uint64_t alignment{};
};

struct BufferAssignment {
  std::uint32_t tensor_id{};
  std::uint32_t slot_id{};
  std::uint64_t bytes{};
  std::uint64_t first_operation{};
  std::uint64_t last_operation{};
};

struct MemoryPlan {
  std::vector<BufferSlot> slots;
  std::vector<BufferAssignment> assignments;
  std::uint64_t total_bytes{};
  std::uint64_t naive_bytes{};

  const BufferAssignment *assignment(std::uint32_t tensor_id) const;
};

MemoryPlan plan_memory(const ir::Program &program,
                       std::uint64_t alignment = 256U,
                       std::uint64_t stream_prefetch_distance = 0U,
                       const std::unordered_set<std::uint32_t>
                           &excluded_internal_tensors = {},
                       const std::unordered_set<std::uint32_t>
                           &replaced_constant_tensors = {},
                       const std::unordered_map<std::uint32_t, std::uint32_t>
                           &tensor_aliases = {},
                       bool verify_program = true);

} // namespace dif::compiler
