#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace dif::compiler {

// Explicit zero-copy layout candidate. DiffIR Reshape remains semantic; the
// execution plan may alias an internal output to its immutable SSA input.
struct ReshapeAliasPlan {
  std::vector<std::uint32_t> operation_ids;
  std::unordered_map<std::uint32_t, std::uint32_t> output_to_root_input;
  std::uint64_t eliminated_materialization_bytes{};
};

ReshapeAliasPlan plan_reshape_aliases(const ir::Program &program);

} // namespace dif::compiler
