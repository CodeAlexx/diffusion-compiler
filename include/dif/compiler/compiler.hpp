#pragma once

#include "dif/ir/ir.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dif::compiler {

struct GeneratedCuda {
  std::string source;
  std::unordered_map<std::uint32_t, std::string> entrypoints;
  // Backend lowering may replace a semantic operation chain with one kernel.
  // launch_inputs records the tensor arguments for such an entrypoint and
  // skipped_operations are semantic nodes subsumed by it.
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> launch_inputs;
  std::unordered_set<std::uint32_t> skipped_operations;
};

GeneratedCuda emit_cuda(const ir::Program &program);

} // namespace dif::compiler
