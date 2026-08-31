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

// Opt-in elementwise region fusion census.  Counts the single-consumer
// pointwise regions -- operations stamped Implementation=2 by
// `difc set-elementwise-fusion` -- that emit_cuda would collapse into one
// kernel each.  Pure analysis; no source is generated.  Unstamped programs
// always report zero regions (fusion is a fingerprinted candidate property,
// never silent global behavior).
struct ElementwiseFusionCensus {
  std::uint64_t regions{};
  std::uint64_t fused_operations{};
  std::uint64_t eliminated_launches{};
};

ElementwiseFusionCensus
census_elementwise_fusion(const ir::Program &program);

} // namespace dif::compiler
