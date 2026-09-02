#pragma once

#include "dif/ir/ir.hpp"
#include "dif/opt/physical_format.hpp"
#include "dif/opt/transform.hpp"
#include "dif/target/profile.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dif::opt {

// A program together with everything needed to execute and reproduce it.
// Transformations may rewrite the graph and the constant values it binds
// (constant folding and quantization both do), so the two travel together.
struct RewriteContext {
  ir::Program program;
  // Constant-role and Input-role tensors. Transforms never touch inputs.
  runtime::TensorMap bindings;
  // Backend-neutral streaming policy consumed by the memory planner.
  std::uint64_t prefetch_distance{};
};

struct MemoryFootprint {
  // Lifetime-aware planned working set, the quantity the memory constraint is
  // applied to.
  std::uint64_t planned_bytes{};
  // Sum of every tensor's aligned size with no reuse.
  std::uint64_t naive_bytes{};
  std::uint64_t resident_constant_bytes{};
  std::uint64_t streamed_constant_bytes{};
};

// Bounds and switches for legality discovery. Discovery never invents a
// transform outside these bounds, so the reachable candidate space is a
// declared property of the run and not an emergent one.
struct DiscoveryOptions {
  bool structural{true};
  bool schedule{true};
  bool numeric{true};
  bool memory{true};
  // Folding an arithmetic operation would bake the reference backend's rounding
  // into the program, so it is off unless a caller accepts that trade.
  bool arithmetic_constant_folding{false};
  std::vector<std::uint64_t> block_sizes{64U, 128U, 256U, 512U};
  // TileM/TileN/TileK are declared DiffIR scheduling attributes that no current
  // backend consumes. Discovery leaves this empty so the search does not spend
  // measurements on behaviourally identical programs; a backend that reads them
  // enables it.
  std::vector<std::uint64_t> tile_shapes;
  std::vector<std::uint64_t> quantization_bits{4U, 5U};
  std::vector<std::uint64_t> quantization_group_sizes{64U};
  std::vector<std::uint64_t> prefetch_distances{1U, 2U, 4U};
  std::vector<ir::DType> precisions{ir::DType::F32, ir::DType::BF16,
                                    ir::DType::F16};
  // Upper bound on how many precision candidates a single discovery call may
  // propose, so one pass over a fifty-block denoiser cannot flood the beam.
  std::size_t max_precision_candidates{8U};
  // Bounded physical-format competition. When non-empty, the precision and
  // quantization candidates above are derived from these formats instead:
  // only formats that are legal on `target` and implemented as search
  // candidates in this build produce transforms. The compiler decides from
  // runtime-discovered capability; product names never enter the decision.
  std::vector<PhysicalFormat> physical_formats;
  std::optional<target::TargetProfile> target;
};

// The status of every requested physical format under the options' target:
// what competes, and why the rest do not. Empty when no formats were
// requested.
std::vector<FormatStatus> format_statuses(const DiscoveryOptions &options);

// Enumerates the transforms that are legal on this context right now. The
// result is deterministic and ordered, so an identical context always yields an
// identical candidate space.
std::vector<Transform> discover(const RewriteContext &context,
                                const DiscoveryOptions &options);

// Applies one transform in place. Throws dif::Error when the transform is not
// legal on this context; a caller may therefore replay a recorded plan and be
// told, rather than silently drift.
void apply(const Transform &transform, RewriteContext &context);

// Hex SHA-256 of the encoded program. Identical to the fingerprint the rest of
// the toolchain uses.
std::string program_fingerprint(const ir::Program &program);

// Hex SHA-256 over the program, the streaming policy, and every bound constant
// value. Two contexts share a candidate fingerprint only when they will execute
// identically, which matters because folding and quantization change constants
// while leaving parts of the graph alone.
std::string candidate_fingerprint(const RewriteContext &context);

MemoryFootprint measure_memory(const RewriteContext &context);

// The typed reference executor used to evaluate constant subgraphs. Exposed so
// that callers and tests can see that folding is a reference-semantics
// operation, not a backend-specific one.
runtime::TensorMap evaluate_constant_operation(const ir::Program &program,
                                               const ir::Operation &operation,
                                               const runtime::TensorMap &bindings);

} // namespace dif::opt
