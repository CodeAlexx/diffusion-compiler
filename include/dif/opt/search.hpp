#pragma once

#include "dif/opt/gate.hpp"
#include "dif/opt/plan.hpp"
#include "dif/opt/rewrite.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/tune/database.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dif::opt {

// What the search minimizes once a candidate has cleared the acceptance gate.
// The objective is declared by the caller before the search starts; nothing a
// transform does can change it.
enum class Objective : std::uint32_t {
  // Minimize measured latency. Planned memory is a hard constraint carried by
  // the gate's memory budget.
  Latency = 1,
  // Minimize planned working set, admitting only candidates whose latency stays
  // within latency_regression_tolerance of the baseline.
  PlannedMemory = 2,
};

std::string_view objective_name(Objective objective);

struct SearchOptions {
  Objective objective{Objective::Latency};
  std::uint32_t warmups{1};
  std::uint32_t iterations{5};
  // CUDA pressure guard applied before preparation and execution. Zero disables
  // an additional reserve; real-device searches should keep enough free memory
  // for the rest of the process and display stack.
  std::uint64_t minimum_free_bytes{};
  // How many accepted candidates survive as parents for the next depth.
  std::uint32_t beam_width{3};
  std::uint32_t max_depth{3};
  // Hard ceiling on measured candidates, so a search is always bounded.
  std::uint32_t max_candidates{96};
  // A candidate must improve the objective by this fraction to displace the
  // incumbent. It exists so measurement noise cannot promote a candidate. When
  // the objective is latency the search widens it to the measured drift, so a
  // requested margin is a floor and never a way to claim a win inside noise.
  double improvement_margin{0.02};
  // How much latency a memory-objective candidate may cost. Widened to the
  // measured drift for the same reason.
  double latency_regression_tolerance{1.05};
  // Discard one untimed run of the base program before the baseline is
  // measured, so first-touch page faults and allocator growth are not charged
  // to the baseline and credited to every candidate after it.
  bool warm_process_before_baseline{true};
  // Re-measure the base program after the search and report how far the machine
  // drifted. A latency claim is only as good as this number.
  bool measure_baseline_drift{true};
  DiscoveryOptions discovery;
};

struct CandidateRecord {
  std::size_t index{};
  // Index of the candidate this one was derived from; the baseline points at
  // itself.
  std::size_t parent{};
  std::uint32_t depth{};
  std::vector<Transform> transforms;
  std::string program_fingerprint;
  std::string candidate_fingerprint;
  std::string backend;
  std::string device;
  std::uint32_t warmups{};
  std::uint32_t iterations{};
  // Preparation is outside the hot latency objective. The actual allocation
  // and free-memory readings are retained beside the planner's estimate so a
  // candidate cannot be promoted on planned memory alone.
  double preparation_milliseconds{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
  std::uint64_t free_bytes_before{};
  std::uint64_t free_bytes_after{};
  std::uint64_t resident_bytes{};
  MemoryFootprint memory;
  NumericalMeasurement numerics;
  Verdict verdict{Verdict::RejectedExecution};
  bool accepted{};
  bool winner{};
  // Verifier or runtime failure text for a rejected candidate.
  std::string diagnostic;
};

struct SearchResult {
  std::string base_program_fingerprint;
  std::string base_fingerprint;
  std::string backend;
  std::string device;
  std::string reference_backend;
  Objective objective{Objective::Latency};
  AcceptanceBars bars;
  // Index 0 is always the baseline.
  std::vector<CandidateRecord> candidates;
  std::size_t winner{};
  // True when the winner beat the baseline by the improvement margin.
  bool improved{};
  std::uint32_t discovered_transforms{};
  // Minimum baseline latency re-measured after every candidate, and its ratio
  // to the baseline measured before them. A ratio far from one means the
  // machine drifted during the search and latency comparisons between
  // candidates carry at least that much error.
  double baseline_recheck_minimum_milliseconds{};
  double baseline_drift_ratio{1.0};
  // The relative error any latency comparison in this search carries, taken
  // from the spread across the baseline's timed iterations and from the drift
  // between the baseline measured before the candidates and after them. With
  // one iteration per candidate the within-run term vanishes and the bound is
  // an underestimate, so a latency claim wants at least a few iterations.
  double latency_noise_bound{1.0};
  // The margin and tolerance actually used for winner selection, after being
  // widened to the noise bound. Reported so a result states the rule it was
  // decided by.
  double effective_improvement_margin{};
  double effective_latency_tolerance{};
  RewriteContext optimized;
  OptimizationPlan plan;
};

// Runs the optimization search.
//
// Reference outputs default to the portable typed CPU executor running the base
// program. A caller may instead supply independently captured source outputs;
// every declared program output must then be present and no non-output tensor
// may be named. The search fails when the base program cannot clear the fixed
// numerical bars on the measurement backend, because no comparison after that
// would mean anything.
//
// `database`, when non-null, receives one measurement per candidate.
SearchResult optimize(
    const RewriteContext &base, runtime::Executor &executor,
    const AcceptanceGate &gate, const SearchOptions &options,
    tune::Database *database,
    const runtime::TensorMap *trusted_reference_outputs = nullptr,
    std::string_view trusted_reference_backend = "external");

// Full JSON record of a search: every candidate's transform sequence,
// fingerprints, timings, memory, numerics, and verdict.
std::string serialize_journal(const SearchResult &result);

} // namespace dif::opt
