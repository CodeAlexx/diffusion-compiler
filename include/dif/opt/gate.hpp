#pragma once

#include "dif/runtime/executor.hpp"

#include <cstdint>
#include <string_view>

namespace dif::opt {

// TRUSTED INFRASTRUCTURE.
//
// This header and gate.cpp define the acceptance oracle. Optimization code
// consumes an AcceptanceGate through a const reference and has no way to relax
// the bars it was constructed with: the bars are captured at construction, the
// class exposes no mutator, and measure()/judge() are const. Search must not be
// given a non-const gate.

struct NumericalMeasurement {
  // Elements actually compared across all reference outputs.
  std::uint64_t compared_elements{};
  // Elements whose declared-dtype payload differs bit for bit. This is
  // reported even when the numerical contract is tolerance-based, so a BF16
  // gate never silently turns "close" into "exact".
  std::uint64_t exact_mismatch_count{};
  // Non-finite values observed in the candidate outputs.
  std::uint64_t nonfinite_count{};
  double max_absolute_error{};
  double cosine_similarity{1.0};
  // sqrt(sum(candidate^2) / sum(reference^2)).
  double norm_ratio{1.0};
  // sqrt(sum((candidate-reference)^2) / sum(reference^2)).
  double relative_l2{};
};

struct AcceptanceBars {
  double max_absolute_error{1.0e-4};
  double min_cosine_similarity{0.999999};
  double min_norm_ratio{0.9999};
  double max_norm_ratio{1.0001};
  double max_relative_l2{1.0e-3};
  // Hard ceiling on the planned device working set of a candidate.
  std::uint64_t memory_budget_bytes{~std::uint64_t{0}};
};

enum class Verdict : std::uint32_t {
  Accepted = 0,
  RejectedVerify = 1,
  RejectedExecution = 2,
  RejectedNonFinite = 3,
  RejectedNumerical = 4,
  RejectedMemory = 5,
  RejectedNotBetter = 6,
};

std::string_view verdict_name(Verdict verdict);

class AcceptanceGate {
public:
  explicit AcceptanceGate(const AcceptanceBars &bars);

  const AcceptanceBars &bars() const noexcept { return bars_; }

  // Compares candidate outputs against the reference outputs of the base
  // program. Fails hard when the candidate omits an output or changes an output
  // shape: a candidate that does not produce the program's declared results is
  // never merely "less accurate".
  NumericalMeasurement measure(const runtime::TensorMap &reference,
                               const runtime::TensorMap &candidate) const;

  // Applies the acceptance order that follows execution: non-finite values,
  // then the numerical bars, then the memory budget. Performance is compared by
  // the search only for candidates this returns Accepted for.
  Verdict judge(const NumericalMeasurement &numerics,
                std::uint64_t planned_memory_bytes) const;

private:
  AcceptanceBars bars_;
};

} // namespace dif::opt
