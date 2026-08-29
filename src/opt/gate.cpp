// TRUSTED INFRASTRUCTURE: the acceptance oracle.
//
// Nothing in this file consults, or may be made to consult, a transform, a
// candidate program, or a search state. It measures a candidate against the
// base program's reference outputs and applies bars fixed by the caller.
#include "dif/opt/gate.hpp"

#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace dif::opt {

std::string_view verdict_name(Verdict verdict) {
  switch (verdict) {
  case Verdict::Accepted:
    return "accepted";
  case Verdict::RejectedVerify:
    return "rejected_verify";
  case Verdict::RejectedExecution:
    return "rejected_execution";
  case Verdict::RejectedNonFinite:
    return "rejected_nonfinite";
  case Verdict::RejectedNumerical:
    return "rejected_numerical";
  case Verdict::RejectedMemory:
    return "rejected_memory";
  case Verdict::RejectedNotBetter:
    return "rejected_not_better";
  }
  fail("unknown acceptance verdict");
}

AcceptanceGate::AcceptanceGate(const AcceptanceBars &bars) : bars_(bars) {
  if (!(bars_.max_absolute_error >= 0.0) ||
      !(bars_.min_cosine_similarity <= 1.0) ||
      !(bars_.min_norm_ratio <= bars_.max_norm_ratio) ||
      !(bars_.max_relative_l2 >= 0.0))
    fail("acceptance bars are not a usable admission region");
}

NumericalMeasurement
AcceptanceGate::measure(const runtime::TensorMap &reference,
                        const runtime::TensorMap &candidate) const {
  if (reference.empty())
    fail("acceptance gate requires at least one reference output");
  NumericalMeasurement result;
  long double dot = 0.0L;
  long double reference_energy = 0.0L;
  long double candidate_energy = 0.0L;
  long double error_energy = 0.0L;
  for (const auto &[id, expected_tensor] : reference) {
    const auto found = candidate.find(id);
    if (found == candidate.end())
      fail("candidate omitted reference output tensor " + std::to_string(id));
    const auto &actual_tensor = found->second;
    if (expected_tensor.dtype != actual_tensor.dtype ||
        expected_tensor.dims != actual_tensor.dims)
      fail("candidate changed the dtype or shape of output tensor " +
           std::to_string(id));
    const auto elements = expected_tensor.element_count();
    if (elements != actual_tensor.element_count())
      fail("candidate output element count differs for tensor " +
           std::to_string(id));
    result.compared_elements += elements;
    // Outputs are read through the typed scalar accessors so a bf16 or f16
    // result is measured at its declared precision rather than refused.
    for (std::uint64_t index = 0; index < elements; ++index) {
      const auto expected_value = runtime::load_float(expected_tensor, index);
      const auto actual_value = runtime::load_float(actual_tensor, index);
      const auto want = static_cast<long double>(expected_value);
      const auto got = static_cast<long double>(actual_value);
      if (!std::isfinite(actual_value)) {
        ++result.nonfinite_count;
        continue;
      }
      const auto difference = static_cast<double>(got - want);
      result.max_absolute_error =
          std::max(result.max_absolute_error, std::abs(difference));
      dot += want * got;
      reference_energy += want * want;
      candidate_energy += got * got;
      error_energy += (got - want) * (got - want);
    }
  }
  if (result.compared_elements == 0U)
    fail("acceptance gate compared no elements");
  const auto denominator = std::sqrt(reference_energy * candidate_energy);
  result.cosine_similarity =
      denominator == 0.0L
          ? (reference_energy == candidate_energy ? 1.0 : 0.0)
          : static_cast<double>(dot / denominator);
  result.norm_ratio =
      reference_energy == 0.0L
          ? (candidate_energy == 0.0L
                 ? 1.0
                 : std::numeric_limits<double>::infinity())
          : std::sqrt(static_cast<double>(candidate_energy / reference_energy));
  result.relative_l2 =
      reference_energy == 0.0L
          ? (error_energy == 0.0L ? 0.0
                                  : std::numeric_limits<double>::infinity())
          : std::sqrt(static_cast<double>(error_energy / reference_energy));
  return result;
}

Verdict AcceptanceGate::judge(const NumericalMeasurement &numerics,
                              std::uint64_t planned_memory_bytes) const {
  if (numerics.nonfinite_count != 0U)
    return Verdict::RejectedNonFinite;
  if (!std::isfinite(numerics.max_absolute_error) ||
      !std::isfinite(numerics.cosine_similarity) ||
      !std::isfinite(numerics.norm_ratio) ||
      !std::isfinite(numerics.relative_l2))
    return Verdict::RejectedNumerical;
  if (numerics.max_absolute_error > bars_.max_absolute_error ||
      numerics.cosine_similarity < bars_.min_cosine_similarity ||
      numerics.norm_ratio < bars_.min_norm_ratio ||
      numerics.norm_ratio > bars_.max_norm_ratio ||
      numerics.relative_l2 > bars_.max_relative_l2)
    return Verdict::RejectedNumerical;
  if (planned_memory_bytes > bars_.memory_budget_bytes)
    return Verdict::RejectedMemory;
  return Verdict::Accepted;
}

} // namespace dif::opt
