#pragma once

#include "dif/opt/rewrite.hpp"
#include "dif/opt/transform.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace dif::opt {

// The reproducible description of one optimized program. A plan plus the base
// program and its bindings is sufficient to rebuild the winning candidate
// byte-for-byte on a clean checkout, which is what makes an optimization result
// auditable rather than anecdotal.
struct OptimizationPlan {
  // ir::fingerprint of the base program, matching the tuning database's
  // program_hash column.
  std::string base_program_fingerprint;
  // Fingerprint of the base program together with its bound constant values.
  std::string base_fingerprint;
  std::string candidate_program_fingerprint;
  std::string candidate_fingerprint;
  std::vector<Transform> transforms;
};

std::string serialize_plan(const OptimizationPlan &plan);
OptimizationPlan parse_plan(std::string_view text);
void write_plan(const OptimizationPlan &plan,
                const std::filesystem::path &path);
OptimizationPlan read_plan(const std::filesystem::path &path);

// Rebuilds the planned candidate from a base context. Fails when the base does
// not match the fingerprint the plan was recorded against, when a transform is
// no longer legal, or when the rebuilt candidate does not reproduce the
// recorded candidate fingerprint.
RewriteContext replay(const OptimizationPlan &plan,
                      const RewriteContext &base);

// Applies the transformation *strategy* from a measured plan to another
// program by clearing every recorded operation/tensor scope before each apply.
// This is deliberately distinct from replay: it does not claim fingerprint
// identity and is only appropriate when the recorded scopes represented all
// legal sites (for example, promoting a one-block all-site strategy to a short
// recurrence chain). Legality is rechecked on the target program.
RewriteContext apply_global_strategy(const OptimizationPlan &plan,
                                     const RewriteContext &target);

// JSON string literal including surrounding quotes, with control characters
// escaped. Shared by the plan writer and the search journal writer.
std::string json_quote(std::string_view text);
// Shortest round-trippable JSON number for a double, or null for a non-finite
// value, so a journal never emits invalid JSON for an overflowing measurement.
std::string json_number(double value);

} // namespace dif::opt
