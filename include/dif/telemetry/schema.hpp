#pragma once

#include "dif/target/profile.hpp"

#include <string>
#include <string_view>

namespace dif::telemetry {

inline constexpr std::string_view kSchemaName =
    "diffusion-compiler-telemetry";
inline constexpr std::uint32_t kSchemaVersion = 1U;

// Phase-A document. Later benchmark/trace/plan/quality documents use the same
// top-level schema, provenance, hardware, and runtime_budget field names.
std::string serialize_probe(const target::TargetProfile &profile,
                            const target::RuntimeBudget &budget);

} // namespace dif::telemetry
