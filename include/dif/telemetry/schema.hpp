#pragma once

#include "dif/runtime/executor.hpp"
#include "dif/target/profile.hpp"
#include "dif/telemetry/document.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace dif::telemetry {

inline constexpr std::string_view kSchemaName =
    "diffusion-compiler-telemetry";
inline constexpr std::uint32_t kSchemaVersion = 1U;

// Every document starts with the same head: `schema`, `kind`, and
// `provenance`. Tools append their own sections after it; shared sections
// keep the same field names in every kind so an agent reads one vocabulary.
Object make_document(std::string_view kind);

// ISO-8601 UTC wall-clock timestamp for `generated_at` fields.
std::string utc_timestamp_now();

Object provenance_section();
Object hardware_section(const target::TargetProfile &profile);
Object runtime_budget_section(const target::RuntimeBudget &budget);
Object launch_telemetry_section(const runtime::LaunchTelemetry &telemetry);
Object pipeline_profile_section(const runtime::PipelineProfile &profile);
Array operation_timings_section(
    const std::vector<runtime::OperationTiming> &timings);
Array trace_events_section(const std::vector<runtime::TraceEvent> &events);
// Per-category count/bytes/host-milliseconds roll-up of a trace event list.
Object trace_attribution_section(
    const std::vector<runtime::TraceEvent> &events);

// Phase-A document kind `device-probe`.
std::string serialize_probe(const target::TargetProfile &profile,
                            const target::RuntimeBudget &budget);

// Document kind `runtime-trace`: one prepared execution's run() with the
// shared runtime sections. The runtime writes this itself when the
// DIF_TRACE_FILE environment variable names a sink.
Object runtime_trace_document(const runtime::RunResult &result,
                              std::string_view program_fingerprint,
                              std::uint64_t operation_count,
                              const runtime::RunOptions &options);

} // namespace dif::telemetry
