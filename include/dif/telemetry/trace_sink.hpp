#pragma once

// Environment-driven runtime trace sink. Any tool that executes through the
// shared runtime can be traced without new command-line flags: when
// DIF_TRACE_FILE names a path, every prepared execution appends one compact
// runtime-trace document per run() to it (JSON lines). DIF_NVTX=1 requests
// NVTX ranges the same way. Neither changes what the runtime submits.

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <filesystem>
#include <string>

namespace dif::telemetry {

inline constexpr const char *kTraceFileVariable = "DIF_TRACE_FILE";
inline constexpr const char *kNvtxVariable = "DIF_NVTX";

std::filesystem::path trace_sink_path();
bool trace_events_requested(const runtime::RunOptions &options);
bool nvtx_ranges_requested(const runtime::RunOptions &options);

// Appends the run's runtime-trace document when a sink is configured.
// Failures to write are reported on stderr and never fail the run.
void append_runtime_trace(const runtime::RunResult &result,
                          const ir::Program &program,
                          const runtime::RunOptions &options);

} // namespace dif::telemetry
