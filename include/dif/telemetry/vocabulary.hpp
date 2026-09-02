#pragma once

// The shared vocabulary every telemetry document uses for document kinds and
// for attributing work. Tools compare against these constants instead of
// inventing local spellings.

#include <string_view>

namespace dif::telemetry {

namespace kind {
inline constexpr std::string_view device_probe = "device-probe";
inline constexpr std::string_view benchmark = "benchmark";
inline constexpr std::string_view trace = "trace";
inline constexpr std::string_view runtime_trace = "runtime-trace";
inline constexpr std::string_view plan_report = "plan-report";
} // namespace kind

// Work attribution categories. A trace event, an aggregate bucket, and a
// benchmark diagnostic all use the same names.
namespace category {
// Process-level stage of a prompt-to-saved-output chain.
inline constexpr std::string_view stage = "stage";
// One executed DiffIR operation (or a fused region standing in for several).
inline constexpr std::string_view operation = "operation";
inline constexpr std::string_view region = "region";
inline constexpr std::string_view gemm = "gemm";
inline constexpr std::string_view attention = "attention";
inline constexpr std::string_view convolution = "convolution";
inline constexpr std::string_view generated_kernel = "generated_kernel";
inline constexpr std::string_view h2d = "h2d";
inline constexpr std::string_view d2h = "d2h";
inline constexpr std::string_view d2d = "d2d";
// Host memcpy from a mapped or owned constant into pinned staging memory.
inline constexpr std::string_view staging = "staging";
// Host blocked on device work (event or stream synchronize).
inline constexpr std::string_view wait = "wait";
// Device-side cross-stream dependency (stream wait event) or event record.
inline constexpr std::string_view synchronization = "synchronization";
inline constexpr std::string_view layout = "layout";
inline constexpr std::string_view allocation = "allocation";
inline constexpr std::string_view filesystem = "filesystem";
inline constexpr std::string_view preparation = "preparation";
} // namespace category

// Cache/process conditions a benchmark must state rather than assume.
namespace condition {
inline constexpr std::string_view fresh_process = "fresh";
inline constexpr std::string_view warm_process = "warm";
inline constexpr std::string_view cold_filesystem = "cold";
inline constexpr std::string_view warm_filesystem = "warm";
inline constexpr std::string_view mixed_filesystem = "mixed";
inline constexpr std::string_view unknown_condition = "unknown";
} // namespace condition

} // namespace dif::telemetry
