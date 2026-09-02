#pragma once

// Fresh-process stage execution with dependency ordering. Every stage is a
// child process; the runner records wall time, exit status, and rusage per
// stage, and the complete wall from the first exec to the last exit.

#include "dif/bench/recipe.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace dif::bench {

struct StageResult {
  std::string name;
  bool started{};
  bool finished{};
  int exit_status{-1};
  bool signaled{};
  int signal{};
  double start_offset_seconds{};
  double wall_seconds{};
  double user_cpu_seconds{};
  double system_cpu_seconds{};
  std::uint64_t max_rss_kib{};
  std::uint64_t major_faults{};
  std::uint64_t minor_faults{};
  // rusage block counts (512-byte units) charged to the child.
  std::uint64_t filesystem_input_blocks{};
  std::uint64_t filesystem_output_blocks{};
  std::vector<std::string> concurrent_with;
  std::filesystem::path log;
};

struct ChainResult {
  bool success{};
  std::string failed_stage;
  double wall_seconds{};
  std::vector<StageResult> stages;
};

// Runs the resolved stages under the given log directory. `before_stage`
// may add stage-specific environment (diftrace uses it for per-stage trace
// sinks); it receives the stage index and returns extra environment pairs.
ChainResult run_chain(
    const ResolvedRecipe &resolved, const std::filesystem::path &log_directory,
    const std::function<std::vector<std::pair<std::string, std::string>>(
        std::size_t)> &before_stage = {});

struct CaptureResult {
  int exit_status{-1};
  std::string output;
};

// Runs one program and captures its stdout (stderr passes through). Used for
// untimed helpers such as difprobe and ffprobe.
CaptureResult run_capture(const std::vector<std::string> &argv);

} // namespace dif::bench
