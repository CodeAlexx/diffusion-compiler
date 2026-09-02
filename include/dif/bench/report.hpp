#pragma once

// Shared benchmark-run mechanics used by difbench and diftrace: one measured
// run of a resolved recipe with page-cache classification, NVML sampling,
// per-stage process accounting, and saved-artifact inspection, rendered as
// telemetry sections with one vocabulary.

#include "dif/bench/artifact.hpp"
#include "dif/bench/gpu_sampler.hpp"
#include "dif/bench/host.hpp"
#include "dif/bench/process.hpp"
#include "dif/bench/recipe.hpp"
#include "dif/telemetry/document.hpp"

#include <functional>
#include <string>
#include <vector>

namespace dif::bench {

struct RunSettings {
  bool drop_file_cache{};
  bool ffprobe{true};
  bool digest_model_files{};
  std::function<std::vector<std::pair<std::string, std::string>>(std::size_t)>
      before_stage;
};

struct RunRecord {
  ChainResult chain;
  std::vector<FileResidency> files_before;
  ResidencySummary residency_before;
  GpuSample gpu;
  ArtifactFacts artifact;
  // "completed", "failed", or "invalid-output".
  std::string status;
  double wall_seconds{};
};

RunRecord execute_run(const ResolvedRecipe &resolved,
                      const RunSettings &settings);

telemetry::Object recipe_section(const ResolvedRecipe &resolved);
telemetry::Object inputs_section(const ResolvedRecipe &resolved,
                                 bool digest_model_files);
telemetry::Object residency_section(const std::vector<FileResidency> &files,
                                    const ResidencySummary &summary);
telemetry::Object stage_section(const StageResult &stage,
                                const ResolvedStage &resolved);
telemetry::Object gpu_section(const GpuSample &sample);
telemetry::Object run_section(const RunRecord &record,
                              const ResolvedRecipe &resolved,
                              const RunSettings &settings);

// Hardware and runtime-budget sections from the build's own difprobe run in
// a fresh process, so the measuring process never holds a CUDA context.
// Falls back to the in-process host probe and says so in `probe_source`.
void probe_sections(const std::filesystem::path &build_directory,
                    telemetry::Object &hardware, telemetry::Object &budget,
                    std::string &probe_source);

std::string file_digest(const std::filesystem::path &path);

} // namespace dif::bench
