#pragma once

// A benchmark recipe is a model-specific validation fixture: the literal
// process chain that turns a prompt into a saved PNG or MP4. The tools that
// consume recipes (difbench, diftrace) stay model-neutral; every model fact
// lives in the recipe file and is recorded verbatim into the report.

#include "dif/telemetry/document.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dif::bench {

inline constexpr const char *kRecipeKind =
    "diffusion-compiler-benchmark-recipe";
inline constexpr int kRecipeVersion = 1;

struct RecipeStage {
  std::string name;
  std::vector<std::string> argv;
  // Names of stages that must have exited successfully before this one may
  // start. Absent in the file means "the previous stage"; an explicit empty
  // list means "may start immediately".
  std::vector<std::string> after;
  std::string cwd;
  std::vector<std::pair<std::string, std::string>> environment;
  // Optional stage cache declaration. `key` lists the files whose identity
  // (plus this stage's argv) determines the stage's outputs; `outputs` lists
  // the files the stage produces. The cache only acts when the runner is
  // given a cache directory (difbench/diftrace --stage-cache DIR); without
  // one the declaration is inert and the historical fresh-process chain
  // runs. A hit is recorded on the stage record and in the run conditions
  // so a cached wall is never mistaken for a cold one.
  struct Cache {
    std::vector<std::string> key;
    std::vector<std::string> outputs;
  };
  bool cacheable{};
  Cache cache;
};

struct RecipeInput {
  std::string name;
  std::string path;
};

struct Comparator {
  bool present{};
  std::string name;
  double wall_seconds{};
  double target_ratio{};
};

struct Recipe {
  std::filesystem::path path;
  std::string name;
  std::string output_kind; // "image" or "video"
  std::string description;
  // Free-form identity facts recorded verbatim into every report.
  telemetry::Object workload;
  Comparator comparator;
  std::map<std::string, std::string> variables;
  std::string prompt_file;
  std::vector<RecipeInput> inputs;
  std::vector<std::string> model_files;
  std::vector<std::string> required_files;
  std::vector<std::string> required_tools;
  std::vector<std::pair<std::string, std::string>> environment;
  std::vector<RecipeStage> stages;
  std::string output;
};

Recipe parse_recipe(const std::filesystem::path &path);

struct ResolveContext {
  std::filesystem::path build_directory;
  std::filesystem::path work_directory;
  std::filesystem::path prompt_file;
  std::map<std::string, std::string> overrides;
};

struct ResolvedStage {
  std::string name;
  std::vector<std::string> argv;
  std::vector<std::size_t> after;
  std::filesystem::path cwd;
  std::vector<std::pair<std::string, std::string>> environment;
  bool cacheable{};
  std::vector<std::filesystem::path> cache_key_files;
  std::vector<std::filesystem::path> cache_outputs;
};

struct ResolvedRecipe {
  Recipe recipe;
  std::filesystem::path build_directory;
  std::filesystem::path work_directory;
  std::filesystem::path prompt_file;
  std::filesystem::path output;
  std::map<std::string, std::string> variables;
  std::vector<RecipeInput> inputs;
  std::vector<std::filesystem::path> model_files;
  std::vector<std::filesystem::path> required_files;
  std::vector<ResolvedStage> stages;
};

// Substitutes ${build}, ${workdir}, ${recipe_dir}, ${prompt_file},
// ${output}, ${input:NAME}, and ${VARIABLE} everywhere, validates stage
// names and dependencies, and fails on any unknown reference or cycle.
ResolvedRecipe resolve_recipe(const Recipe &recipe,
                              const ResolveContext &context);

struct PreflightProblem {
  std::string kind; // "missing-file", "missing-tool", "missing-binary"
  std::string subject;
};

std::vector<PreflightProblem> preflight(const ResolvedRecipe &resolved);

} // namespace dif::bench
