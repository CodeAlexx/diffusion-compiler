// Stage cache tests: a cached stage is restored from the cache directory on
// an identical key and never started; any key-file or argv change misses.
#include "dif/bench/process.hpp"
#include "dif/bench/recipe.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

std::filesystem::path workspace() {
  const auto path = std::filesystem::temp_directory_path() / "dif-bench-tests";
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

void write(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

std::string read(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

} // namespace

int main() {
  try {
    const auto base = workspace();
    write(base / "prompt.txt", "a lamb\n");
    write(base / "key.txt", "v1\n");
    const std::string recipe_text = R"({
  "kind": "diffusion-compiler-benchmark-recipe", "version": 1,
  "name": "cache-test", "output_kind": "image", "description": "stage cache test",
  "workload": {}, "variables": {"KEY": "${recipe_dir}/key.txt"},
  "prompt": {"file": "${recipe_dir}/prompt.txt"},
  "inputs": [], "model_files": [], "required_files": [], "required_tools": ["sh"],
  "stages": [
    {"name": "encode", "after": [],
     "cache": {"key": ["${prompt_file}", "${KEY}"], "outputs": ["${workdir}/cond.txt"]},
     "argv": ["sh", "-c", "cat ${prompt_file} ${KEY} > ${workdir}/cond.txt; echo ran-encode >> ${workdir}/marker.txt"]},
    {"name": "decode", "argv": ["sh", "-c", "cat ${workdir}/cond.txt > ${workdir}/out.txt"]}
  ],
  "output": "${workdir}/out.txt"
})";
    write(base / "recipe.json", recipe_text);
    const auto recipe = dif::bench::parse_recipe(base / "recipe.json");
    expect(recipe.stages.size() == 2U && recipe.stages[0].cacheable &&
               !recipe.stages[1].cacheable,
           "cache declaration parsed on the encode stage only");

    dif::bench::StageCachePolicy policy;
    policy.directory = base / "cache";
    const auto run = [&](const std::string &name, bool cached) {
      dif::bench::ResolveContext context;
      context.build_directory = base;
      context.work_directory = base / name;
      std::filesystem::create_directories(context.work_directory);
      const auto resolved = dif::bench::resolve_recipe(recipe, context);
      expect(resolved.stages[0].cache_key_files.size() == 2U &&
                 resolved.stages[0].cache_outputs.size() == 1U,
             "cache key files and outputs resolve");
      return dif::bench::run_chain(resolved, context.work_directory / "logs", {},
                                   cached ? policy : dif::bench::StageCachePolicy{});
    };

    const auto disabled = run("run-disabled", false);
    expect(disabled.success && disabled.stages[0].cache_status == "disabled",
           "without a cache directory the declaration is inert");
    const auto first = run("run-1", true);
    expect(first.success && first.stages[0].cache_status == "miss" &&
               first.stages[1].cache_status == "none",
           "first cached run misses and stores");
    expect(std::filesystem::exists(base / "run-1" / "marker.txt"),
           "first run executed the encode process");
    const auto second = run("run-2", true);
    expect(second.success && second.stages[0].cache_status == "hit" &&
               !second.stages[0].cache_key.empty() &&
               second.stages[0].cache_key == first.stages[0].cache_key,
           "second cached run hits with the same key");
    expect(!std::filesystem::exists(base / "run-2" / "marker.txt"),
           "a hit never starts the process");
    expect(read(base / "run-2" / "out.txt") == read(base / "run-1" / "out.txt") &&
               read(base / "run-2" / "out.txt") == "a lamb\nv1\n",
           "restored outputs feed the dependent stage byte-identically");
    write(base / "key.txt", "v2\n");
    const auto third = run("run-3", true);
    expect(third.success && third.stages[0].cache_status == "miss" &&
               third.stages[0].cache_key != first.stages[0].cache_key &&
               read(base / "run-3" / "out.txt") == "a lamb\nv2\n",
           "a changed key file misses and recomputes");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: exception: " << error.what() << "\n";
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " bench test failure(s)\n";
    return 1;
  }
  std::cout << "BENCH_TESTS PASS\n";
  return 0;
}
