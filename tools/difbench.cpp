// difbench: the canonical literal-prompt-to-saved-output benchmark boundary.
//
// A recipe names the fresh-process chain for one model workload; difbench
// owns the timer, the process boundaries, the cache-condition evidence, and
// the saved-artifact check. Total complete wall is the only acceptance
// metric; per-stage timings are diagnostics and are never summed into a
// replacement total.

#include "dif/bench/report.hpp"
#include "dif/support/error.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: difbench run RECIPE.json --workdir DIR [--build DIR]\n"
         "                    [--prompt-file FILE] [--set VAR=VALUE ...]\n"
         "                    [--repeat N] [--cooldown-seconds S]\n"
         "                    [--drop-file-cache] [--digest-model-files]\n"
         "                    [--no-ffprobe] [--gpu-lock FILE] [--label TEXT]\n"
         "                    [--json] [--report FILE]\n"
         "       difbench show RECIPE.json [--workdir DIR] [--build DIR]\n"
         "                    [--prompt-file FILE] [--set VAR=VALUE ...] [--json]\n"
         "       difbench inspect OUTPUT.png|OUTPUT.mp4 [--no-ffprobe] [--json]\n"
         "\n"
         "The timer starts before the first stage process is created and stops\n"
         "when the last stage exits; the saved output is then verified. Each\n"
         "repetition uses DIR/run-N. Defaults: one run, 10 s cooldown, ffprobe\n"
         "when installed. Keep repetitions few: the GPU power cap is deliberate.\n";
}

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

struct Options {
  std::string command;
  std::filesystem::path recipe;
  std::filesystem::path workdir;
  std::filesystem::path build;
  std::filesystem::path prompt_file;
  std::map<std::string, std::string> overrides;
  std::uint32_t repeat{1U};
  double cooldown_seconds{10.0};
  bool drop_file_cache{};
  bool digest_model_files{};
  bool ffprobe{true};
  std::filesystem::path gpu_lock;
  std::string label;
  bool json{};
  std::filesystem::path report;
  std::filesystem::path artifact;
};

Options parse(int argc, char **argv) {
  Options options;
  if (argc < 3) {
    usage();
    std::exit(2);
  }
  options.command = argv[1];
  if (options.command == "inspect")
    options.artifact = argv[2];
  else
    options.recipe = argv[2];
  for (int index = 3; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value after " + option);
      return argv[++index];
    };
    if (option == "--workdir")
      options.workdir = value();
    else if (option == "--build")
      options.build = value();
    else if (option == "--prompt-file")
      options.prompt_file = value();
    else if (option == "--set") {
      const auto text = value();
      const auto split = text.find('=');
      if (split == std::string::npos || split == 0U)
        dif::fail("--set expects VAR=VALUE");
      options.overrides[text.substr(0, split)] = text.substr(split + 1U);
    } else if (option == "--repeat")
      options.repeat = static_cast<std::uint32_t>(number(value(), "repeat"));
    else if (option == "--cooldown-seconds")
      options.cooldown_seconds = std::stod(value());
    else if (option == "--drop-file-cache")
      options.drop_file_cache = true;
    else if (option == "--digest-model-files")
      options.digest_model_files = true;
    else if (option == "--no-ffprobe")
      options.ffprobe = false;
    else if (option == "--gpu-lock")
      options.gpu_lock = value();
    else if (option == "--label")
      options.label = value();
    else if (option == "--json")
      options.json = true;
    else if (option == "--report")
      options.report = value();
    else if (option == "--help" || option == "-h") {
      usage();
      std::exit(0);
    } else
      dif::fail("unknown difbench option: " + option);
  }
  if (options.build.empty()) {
    // Default to the directory holding this binary, which is the build tree.
    char buffer[4096];
    const auto count = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1U);
    if (count > 0) {
      buffer[count] = '\0';
      options.build = std::filesystem::path(buffer).parent_path();
    } else {
      options.build = std::filesystem::current_path();
    }
  }
  return options;
}

bool ffprobe_available() {
  const char *path = std::getenv("PATH");
  if (!path)
    return false;
  std::stringstream entries(path);
  std::string entry;
  while (std::getline(entries, entry, ':'))
    if (!entry.empty() &&
        std::filesystem::exists(std::filesystem::path(entry) / "ffprobe"))
      return true;
  return false;
}

dif::bench::ResolvedRecipe resolve(const Options &options,
                                   const dif::bench::Recipe &recipe,
                                   const std::filesystem::path &workdir) {
  dif::bench::ResolveContext context;
  context.build_directory = options.build;
  context.work_directory = workdir;
  context.prompt_file = options.prompt_file;
  context.overrides = options.overrides;
  return dif::bench::resolve_recipe(recipe, context);
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    dif::fail("cannot write " + path.string());
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string seconds(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  return out.str();
}

int command_inspect(const Options &options) {
  const auto facts = dif::bench::inspect_artifact(options.artifact);
  auto document = dif::telemetry::make_document("artifact");
  document.set("artifact",
               dif::bench::artifact_section(
                   options.artifact, facts,
                   options.ffprobe && ffprobe_available()));
  if (options.json) {
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return facts.exists ? 0 : 1;
  }
  std::cout << "path      " << options.artifact.string() << "\n"
            << "exists    " << (facts.exists ? "yes" : "no") << "\n"
            << "bytes     " << facts.bytes << "\n"
            << "sha256    " << facts.sha256 << "\n"
            << "format    " << facts.format << "\n";
  if (facts.format == "png")
    std::cout << "image     " << facts.width << "x" << facts.height
              << " depth=" << facts.bit_depth << " color_type="
              << facts.color_type << "\n";
  if (facts.format == "mp4")
    std::cout << "video     brand=" << facts.major_brand
              << " duration=" << seconds(facts.duration_seconds)
              << " s tracks=" << facts.track_count << "\n";
  return facts.exists ? 0 : 1;
}

int command_show(const Options &options) {
  const auto recipe = dif::bench::parse_recipe(options.recipe);
  const auto workdir =
      options.workdir.empty() ? std::filesystem::path("WORKDIR")
                              : options.workdir;
  const auto resolved = resolve(options, recipe, workdir);
  const auto problems = dif::bench::preflight(resolved);
  if (options.json) {
    auto document = dif::telemetry::make_document("benchmark-recipe");
    document.set("recipe", dif::bench::recipe_section(resolved));
    document.set("workload", resolved.recipe.workload);
    document.set("inputs", dif::bench::inputs_section(resolved, false));
    dif::telemetry::Array stages;
    for (const auto &stage : resolved.stages) {
      dif::telemetry::Object entry;
      entry.set("name", stage.name);
      dif::telemetry::Array after;
      for (const auto dependency : stage.after)
        after.push_back(resolved.stages[dependency].name);
      entry.set("after", std::move(after));
      dif::telemetry::Array argv;
      for (const auto &argument : stage.argv)
        argv.push_back(argument);
      entry.set("argv", std::move(argv));
      stages.push_back(std::move(entry));
    }
    document.set("stages", std::move(stages));
    document.set("output", resolved.output.string());
    dif::telemetry::Array preflight;
    for (const auto &problem : problems) {
      dif::telemetry::Object entry;
      entry.set("kind", problem.kind);
      entry.set("subject", problem.subject);
      preflight.push_back(std::move(entry));
    }
    document.set("preflight", std::move(preflight));
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return problems.empty() ? 0 : 2;
  }
  std::cout << "recipe    " << resolved.recipe.name << " ("
            << resolved.recipe.output_kind << ")\n"
            << "output    " << resolved.output.string() << "\n"
            << "prompt    " << resolved.prompt_file.string() << "\n";
  for (const auto &stage : resolved.stages) {
    std::cout << "stage     " << stage.name;
    if (!stage.after.empty()) {
      std::cout << " (after";
      for (const auto dependency : stage.after)
        std::cout << " " << resolved.stages[dependency].name;
      std::cout << ")";
    }
    std::cout << "\n";
    for (const auto &argument : stage.argv)
      std::cout << "            " << argument << "\n";
  }
  for (const auto &problem : problems)
    std::cout << "PREFLIGHT " << problem.kind << " " << problem.subject << "\n";
  std::cout << (problems.empty() ? "preflight OK\n" : "preflight FAILED\n");
  return problems.empty() ? 0 : 2;
}

int command_run(const Options &options) {
  if (options.workdir.empty())
    dif::fail("difbench run requires --workdir");
  if (options.repeat == 0U)
    dif::fail("--repeat must be at least 1");
  const auto recipe = dif::bench::parse_recipe(options.recipe);
  const auto base = std::filesystem::absolute(options.workdir);
  std::error_code error;
  if (std::filesystem::exists(base, error) &&
      !std::filesystem::is_empty(base, error))
    dif::fail("refusing nonempty work directory: " + base.string());
  std::filesystem::create_directories(base);

  auto document = dif::telemetry::make_document(dif::telemetry::kind::benchmark);
  auto first = resolve(options, recipe, base / "run-1");
  const auto problems = dif::bench::preflight(first);
  {
    dif::telemetry::Object benchmark;
    benchmark.set("boundary", "literal-prompt-to-saved-output");
    benchmark.set("acceptance_metric", "complete_wall_seconds");
    benchmark.set("label", options.label);
    benchmark.set("host", dif::bench::hostname());
    benchmark.set("build_directory", first.build_directory.string());
    benchmark.set("work_directory", base.string());
    benchmark.set("recipe", dif::bench::recipe_section(first));
    benchmark.set("workload", first.recipe.workload);
    if (first.recipe.comparator.present) {
      dif::telemetry::Object comparator;
      comparator.set("name", first.recipe.comparator.name);
      comparator.set("wall_seconds", first.recipe.comparator.wall_seconds);
      comparator.set("target_ratio", first.recipe.comparator.target_ratio);
      benchmark.set("comparator", std::move(comparator));
    } else {
      benchmark.set("comparator", nullptr);
    }
    document.set("benchmark", std::move(benchmark));
  }
  document.set("inputs",
               dif::bench::inputs_section(first, options.digest_model_files));
  {
    dif::telemetry::Array preflight;
    for (const auto &problem : problems) {
      dif::telemetry::Object entry;
      entry.set("kind", problem.kind);
      entry.set("subject", problem.subject);
      preflight.push_back(std::move(entry));
    }
    document.set("preflight", std::move(preflight));
  }
  if (!problems.empty()) {
    document.set("status", "refused");
    const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
    write_text(base / "difbench.json", text);
    if (!options.report.empty())
      write_text(options.report, text);
    if (options.json)
      std::cout << text;
    else {
      for (const auto &problem : problems)
        std::cerr << "PREFLIGHT " << problem.kind << " " << problem.subject
                  << "\n";
      std::cerr << "difbench: refused; the recipe is not runnable on this "
                   "host\n";
    }
    return 2;
  }

  int lock_descriptor = -1;
  if (!options.gpu_lock.empty()) {
    lock_descriptor = ::open(options.gpu_lock.c_str(), O_RDWR | O_CREAT, 0644);
    if (lock_descriptor < 0 || ::flock(lock_descriptor, LOCK_EX | LOCK_NB) != 0)
      dif::fail("benchmark GPU lock is held: " + options.gpu_lock.string());
  }

  dif::telemetry::Object hardware;
  dif::telemetry::Object budget;
  std::string probe_source;
  dif::bench::probe_sections(first.build_directory, hardware, budget,
                             probe_source);
  document.set("hardware", std::move(hardware));
  document.set("runtime_budget", std::move(budget));
  {
    dif::telemetry::Object conditions;
    conditions.set("probe_source", probe_source);
    conditions.set("process", dif::telemetry::condition::fresh_process);
    conditions.set("repeat", options.repeat);
    conditions.set("cooldown_seconds", options.cooldown_seconds);
    conditions.set("drop_file_cache", options.drop_file_cache);
    conditions.set("gpu_lock", options.gpu_lock.string());
    document.set("conditions", std::move(conditions));
  }

  dif::bench::RunSettings settings;
  settings.drop_file_cache = options.drop_file_cache;
  settings.ffprobe = options.ffprobe && ffprobe_available();
  settings.digest_model_files = options.digest_model_files;

  dif::telemetry::Array runs;
  std::vector<double> completed_walls;
  std::string first_filesystem_condition;
  for (std::uint32_t index = 1; index <= options.repeat; ++index) {
    if (index > 1U && options.cooldown_seconds > 0.0)
      std::this_thread::sleep_for(
          std::chrono::duration<double>(options.cooldown_seconds));
    const auto resolved = resolve(options, recipe, base / ("run-" + std::to_string(index)));
    if (!options.json)
      std::cerr << "difbench: run " << index << "/" << options.repeat
                << " starting (" << resolved.stages.size() << " stages)\n";
    const auto record = dif::bench::execute_run(resolved, settings);
    if (index == 1U)
      first_filesystem_condition = record.residency_before.condition;
    auto section = dif::bench::run_section(record, resolved, settings);
    section.set("index", index);
    if (record.status == "completed")
      completed_walls.push_back(record.wall_seconds);
    if (!options.json) {
      std::cerr << "difbench: run " << index << " " << record.status
                << " wall=" << seconds(record.wall_seconds) << " s";
      if (!record.chain.failed_stage.empty())
        std::cerr << " failed_stage=" << record.chain.failed_stage;
      std::cerr << "\n";
    }
    runs.push_back(std::move(section));
    // Preserve evidence after every run so an interrupted series still
    // leaves a valid document behind.
    auto partial = document;
    partial.set("runs", runs);
    write_text(base / "difbench.json",
               dif::telemetry::serialize(dif::telemetry::Value(partial)));
  }
  document.set("runs", std::move(runs));

  dif::telemetry::Object summary;
  summary.set("completed_runs", completed_walls.size());
  summary.set("requested_runs", options.repeat);
  summary.set("filesystem_condition_first_run", first_filesystem_condition);
  double minimum = 0.0;
  double median = 0.0;
  double maximum = 0.0;
  if (!completed_walls.empty()) {
    auto sorted = completed_walls;
    std::sort(sorted.begin(), sorted.end());
    minimum = sorted.front();
    maximum = sorted.back();
    median = sorted.size() % 2U == 1U
                 ? sorted[sorted.size() / 2U]
                 : 0.5 * (sorted[sorted.size() / 2U - 1U] +
                          sorted[sorted.size() / 2U]);
  }
  dif::telemetry::Object walls;
  walls.set("minimum",
            dif::telemetry::nullable_number(!completed_walls.empty(), minimum));
  walls.set("median",
            dif::telemetry::nullable_number(!completed_walls.empty(), median));
  walls.set("maximum",
            dif::telemetry::nullable_number(!completed_walls.empty(), maximum));
  dif::telemetry::Array all;
  for (const auto wall : completed_walls)
    all.push_back(wall);
  walls.set("all", std::move(all));
  summary.set("complete_wall_seconds", std::move(walls));
  std::string verdict;
  if (completed_walls.empty()) {
    verdict = "no run completed with a valid saved output";
    summary.set("comparator", nullptr);
  } else {
    verdict = std::to_string(completed_walls.size()) + " of " +
              std::to_string(options.repeat) +
              " runs completed; minimum complete wall " + seconds(minimum) +
              " s";
    if (first.recipe.comparator.present) {
      dif::telemetry::Object comparator;
      const auto ratio = first.recipe.comparator.wall_seconds / minimum;
      comparator.set("name", first.recipe.comparator.name);
      comparator.set("wall_seconds", first.recipe.comparator.wall_seconds);
      comparator.set("ratio_minimum", ratio);
      comparator.set("ratio_median",
                     first.recipe.comparator.wall_seconds / median);
      verdict += "; " + seconds(ratio) + "x versus " +
                 first.recipe.comparator.name;
      if (first.recipe.comparator.target_ratio > 0.0) {
        const bool meets = ratio >= first.recipe.comparator.target_ratio;
        comparator.set("target_ratio", first.recipe.comparator.target_ratio);
        comparator.set("meets_target", meets);
        comparator.set("target_wall_seconds",
                       first.recipe.comparator.wall_seconds /
                           first.recipe.comparator.target_ratio);
        verdict += "; target " + seconds(first.recipe.comparator.target_ratio) +
                   "x " + (meets ? "MET" : "NOT met");
      }
      summary.set("comparator", std::move(comparator));
    } else {
      summary.set("comparator", nullptr);
    }
  }
  summary.set("verdict", verdict);
  document.set("summary", std::move(summary));
  document.set("status", completed_walls.empty() ? "failed" : "completed");

  const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
  write_text(base / "difbench.json", text);
  if (!options.report.empty())
    write_text(options.report, text);
  if (lock_descriptor >= 0)
    ::close(lock_descriptor);
  if (options.json) {
    std::cout << text;
  } else {
    std::cout << "DIFBENCH recipe=" << first.recipe.name
              << " kind=" << first.recipe.output_kind
              << " runs=" << completed_walls.size() << "/" << options.repeat
              << " filesystem=" << first_filesystem_condition << "\n";
    for (const auto &run : document.find("runs")->array()) {
      const auto &object = run.object();
      std::cout << "RUN index=" << object.find("index")->number()
                << " status=" << object.find("status")->string()
                << " wall_s=" << seconds(object.find("wall_seconds")->number())
                << "\n";
      for (const auto &stage : object.find("stages")->array()) {
        const auto &entry = stage.object();
        std::cout << "  STAGE " << std::left << std::setw(22)
                  << entry.find("name")->string() << " wall_s="
                  << seconds(entry.find("wall_seconds")->number())
                  << " start_s="
                  << seconds(entry.find("start_offset_seconds")->number())
                  << " exit=" << entry.find("exit_status")->number() << "\n";
      }
      const auto &output = run.object().find("output")->object();
      std::cout << "  OUTPUT format=" << output.find("format")->string()
                << " bytes=" << output.find("bytes")->number()
                << " sha256=" << output.find("sha256")->string() << "\n";
      const auto &gpu = run.object().find("gpu")->object();
      if (gpu.find("available")->boolean())
        std::cout << "  GPU peak_used_bytes="
                  << gpu.find("peak_used_memory_bytes")->number()
                  << " mean_w=" << seconds(gpu.find("mean_power_watts")->number())
                  << " max_w=" << seconds(gpu.find("max_power_watts")->number())
                  << " limit_w="
                  << seconds(gpu.find("power_limit_watts")->number()) << "\n";
    }
    std::cout << "SUMMARY " << verdict << "\n"
              << "report  " << (base / "difbench.json").string() << "\n";
  }
  return completed_walls.empty() ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    if (options.command == "run")
      return command_run(options);
    if (options.command == "show")
      return command_show(options);
    if (options.command == "inspect")
      return command_inspect(options);
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "difbench: " << error.what() << "\n";
    return 1;
  }
}
