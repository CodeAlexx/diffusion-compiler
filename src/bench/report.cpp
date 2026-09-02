#include "dif/bench/report.hpp"

// GCC 13 at -O3 reports a spurious maybe-uninitialized inside std::variant's
// move constructor when a telemetry::Value temporary is pushed into an Array
// that reallocates. The value is fully constructed before the move; the
// diagnostic has no source-level cause, so it is silenced for this file only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "dif/runtime/device_probe.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include <filesystem>

namespace dif::bench {

std::string file_digest(const std::filesystem::path &path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error))
    return {};
  return hex_digest(sha256_file(path));
}

RunRecord execute_run(const ResolvedRecipe &resolved,
                      const RunSettings &settings) {
  RunRecord record;
  std::filesystem::create_directories(resolved.work_directory);
  if (settings.drop_file_cache) {
    for (const auto &file : resolved.model_files)
      drop_file_cache(file);
    for (const auto &input : resolved.inputs)
      drop_file_cache(input.path);
  }
  for (const auto &file : resolved.model_files)
    record.files_before.push_back(measure_residency(file));
  record.residency_before = summarize_residency(record.files_before);
  GpuSampler sampler;
  sampler.start();
  record.chain = run_chain(resolved, resolved.work_directory / "logs",
                           settings.before_stage);
  record.gpu = sampler.stop();
  record.wall_seconds = record.chain.wall_seconds;
  record.artifact = inspect_artifact(resolved.output);
  if (!record.chain.success)
    record.status = "failed";
  else if (!artifact_matches_kind(record.artifact, resolved.recipe.output_kind))
    record.status = "invalid-output";
  else
    record.status = "completed";
  return record;
}

telemetry::Object recipe_section(const ResolvedRecipe &resolved) {
  telemetry::Object out;
  out.set("name", resolved.recipe.name);
  out.set("path", resolved.recipe.path.string());
  out.set("sha256", file_digest(resolved.recipe.path));
  out.set("output_kind", resolved.recipe.output_kind);
  out.set("description", resolved.recipe.description);
  telemetry::Object variables;
  for (const auto &[key, value] : resolved.variables)
    variables.set(key, value);
  out.set("variables", std::move(variables));
  out.set("stage_count", resolved.stages.size());
  telemetry::Array names;
  for (const auto &stage : resolved.stages)
    names.push_back(stage.name);
  out.set("stages", std::move(names));
  return out;
}

telemetry::Object inputs_section(const ResolvedRecipe &resolved,
                                 bool digest_model_files) {
  telemetry::Object out;
  telemetry::Object prompt;
  prompt.set("path", resolved.prompt_file.string());
  std::error_code error;
  if (!resolved.prompt_file.empty() &&
      std::filesystem::is_regular_file(resolved.prompt_file, error)) {
    prompt.set("bytes", std::filesystem::file_size(resolved.prompt_file));
    prompt.set("sha256", file_digest(resolved.prompt_file));
  } else {
    prompt.set("bytes", nullptr);
    prompt.set("sha256", nullptr);
  }
  out.set("prompt", std::move(prompt));
  telemetry::Array files;
  for (const auto &input : resolved.inputs) {
    telemetry::Object entry;
    entry.set("name", input.name);
    entry.set("path", input.path);
    if (std::filesystem::is_regular_file(input.path, error)) {
      entry.set("bytes", std::filesystem::file_size(input.path));
      entry.set("sha256", file_digest(input.path));
    } else {
      entry.set("bytes", nullptr);
      entry.set("sha256", nullptr);
    }
    files.push_back(std::move(entry));
  }
  out.set("files", std::move(files));
  telemetry::Array models;
  for (const auto &file : resolved.model_files) {
    telemetry::Object entry;
    entry.set("path", file.string());
    const bool exists = std::filesystem::is_regular_file(file, error);
    entry.set("exists", exists);
    if (exists) {
      entry.set("bytes", std::filesystem::file_size(file));
      if (digest_model_files)
        entry.set("sha256", file_digest(file));
      else
        entry.set("sha256", nullptr);
    } else {
      entry.set("bytes", nullptr);
      entry.set("sha256", nullptr);
    }
    models.push_back(std::move(entry));
  }
  out.set("model_files", std::move(models));
  return out;
}

telemetry::Object residency_section(const std::vector<FileResidency> &files,
                                    const ResidencySummary &summary) {
  telemetry::Object out;
  out.set("condition", summary.condition);
  out.set("total_bytes", summary.total_bytes);
  out.set("resident_bytes", summary.resident_bytes);
  out.set("resident_fraction", summary.resident_fraction);
  telemetry::Array entries;
  for (const auto &file : files) {
    telemetry::Object entry;
    entry.set("path", file.path.string());
    entry.set("exists", file.exists);
    entry.set("bytes", file.bytes);
    entry.set("resident_bytes", file.resident_bytes);
    entries.push_back(std::move(entry));
  }
  out.set("files", std::move(entries));
  return out;
}

telemetry::Object stage_section(const StageResult &stage,
                                const ResolvedStage &resolved) {
  telemetry::Object out;
  out.set("name", stage.name);
  out.set("category", telemetry::category::stage);
  out.set("started", stage.started);
  out.set("finished", stage.finished);
  out.set("exit_status", stage.exit_status);
  if (stage.signaled)
    out.set("signal", stage.signal);
  out.set("start_offset_seconds", stage.start_offset_seconds);
  out.set("wall_seconds", stage.wall_seconds);
  out.set("user_cpu_seconds", stage.user_cpu_seconds);
  out.set("system_cpu_seconds", stage.system_cpu_seconds);
  out.set("max_rss_kib", stage.max_rss_kib);
  out.set("major_faults", stage.major_faults);
  out.set("minor_faults", stage.minor_faults);
  out.set("filesystem_input_bytes", stage.filesystem_input_blocks * 512U);
  out.set("filesystem_output_bytes", stage.filesystem_output_blocks * 512U);
  telemetry::Array concurrent;
  for (const auto &name : stage.concurrent_with)
    concurrent.push_back(name);
  out.set("concurrent_with", std::move(concurrent));
  telemetry::Array after;
  for (const auto dependency : resolved.after)
    after.push_back(dependency);
  out.set("after", std::move(after));
  telemetry::Array argv;
  for (const auto &argument : resolved.argv)
    argv.push_back(argument);
  out.set("argv", std::move(argv));
  out.set("log", stage.log.string());
  return out;
}

telemetry::Object gpu_section(const GpuSample &sample) {
  telemetry::Object out;
  out.set("available", sample.available);
  if (!sample.available)
    return out;
  out.set("product_name", sample.product_name);
  out.set("power_limit_watts", sample.power_limit_watts);
  out.set("total_memory_bytes", sample.total_memory_bytes);
  out.set("used_memory_bytes_before", sample.used_memory_bytes_before);
  out.set("peak_used_memory_bytes", sample.peak_used_memory_bytes);
  out.set("mean_power_watts", sample.mean_power_watts);
  out.set("max_power_watts", sample.max_power_watts);
  out.set("max_temperature_celsius", sample.max_temperature_celsius);
  out.set("sample_count", sample.sample_count);
  out.set("sample_interval_seconds", sample.interval_seconds);
  return out;
}

telemetry::Object run_section(const RunRecord &record,
                              const ResolvedRecipe &resolved,
                              const RunSettings &settings) {
  telemetry::Object out;
  out.set("status", record.status);
  out.set("workdir", resolved.work_directory.string());
  out.set("wall_seconds", record.wall_seconds);
  out.set("failed_stage", telemetry::nullable_string(record.chain.failed_stage));
  telemetry::Object conditions;
  conditions.set("process", telemetry::condition::fresh_process);
  conditions.set("filesystem", record.residency_before.condition);
  conditions.set("drop_file_cache", settings.drop_file_cache);
  out.set("conditions", std::move(conditions));
  out.set("filesystem_before",
          residency_section(record.files_before, record.residency_before));
  telemetry::Array stages;
  for (std::size_t index = 0; index < record.chain.stages.size(); ++index)
    stages.push_back(
        stage_section(record.chain.stages[index], resolved.stages[index]));
  out.set("stages", std::move(stages));
  out.set("output", artifact_section(resolved.output, record.artifact,
                                     settings.ffprobe));
  out.set("gpu", gpu_section(record.gpu));
  return out;
}

void probe_sections(const std::filesystem::path &build_directory,
                    telemetry::Object &hardware, telemetry::Object &budget,
                    std::string &probe_source) {
  const auto difprobe = build_directory / "difprobe";
  std::error_code error;
  if (std::filesystem::is_regular_file(difprobe, error)) {
    const auto capture = run_capture({difprobe.string(), "--json"});
    if (capture.exit_status == 0) {
      try {
        const auto parsed = json::parse(capture.output);
        const auto *hardware_value = parsed.find("hardware");
        const auto *budget_value = parsed.find("runtime_budget");
        if (hardware_value && budget_value) {
          hardware = telemetry::from_parsed(*hardware_value).object();
          budget = telemetry::from_parsed(*budget_value).object();
          probe_source = "difprobe";
          return;
        }
      } catch (const std::exception &) {
      }
    }
  }
  const auto report =
      runtime::probe_device(runtime::ProbeBackend::Host, 0);
  hardware = telemetry::hardware_section(report.target);
  budget = telemetry::runtime_budget_section(report.budget);
  probe_source = "host-fallback";
}

} // namespace dif::bench
