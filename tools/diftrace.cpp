// diftrace: compiler-aware attribution of where a complete generation's
// wall time goes. Recipe mode traces every stage process of a benchmark
// chain through the runtime's environment trace sink and merges the stage
// walls with the attributed runtime events; program mode traces a single
// DiffIR program in-process. Neither replaces Nsight Systems: with --nvtx
// the same operation ranges are pushed so an nsys timeline correlates.

#include "dif/bench/report.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/trace_sink.hpp"
#include "dif/telemetry/vocabulary.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <unistd.h>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: diftrace recipe RECIPE.json --workdir DIR [--build DIR]\n"
         "                     [--prompt-file FILE] [--set VAR=VALUE ...]\n"
         "                     [--nvtx] [--no-ffprobe] [--json] [--report FILE]\n"
         "                     [--stage-cache DIR]\n"
         "       diftrace program --backend cpu|cuda --program FILE.difir\n"
         "                     [--weight-bundle FILE.difbind] [--input ID=FILE ...]\n"
         "                     [--warmups N] [--iterations N] [--nvtx]\n"
         "                     [--profile-pipeline] [--trace-ops] [--json] [--report FILE]\n"
         "       diftrace merge TRACE.jsonl [--json]\n"
         "\n"
         "Recipe mode sets DIF_TRACE_FILE per stage so every prepared execution\n"
         "in every stage appends a runtime-trace document; the merged report\n"
         "attributes the complete wall to stages and the stage time to GEMM,\n"
         "attention, generated kernels, copies, staging, waits, and layout.\n"
         "Nsight correlation: run the same command under nsys with --nvtx.\n";
}

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

std::string seconds(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  return out.str();
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    dif::fail("cannot write " + path.string());
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// --- Aggregation over runtime-trace documents -----------------------------

struct Bucket {
  std::uint64_t count{};
  std::uint64_t bytes{};
  double host_ms{};
};

struct OpcodeMix {
  std::uint64_t submissions{};
  double host_ms{};
};

struct StageAggregate {
  std::uint64_t documents{};
  double preparation_ms{};
  double run_wall_ms{};
  double mean_ms_sum{};
  std::uint64_t iterations{};
  std::map<std::string, Bucket> run_attribution;
  std::map<std::string, Bucket> preparation_attribution;
  dif::runtime::LaunchTelemetry launches;
  std::map<std::string, OpcodeMix> opcode_mix;
  double operation_kernel_ms{};
  double attention_kernel_ms{};
  double streamed_host_stage_ms{};
  double streamed_host_wait_ms{};
  double resident_upload_ms{};
  std::uint64_t streamed_weight_bytes{};
  std::uint64_t resident_weight_bytes{};
};

double number_or_zero(const dif::json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value || !std::holds_alternative<double>(value->storage))
    return 0.0;
  return value->number();
}

std::uint64_t count_or_zero(const dif::json::Value &object, const char *key) {
  return static_cast<std::uint64_t>(number_or_zero(object, key));
}

void accumulate_buckets(std::map<std::string, Bucket> &into,
                        const dif::json::Value *section) {
  if (!section || !section->is_object())
    return;
  for (const auto &[name, entry] : section->object()) {
    if (!entry.is_object())
      continue;
    auto &bucket = into[name];
    bucket.count += count_or_zero(entry, "count");
    bucket.bytes += count_or_zero(entry, "bytes");
    bucket.host_ms += number_or_zero(entry, "host_ms");
  }
}

void accumulate_launches(dif::runtime::LaunchTelemetry &into,
                         const dif::json::Value *section) {
  if (!section || !section->is_object())
    return;
  into.kernel_launches += count_or_zero(*section, "kernel_launches");
  into.cublaslt_matmuls += count_or_zero(*section, "cublaslt_matmuls");
  into.cublas_gemms += count_or_zero(*section, "cublas_gemms");
  into.cudnn_attention_dispatches +=
      count_or_zero(*section, "cudnn_attention_dispatches");
  into.cudnn_convolution_dispatches +=
      count_or_zero(*section, "cudnn_convolution_dispatches");
  into.cutlass_launches += count_or_zero(*section, "cutlass_launches");
  into.ck_attention_dispatches +=
      count_or_zero(*section, "ck_attention_dispatches");
  into.h2d_copies += count_or_zero(*section, "h2d_copies");
  into.h2d_bytes += count_or_zero(*section, "h2d_bytes");
  into.d2h_copies += count_or_zero(*section, "d2h_copies");
  into.d2h_bytes += count_or_zero(*section, "d2h_bytes");
  into.d2d_copies += count_or_zero(*section, "d2d_copies");
  into.d2d_bytes += count_or_zero(*section, "d2d_bytes");
  into.event_records += count_or_zero(*section, "event_records");
  into.stream_wait_events += count_or_zero(*section, "stream_wait_events");
  into.host_event_synchronizes +=
      count_or_zero(*section, "host_event_synchronizes");
  into.host_stream_synchronizes +=
      count_or_zero(*section, "host_stream_synchronizes");
  into.device_mem_allocs += count_or_zero(*section, "device_mem_allocs");
  into.pinned_mem_allocs += count_or_zero(*section, "pinned_mem_allocs");
}

void accumulate_document(StageAggregate &aggregate,
                         const dif::json::Value &document) {
  ++aggregate.documents;
  // A prepared execution that runs many times emits one document per run,
  // each describing the same one-time preparation; count it once. Documents
  // without the flag (older writers) keep the historical per-document sum.
  bool preparation_reported = true;
  if (const auto *execution = document.find("execution")) {
    if (const auto *flag = execution->find("preparation_reported");
        flag && std::holds_alternative<bool>(flag->storage))
      preparation_reported = flag->boolean();
    if (preparation_reported)
      aggregate.preparation_ms +=
          number_or_zero(*execution, "preparation_ms");
    aggregate.mean_ms_sum += number_or_zero(*execution, "mean_ms");
    aggregate.iterations += count_or_zero(*execution, "iterations");
  }
  if (const auto *trace = document.find("trace")) {
    aggregate.run_wall_ms += number_or_zero(*trace, "run_wall_ms");
    accumulate_buckets(aggregate.run_attribution,
                       trace->find("run_attribution"));
    if (preparation_reported)
      accumulate_buckets(aggregate.preparation_attribution,
                         trace->find("preparation_attribution"));
    if (const auto *events = trace->find("run_events");
        events && events->is_array()) {
      for (const auto &event : events->array()) {
        const auto *category = event.find("category");
        if (!category || category->string() != "operation")
          continue;
        const auto *opcode = event.find("opcode");
        auto &mix = aggregate.opcode_mix[opcode ? opcode->string() : "?"];
        ++mix.submissions;
        mix.host_ms += number_or_zero(event, "host_end_ms") -
                       number_or_zero(event, "host_start_ms");
      }
    }
  }
  accumulate_launches(aggregate.launches,
                      document.find("run_launch_telemetry"));
  if (const auto *profile = document.find("pipeline_profile")) {
    aggregate.operation_kernel_ms +=
        number_or_zero(*profile, "operation_kernel_ms");
    aggregate.attention_kernel_ms +=
        number_or_zero(*profile, "attention_kernel_ms");
    aggregate.streamed_host_stage_ms +=
        number_or_zero(*profile, "streamed_host_stage_ms");
    aggregate.streamed_host_wait_ms +=
        number_or_zero(*profile, "streamed_host_wait_ms");
    aggregate.resident_upload_ms +=
        number_or_zero(*profile, "resident_upload_ms");
    aggregate.streamed_weight_bytes +=
        count_or_zero(*profile, "streamed_weight_bytes");
    aggregate.resident_weight_bytes = std::max(
        aggregate.resident_weight_bytes,
        count_or_zero(*profile, "resident_weight_bytes"));
  }
}

StageAggregate aggregate_sink(const std::filesystem::path &path) {
  StageAggregate aggregate;
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return aggregate;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty())
      continue;
    try {
      accumulate_document(aggregate, dif::json::parse(line));
    } catch (const std::exception &error) {
      std::cerr << "diftrace: skipping malformed trace line in "
                << path.string() << ": " << error.what() << "\n";
    }
  }
  return aggregate;
}

dif::telemetry::Object buckets_section(const std::map<std::string, Bucket> &buckets) {
  dif::telemetry::Object out;
  for (const auto &[name, bucket] : buckets) {
    dif::telemetry::Object entry;
    entry.set("count", bucket.count);
    entry.set("bytes", bucket.bytes);
    entry.set("host_ms", bucket.host_ms);
    out.set(name, std::move(entry));
  }
  return out;
}

dif::telemetry::Object aggregate_section(const StageAggregate &aggregate) {
  dif::telemetry::Object out;
  out.set("runtime_trace_documents", aggregate.documents);
  out.set("preparation_ms", aggregate.preparation_ms);
  out.set("run_wall_ms", aggregate.run_wall_ms);
  out.set("iterations", aggregate.iterations);
  out.set("run_attribution", buckets_section(aggregate.run_attribution));
  out.set("preparation_attribution",
          buckets_section(aggregate.preparation_attribution));
  out.set("run_launch_telemetry",
          dif::telemetry::launch_telemetry_section(aggregate.launches));
  dif::telemetry::Object profile;
  profile.set("operation_kernel_ms", aggregate.operation_kernel_ms);
  profile.set("attention_kernel_ms", aggregate.attention_kernel_ms);
  profile.set("streamed_host_stage_ms", aggregate.streamed_host_stage_ms);
  profile.set("streamed_host_wait_ms", aggregate.streamed_host_wait_ms);
  profile.set("resident_upload_ms", aggregate.resident_upload_ms);
  profile.set("streamed_weight_bytes", aggregate.streamed_weight_bytes);
  profile.set("resident_weight_bytes", aggregate.resident_weight_bytes);
  out.set("pipeline_profile_totals", std::move(profile));
  std::vector<std::pair<std::string, OpcodeMix>> mix(
      aggregate.opcode_mix.begin(), aggregate.opcode_mix.end());
  std::sort(mix.begin(), mix.end(), [](const auto &left, const auto &right) {
    if (left.second.host_ms != right.second.host_ms)
      return left.second.host_ms > right.second.host_ms;
    return left.first < right.first;
  });
  dif::telemetry::Array opcodes;
  for (const auto &[opcode, entry] : mix) {
    dif::telemetry::Object item;
    item.set("opcode", opcode);
    item.set("submissions", entry.submissions);
    item.set("host_submit_ms", entry.host_ms);
    opcodes.push_back(std::move(item));
  }
  out.set("operation_mix", std::move(opcodes));
  return out;
}

// --- recipe mode ------------------------------------------------------------

struct RecipeOptions {
  std::filesystem::path recipe;
  std::filesystem::path workdir;
  std::filesystem::path build;
  std::filesystem::path prompt_file;
  std::map<std::string, std::string> overrides;
  bool nvtx{};
  bool ffprobe{true};
  std::filesystem::path stage_cache;
  bool json{};
  std::filesystem::path report;
};

std::filesystem::path default_build() {
  char buffer[4096];
  const auto count = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1U);
  if (count > 0) {
    buffer[count] = '\0';
    return std::filesystem::path(buffer).parent_path();
  }
  return std::filesystem::current_path();
}

int command_recipe(int argc, char **argv) {
  RecipeOptions options;
  if (argc < 3) {
    usage();
    return 2;
  }
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
    } else if (option == "--nvtx")
      options.nvtx = true;
    else if (option == "--stage-cache")
      options.stage_cache = std::filesystem::path(value());
    else if (option == "--no-ffprobe")
      options.ffprobe = false;
    else if (option == "--json")
      options.json = true;
    else if (option == "--report")
      options.report = value();
    else
      dif::fail("unknown diftrace option: " + option);
  }
  if (options.workdir.empty())
    dif::fail("diftrace recipe requires --workdir");
  if (options.build.empty())
    options.build = default_build();
  const auto recipe = dif::bench::parse_recipe(options.recipe);
  const auto base = std::filesystem::absolute(options.workdir);
  std::error_code error;
  if (std::filesystem::exists(base, error) &&
      !std::filesystem::is_empty(base, error))
    dif::fail("refusing nonempty work directory: " + base.string());
  dif::bench::ResolveContext context;
  context.build_directory = options.build;
  context.work_directory = base;
  context.prompt_file = options.prompt_file;
  context.overrides = options.overrides;
  const auto resolved = dif::bench::resolve_recipe(recipe, context);
  const auto problems = dif::bench::preflight(resolved);
  if (!problems.empty()) {
    for (const auto &problem : problems)
      std::cerr << "PREFLIGHT " << problem.kind << " " << problem.subject
                << "\n";
    dif::fail("recipe is not runnable on this host");
  }
  const auto trace_directory = base / "trace";
  std::filesystem::create_directories(trace_directory);
  dif::bench::RunSettings settings;
  settings.ffprobe = options.ffprobe;
  settings.stage_cache.directory = options.stage_cache;
  settings.before_stage = [&](std::size_t index) {
    std::vector<std::pair<std::string, std::string>> environment;
    environment.emplace_back(
        dif::telemetry::kTraceFileVariable,
        (trace_directory / (resolved.stages[index].name + ".jsonl")).string());
    if (options.nvtx)
      environment.emplace_back(dif::telemetry::kNvtxVariable, "1");
    return environment;
  };
  if (!options.json)
    std::cerr << "diftrace: tracing " << resolved.stages.size()
              << " stages into " << trace_directory.string() << "\n";
  const auto record = dif::bench::execute_run(resolved, settings);

  auto document = dif::telemetry::make_document(dif::telemetry::kind::trace);
  {
    dif::telemetry::Object trace;
    trace.set("mode", "recipe");
    trace.set("boundary", "literal-prompt-to-saved-output");
    trace.set("nvtx_ranges", options.nvtx);
    trace.set("nvtx_range_naming", "op<id> <opcode>; dif::prepare; dif::run");
    trace.set("trace_directory", trace_directory.string());
    trace.set("recipe", dif::bench::recipe_section(resolved));
    trace.set("workload", resolved.recipe.workload);
    document.set("trace", std::move(trace));
  }
  dif::telemetry::Object hardware;
  dif::telemetry::Object budget;
  std::string probe_source;
  dif::bench::probe_sections(resolved.build_directory, hardware, budget,
                             probe_source);
  document.set("hardware", std::move(hardware));
  document.set("runtime_budget", std::move(budget));
  document.set("inputs", dif::bench::inputs_section(resolved, false));
  auto run = dif::bench::run_section(record, resolved, settings);
  run.set("index", 1);
  document.set("run", std::move(run));

  // Per-stage attribution and the whole-chain roll-up.
  std::map<std::string, Bucket> chain_buckets;
  dif::runtime::LaunchTelemetry chain_launches;
  dif::telemetry::Array stages;
  std::vector<std::pair<std::string, double>> stage_walls;
  for (std::size_t index = 0; index < resolved.stages.size(); ++index) {
    const auto &stage = record.chain.stages[index];
    const auto sink = trace_directory / (stage.name + ".jsonl");
    const auto aggregate = aggregate_sink(sink);
    dif::telemetry::Object entry;
    entry.set("name", stage.name);
    entry.set("wall_seconds", stage.wall_seconds);
    entry.set("start_offset_seconds", stage.start_offset_seconds);
    entry.set("exit_status", stage.exit_status);
    dif::telemetry::Array concurrent;
    for (const auto &name : stage.concurrent_with)
      concurrent.push_back(name);
    entry.set("concurrent_with", std::move(concurrent));
    entry.set("trace_sink", sink.string());
    entry.set("runtime", aggregate_section(aggregate));
    // Filesystem attribution from the child's own accounting.
    dif::telemetry::Object filesystem;
    filesystem.set("input_bytes", stage.filesystem_input_blocks * 512U);
    filesystem.set("output_bytes", stage.filesystem_output_blocks * 512U);
    filesystem.set("major_faults", stage.major_faults);
    filesystem.set("minor_faults", stage.minor_faults);
    entry.set("filesystem", std::move(filesystem));
    stages.push_back(std::move(entry));
    for (const auto &[name, bucket] : aggregate.run_attribution) {
      auto &into = chain_buckets[name];
      into.count += bucket.count;
      into.bytes += bucket.bytes;
      into.host_ms += bucket.host_ms;
    }
    for (const auto &[name, bucket] : aggregate.preparation_attribution) {
      auto &into = chain_buckets[name];
      into.count += bucket.count;
      into.bytes += bucket.bytes;
      into.host_ms += bucket.host_ms;
    }
    chain_launches.kernel_launches += aggregate.launches.kernel_launches;
    chain_launches.cublaslt_matmuls += aggregate.launches.cublaslt_matmuls;
    chain_launches.cublas_gemms += aggregate.launches.cublas_gemms;
    chain_launches.cudnn_attention_dispatches +=
        aggregate.launches.cudnn_attention_dispatches;
    chain_launches.cudnn_convolution_dispatches +=
        aggregate.launches.cudnn_convolution_dispatches;
    chain_launches.cutlass_launches += aggregate.launches.cutlass_launches;
    chain_launches.ck_attention_dispatches +=
        aggregate.launches.ck_attention_dispatches;
    chain_launches.h2d_copies += aggregate.launches.h2d_copies;
    chain_launches.h2d_bytes += aggregate.launches.h2d_bytes;
    chain_launches.d2h_copies += aggregate.launches.d2h_copies;
    chain_launches.d2h_bytes += aggregate.launches.d2h_bytes;
    chain_launches.d2d_copies += aggregate.launches.d2d_copies;
    chain_launches.d2d_bytes += aggregate.launches.d2d_bytes;
    chain_launches.event_records += aggregate.launches.event_records;
    chain_launches.stream_wait_events +=
        aggregate.launches.stream_wait_events;
    chain_launches.host_event_synchronizes +=
        aggregate.launches.host_event_synchronizes;
    chain_launches.host_stream_synchronizes +=
        aggregate.launches.host_stream_synchronizes;
    stage_walls.emplace_back(stage.name, stage.wall_seconds);
  }
  document.set("stages", std::move(stages));

  // Largest contributors to the complete wall: stage walls on the actual
  // timeline. Concurrent stages are reported with their overlap partners;
  // they are never summed into a replacement total.
  std::sort(stage_walls.begin(), stage_walls.end(),
            [](const auto &left, const auto &right) {
              return left.second > right.second;
            });
  dif::telemetry::Object attribution;
  attribution.set("complete_wall_seconds", record.wall_seconds);
  attribution.set("note",
                  "stage walls overlap when concurrent and must not be summed; "
                  "the complete wall is the only acceptance metric");
  dif::telemetry::Array ranked;
  for (const auto &[name, wall] : stage_walls) {
    dif::telemetry::Object entry;
    entry.set("stage", name);
    entry.set("wall_seconds", wall);
    entry.set("fraction_of_complete_wall",
              record.wall_seconds > 0.0 ? wall / record.wall_seconds : 0.0);
    ranked.push_back(std::move(entry));
  }
  attribution.set("stages_by_wall", std::move(ranked));
  attribution.set("runtime_categories", buckets_section(chain_buckets));
  attribution.set("runtime_launch_telemetry",
                  dif::telemetry::launch_telemetry_section(chain_launches));
  document.set("attribution", std::move(attribution));
  document.set("status", record.status);

  const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
  write_text(base / "diftrace.json", text);
  if (!options.report.empty())
    write_text(options.report, text);
  if (options.json) {
    std::cout << text;
  } else {
    std::cout << "DIFTRACE recipe=" << resolved.recipe.name
              << " status=" << record.status
              << " complete_wall_s=" << seconds(record.wall_seconds) << "\n";
    for (const auto &[name, wall] : stage_walls)
      std::cout << "  STAGE " << std::left << std::setw(22) << name
                << " wall_s=" << seconds(wall) << "\n";
    for (const auto &[name, bucket] : chain_buckets)
      std::cout << "  RUNTIME " << std::left << std::setw(18) << name
                << " count=" << bucket.count << " bytes=" << bucket.bytes
                << " host_ms=" << seconds(bucket.host_ms) << "\n";
    std::cout << "report  " << (base / "diftrace.json").string() << "\n";
  }
  return record.status == "completed" ? 0 : 1;
}

// --- program mode -----------------------------------------------------------

std::pair<std::uint32_t, std::filesystem::path> binding(const std::string &text) {
  const auto split = text.find('=');
  if (split == std::string::npos || split == 0 || split + 1 >= text.size())
    dif::fail("tensor binding must be ID=PATH");
  return {static_cast<std::uint32_t>(number(text.substr(0, split), "tensor id")),
          text.substr(split + 1)};
}

int command_program(int argc, char **argv) {
  std::string backend = "cpu";
  std::filesystem::path program_path;
  std::filesystem::path weight_bundle;
  std::vector<std::pair<std::uint32_t, std::filesystem::path>> inputs;
  dif::runtime::RunOptions options;
  options.warmups = 1U;
  options.iterations = 1U;
  bool json = false;
  std::filesystem::path report;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value after " + option);
      return argv[++index];
    };
    if (option == "--backend")
      backend = value();
    else if (option == "--program")
      program_path = value();
    else if (option == "--weight-bundle")
      weight_bundle = value();
    else if (option == "--input")
      inputs.push_back(binding(value()));
    else if (option == "--warmups")
      options.warmups = static_cast<std::uint32_t>(number(value(), "warmups"));
    else if (option == "--iterations")
      options.iterations =
          static_cast<std::uint32_t>(number(value(), "iterations"));
    else if (option == "--nvtx")
      options.nvtx_ranges = true;
    else if (option == "--profile-pipeline")
      options.profile_pipeline = true;
    else if (option == "--trace-ops")
      options.trace_operations = true;
    else if (option == "--json")
      json = true;
    else if (option == "--report")
      report = value();
    else
      dif::fail("unknown diftrace option: " + option);
  }
  if (program_path.empty())
    dif::fail("diftrace program requires --program");
  options.trace_events = true;
  options.minimum_free_bytes = 0U;
  const auto program = dif::ir::read_file(program_path);
  dif::runtime::TensorMap bindings;
  if (!weight_bundle.empty()) {
    const auto bundle = dif::weights::read_weight_bundle(weight_bundle);
    bindings = dif::weights::load_weight_bundle(bundle, program, false);
  }
  for (const auto &[id, path] : inputs)
    bindings.insert_or_assign(id, dif::runtime::read_tensor(path));
  std::unique_ptr<dif::runtime::Executor> executor;
  if (backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("diftrace backend accepts cpu or cuda");
  auto prepared = executor->prepare(program, bindings, options);
  auto result = prepared->run(bindings, options);
  result.preparation_milliseconds = prepared->preparation_milliseconds();
  result.resident_bytes = prepared->resident_bytes();
  auto document = dif::telemetry::runtime_trace_document(
      result, dif::hex_digest(dif::ir::fingerprint(program)),
      program.operations.size(), options);
  document.set("kind", dif::telemetry::kind::trace);
  {
    dif::telemetry::Object mode;
    mode.set("mode", "program");
    mode.set("program_path", std::filesystem::absolute(program_path).string());
    mode.set("nvtx_ranges", options.nvtx_ranges);
    mode.set("nvtx_range_naming", "op<id> <opcode>; dif::prepare; dif::run");
    document.set("trace_mode", std::move(mode));
  }
  std::map<std::string, OpcodeMix> mix;
  for (const auto &event : result.trace_events) {
    if (event.category != dif::telemetry::category::operation)
      continue;
    auto &entry = mix[event.opcode];
    ++entry.submissions;
    entry.host_ms += event.host_end_ms - event.host_start_ms;
  }
  std::vector<std::pair<std::string, OpcodeMix>> ordered(mix.begin(), mix.end());
  std::sort(ordered.begin(), ordered.end(),
            [](const auto &left, const auto &right) {
              return left.second.host_ms > right.second.host_ms;
            });
  dif::telemetry::Array opcodes;
  for (const auto &[opcode, entry] : ordered) {
    dif::telemetry::Object item;
    item.set("opcode", opcode);
    item.set("submissions", entry.submissions);
    item.set("host_submit_ms", entry.host_ms);
    opcodes.push_back(std::move(item));
  }
  document.set("operation_mix", std::move(opcodes));
  const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
  if (!report.empty())
    write_text(report, text);
  if (json) {
    std::cout << text;
    return 0;
  }
  std::cout << "DIFTRACE program=" << program_path.string()
            << " backend=" << result.backend_name
            << " operations=" << program.operations.size()
            << " prepare_ms=" << result.preparation_milliseconds
            << " mean_ms=" << result.mean_milliseconds
            << " events=" << result.trace_events.size() << "\n";
  const auto attribution =
      dif::telemetry::trace_attribution_section(result.trace_events);
  for (const auto &[name, bucket] : attribution.members()) {
    const auto &object = bucket.object();
    std::cout << "  RUNTIME " << std::left << std::setw(18) << name
              << " count=" << object.find("count")->number()
              << " bytes=" << object.find("bytes")->number()
              << " host_ms=" << seconds(object.find("host_ms")->number())
              << "\n";
  }
  for (const auto &[opcode, entry] : ordered)
    std::cout << "  OPCODE " << std::left << std::setw(24) << opcode
              << " submissions=" << entry.submissions
              << " host_submit_ms=" << seconds(entry.host_ms) << "\n";
  return 0;
}

int command_merge(int argc, char **argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const std::filesystem::path sink = argv[2];
  bool json = false;
  for (int index = 3; index < argc; ++index)
    if (std::string(argv[index]) == "--json")
      json = true;
  const auto aggregate = aggregate_sink(sink);
  auto document = dif::telemetry::make_document(dif::telemetry::kind::trace);
  dif::telemetry::Object mode;
  mode.set("mode", "merge");
  mode.set("trace_sink", sink.string());
  document.set("trace_mode", std::move(mode));
  document.set("runtime", aggregate_section(aggregate));
  if (json) {
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return 0;
  }
  std::cout << "DIFTRACE merge sink=" << sink.string()
            << " documents=" << aggregate.documents
            << " preparation_ms=" << seconds(aggregate.preparation_ms)
            << " run_wall_ms=" << seconds(aggregate.run_wall_ms) << "\n";
  for (const auto &[name, bucket] : aggregate.run_attribution)
    std::cout << "  RUNTIME " << std::left << std::setw(18) << name
              << " count=" << bucket.count << " bytes=" << bucket.bytes
              << " host_ms=" << seconds(bucket.host_ms) << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "recipe")
      return command_recipe(argc, argv);
    if (command == "program")
      return command_program(argc, argv);
    if (command == "merge")
      return command_merge(argc, argv);
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "diftrace: " << error.what() << "\n";
    return 1;
  }
}
