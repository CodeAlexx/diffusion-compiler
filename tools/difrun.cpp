#include "dif/backend/plugin.hpp"
#include "dif/ir/codec.hpp"
#include "dif/opt/plan.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <sys/resource.h>

namespace {

std::pair<std::uint32_t, std::filesystem::path> binding(const std::string &text) {
  const auto split = text.find('=');
  if (split == std::string::npos || split == 0 || split + 1 >= text.size())
    dif::fail("tensor binding must be ID=PATH");
  char *end = nullptr;
  const auto id = std::strtoul(text.substr(0, split).c_str(), &end, 10);
  if (!end || *end != '\0' || id == 0)
    dif::fail("tensor binding has invalid id");
  return {static_cast<std::uint32_t>(id), text.substr(split + 1)};
}

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

void usage() {
  std::cerr << "usage: difrun --backend cpu|cuda --program FILE.difir"
               " [--backend-plugin FILE.so]"
               " [--optimization-plan FILE.difplan]"
               " [--weight-bundle FILE.difbind] [--verify-shards]"
               " --input ID=FILE [--input ...] --output ID=FILE [--output ...]"
               " [--warmups N] [--iterations N] [--min-free-mib N]"
               " [--session-runs N] [--cache-dir DIR] [--trace-ops]"
               " [--profile-pipeline] [--serial-streaming] [--map-inputs]\n";
  std::cerr << "input PATH may be FILE.diftensor or SHARD.safetensors::TENSOR_NAME\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string backend = "cpu";
    std::filesystem::path program_path;
    std::filesystem::path backend_plugin;
    std::filesystem::path optimization_plan;
    std::filesystem::path weight_bundle;
    std::unordered_map<std::uint32_t, std::filesystem::path> input_paths;
    std::unordered_map<std::uint32_t, std::filesystem::path> output_paths;
    dif::runtime::RunOptions options;
    bool map_inputs = false;
    bool verify_shards = false;
    std::uint32_t session_runs = 1U;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--backend" && i + 1 < argc)
        backend = argv[++i];
      else if (option == "--backend-plugin" && i + 1 < argc)
        backend_plugin = argv[++i];
      else if (option == "--optimization-plan" && i + 1 < argc)
        optimization_plan = argv[++i];
      else if (option == "--weight-bundle" && i + 1 < argc)
        weight_bundle = argv[++i];
      else if (option == "--program" && i + 1 < argc)
        program_path = argv[++i];
      else if (option == "--input" && i + 1 < argc) {
        auto [id, path] = binding(argv[++i]);
        input_paths[id] = std::move(path);
      } else if (option == "--output" && i + 1 < argc) {
        auto [id, path] = binding(argv[++i]);
        output_paths[id] = std::move(path);
      } else if (option == "--warmups" && i + 1 < argc)
        options.warmups = static_cast<std::uint32_t>(number(argv[++i], "warmups"));
      else if (option == "--iterations" && i + 1 < argc)
        options.iterations = static_cast<std::uint32_t>(number(argv[++i], "iterations"));
      else if (option == "--session-runs" && i + 1 < argc)
        session_runs = static_cast<std::uint32_t>(number(argv[++i], "session runs"));
      else if (option == "--min-free-mib" && i + 1 < argc)
        options.minimum_free_bytes = number(argv[++i], "minimum free memory") * 1024ULL * 1024ULL;
      else if (option == "--cache-dir" && i + 1 < argc)
        options.cache_directory = argv[++i];
      else if (option == "--trace-ops")
        options.trace_operations = true;
      else if (option == "--profile-pipeline")
        options.profile_pipeline = true;
      else if (option == "--serial-streaming")
        options.overlap_streaming = false;
      else if (option == "--map-inputs")
        map_inputs = true;
      else if (option == "--verify-shards")
        verify_shards = true;
      else {
        usage();
        return 2;
      }
    }
    if (program_path.empty() || output_paths.empty()) {
      usage();
      return 2;
    }
    if (session_runs == 0U)
      dif::fail("session runs must be nonzero");
    auto program = dif::ir::read_file(program_path);
    std::string optimization_candidate = "none";
    std::uint64_t optimization_prefetch_distance =
        options.overlap_streaming ? 1U : 0U;
    if (!optimization_plan.empty()) {
      const auto plan = dif::opt::read_plan(optimization_plan);
      auto candidate = dif::opt::replay_candidate(program, plan);
      optimization_candidate =
          dif::hex_digest(candidate.candidate_fingerprint);
      optimization_prefetch_distance =
          candidate.policy.stream_prefetch_distance;
      options.overlap_streaming = optimization_prefetch_distance != 0U;
      program = std::move(candidate.program);
    }
    dif::runtime::TensorMap inputs;
    if (!weight_bundle.empty()) {
      const auto bundle = dif::weights::read_weight_bundle(weight_bundle);
      inputs = dif::weights::load_weight_bundle(bundle, program, verify_shards);
    }
    std::unordered_map<std::string, dif::weights::SafeTensorFile> shards;
    for (const auto &[id, path] : input_paths) {
      if (inputs.contains(id))
        dif::fail("manual input duplicates weight bundle tensor " +
                  std::to_string(id));
      const auto specification = path.string();
      const auto marker = specification.find("::");
      if (marker == std::string::npos) {
        inputs.emplace(id, map_inputs ? dif::runtime::map_tensor(path)
                                      : dif::runtime::read_tensor(path));
        continue;
      }
      const auto shard_path = specification.substr(0, marker);
      const auto tensor_name = specification.substr(marker + 2U);
      if (shard_path.empty() || tensor_name.empty())
        dif::fail("SafeTensors binding must be PATH::TENSOR_NAME");
      auto found = shards.find(shard_path);
      if (found == shards.end())
        found = shards.emplace(shard_path,
                               dif::weights::read_safetensors(shard_path))
                    .first;
      inputs.emplace(id, dif::weights::map_safetensor(found->second, tensor_name));
    }

    std::unique_ptr<dif::runtime::Executor> executor;
    if (!backend_plugin.empty())
      executor = dif::backend::make_plugin_executor(backend_plugin);
    else if (backend == "cpu")
      executor = dif::runtime::make_cpu_executor();
    else if (backend == "cuda")
      executor = dif::runtime::make_cuda_executor();
    else
      dif::fail("unknown backend: " + backend);
    auto prepared = executor->prepare(program, inputs, options);
    std::vector<double> session_elapsed;
    session_elapsed.reserve(session_runs);
    dif::runtime::RunResult result;
    for (std::uint32_t run = 0; run < session_runs; ++run) {
      result = prepared->run(inputs, options);
      session_elapsed.push_back(result.mean_milliseconds);
    }
    result.preparation_milliseconds = prepared->preparation_milliseconds();
    result.resident_bytes = prepared->resident_bytes();
    const auto session_mean =
        std::accumulate(session_elapsed.begin(), session_elapsed.end(), 0.0) /
        static_cast<double>(session_elapsed.size());
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
      dif::fail("getrusage failed");
    for (const auto &[id, path] : output_paths) {
      const auto it = result.outputs.find(id);
      if (it == result.outputs.end())
        dif::fail("requested output id is not a program output: " + std::to_string(id));
      dif::runtime::write_tensor(it->second, path);
    }
    std::cout << "RESULT backend=" << result.backend_name << " device=\""
              << result.device_name << "\" prepare_ms="
              << result.preparation_milliseconds << " mean_ms="
              << result.mean_milliseconds
              << " min_ms=" << result.minimum_milliseconds
              << " max_ms=" << result.maximum_milliseconds
              << " free_before=" << result.free_bytes_before
              << " free_after=" << result.free_bytes_after
              << " resident_bytes=" << result.resident_bytes
              << " session_runs=" << session_runs
              << " session_mean_ms=" << session_mean
              << " host_max_rss_kib=" << usage.ru_maxrss
              << " source_hash=" << result.generated_source_hash << "\n";
    if (!optimization_plan.empty())
      std::cout << "OPTIMIZATION_PLAN path=" << optimization_plan.string()
                << " candidate_fingerprint=" << optimization_candidate
                << " prefetch_distance="
                << optimization_prefetch_distance << "\n";
    if (result.pipeline_profile.enabled) {
      const auto &profile = result.pipeline_profile;
      std::cout << "PIPELINE_PROFILE iterations="
                << profile.measured_iterations
                << " resident_weight_bytes=" << profile.resident_weight_bytes
                << " resident_upload_ms="
                << profile.resident_upload_milliseconds
                << " streamed_weight_bytes=" << profile.streamed_weight_bytes
                << " streamed_host_stage_ms="
                << profile.streamed_host_stage_milliseconds
                << " streamed_host_wait_ms="
                << profile.streamed_host_wait_milliseconds
                << " streamed_h2d_ms=" << profile.streamed_h2d_milliseconds
                << " operation_kernel_ms="
                << profile.operation_kernel_milliseconds
                << " attention_kernel_ms="
                << profile.attention_kernel_milliseconds
                << " non_kernel_device_timeline_ms="
                << profile.non_kernel_device_timeline_milliseconds << "\n";
    }
    for (const auto &timing : result.operation_timings) {
      std::cout << "OP id=" << timing.operation_id
                << " opcode=" << dif::ir::opcode_name(timing.opcode)
                << " mean_ms=" << timing.mean_milliseconds
                << " min_ms=" << timing.minimum_milliseconds
                << " max_ms=" << timing.maximum_milliseconds << "\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difrun: " << error.what() << "\n";
    return 1;
  }
}
