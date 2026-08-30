#include "dif/backend/plugin.hpp"
#include "dif/ir/codec.hpp"
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

dif::runtime::CutlassLinearChoice cutlass_choice(const std::string &text) {
  const auto split = text.find(':');
  if (split == std::string::npos || split == 0U || split + 1U >= text.size())
    dif::fail("CUTLASS Linear choice must be OP_ID:SCHEDULE_ID");
  return {static_cast<std::uint32_t>(
              number(text.substr(0U, split), "CUTLASS Linear operation id")),
          static_cast<std::uint32_t>(
              number(text.substr(split + 1U), "CUTLASS schedule id"))};
}

dif::runtime::LinearAlgorithmChoice linear_choice(const std::string &text) {
  const auto parsed = cutlass_choice(text);
  return {parsed.operation_id, parsed.schedule};
}

void usage() {
  std::cerr << "usage: difrun --backend cpu|cuda --program FILE.difir"
               " [--backend-plugin FILE.so]"
               " [--weight-bundle FILE.difbind] [--verify-shards]"
               " --input ID=FILE [--input ...] --output ID=FILE [--output ...]"
               " [--warmups N] [--iterations N] [--min-free-mib N]"
               " [--session-runs N] [--cache-dir DIR] [--trace-ops]"
               " [--tune-linear OP_ID] [--linear-tune-warmups N]"
               " [--linear-tune-iterations N] [--linear-tune-sessions N]"
               " [--expand-linear-algorithms]"
               " [--select-linear-algorithm OP_ID:HEURISTIC_RANK]"
               " [--fuse-linear-swiglu OP_ID]"
               " [--cutlass-linear OP_ID:SCHEDULE_ID]"
               " [--h3-w8a8-cache FILE.safetensors] [--h3-w8a8-layer N]"
               " [--h3-groupwise-cache FILE.safetensors]"
               " [--h3-groupwise-layer N]"
               " [--h3-modulation-cache FILE.safetensors]"
               " [--h3-modulation-input FILE.diftensor]"
               " [--h3-modulation-layer N]"
               " [--h3-ck-attention-dso FILE.so]"
               " [--profile-pipeline] [--serial-streaming] [--map-inputs]\n";
  std::cerr << "input PATH may be FILE.diftensor or SHARD.safetensors::TENSOR_NAME\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string backend = "cpu";
    std::filesystem::path program_path;
    std::filesystem::path backend_plugin;
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
      else if (option == "--tune-linear" && i + 1 < argc)
        options.tune_linear_operations.push_back(static_cast<std::uint32_t>(
            number(argv[++i], "Linear operation id")));
      else if (option == "--linear-tune-warmups" && i + 1 < argc)
        options.linear_tuning_warmups = static_cast<std::uint32_t>(
            number(argv[++i], "Linear tuning warmups"));
      else if (option == "--linear-tune-iterations" && i + 1 < argc)
        options.linear_tuning_iterations = static_cast<std::uint32_t>(
            number(argv[++i], "Linear tuning iterations"));
      else if (option == "--linear-tune-sessions" && i + 1 < argc)
        options.linear_tuning_sessions = static_cast<std::uint32_t>(
            number(argv[++i], "Linear tuning sessions"));
      else if (option == "--expand-linear-algorithms")
        options.expand_linear_algorithms = true;
      else if (option == "--select-linear-algorithm" && i + 1 < argc)
        options.linear_algorithm_choices.push_back(linear_choice(argv[++i]));
      else if (option == "--fuse-linear-swiglu" && i + 1 < argc)
        options.fuse_linear_swiglu_operations.push_back(
            static_cast<std::uint32_t>(
                number(argv[++i], "fused Linear operation id")));
      else if (option == "--cutlass-linear" && i + 1 < argc)
        options.cutlass_linear_operations.push_back(
            cutlass_choice(argv[++i]));
      else if (option == "--h3-w8a8-cache" && i + 1 < argc)
        options.h3_w8a8_cache = argv[++i];
      else if (option == "--h3-w8a8-layer" && i + 1 < argc)
        options.h3_w8a8_layer = static_cast<std::uint32_t>(
            number(argv[++i], "H3 W8A8 layer"));
      else if (option == "--h3-groupwise-cache" && i + 1 < argc)
        options.h3_groupwise_cache = argv[++i];
      else if (option == "--h3-groupwise-layer" && i + 1 < argc)
        options.h3_groupwise_layer = static_cast<std::uint32_t>(
            number(argv[++i], "H3 groupwise INT8 layer"));
      else if (option == "--h3-modulation-cache" && i + 1 < argc)
        options.h3_modulation_cache = argv[++i];
      else if (option == "--h3-modulation-input" && i + 1 < argc)
        options.h3_modulation_input = argv[++i];
      else if (option == "--h3-modulation-layer" && i + 1 < argc)
        options.h3_modulation_layer = static_cast<std::uint32_t>(
            number(argv[++i], "H3 modulation layer"));
      else if (option == "--h3-ck-attention-dso" && i + 1 < argc)
        options.h3_ck_attention_dso = argv[++i];
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
    const auto program = dif::ir::read_file(program_path);
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
      std::cout << "SESSION_RESULT index=" << run
                << " mean_ms=" << result.mean_milliseconds
                << " min_ms=" << result.minimum_milliseconds
                << " max_ms=" << result.maximum_milliseconds << "\n";
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
    for (const auto &tuning : result.linear_tuning_results) {
      std::cout << "LINEAR_TUNING op=" << tuning.operation_id
                << " selected_rank=" << tuning.selected_heuristic_index
                << " selected_algorithm=" << tuning.selected_algorithm_id
                << " default_mean_ms=" << tuning.default_mean_milliseconds
                << " selected_mean_ms=" << tuning.selected_mean_milliseconds
                << " noise_ms=" << tuning.observed_noise_milliseconds
                << " changed=" << (tuning.changed_from_default ? 1 : 0)
                << " decision=" << tuning.decision
                << " tuning_ms=" << tuning.tuning_milliseconds << "\n";
      for (const auto &candidate : tuning.candidates)
        std::cout << "LINEAR_ALGORITHM op=" << tuning.operation_id
                  << " rank=" << candidate.heuristic_index
                  << " algorithm=" << candidate.algorithm_id
                  << " tile_id=" << candidate.tile_id
                  << " stages_id=" << candidate.stages_id
                  << " split_k=" << candidate.split_k
                  << " reduction_scheme=" << candidate.reduction_scheme
                  << " cta_swizzle=" << candidate.cta_swizzle
                  << " custom_option=" << candidate.custom_option
                  << " waves_count=" << candidate.waves_count
                  << " workspace_bytes=" << candidate.workspace_bytes
                  << " mean_ms=" << candidate.mean_milliseconds
                  << " session_min_ms="
                  << candidate.minimum_session_milliseconds
                  << " session_max_ms="
                  << candidate.maximum_session_milliseconds << "\n";
    }
    for (const auto &selection : result.selected_linear_algorithms)
      std::cout << "LINEAR_SELECTION op=" << selection.operation_id
                << " heuristic_rank=" << selection.heuristic_rank << "\n";
    for (const auto &fusion : result.primitive_fusions)
      std::cout << "PRIMITIVE_FUSION linear_op="
                << fusion.linear_operation_id
                << " swiglu_op=" << fusion.swiglu_operation_id
                << " eliminated_intermediate_bytes="
                << fusion.eliminated_intermediate_bytes
                << " implementation=" << fusion.implementation << "\n";
    for (const auto &primitive : result.gemm_primitives)
      std::cout << "GEMM_PRIMITIVE op=" << primitive.operation_id
                << " schedule=" << primitive.schedule
                << " implementation=" << primitive.implementation
                << " threadblock=" << primitive.threadblock_m << "x"
                << primitive.threadblock_n << "x"
                << primitive.threadblock_k << " warp=" << primitive.warp_m
                << "x" << primitive.warp_n << "x" << primitive.warp_k
                << " stages=" << primitive.stages
                << " threads=" << primitive.threads_per_block
                << " registers_per_thread="
                << primitive.registers_per_thread
                << " static_shared_bytes=" << primitive.static_shared_bytes
                << " dynamic_shared_bytes=" << primitive.dynamic_shared_bytes
                << " max_dynamic_shared_bytes="
                << primitive.maximum_dynamic_shared_bytes << "\n";
    for (const auto &primitive : result.h3_w8a8_mlps)
      std::cout << "H3_W8A8_MLP fc1_op=" << primitive.fc1_operation_id
                << " swiglu_op=" << primitive.swiglu_operation_id
                << " fc2_op=" << primitive.fc2_operation_id
                << " residual_op=" << primitive.residual_operation_id
                << " layer=" << primitive.layer
                << " chunk_rows=" << primitive.chunk_rows
                << " quantized_weight_bytes="
                << primitive.quantized_weight_bytes
                << " scratch_bytes=" << primitive.scratch_bytes
                << " eliminated_intermediate_bytes="
                << primitive.eliminated_intermediate_bytes
                << " classification=" << primitive.classification
                << " implementation=" << primitive.implementation
                << " cache=\"" << primitive.cache_path << "\"\n";
    for (const auto &primitive : result.h3_w8a8_attentions)
      std::cout << "H3_W8A8_ATTENTION qkv_layout_op="
                << primitive.qkv_layout_operation_id
                << " qkv_linear_ops="
                << primitive.qkv_linear_operation_ids.at(0) << ","
                << primitive.qkv_linear_operation_ids.at(1) << ","
                << primitive.qkv_linear_operation_ids.at(2)
                << " output_linear_op="
                << primitive.output_linear_operation_id
                << " residual_op=" << primitive.residual_operation_id
                << " layer=" << primitive.layer
                << " chunk_rows=" << primitive.chunk_rows
                << " quantized_weight_bytes="
                << primitive.quantized_weight_bytes
                << " scratch_bytes=" << primitive.scratch_bytes
                << " eliminated_intermediate_bytes="
                << primitive.eliminated_intermediate_bytes
                << " classification=" << primitive.classification
                << " implementation=" << primitive.implementation
                << " cache=\"" << primitive.cache_path << "\"\n";
    for (const auto &primitive : result.h3_ck_attentions)
      std::cout << "H3_CK_ATTENTION op=" << primitive.operation_id
                << " target_sm=" << primitive.target_sm
                << " scratch_bytes=" << primitive.scratch_bytes
                << " classification=" << primitive.classification
                << " implementation=" << primitive.implementation
                << " dso=\"" << primitive.dso_path << "\"\n";
    for (const auto &primitive : result.h3_groupwise_int8)
      std::cout << "H3_GROUPWISE_INT8 qkv_layout_op="
                << primitive.qkv_layout_operation_id
                << " output_linear_op=" << primitive.output_linear_operation_id
                << " fc1_op=" << primitive.fc1_operation_id
                << " fc2_op=" << primitive.fc2_operation_id
                << " layer=" << primitive.layer
                << " group_sizes=" << primitive.group_sizes.at(0) << ","
                << primitive.group_sizes.at(1) << ","
                << primitive.group_sizes.at(2) << ","
                << primitive.group_sizes.at(3)
                << " quantized_weight_bytes="
                << primitive.quantized_weight_bytes
                << " scratch_bytes=" << primitive.scratch_bytes
                << " classification=" << primitive.classification
                << " implementation=" << primitive.implementation
                << " cache=\"" << primitive.cache_path << "\"\n";
    for (const auto &primitive : result.h3_modulation_caches)
      std::cout << "H3_MODULATION_CACHE linear_op="
                << primitive.linear_operation_id
                << " select_op=" << primitive.select_operation_id
                << " layer=" << primitive.layer
                << " cache_bytes=" << primitive.cache_bytes
                << " replaced_weight_bytes="
                << primitive.replaced_weight_bytes
                << " classification=" << primitive.classification
                << " implementation=" << primitive.implementation
                << " cache=\"" << primitive.cache_path << "\""
                << " input=\"" << primitive.input_path << "\"\n";
    if (result.pipeline_profile.enabled) {
      const auto &profile = result.pipeline_profile;
      std::cout << "PIPELINE_PROFILE iterations="
                << profile.measured_iterations
                << " resident_weight_bytes=" << profile.resident_weight_bytes
                << " resident_host_prefault_ms="
                << profile.resident_host_prefault_milliseconds
                << " resident_minor_faults="
                << profile.resident_minor_page_faults
                << " resident_major_faults="
                << profile.resident_major_page_faults
                << " resident_h2d_ms="
                << profile.resident_h2d_milliseconds
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
