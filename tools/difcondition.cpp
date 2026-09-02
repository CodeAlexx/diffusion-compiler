// difcondition — native shared-Qwen diffusion text conditioning.
//
// Builds a Qwen3/Qwen3-VL conditioner as one DiffIR program, binds it straight
// to the checkpoint shards (no Python loader), executes it, and writes the raw
// hidden-state conditioning the selected model frontend consumes.
//
//   difcondition program  --checkpoint DIR --sequence N --output P.difir
//                         [--layers N] [--attention 1|2]
//   difcondition bundle   --checkpoint DIR --program P.difir --output B.difbind
//   difcondition run      --program P.difir --bundle B.difbind --ids I.diftensor
//                         --output C.diftensor [--backend cuda|cpu]
//                         [--cache-dir DIR] [--min-free-mib N] [--depth N]
//
// The `--depth` form of `program` builds a truncated parity ladder so a
// depth-k program can be compared against the oracle's raw hidden_states[k].

#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/frontend/krea2.hpp"
#include "dif/frontend/flux2.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string command;
  fs::path checkpoint;
  fs::path program;
  fs::path bundle;
  fs::path ids;
  fs::path inputs;
  fs::path vision_inputs;
  fs::path vision_outputs;
  fs::path output;
  fs::path report;
  fs::path cache_directory;
  fs::path convrot_int8_checkpoint;
  fs::path sealed_bundle;
  std::string backend{"cuda"};
  std::uint64_t sequence{};
  std::uint64_t layers{};
  std::uint64_t attention{2};
  std::uint64_t vision_tokens{};
  std::uint64_t minimum_free_mib{4096};
  std::uint64_t streamed_stage_threads{1};
  bool krea2{};
  bool flux2{};
  bool capture_first_layer{};
  bool profile_pipeline{};
  bool resident_streamed{};
  bool convrot_int8_weight_only_quality{};
  std::uint32_t convrot_int8_linear_count{};
};

void usage() {
  std::cerr
      << "usage: difcondition program --checkpoint DIR --sequence N --output "
         "FILE.difir [--layers N] [--attention 1|2] [--krea2]\n"
         "       difcondition bundle --checkpoint DIR --program FILE.difir "
         "--output FILE.difbind [--sealed-bundle FILE.difbind]\n"
         "       difcondition run --program FILE.difir --bundle FILE.difbind "
         "--ids FILE.diftensor --output FILE.diftensor [--backend cuda|cpu] "
         "[--cache-dir DIR] [--min-free-mib N] "
         "[--convrot-int8-checkpoint FILE] [--convrot-int8-linear-count N] "
         "[--convrot-int8-weight-only-quality] "
         "[--streamed-stage-threads N]\n"
         "       difcondition run --program FILE.difir --bundle FILE.difbind "
         "--vision-inputs PRESENTATION.safetensors --vision-outputs "
         "VISION.safetensors --vision-tokens N --output FILE.diftensor "
         "[--cache-dir DIR] [--min-free-mib N]\n"
         "       difcondition run --krea2 --program FILE.difir --bundle "
         "FILE.difbind --inputs creator.safetensors --output "
         "native.safetensors [--cache-dir DIR] [--min-free-mib N] "
         "[--convrot-int8-checkpoint FILE] [--convrot-int8-linear-count N] "
         "[--convrot-int8-weight-only-quality] "
         "[--streamed-stage-threads N]\n";
  std::cerr
      << "       add --flux2 for the FLUX.2 [klein] 9B Qwen3-8B frontend; "
         "use --inputs from diftokenize --flux2-inputs-out\n";
}

std::uint64_t number(const std::string &text, const char *label) {
  try {
    return static_cast<std::uint64_t>(std::stoull(text));
  } catch (const std::exception &) {
    dif::fail(std::string("invalid ") + label + ": " + text);
  }
  return 0U;
}

Options parse(int argc, char **argv) {
  if (argc < 2) {
    usage();
    dif::fail("difcondition requires a command");
  }
  Options options;
  options.command = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&](const char *name) -> std::string {
      if (index + 1 >= argc)
        dif::fail(std::string(name) + " requires a value");
      return argv[++index];
    };
    if (option == "--checkpoint")
      options.checkpoint = value("--checkpoint");
    else if (option == "--program")
      options.program = value("--program");
    else if (option == "--bundle")
      options.bundle = value("--bundle");
    else if (option == "--ids")
      options.ids = value("--ids");
    else if (option == "--inputs")
      options.inputs = value("--inputs");
    else if (option == "--vision-inputs")
      options.vision_inputs = value("--vision-inputs");
    else if (option == "--vision-outputs")
      options.vision_outputs = value("--vision-outputs");
    else if (option == "--output")
      options.output = value("--output");
    else if (option == "--report")
      options.report = value("--report");
    else if (option == "--backend")
      options.backend = value("--backend");
    else if (option == "--cache-dir")
      options.cache_directory = value("--cache-dir");
    else if (option == "--convrot-int8-checkpoint")
      options.convrot_int8_checkpoint =
          value("--convrot-int8-checkpoint");
    else if (option == "--sealed-bundle")
      options.sealed_bundle = value("--sealed-bundle");
    else if (option == "--convrot-int8-weight-only-quality")
      options.convrot_int8_weight_only_quality = true;
    else if (option == "--convrot-int8-linear-count")
      options.convrot_int8_linear_count = static_cast<std::uint32_t>(
          number(value("--convrot-int8-linear-count"),
                 "ConvRot INT8 Linear count"));
    else if (option == "--sequence")
      options.sequence = number(value("--sequence"), "sequence");
    else if (option == "--layers")
      options.layers = number(value("--layers"), "layers");
    else if (option == "--attention")
      options.attention = number(value("--attention"), "attention");
    else if (option == "--vision-tokens")
      options.vision_tokens = number(value("--vision-tokens"), "vision tokens");
    else if (option == "--min-free-mib")
      options.minimum_free_mib = number(value("--min-free-mib"), "min free MiB");
    else if (option == "--streamed-stage-threads")
      options.streamed_stage_threads =
          number(value("--streamed-stage-threads"), "streamed stage threads");
    else if (option == "--krea2")
      options.krea2 = true;
    else if (option == "--flux2")
      options.flux2 = true;
    else if (option == "--capture-first-layer")
      options.capture_first_layer = true;
    else if (option == "--profile-pipeline")
      options.profile_pipeline = true;
    else if (option == "--resident-streamed")
      options.resident_streamed = true;
    else {
      usage();
      dif::fail("unknown option: " + option);
    }
  }
  if (options.krea2 && options.flux2)
    dif::fail("--krea2 and --flux2 are mutually exclusive");
  return options;
}

dif::frontend::Qwen3VlConditionerConfig
config_for(const Options &options, std::uint64_t reconstructed_layers = 0U) {
  const auto requested_layers = reconstructed_layers != 0U
                                    ? reconstructed_layers
                                    : options.layers;
  if (options.flux2) {
    auto config = dif::frontend::make_flux2_klein_9b_conditioner_config(
        requested_layers == 0U ? 27U : requested_layers);
    config.attention_implementation = options.attention;
    config.capture_first_layer_boundaries = options.capture_first_layer;
    return config;
  }
  if (options.krea2) {
    auto config = dif::frontend::make_krea2_conditioner_config();
    if (requested_layers != 0U)
      config.executed_layers = requested_layers;
    config.attention_implementation = options.attention;
    return config;
  }
  dif::frontend::Qwen3VlConditionerConfig config;
  if (requested_layers != 0U)
    config.executed_layers = requested_layers;
  config.attention_implementation = options.attention;
  config.vision_token_count = options.vision_tokens;
  return config;
}

std::uint64_t program_sequence(const dif::ir::Program &program) {
  for (const auto &tensor : program.tensors)
    if (tensor.has_role(dif::ir::TensorRole::Input) &&
        tensor.dtype == dif::ir::DType::I32 && tensor.dims.size() == 1U)
      return tensor.dims[0];
  dif::fail("conditioner program has no one-dimensional token-id input");
  return 0U;
}

void command_program(const Options &options) {
  if (options.sequence == 0U || options.output.empty())
    dif::fail("difcondition program requires --sequence and --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  if (!options.report.empty() && fs::exists(options.report))
    dif::fail("refusing to overwrite " + options.report.string());
  const auto build = dif::frontend::build_qwen3vl_conditioner_program(
      options.sequence, config_for(options));
  dif::ir::verify(build.program);
  dif::ir::write_file(build.program, options.output);
  std::cout << "CONDITIONER_PROGRAM path=" << options.output.string()
            << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(build.program))
            << " sequence=" << options.sequence
            << " layers=" << build.attention_operations
            << " operations=" << build.program.operations.size()
            << " linears=" << build.linear_operations
            << " attentions=" << build.attention_operations
            << " streamed_weights=" << build.bindings.size()
            << " input_id=" << build.token_ids_input_id
            << " output_id=" << build.conditioning_output_id
            << " outputs=" << build.conditioning_output_ids.size()
            << " vision_tokens=" << options.vision_tokens
            << " family="
            << (options.flux2 ? "flux2" : options.krea2 ? "krea2" : "h3")
            << "\n";
}

// Bind every streamed weight straight to the shard that already holds it:
// the checkpoint is the source of truth and is never copied or rewritten.
void command_bundle(const Options &options) {
  if (options.checkpoint.empty() || options.program.empty() ||
      options.output.empty())
    dif::fail("difcondition bundle requires --checkpoint, --program, --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  if (!options.report.empty() && fs::exists(options.report))
    dif::fail("refusing to overwrite " + options.report.string());
  const auto program = dif::ir::read_file(options.program);
  dif::ir::verify(program);

  // Rebuild the frontend description to recover tensor-id -> weight-name.
  const auto sequence = program_sequence(program);
  std::uint64_t layers = 0U;
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::Attention)
      ++layers;
  auto config = config_for(options, layers);
  const auto build =
      dif::frontend::build_qwen3vl_conditioner_program(sequence, config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("conditioner program does not reconstruct from its own geometry");

  const auto index_path = options.krea2
                              ? options.checkpoint / "model.safetensors.index.json"
                              : options.checkpoint / "text_encoder" /
                                    "model.safetensors.index.json";
  const auto index = dif::weights::read_safetensors_index(index_path);
  const auto index_fingerprint = dif::sha256_file(index_path);
  std::optional<dif::weights::WeightBundle> sealed;
  if (!options.sealed_bundle.empty()) {
    sealed = dif::weights::read_weight_bundle(options.sealed_bundle);
    if (sealed->index_fingerprint != index_fingerprint)
      dif::fail("sealed conditioner bundle targets a different checkpoint index");
  }

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  bundle.index_fingerprint = index_fingerprint;
  std::map<fs::path, std::uint32_t> shard_indices;
  std::map<fs::path, dif::weights::SafeTensorFile> shard_files;
  for (const auto &binding : build.bindings) {
    const auto entry = index.weight_map.find(binding.name);
    if (entry == index.weight_map.end())
      dif::fail("checkpoint index has no tensor named " + binding.name);
    const auto shard_path =
        fs::absolute(index_path.parent_path() / entry->second)
            .lexically_normal();
    auto known = shard_indices.find(shard_path);
    if (known == shard_indices.end()) {
      const auto shard_index = static_cast<std::uint32_t>(bundle.shards.size());
      const auto shard_size = fs::file_size(shard_path);
      auto digest = dif::Sha256Digest{};
      if (sealed) {
        const auto receipt = std::find_if(
            sealed->shards.begin(), sealed->shards.end(),
            [&](const dif::weights::BundleShard &candidate) {
              return fs::absolute(candidate.path).lexically_normal() ==
                     shard_path;
            });
        if (receipt == sealed->shards.end())
          dif::fail("sealed conditioner bundle has no receipt for " +
                    shard_path.string());
        if (receipt->file_size != shard_size)
          dif::fail("sealed conditioner shard size changed for " +
                    shard_path.string());
        digest = receipt->digest;
      } else {
        digest = dif::sha256_file(shard_path);
      }
      bundle.shards.push_back({shard_path, shard_size, digest});
      shard_files.emplace(shard_path,
                          dif::weights::read_safetensors(shard_path));
      known = shard_indices.emplace(shard_path, shard_index).first;
    }
    const auto &file = shard_files.at(shard_path);
    const auto *tensor = file.find(binding.name);
    if (!tensor)
      dif::fail("shard is missing indexed tensor " + binding.name);
    bundle.bindings.push_back({binding.tensor_id, known->second, binding.name,
                               tensor->dtype, tensor->dims,
                               tensor->file_offset, tensor->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, program, false);
  dif::weights::write_weight_bundle(bundle, options.output);
  std::cout << "CONDITIONER_BUNDLE path=" << options.output.string()
            << " program=" << dif::hex_digest(bundle.program_fingerprint)
            << " index=" << dif::hex_digest(bundle.index_fingerprint)
            << " shards=" << bundle.shards.size()
            << " bindings=" << bundle.bindings.size() << "\n";
}

void command_run(const Options &options) {
  const bool structured_inputs = options.krea2 || options.flux2;
  const bool missing_inputs =
      structured_inputs
          ? options.inputs.empty()
          : (options.vision_tokens != 0U
                 ? (options.vision_inputs.empty() ||
                    options.vision_outputs.empty())
                 : options.ids.empty());
  if (options.program.empty() || options.bundle.empty() || missing_inputs ||
      options.output.empty())
    dif::fail("difcondition run requires --program, --bundle, --output and "
              "either --ids, vision inputs, or structured --inputs");
  if (options.convrot_int8_weight_only_quality &&
      options.convrot_int8_checkpoint.empty())
    dif::fail("ConvRot weight-only quality requires a ConvRot checkpoint");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto program = dif::ir::read_file(options.program);
  dif::ir::verify(program);
  const auto bundle = dif::weights::read_weight_bundle(options.bundle);
  dif::weights::verify_weight_bundle(bundle, program, false);

  auto inputs = dif::weights::load_weight_bundle(bundle, program, false);
  // Rotary positions and inverse frequencies are compiler-derived constants,
  // not checkpoint tensors: rebuild the frontend description (fingerprint-
  // checked) and supply them alongside the bundle's weights.
  const auto sequence = program_sequence(program);
  std::uint64_t layers = 0U;
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::Attention)
      ++layers;
  auto config = config_for(options, layers);
  auto build =
      dif::frontend::build_qwen3vl_conditioner_program(sequence, config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("conditioner program does not reconstruct from its own geometry");
  for (auto &generated : build.generated_constants)
    inputs.insert_or_assign(generated.first, std::move(generated.second));

  std::uint64_t token_count = sequence;
  if (structured_inputs) {
    const auto fixture = dif::weights::read_safetensors(options.inputs);
    auto ids = dif::weights::map_safetensor(fixture, "input_ids");
    if (ids.dtype != dif::ir::DType::I32 ||
        ids.element_count() != sequence)
      dif::fail("conditioner input_ids must be I32 with the program sequence count");
    ids.dims = {sequence};
    ids.validate();
    inputs.insert_or_assign(build.token_ids_input_id, std::move(ids));
    inputs.insert_or_assign(
        build.attention_mask_input_id,
        dif::weights::map_safetensor(fixture, "attention_mask"));
    inputs.insert_or_assign(build.position_ids_input_id,
                            dif::weights::map_safetensor(fixture,
                                                        "position_ids"));
  } else if (options.vision_tokens != 0U) {
    const auto presentation =
        dif::weights::read_safetensors(options.vision_inputs);
    const auto vision = dif::weights::read_safetensors(options.vision_outputs);
    auto ids = dif::weights::map_safetensor(presentation, "input_ids");
    if (ids.dtype != dif::ir::DType::I32 || ids.dims.size() != 1U ||
        ids.dims[0] != sequence)
      dif::fail("multimodal H3 input_ids must be I32 [S]");
    inputs.insert_or_assign(build.token_ids_input_id, std::move(ids));
    inputs.insert_or_assign(
        build.vision_destination_map_input_id,
        dif::weights::map_safetensor(presentation,
                                     "vision_destination_map"));
    inputs.insert_or_assign(
        build.visual_positions_input_id,
        dif::weights::map_safetensor(presentation, "visual_positions"));
    inputs.insert_or_assign(
        build.vision_embeddings_input_id,
        dif::weights::map_safetensor(vision, "vision_embeds"));
    for (std::size_t index = 0U;
         index < build.vision_deepstack_input_ids.size(); ++index)
      inputs.insert_or_assign(
          build.vision_deepstack_input_ids[index],
          dif::weights::map_safetensor(vision,
                                       "deepstack_" + std::to_string(index)));
    token_count = sequence;
  } else {
    const auto ids = dif::runtime::read_tensor(options.ids);
    if (ids.dtype != dif::ir::DType::I32 || ids.dims.size() != 1U ||
        ids.dims[0] != sequence)
      dif::fail("token ids must be an I32 [S] tensor matching the program");
    inputs.insert_or_assign(build.token_ids_input_id, ids);
    token_count = ids.dims[0];
  }

  dif::runtime::RunOptions run_options;
  run_options.warmups = 0U;
  run_options.iterations = 1U;
  run_options.cache_directory = options.cache_directory;
  run_options.minimum_free_bytes = options.minimum_free_mib * 1024ULL * 1024ULL;
  run_options.profile_pipeline = options.profile_pipeline;
  run_options.convrot_int8_checkpoint = options.convrot_int8_checkpoint;
  run_options.convrot_int8_linear_count = options.convrot_int8_linear_count;
  run_options.convrot_int8_weight_only_quality =
      options.convrot_int8_weight_only_quality;
  run_options.streamed_stage_threads =
      static_cast<std::uint32_t>(options.streamed_stage_threads);
  if (options.resident_streamed) {
    for (const auto &tensor : program.tensors)
      if (tensor.has_role(dif::ir::TensorRole::Constant) &&
          tensor.has_role(dif::ir::TensorRole::Streamed))
        run_options.resident_streamed_constants.push_back(tensor.id);
  }

  auto backend = options.backend == "cpu" ? dif::runtime::make_cpu_executor()
                                          : dif::runtime::make_cuda_executor();
  const auto start = std::chrono::steady_clock::now();
  const auto result = backend->run(program, inputs, run_options);
  const auto wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  const auto &conditioning = result.outputs.at(build.conditioning_output_id);
  if (options.krea2 || options.flux2) {
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    if (options.flux2) {
      specs.push_back(
          {"conditioning", conditioning.dtype, conditioning.dims});
      for (const auto &[name, tensor_id] : build.first_layer_boundaries) {
        const auto &tensor = result.outputs.at(tensor_id);
        specs.push_back({"boundary_" + name, tensor.dtype, tensor.dims});
      }
    }
    for (std::size_t index = 0; index < build.conditioning_output_ids.size();
         ++index) {
      const auto &tensor = result.outputs.at(build.conditioning_output_ids[index]);
      specs.push_back({"tap_" + (index < 10U ? std::string("0") : std::string()) +
                           std::to_string(index),
                       tensor.dtype, tensor.dims});
    }
    dif::weights::SafeTensorWriter writer(options.output, std::move(specs));
    if (options.flux2)
      writer.append("conditioning",
                    std::span<const std::uint8_t>(conditioning.data(),
                                                  conditioning.byte_size()));
    if (options.flux2)
      for (const auto &[name, tensor_id] : build.first_layer_boundaries) {
        const auto &tensor = result.outputs.at(tensor_id);
        writer.append("boundary_" + name,
                      std::span<const std::uint8_t>(tensor.data(),
                                                    tensor.byte_size()));
      }
    for (std::size_t index = 0; index < build.conditioning_output_ids.size();
         ++index) {
      const auto &tensor = result.outputs.at(build.conditioning_output_ids[index]);
      const auto name = "tap_" +
                        (index < 10U ? std::string("0") : std::string()) +
                        std::to_string(index);
      writer.append(name, std::span<const std::uint8_t>(tensor.data(),
                                                        tensor.byte_size()));
    }
    (void)writer.finish();
  } else {
    dif::runtime::write_tensor(conditioning, options.output);
  }

  const auto payload_hash = dif::hex_digest(dif::sha256(
      {conditioning.data(),
       static_cast<std::size_t>(conditioning.byte_size())}));
  if (!options.report.empty()) {
    const auto &profile = result.pipeline_profile;
    const auto &telemetry = result.run_telemetry;
    std::ofstream report(options.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"backend\": " << std::quoted(result.backend_name)
           << ",\n  \"device\": " << std::quoted(result.device_name)
           << ",\n  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(program))
           << "\",\n  \"payload_sha256\": \"" << payload_hash
           << "\",\n  \"tokens\": " << token_count
           << ",\n  \"preparation_ms\": " << result.preparation_milliseconds
           << ",\n  \"execution_ms\": " << result.mean_milliseconds
           << ",\n  \"wall_seconds\": " << wall
           << ",\n  \"resident_bytes\": " << result.resident_bytes
           << ",\n  \"resident_streamed_constants\": "
           << run_options.resident_streamed_constants.size()
           << ",\n  \"pipeline_profile\": {"
           << "\"enabled\":" << (options.profile_pipeline ? "true" : "false")
           << ",\"resident_weight_bytes\":" << profile.resident_weight_bytes
           << ",\"resident_host_prefault_ms\":"
           << profile.resident_host_prefault_milliseconds
           << ",\"resident_h2d_ms\":" << profile.resident_h2d_milliseconds
           << ",\"streamed_weight_bytes\":" << profile.streamed_weight_bytes
           << ",\"streamed_host_stage_ms\":"
           << profile.streamed_host_stage_milliseconds
           << ",\"streamed_host_wait_ms\":"
           << profile.streamed_host_wait_milliseconds
           << ",\"streamed_h2d_ms\":" << profile.streamed_h2d_milliseconds
           << ",\"operation_kernel_ms\":"
           << profile.operation_kernel_milliseconds
           << ",\"attention_kernel_ms\":"
           << profile.attention_kernel_milliseconds
           << ",\"non_kernel_device_timeline_ms\":"
           << profile.non_kernel_device_timeline_milliseconds << "},\n"
           << "  \"telemetry\": {\"kernel_launches\":"
           << telemetry.kernel_launches
           << ",\"cublaslt_matmuls\":" << telemetry.cublaslt_matmuls
           << ",\"cudnn_attention_dispatches\":"
           << telemetry.cudnn_attention_dispatches
           << ",\"h2d_copies\":" << telemetry.h2d_copies
           << ",\"h2d_bytes\":" << telemetry.h2d_bytes
           << ",\"d2h_copies\":" << telemetry.d2h_copies
           << ",\"d2h_bytes\":" << telemetry.d2h_bytes
           << ",\"event_records\":" << telemetry.event_records
           << ",\"stream_wait_events\":" << telemetry.stream_wait_events
           << ",\"host_event_synchronizes\":"
           << telemetry.host_event_synchronizes
           << ",\"host_stream_synchronizes\":"
           << telemetry.host_stream_synchronizes << "},\n"
           << "  \"operations\": [";
    for (std::size_t index = 0U; index < result.operation_timings.size();
         ++index) {
      if (index != 0U)
        report << ',';
      const auto &timing = result.operation_timings[index];
      report << "{\"operation_id\":" << timing.operation_id
             << ",\"opcode\":" << std::quoted(dif::ir::opcode_name(timing.opcode))
             << ",\"mean_ms\":" << timing.mean_milliseconds
             << ",\"min_ms\":" << timing.minimum_milliseconds
             << ",\"max_ms\":" << timing.maximum_milliseconds << '}';
    }
    report << "]\n}\n";
    if (!report)
      dif::fail("failed to write conditioner report " +
                options.report.string());
  }

  std::cout << "CONDITIONING PASS backend=" << result.backend_name
            << " tokens=" << token_count << " shape=[" << conditioning.dims[0]
            << "," << conditioning.dims[1] << "]"
            << " payload_sha256=" << payload_hash
            << " prepare_ms=" << result.preparation_milliseconds
            << " run_ms=" << result.mean_milliseconds << " wall_s=" << wall
            << " resident_bytes=" << result.resident_bytes << "\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    if (options.command == "program")
      command_program(options);
    else if (options.command == "bundle")
      command_bundle(options);
    else if (options.command == "run")
      command_run(options);
    else {
      usage();
      dif::fail("unknown command: " + options.command);
    }
  } catch (const std::exception &error) {
    std::cerr << "difcondition: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
