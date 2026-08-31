// difcondition — native MiniMax-H3 text conditioning.
//
// Builds the Qwen3-VL text-tower conditioner as one DiffIR program, binds it
// straight to the checkpoint's own text_encoder shards (no derived copy, no
// Python loader), executes it, and writes the [S, hidden] BF16 conditioning
// the H3 denoiser consumes.
//
//   difcondition program  --checkpoint DIR --sequence N --output P.difir
//                         [--layers N] [--attention 1|2]
//   difcondition bundle   --checkpoint DIR --program P.difir --output B.difbind
//   difcondition run      --program P.difir --bundle B.difbind --ids I.diftensor
//                         --output C.diftensor [--backend cuda|cpu]
//                         [--cache-dir DIR] [--min-free-mib N] [--depth N]
//
// The `--depth` form of `program` builds a truncated tower (the parity ladder
// of docs/QWEN3VL_CONDITIONER_PLAN.md §5) so a depth-k program can be compared
// against the oracle's raw hidden_states[k].

#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/frontend/krea2.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
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
  fs::path output;
  fs::path cache_directory;
  std::string backend{"cuda"};
  std::uint64_t sequence{};
  std::uint64_t layers{50};
  std::uint64_t attention{2};
  std::uint64_t minimum_free_mib{4096};
  bool krea2{};
};

void usage() {
  std::cerr
      << "usage: difcondition program --checkpoint DIR --sequence N --output "
         "FILE.difir [--layers N] [--attention 1|2] [--krea2]\n"
         "       difcondition bundle --checkpoint DIR --program FILE.difir "
         "--output FILE.difbind\n"
         "       difcondition run --program FILE.difir --bundle FILE.difbind "
         "--ids FILE.diftensor --output FILE.diftensor [--backend cuda|cpu] "
         "[--cache-dir DIR] [--min-free-mib N]\n"
         "       difcondition run --krea2 --program FILE.difir --bundle "
         "FILE.difbind --inputs creator.safetensors --output "
         "native.safetensors [--cache-dir DIR] [--min-free-mib N]\n";
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
    else if (option == "--output")
      options.output = value("--output");
    else if (option == "--backend")
      options.backend = value("--backend");
    else if (option == "--cache-dir")
      options.cache_directory = value("--cache-dir");
    else if (option == "--sequence")
      options.sequence = number(value("--sequence"), "sequence");
    else if (option == "--layers")
      options.layers = number(value("--layers"), "layers");
    else if (option == "--attention")
      options.attention = number(value("--attention"), "attention");
    else if (option == "--min-free-mib")
      options.minimum_free_mib = number(value("--min-free-mib"), "min free MiB");
    else if (option == "--krea2")
      options.krea2 = true;
    else {
      usage();
      dif::fail("unknown option: " + option);
    }
  }
  return options;
}

dif::frontend::Qwen3VlConditionerConfig config_for(const Options &options) {
  if (options.krea2)
    return dif::frontend::make_krea2_conditioner_config();
  dif::frontend::Qwen3VlConditionerConfig config;
  config.executed_layers = options.layers;
  config.attention_implementation = options.attention;
  return config;
}

void command_program(const Options &options) {
  if (options.sequence == 0U || options.output.empty())
    dif::fail("difcondition program requires --sequence and --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto build = dif::frontend::build_qwen3vl_conditioner_program(
      options.sequence, config_for(options));
  dif::ir::verify(build.program);
  dif::ir::write_file(build.program, options.output);
  std::cout << "CONDITIONER_PROGRAM path=" << options.output.string()
            << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(build.program))
            << " sequence=" << options.sequence << " layers=" << options.layers
            << " operations=" << build.program.operations.size()
            << " linears=" << build.linear_operations
            << " attentions=" << build.attention_operations
            << " streamed_weights=" << build.bindings.size()
            << " input_id=" << build.token_ids_input_id
            << " output_id=" << build.conditioning_output_id
            << " outputs=" << build.conditioning_output_ids.size()
            << " family=" << (options.krea2 ? "krea2" : "h3") << "\n";
}

// Bind every streamed weight straight to the shard that already holds it:
// the checkpoint is the source of truth and is never copied or rewritten.
void command_bundle(const Options &options) {
  if (options.checkpoint.empty() || options.program.empty() ||
      options.output.empty())
    dif::fail("difcondition bundle requires --checkpoint, --program, --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto program = dif::ir::read_file(options.program);
  dif::ir::verify(program);

  // Rebuild the frontend description to recover tensor-id -> weight-name.
  std::uint64_t sequence = 0U;
  for (const auto &tensor : program.tensors)
    if ((tensor.roles & static_cast<std::uint32_t>(dif::ir::TensorRole::Input)) !=
        0U)
      sequence = tensor.dims.at(0);
  if (sequence == 0U)
    dif::fail("conditioner program has no token-id input");
  std::uint64_t layers = 0U;
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::Attention)
      ++layers;
  auto config = config_for(options);
  config.executed_layers = layers;
  const auto build =
      dif::frontend::build_qwen3vl_conditioner_program(sequence, config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("conditioner program does not reconstruct from its own geometry");

  const auto index_path = options.krea2
                              ? options.checkpoint / "model.safetensors.index.json"
                              : options.checkpoint / "text_encoder" /
                                    "model.safetensors.index.json";
  const auto index = dif::weights::read_safetensors_index(index_path);

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  bundle.index_fingerprint = dif::sha256_file(index_path);
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
      bundle.shards.push_back({shard_path, fs::file_size(shard_path),
                               dif::sha256_file(shard_path)});
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
  if (options.program.empty() || options.bundle.empty() ||
      (options.krea2 ? options.inputs.empty() : options.ids.empty()) ||
      options.output.empty())
    dif::fail("difcondition run requires --program, --bundle, --output and "
              "either --ids or Krea 2 --inputs");
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
  std::uint64_t sequence = 0U;
  for (const auto &tensor : program.tensors)
    if ((tensor.roles & static_cast<std::uint32_t>(dif::ir::TensorRole::Input)) !=
            0U &&
        tensor.dtype == dif::ir::DType::I32 && tensor.dims.size() == 1U)
      sequence = tensor.dims[0];
  std::uint64_t layers = 0U;
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::Attention)
      ++layers;
  auto config = config_for(options);
  config.executed_layers = layers;
  auto build =
      dif::frontend::build_qwen3vl_conditioner_program(sequence, config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("conditioner program does not reconstruct from its own geometry");
  for (auto &generated : build.generated_constants)
    inputs.insert_or_assign(generated.first, std::move(generated.second));

  std::uint64_t token_count = sequence;
  if (options.krea2) {
    const auto fixture = dif::weights::read_safetensors(options.inputs);
    auto ids = dif::weights::map_safetensor(fixture, "input_ids");
    if (ids.dtype != dif::ir::DType::I32 ||
        ids.element_count() != sequence)
      dif::fail("Krea 2 input_ids must be I32 with the program sequence count");
    ids.dims = {sequence};
    ids.validate();
    inputs.insert_or_assign(build.token_ids_input_id, std::move(ids));
    inputs.insert_or_assign(
        build.attention_mask_input_id,
        dif::weights::map_safetensor(fixture, "attention_mask"));
    inputs.insert_or_assign(build.position_ids_input_id,
                            dif::weights::map_safetensor(fixture,
                                                        "position_ids"));
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

  auto backend = options.backend == "cpu" ? dif::runtime::make_cpu_executor()
                                          : dif::runtime::make_cuda_executor();
  const auto start = std::chrono::steady_clock::now();
  const auto result = backend->run(program, inputs, run_options);
  const auto wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  const auto &conditioning = result.outputs.at(build.conditioning_output_id);
  if (options.krea2) {
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (std::size_t index = 0; index < build.conditioning_output_ids.size();
         ++index) {
      const auto &tensor = result.outputs.at(build.conditioning_output_ids[index]);
      specs.push_back({"tap_" + (index < 10U ? std::string("0") : std::string()) +
                           std::to_string(index),
                       tensor.dtype, tensor.dims});
    }
    dif::weights::SafeTensorWriter writer(options.output, std::move(specs));
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

  std::cout << "CONDITIONING PASS backend=" << result.backend_name
            << " tokens=" << token_count << " shape=[" << conditioning.dims[0]
            << "," << conditioning.dims[1] << "]"
            << " payload_sha256="
            << dif::hex_digest(dif::sha256(
                   {conditioning.data(),
                    static_cast<std::size_t>(conditioning.byte_size())}))
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
