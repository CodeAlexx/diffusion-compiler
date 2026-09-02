// difh3vision — native Qwen3-VL vision tower and MiniMax-H3 image presentation.
//
// The vision tower is one ordinary DiffIR program over shared Linear,
// LayerNorm, GELU, RotaryApply, Attention, layout, and residual operations.
// This CLI owns only checkpoint binding and H3's creator-defined presentation.

#include "dif/frontend/qwen3vl_vision.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/text/qwen_bpe_tokenizer.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string command;
  fs::path checkpoint;
  fs::path processor;
  std::vector<fs::path> images;
  fs::path prompt;
  fs::path program;
  fs::path bundle;
  fs::path inputs;
  std::vector<fs::path> vision_inputs;
  fs::path output;
  fs::path ids_output;
  fs::path tags_output;
  fs::path report;
  fs::path cache_directory;
  std::string backend{"cuda"};
  std::uint64_t attention_implementation{2U};
  std::uint64_t grid_t{1U};
  std::uint64_t grid_h{};
  std::uint64_t grid_w{};
  std::uint64_t minimum_free_mib{4096U};
  std::uint64_t streamed_stage_threads{1U};
  std::uint64_t warmups{};
  std::uint32_t cudnn_attention_heuristic{};
  std::vector<std::uint32_t> capture_tensors;
  fs::path capture_directory;
  bool trace{};
  bool profile_pipeline{};
  bool resident_streamed{};
  bool overlap_streaming{true};
  bool strip_trailing_newline{};
  bool verify_repeat{};
  bool deterministic_linear{};
};

void usage() {
  std::cerr
      << "usage: difh3vision program --grid-h N --grid-w N --output V.difir"
         " [--grid-t N] [--trace]\n"
         "       difh3vision bundle --checkpoint TEXT_ENCODER_DIR --program"
         " V.difir --grid-h N --grid-w N --output V.difbind [--grid-t N]"
         " [--trace]\n"
         "       difh3vision inputs --checkpoint TEXT_ENCODER_DIR --processor"
         " PROCESSOR_DIR --image FIRST.png [--image LAST.png]"
         " --prompt-file PROMPT --grid-h N --grid-w N"
         " --output INPUTS.safetensors [--grid-t N]\n"
         "         [--strip-trailing-newline] [--ids-out IDS.diftensor]"
         " [--tags-out TAGS.diftensor]\n"
         "       difh3vision run --program V.difir --bundle V.difbind --inputs"
         " INPUTS.safetensors --grid-h N --grid-w N --output"
         " VISION.safetensors [--grid-t N] [--backend cuda|cpu] [--trace]"
         " [--cache-dir DIR] [--min-free-mib N] [--resident-streamed]"
         " [--no-overlap-streaming] [--profile-pipeline]"
         " [--warmups N]"
         " [--attention generated|cudnn]"
         " [--deterministic-linear]"
         " [--verify-repeat]"
         " [--cudnn-attention-heuristic a|b|fallback|autotune|deterministic]"
         " [--capture-tensor ID --capture-dir DIR]"
         " [--report FILE.json]\n"
         "       difh3vision combine --vision-input FIRST.safetensors"
         " --vision-input LAST.safetensors --output BOTH.safetensors\n";
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
    dif::fail("difh3vision requires a command");
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
    else if (option == "--processor")
      options.processor = value("--processor");
    else if (option == "--image")
      options.images.emplace_back(value("--image"));
    else if (option == "--prompt-file")
      options.prompt = value("--prompt-file");
    else if (option == "--program")
      options.program = value("--program");
    else if (option == "--bundle")
      options.bundle = value("--bundle");
    else if (option == "--inputs")
      options.inputs = value("--inputs");
    else if (option == "--vision-input")
      options.vision_inputs.emplace_back(value("--vision-input"));
    else if (option == "--output")
      options.output = value("--output");
    else if (option == "--ids-out")
      options.ids_output = value("--ids-out");
    else if (option == "--tags-out")
      options.tags_output = value("--tags-out");
    else if (option == "--report")
      options.report = value("--report");
    else if (option == "--backend")
      options.backend = value("--backend");
    else if (option == "--attention") {
      const auto name = value("--attention");
      if (name == "generated")
        options.attention_implementation = 1U;
      else if (name == "cudnn")
        options.attention_implementation = 2U;
      else
        dif::fail("unknown vision attention implementation: " + name);
    }
    else if (option == "--cache-dir")
      options.cache_directory = value("--cache-dir");
    else if (option == "--grid-t")
      options.grid_t = number(value("--grid-t"), "grid t");
    else if (option == "--grid-h")
      options.grid_h = number(value("--grid-h"), "grid h");
    else if (option == "--grid-w")
      options.grid_w = number(value("--grid-w"), "grid w");
    else if (option == "--min-free-mib")
      options.minimum_free_mib = number(value("--min-free-mib"), "min free MiB");
    else if (option == "--streamed-stage-threads")
      options.streamed_stage_threads =
          number(value("--streamed-stage-threads"), "streamed stage threads");
    else if (option == "--warmups")
      options.warmups = number(value("--warmups"), "warmups");
    else if (option == "--capture-tensor") {
      const auto id = number(value("--capture-tensor"), "capture tensor");
      if (id > std::numeric_limits<std::uint32_t>::max())
        dif::fail("capture tensor id exceeds U32 range");
      options.capture_tensors.push_back(static_cast<std::uint32_t>(id));
    }
    else if (option == "--capture-dir")
      options.capture_directory = value("--capture-dir");
    else if (option == "--cudnn-attention-heuristic") {
      const auto name = value("--cudnn-attention-heuristic");
      if (name == "a")
        options.cudnn_attention_heuristic = 0U;
      else if (name == "b")
        options.cudnn_attention_heuristic = 1U;
      else if (name == "fallback")
        options.cudnn_attention_heuristic = 2U;
      else if (name == "autotune")
        options.cudnn_attention_heuristic = 3U;
      else if (name == "deterministic")
        options.cudnn_attention_heuristic = 4U;
      else
        dif::fail("unknown cuDNN attention heuristic: " + name);
    }
    else if (option == "--trace")
      options.trace = true;
    else if (option == "--profile-pipeline")
      options.profile_pipeline = true;
    else if (option == "--resident-streamed")
      options.resident_streamed = true;
    else if (option == "--no-overlap-streaming")
      options.overlap_streaming = false;
    else if (option == "--strip-trailing-newline")
      options.strip_trailing_newline = true;
    else if (option == "--verify-repeat")
      options.verify_repeat = true;
    else if (option == "--deterministic-linear")
      options.deterministic_linear = true;
    else {
      usage();
      dif::fail("unknown option: " + option);
    }
  }
  if (options.command != "combine" &&
      (options.grid_t == 0U || options.grid_h == 0U || options.grid_w == 0U))
    dif::fail("difh3vision requires positive --grid-t/--grid-h/--grid-w");
  if (options.backend != "cuda" && options.backend != "cpu")
    dif::fail("difh3vision backend must be cuda or cpu");
  if (options.warmups > std::numeric_limits<std::uint32_t>::max())
    dif::fail("vision warmups exceed U32 range");
  if (options.capture_tensors.empty() != options.capture_directory.empty())
    dif::fail("vision capture requires both --capture-tensor and --capture-dir");
  return options;
}

dif::frontend::Qwen3VlVisionConfig config_for(const Options &options) {
  dif::frontend::Qwen3VlVisionConfig config;
  config.trace_outputs = options.trace;
  config.attention_implementation = options.attention_implementation;
  return config;
}

std::string read_text(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    dif::fail("failed to open prompt " + path.string());
  std::string text((std::istreambuf_iterator<char>(stream)),
                   std::istreambuf_iterator<char>());
  if (stream.bad())
    dif::fail("failed to read prompt " + path.string());
  return text;
}

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

void write_named_tensors(
    const fs::path &path,
    const std::vector<std::pair<std::string, const dif::runtime::Tensor *>>
        &tensors) {
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(tensors.size());
  for (const auto &[name, tensor] : tensors)
    specs.push_back({name, tensor->dtype, tensor->dims});
  dif::weights::SafeTensorWriter writer(path, std::move(specs));
  for (const auto &[name, tensor] : tensors)
    writer.append(name, std::span<const std::uint8_t>(tensor->data(),
                                                      tensor->byte_size()));
  (void)writer.finish();
}

std::int32_t special_id(const dif::text::QwenBpeTokenizer &tokenizer,
                        std::string_view content) {
  const auto found = std::find_if(
      tokenizer.added_tokens().begin(), tokenizer.added_tokens().end(),
      [&](const auto &token) { return token.content == content; });
  if (found == tokenizer.added_tokens().end())
    dif::fail("H3 tokenizer has no special token " + std::string(content));
  return found->id;
}

void append_ids(std::vector<std::int32_t> &ids, std::vector<std::int32_t> &tags,
                const std::vector<std::int32_t> &part, std::int32_t tag) {
  ids.insert(ids.end(), part.begin(), part.end());
  tags.insert(tags.end(), part.size(), tag);
}

void command_program(const Options &options) {
  if (options.output.empty())
    dif::fail("difh3vision program requires --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto build = dif::frontend::build_qwen3vl_vision_program(
      options.grid_t, options.grid_h, options.grid_w, config_for(options));
  dif::ir::verify(build.program);
  dif::ir::write_file(build.program, options.output);
  std::cout << "H3_VISION_PROGRAM path=" << options.output
            << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(build.program))
            << " grid=[" << options.grid_t << ',' << options.grid_h << ','
            << options.grid_w << "] operations=" << build.program.operations.size()
            << " linears=" << build.linear_operations
            << " attentions=" << build.attention_operations
            << " bindings=" << build.bindings.size() << '\n';
}

void command_bundle(const Options &options) {
  if (options.checkpoint.empty() || options.program.empty() ||
      options.output.empty())
    dif::fail("difh3vision bundle requires --checkpoint, --program, --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto program = dif::ir::read_file(options.program);
  dif::ir::verify(program);
  const auto build = dif::frontend::build_qwen3vl_vision_program(
      options.grid_t, options.grid_h, options.grid_w, config_for(options));
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("vision program does not reconstruct from supplied geometry");
  const auto index_path = options.checkpoint / "model.safetensors.index.json";
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
        fs::absolute(index_path.parent_path() / entry->second).lexically_normal();
    auto known = shard_indices.find(shard_path);
    if (known == shard_indices.end()) {
      const auto shard_index = static_cast<std::uint32_t>(bundle.shards.size());
      bundle.shards.push_back({shard_path, fs::file_size(shard_path),
                               dif::sha256_file(shard_path)});
      shard_files.emplace(shard_path, dif::weights::read_safetensors(shard_path));
      known = shard_indices.emplace(shard_path, shard_index).first;
    }
    const auto &file = shard_files.at(shard_path);
    const auto *tensor = file.find(binding.name);
    if (!tensor)
      dif::fail("checkpoint shard is missing " + binding.name);
    bundle.bindings.push_back({binding.tensor_id, known->second, binding.name,
                               tensor->dtype, tensor->dims,
                               tensor->file_offset, tensor->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, program, false);
  dif::weights::write_weight_bundle(bundle, options.output);
  std::cout << "H3_VISION_BUNDLE path=" << options.output
            << " index=" << dif::hex_digest(bundle.index_fingerprint)
            << " shards=" << bundle.shards.size()
            << " bindings=" << bundle.bindings.size() << '\n';
}

dif::runtime::Tensor load_position_table(const fs::path &checkpoint) {
  constexpr std::string_view name = "model.visual.pos_embed.weight";
  const auto index_path = checkpoint / "model.safetensors.index.json";
  const auto index = dif::weights::read_safetensors_index(index_path);
  const auto entry = index.weight_map.find(name);
  if (entry == index.weight_map.end())
    dif::fail("checkpoint index has no Qwen3-VL position table");
  const auto file = dif::weights::read_safetensors(index_path.parent_path() /
                                                    entry->second);
  return dif::weights::map_safetensor(file, name);
}

void command_inputs(const Options &options) {
  if (options.checkpoint.empty() || options.processor.empty() ||
      options.images.empty() || options.prompt.empty() || options.output.empty())
    dif::fail(
        "difh3vision inputs requires checkpoint, processor, image(s), prompt, output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  if ((!options.ids_output.empty() && fs::exists(options.ids_output)) ||
      (!options.tags_output.empty() && fs::exists(options.tags_output)))
    dif::fail("refusing to overwrite H3 presentation tensor output");
  const auto config = config_for(options);
  std::vector<dif::runtime::Tensor> pixels;
  pixels.reserve(options.images.size());
  for (const auto &path : options.images) {
    const auto image = dif::read_png_rgb8(path);
    if (image.height != options.grid_h * config.patch_size ||
        image.width != options.grid_w * config.patch_size)
      dif::fail("H3 vision canvas dimensions do not match grid geometry");
    pixels.push_back(
        dif::frontend::qwen3vl_vision_image_patch_rows(image, config));
  }
  const auto table = load_position_table(options.checkpoint);
  auto positions = dif::frontend::qwen3vl_vision_position_embeddings(
      table, options.grid_t, options.grid_h, options.grid_w, config);
  const auto tokenizer = dif::text::QwenBpeTokenizer::load(
      options.processor / "tokenizer.json",
      options.processor / "tokenizer_config.json");
  const auto vision_start = special_id(tokenizer, "<|vision_start|>");
  const auto vision_end = special_id(tokenizer, "<|vision_end|>");
  const auto image_pad = special_id(tokenizer, "<|image_pad|>");
  const auto merge = config.spatial_merge_size * config.spatial_merge_size;
  const auto vision_tokens = options.grid_t * options.grid_h * options.grid_w /
                             merge;
  std::vector<std::int32_t> ids;
  std::vector<std::int32_t> tags;
  std::vector<std::int32_t> visual_positions;
  visual_positions.reserve(static_cast<std::size_t>(
      vision_tokens * static_cast<std::uint64_t>(options.images.size())));
  for (std::size_t image = 0U; image < options.images.size(); ++image) {
    append_ids(ids, tags,
               tokenizer.encode("<Picture " + std::to_string(image + 1U) +
                                ">: "),
               1);
    ids.push_back(vision_start);
    tags.push_back(0);
    const auto first_visual = ids.size();
    ids.insert(ids.end(), static_cast<std::size_t>(vision_tokens), image_pad);
    tags.insert(tags.end(), static_cast<std::size_t>(vision_tokens), 0);
    for (std::uint64_t index = 0U; index < vision_tokens; ++index)
      visual_positions.push_back(
          static_cast<std::int32_t>(first_visual + index));
    ids.push_back(vision_end);
    tags.push_back(0);
  }
  auto prompt = read_text(options.prompt);
  if (options.strip_trailing_newline && !prompt.empty() && prompt.back() == '\n') {
    prompt.pop_back();
    if (!prompt.empty() && prompt.back() == '\r')
      prompt.pop_back();
  }
  append_ids(ids, tags, tokenizer.encode(prompt), 1);
  std::vector<std::int32_t> destination_map(ids.size(), -1);
  for (std::size_t index = 0U; index < visual_positions.size(); ++index)
    destination_map[static_cast<std::size_t>(visual_positions[index])] =
        static_cast<std::int32_t>(index);
  auto ids_tensor = i32_tensor({ids.size()}, ids);
  auto tags_tensor = i32_tensor({tags.size()}, tags);
  auto map_tensor = i32_tensor({destination_map.size()}, destination_map);
  auto visual_tensor = i32_tensor({visual_positions.size()}, visual_positions);
  std::vector<std::pair<std::string, const dif::runtime::Tensor *>> tensors{
      {"input_ids", &ids_tensor},
      {"token_tags", &tags_tensor},
      {"vision_destination_map", &map_tensor},
      {"visual_positions", &visual_tensor},
      {"position_embeddings", &positions}};
  for (std::size_t image = 0U; image < pixels.size(); ++image) {
    const auto name = pixels.size() == 1U
                          ? std::string("pixel_patches")
                          : "pixel_patches_" + std::to_string(image);
    tensors.push_back({name, &pixels[image]});
  }
  write_named_tensors(options.output, tensors);
  if (!options.ids_output.empty())
    dif::runtime::write_tensor(ids_tensor, options.ids_output);
  if (!options.tags_output.empty())
    dif::runtime::write_tensor(tags_tensor, options.tags_output);
  std::cout << "H3_REF_PRESENTATION path=" << options.output
            << " sequence=" << ids.size() << " images=" << options.images.size()
            << " vision_tokens_per_image=" << vision_tokens << " canvas="
            << options.grid_w * config.patch_size << 'x'
            << options.grid_h * config.patch_size << " prompt_sha256="
            << dif::hex_digest(dif::sha256(
                   {reinterpret_cast<const std::uint8_t *>(prompt.data()),
                    prompt.size()}))
            << '\n';
}

void append_rows(dif::runtime::Tensor &destination,
                 const dif::runtime::Tensor &source) {
  source.validate();
  if (source.dims.size() != 2U)
    dif::fail("H3 vision output must be rank two");
  if (destination.dims.empty()) {
    destination = dif::runtime::Tensor{source.dtype, source.dims, {}};
    destination.bytes.resize(static_cast<std::size_t>(source.byte_size()));
    std::memcpy(destination.bytes.data(), source.data(),
                static_cast<std::size_t>(source.byte_size()));
    return;
  }
  if (destination.dtype != source.dtype || destination.dims.size() != 2U ||
      destination.dims[1] != source.dims[1])
    dif::fail("H3 vision images produced incompatible output tensors");
  const auto source_bytes = static_cast<std::size_t>(source.byte_size());
  if (source_bytes > destination.bytes.max_size() - destination.bytes.size())
    dif::fail("H3 vision output byte size overflows");
  const auto byte_offset = destination.bytes.size();
  destination.bytes.resize(byte_offset + source_bytes);
  std::memcpy(destination.bytes.data() + byte_offset, source.data(),
              source_bytes);
  destination.dims[0] += source.dims[0];
  destination.validate();
}

void command_combine(const Options &options) {
  if (options.vision_inputs.size() < 2U || options.output.empty())
    dif::fail("difh3vision combine requires at least two --vision-input files and --output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  std::vector<std::string> names{"vision_embeds", "deepstack_0",
                                 "deepstack_1", "deepstack_2"};
  const auto first = dif::weights::read_safetensors(options.vision_inputs[0]);
  for (std::size_t index = 0U;; ++index) {
    const auto name = "trace_" + std::to_string(index);
    if (!first.find(name))
      break;
    names.push_back(name);
  }
  std::map<std::string, dif::runtime::Tensor, std::less<>> combined;
  for (const auto &path : options.vision_inputs) {
    const auto file = dif::weights::read_safetensors(path);
    for (const auto &name : names) {
      if (!file.find(name))
        dif::fail("H3 vision input is missing tensor " + name + ": " +
                  path.string());
      append_rows(combined[name], dif::weights::map_safetensor(file, name));
    }
  }
  std::vector<std::pair<std::string, const dif::runtime::Tensor *>> output;
  output.reserve(names.size());
  for (const auto &name : names)
    output.push_back({name, &combined.at(name)});
  write_named_tensors(options.output, output);
  const auto &embeds = combined.at("vision_embeds");
  const auto hash = dif::hex_digest(dif::sha256(
      {embeds.data(), static_cast<std::size_t>(embeds.byte_size())}));
  std::cout << "H3_VISION_COMBINE path=" << options.output
            << " images=" << options.vision_inputs.size() << " shape=["
            << embeds.dims.at(0) << ',' << embeds.dims.at(1)
            << "] payload_sha256=" << hash << '\n';
}

void command_run(const Options &options) {
  if (options.program.empty() || options.bundle.empty() ||
      options.inputs.empty() || options.output.empty())
    dif::fail("difh3vision run requires program, bundle, inputs, output");
  if (fs::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  const auto program = dif::ir::read_file(options.program);
  dif::ir::verify(program);
  const auto bundle = dif::weights::read_weight_bundle(options.bundle);
  dif::weights::verify_weight_bundle(bundle, program, false);
  auto tensors = dif::weights::load_weight_bundle(bundle, program, false);
  auto build = dif::frontend::build_qwen3vl_vision_program(
      options.grid_t, options.grid_h, options.grid_w, config_for(options));
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("vision program does not reconstruct from supplied geometry");
  for (auto &constant : build.generated_constants)
    tensors.insert_or_assign(constant.first, std::move(constant.second));
  const auto input_file = dif::weights::read_safetensors(options.inputs);
  std::vector<std::string> pixel_names;
  if (input_file.find("pixel_patches")) {
    pixel_names.push_back("pixel_patches");
  } else {
    for (std::size_t index = 0U;; ++index) {
      const auto name = "pixel_patches_" + std::to_string(index);
      if (!input_file.find(name))
        break;
      pixel_names.push_back(name);
    }
  }
  if (pixel_names.empty())
    dif::fail("H3 vision inputs contain no pixel patch tensor");
  tensors.insert_or_assign(
      build.pixel_patches_input_id,
      dif::weights::map_safetensor(input_file, pixel_names.front()));
  tensors.insert_or_assign(
      build.position_embeddings_input_id,
      dif::weights::map_safetensor(input_file, "position_embeddings"));
  dif::runtime::RunOptions run_options;
  run_options.warmups = static_cast<std::uint32_t>(options.warmups);
  run_options.iterations = 1U;
  run_options.cache_directory = options.cache_directory;
  run_options.minimum_free_bytes = options.minimum_free_mib * 1024ULL * 1024ULL;
  run_options.cudnn_attention_heuristic = options.cudnn_attention_heuristic;
  run_options.deterministic_linear_algorithms = options.deterministic_linear;
  run_options.profile_pipeline = options.profile_pipeline;
  run_options.overlap_streaming = options.overlap_streaming;
  run_options.streamed_stage_threads =
      static_cast<std::uint32_t>(options.streamed_stage_threads);
  run_options.capture_intermediate_tensors = options.capture_tensors;
  if (options.resident_streamed)
    for (const auto &tensor : program.tensors)
      if (tensor.has_role(dif::ir::TensorRole::Constant) &&
          tensor.has_role(dif::ir::TensorRole::Streamed))
        run_options.resident_streamed_constants.push_back(tensor.id);
  const auto start = std::chrono::steady_clock::now();
  std::vector<dif::runtime::Tensor> combined(
      1U + build.deepstack_output_ids.size() + build.trace_output_ids.size());
  double preparation_ms = 0.0;
  double execution_ms = 0.0;
  std::string backend_name;
  std::string device_name;
  std::uint64_t resident_bytes = 0U;
  auto backend = options.backend == "cpu"
                     ? dif::runtime::make_cpu_executor()
                     : dif::runtime::make_cuda_executor();
  auto prepared = backend->prepare(program, tensors, run_options);
  preparation_ms = prepared->preparation_milliseconds();
  if (!options.capture_directory.empty()) {
    if (fs::exists(options.capture_directory) &&
        !fs::is_empty(options.capture_directory))
      dif::fail("vision capture directory is not empty: " +
                options.capture_directory.string());
    fs::create_directories(options.capture_directory);
  }
  for (std::size_t image = 0U; image < pixel_names.size(); ++image) {
    if (image != 0U)
      tensors.insert_or_assign(
          build.pixel_patches_input_id,
          dif::weights::map_safetensor(input_file, pixel_names[image]));
    const auto result = prepared->run(tensors, run_options);
    for (const auto tensor_id : options.capture_tensors) {
      const auto captured = result.captured_intermediates.find(tensor_id);
      if (captured == result.captured_intermediates.end())
        dif::fail("requested vision capture was not returned for tensor " +
                  std::to_string(tensor_id));
      dif::runtime::write_tensor(
          captured->second,
          options.capture_directory /
              ("image-" + std::to_string(image) + "-tensor-" +
               std::to_string(tensor_id) + ".diftensor"));
    }
    if (options.verify_repeat) {
      const auto repeated = prepared->run(tensors, run_options);
      std::vector<std::uint32_t> output_ids = build.trace_output_ids;
      output_ids.insert(output_ids.end(), build.deepstack_output_ids.begin(),
                        build.deepstack_output_ids.end());
      output_ids.push_back(build.embeds_output_id);
      for (const auto output_id : output_ids) {
        const auto &first = result.outputs.at(output_id);
        const auto &second = repeated.outputs.at(output_id);
        if (first.dtype != second.dtype || first.dims != second.dims ||
            first.bytes.size() != second.bytes.size())
          dif::fail("repeated prepared vision execution changed tensor metadata at output " +
                    std::to_string(output_id));
        const auto mismatch = std::mismatch(first.bytes.begin(), first.bytes.end(),
                                            second.bytes.begin());
        if (mismatch.first != first.bytes.end()) {
          const auto first_offset = static_cast<std::size_t>(
              std::distance(first.bytes.begin(), mismatch.first));
          const auto mismatches = static_cast<std::size_t>(std::count_if(
              first.bytes.begin(), first.bytes.end(),
              [&, offset = std::size_t{0U}](std::uint8_t value) mutable {
                return value != second.bytes[offset++];
              }));
          dif::fail("repeated prepared vision execution diverged at output " +
                    std::to_string(output_id) + " first_byte=" +
                    std::to_string(first_offset) + " mismatch_bytes=" +
                    std::to_string(mismatches));
        }
      }
      std::cout << "H3_VISION_REPEAT PASS outputs=" << output_ids.size()
                << '\n';
    }
    backend_name = result.backend_name;
    device_name = result.device_name;
    resident_bytes = result.resident_bytes;
    execution_ms += result.mean_milliseconds;
    append_rows(combined[0U], result.outputs.at(build.embeds_output_id));
    for (std::size_t index = 0U;
         index < build.deepstack_output_ids.size(); ++index)
      append_rows(combined[1U + index],
                  result.outputs.at(build.deepstack_output_ids[index]));
    for (std::size_t index = 0U; index < build.trace_output_ids.size(); ++index)
      append_rows(combined[1U + build.deepstack_output_ids.size() + index],
                  result.outputs.at(build.trace_output_ids[index]));
  }
  const auto wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  std::vector<std::pair<std::string, const dif::runtime::Tensor *>> outputs;
  outputs.push_back({"vision_embeds", &combined[0U]});
  for (std::size_t index = 0U; index < build.deepstack_output_ids.size(); ++index)
    outputs.push_back({"deepstack_" + std::to_string(index),
                       &combined[1U + index]});
  for (std::size_t index = 0U; index < build.trace_output_ids.size(); ++index)
    outputs.push_back({"trace_" + std::to_string(index),
                       &combined[1U + build.deepstack_output_ids.size() + index]});
  write_named_tensors(options.output, outputs);
  const auto &embeds = combined[0U];
  const auto hash = dif::hex_digest(dif::sha256(
      {embeds.data(), static_cast<std::size_t>(embeds.byte_size())}));
  if (!options.report.empty()) {
    std::ofstream report(options.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"backend\": " << std::quoted(backend_name)
           << ",\n  \"device\": " << std::quoted(device_name)
           << ",\n  \"images\": " << pixel_names.size()
           << ",\n  \"diffir_fingerprint\": \""
           << dif::hex_digest(dif::ir::fingerprint(program))
           << "\",\n  \"payload_sha256\": \"" << hash
           << "\",\n  \"preparation_ms\": " << preparation_ms
           << ",\n  \"execution_ms\": " << execution_ms
           << ",\n  \"wall_seconds\": " << wall
           << ",\n  \"resident_bytes\": " << resident_bytes
           << "\n}\n";
    if (!report)
      dif::fail("failed to write H3 vision report");
  }
  std::cout << "H3_VISION PASS backend=" << backend_name << " images="
            << pixel_names.size() << " shape=["
            << embeds.dims.at(0) << ',' << embeds.dims.at(1)
            << "] payload_sha256=" << hash
            << " prepare_ms=" << preparation_ms << " run_ms=" << execution_ms
            << " wall_s=" << wall << " resident_bytes=" << resident_bytes
            << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    if (options.command == "program")
      command_program(options);
    else if (options.command == "bundle")
      command_bundle(options);
    else if (options.command == "inputs")
      command_inputs(options);
    else if (options.command == "run")
      command_run(options);
    else if (options.command == "combine")
      command_combine(options);
    else {
      usage();
      dif::fail("unknown command: " + options.command);
    }
  } catch (const std::exception &error) {
    std::cerr << "difh3vision: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
