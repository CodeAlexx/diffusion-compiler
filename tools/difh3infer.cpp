#include "dif/backend/plugin.hpp"
#include "dif/frontend/h3_conditioning.hpp"
#include "dif/frontend/h3_latents.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/bundle.hpp"

#include <algorithm>
#include <cerrno>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
  std::string backend{"cuda"};
  std::string sampler{"euler"};
  std::filesystem::path backend_plugin;
  std::filesystem::path denoiser_program;
  std::filesystem::path denoiser_bundle;
  std::filesystem::path vae_program;
  std::filesystem::path vae_bundle;
  std::filesystem::path text_tags;
  std::filesystem::path text;
  std::filesystem::path video;
  std::filesystem::path audio;
  std::filesystem::path video_sigmas;
  std::filesystem::path audio_sigmas;
  std::filesystem::path output_latent;
  std::filesystem::path output_video_rows;
  std::filesystem::path output_audio;
  std::filesystem::path output_audio_latent;
  std::filesystem::path output_handoff;
  std::filesystem::path output_raw;
  std::filesystem::path output_decoded;
  std::filesystem::path first_evaluation_input_directory;
  std::filesystem::path capture_denoiser_directory;
  std::filesystem::path cache_directory;
  std::filesystem::path h3_w8a8_cache;
  std::filesystem::path h3_convrot_int8_checkpoint;
  std::filesystem::path h3_groupwise_cache;
  std::filesystem::path h3_ck_attention_dso;
  bool h3_owned_attention{false};
  bool h3_owned_attention_center_k{false};
  std::uint64_t h3_resident_readahead_mib{std::numeric_limits<std::uint64_t>::max()};
  bool h3_resident_mapped_copy{false}; // --h3-resident-mapped-copy: no direct IO
  // --h3-int8-attention-first-step N: evaluations before N run exact cuDNN
  // attention, later ones the INT8 route.
  std::uint32_t h3_int8_attention_first_step{};
  // Persistent denoiser: --serve SOCKET keeps this process alive after the
  // first request and serves further requests over a Unix socket with the
  // prepared denoiser and its resident weights kept on the device.
  std::filesystem::path serve_socket;
  // Every argv token that can change what prepare() builds, in order. A served
  // request whose signature differs from the prepared one is refused.
  std::string prepare_signature;
  std::filesystem::path h3_modulation_cache;
  std::filesystem::path h3_modulation_source_index;
  std::uint32_t h3_w8a8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_convrot_int8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_convrot_int8_attention_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_convrot_int8_mlp_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_groupwise_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_int8_mlp_chunk_rows{1024U};
  std::uint32_t h3_int8_attention_first_layer{};
  std::uint32_t h3_int8_attention_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::vector<std::uint32_t> resident_streamed_constants;
  bool h3_int8_cublaslt{false};
  std::uint32_t h3_int8_cublaslt_heuristic_rank{};
  bool h3_int8_cublaslt_tune{false};
  bool h3_int8_cutlass_scaled_fc1{false};
  bool h3_int8_cutlass_scaled_all{false};
  std::uint32_t h3_int8_convrot_scale_chunk{};
  bool h3_int8_convrot_global_activation_scale{false};
  bool h3_convrot_bf16_audio_rows{false};
  bool h3_int8_compact_adaln{false};
  bool h3_cache_text_refiner{false};
  std::uint32_t cudnn_attention_heuristic{};
  std::uint32_t h3_modulation_steps{};
  std::uint32_t h3_modulation_first_step{};
  std::uint64_t all_text_tokens{};
  std::uint64_t latent_frames{};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t audio_latents{};
  std::uint64_t patch_height{2U};
  std::uint64_t patch_width{2U};
  std::uint32_t schedule_points{};
  std::uint32_t simple_evaluations{};
  std::uint32_t maximum_evaluations{};
  std::vector<dif::frontend::H3KeyframeAnchor> keyframes;
  std::vector<dif::frontend::H3ReferenceGeometry> references;
  std::vector<std::uint32_t> capture_denoiser_tensors;
  std::uint64_t minimum_free_bytes{4096ULL * 1024ULL * 1024ULL};
  bool verify_shards{false};
  bool profile_pipeline{false};
  // Shared streamed-weight staging policy (Wave-1 runtime knobs). Defaults
  // mirror RunOptions exactly, so an unflagged run is the historical path.
  bool streamed_keep_pages{false};
  bool pipelined_resident_upload{false};
  bool lazy_resident_upload{false};
  bool keep_resident_host_pages{false};
  std::uint32_t streamed_staging_buffers{};
  std::uint32_t streamed_prefetch_depth{};
  std::uint32_t streamed_stage_threads{};
  std::uint64_t streamed_pinned_budget_mib{};
  bool pinned_io{false};
  bool denoise_only{false};
};

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (text.empty() || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

void usage() {
  std::cerr
      << "usage: difh3infer --backend cpu|cuda --sampler euler|res_multistep --denoiser-program FILE.difir --denoiser-bundle FILE.difbind (--text-tags FILE.diftensor | --all-text-tokens N) --text FILE.diftensor --video FILE.diftensor --audio FILE.diftensor (--simple-steps N | --schedule-points N | --video-sigmas FILE.diftensor --audio-sigmas FILE.diftensor) --latent-t N --latent-h N --latent-w N --audio-latents N [--keyframes none|first|last|first-last | --reference-geometry KIND:T:H:W:A ...] --output-latent FILE.diftensor [--output-video-rows FILE.diftensor] --output-audio FILE.diftensor [--output-audio-latent FILE.diftensor] [--output-handoff latents.safetensors] [--h3-w8a8-cache FILE.safetensors --h3-w8a8-resident-layers N | --h3-convrot-int8-checkpoint FILE.safetensors [--h3-convrot-int8-layers N | --h3-convrot-int8-attention-layers N --h3-convrot-int8-mlp-layers N] --h3-convrot-int8-resident-layers N [--h3-convrot-bf16-audio-rows] | --h3-groupwise-cache FILE.safetensors --h3-groupwise-layers N] [--h3-int8-mlp-chunk-rows N] [--h3-int8-cublaslt --h3-int8-cublaslt-rank N --h3-int8-cublaslt-tune] [--h3-int8-cutlass-scaled-fc1] [--h3-int8-cutlass-scaled-all | --h3-int8-convrot-scale-chunk N] [--h3-int8-compact-adaln] [--h3-cache-text-refiner] [--resident-streamed-constant TENSOR_ID ...] [--cudnn-attention-heuristic a|b|fallback|autotune] [--h3-modulation-cache FILE.safetensors --h3-modulation-source-index FILE.index.json [--h3-modulation-steps N]] [(--h3-ck-attention-dso FILE.so | --h3-owned-attention [--h3-owned-attention-center-k]) --h3-int8-attention-first-layer N --h3-int8-attention-layers N] [--h3-int8-attention-first-step N] [--denoise-only | --vae-program FILE.difir --vae-bundle FILE.difbind --output-raw FILE.diftensor --output-decoded FILE.diftensor] [--first-eval-input-dir DIR] [--capture-denoiser-dir DIR --capture-denoiser-tensor ID ...] [--max-evaluations N] [--patch-h N] [--patch-w N] [--backend-plugin FILE.so] [--verify-shards] [--profile-pipeline] [--streamed-keep-pages] [--pipelined-resident-upload | --lazy-resident-upload] [--h3-resident-readahead-mib N] [--h3-resident-mapped-copy] [--keep-resident-host-pages] [--streamed-staging-buffers N] [--streamed-prefetch-depth N] [--streamed-stage-threads N] [--streamed-pinned-budget-mib N] [--pinned-io] [--cache-dir DIR] [--min-free-mib N] [--serve SOCKET | --connect SOCKET]\n";
}

std::vector<dif::frontend::H3KeyframeAnchor>
keyframes(const std::string &name) {
  using dif::frontend::H3KeyframeAnchor;
  if (name == "none")
    return {};
  if (name == "first")
    return {H3KeyframeAnchor::First};
  if (name == "last")
    return {H3KeyframeAnchor::Last};
  if (name == "first-last")
    return {H3KeyframeAnchor::First, H3KeyframeAnchor::Last};
  dif::fail("keyframes must be none, first, last, or first-last");
}

dif::frontend::H3ReferenceGeometry
reference_geometry(const std::string &specification) {
  std::array<std::string, 5> fields;
  std::size_t start = 0U;
  for (std::size_t field = 0U; field < fields.size(); ++field) {
    const auto separator = specification.find(':', start);
    if ((field + 1U < fields.size() && separator == std::string::npos) ||
        (field + 1U == fields.size() && separator != std::string::npos))
      dif::fail("reference geometry must be KIND:T:H:W:A");
    fields[field] = specification.substr(
        start, separator == std::string::npos ? separator : separator - start);
    if (separator == std::string::npos)
      break;
    start = separator + 1U;
  }
  dif::frontend::H3ReferenceKind kind;
  if (fields[0] == "image")
    kind = dif::frontend::H3ReferenceKind::Image;
  else if (fields[0] == "video")
    kind = dif::frontend::H3ReferenceKind::Video;
  else if (fields[0] == "audio")
    kind = dif::frontend::H3ReferenceKind::Audio;
  else
    dif::fail("reference kind must be image, video, or audio");
  return {kind, number(fields[1], "reference latent frames"),
          number(fields[2], "reference latent height"),
          number(fields[3], "reference latent width"),
          number(fields[4], "reference audio latents")};
}

std::vector<std::uint32_t>
h3_text_refiner_invariant_operations(const dif::ir::Program &program) {
  const auto *text = program.tensor(3U);
  if (!text || !text->has_role(dif::ir::TensorRole::Input) ||
      text->dims.size() != 2U)
    dif::fail("H3 text-refiner cache requires canonical text input tensor 3");

  std::unordered_map<std::uint32_t, const dif::ir::Operation *> producers;
  for (const auto &operation : program.operations)
    for (const auto output : operation.outputs)
      producers.emplace(output, &operation);

  for (const auto &update : program.operations) {
    if (update.opcode != dif::ir::Opcode::IndexedUpdateRows ||
        update.inputs.size() != 3U)
      continue;
    const auto *candidate = program.tensor(update.inputs.at(1));
    if (!candidate || candidate->dims.size() != 2U ||
        candidate->dims.front() != text->dims.front())
      continue;

    std::unordered_set<std::uint32_t> operation_ids;
    std::unordered_set<std::uint32_t> external_inputs;
    std::function<void(std::uint32_t)> visit = [&](std::uint32_t tensor_id) {
      const auto produced = producers.find(tensor_id);
      if (produced == producers.end()) {
        const auto *description = program.tensor(tensor_id);
        if (!description)
          dif::fail("H3 text-refiner dependency references a missing tensor");
        if (description->has_role(dif::ir::TensorRole::Input))
          external_inputs.insert(tensor_id);
        else if (!description->has_role(dif::ir::TensorRole::Constant))
          dif::fail("H3 text-refiner dependency is neither input nor constant");
        return;
      }
      if (!operation_ids.insert(produced->second->id).second)
        return;
      for (const auto input : produced->second->inputs)
        visit(input);
    };
    visit(update.inputs.at(1));
    if (external_inputs != std::unordered_set<std::uint32_t>{3U})
      continue;
    std::vector<std::uint32_t> result;
    result.reserve(operation_ids.size());
    for (const auto &operation : program.operations)
      if (operation_ids.contains(operation.id))
        result.push_back(operation.id);
    if (result.empty())
      dif::fail("H3 text-refiner cache found an empty invariant region");
    return result;
  }
  dif::fail("H3 text-refiner cache could not identify the creator dependency frontier");
}

Options parse(int argc, char **argv) {
  Options options;
  // Options that only shape one request (inputs, schedule, geometry validated
  // against the prepared program, outputs, diagnostics). Everything else can
  // change what prepare() builds and is folded into prepare_signature.
  static const std::unordered_set<std::string> per_request_options = {
      "--sampler",          "--vae-program",        "--vae-bundle",
      "--text-tags",        "--all-text-tokens",    "--text",
      "--video",            "--audio",              "--video-sigmas",
      "--audio-sigmas",     "--schedule-points",    "--steps",
      "--simple-steps",     "--latent-t",           "--latent-h",
      "--latent-w",         "--audio-latents",      "--patch-h",
      "--patch-w",          "--keyframes",          "--reference-geometry",
      "--output-latent",    "--output-video-rows",  "--output-audio",
      "--output-audio-latent", "--output-handoff",  "--output-raw",
      "--output-decoded",   "--first-eval-input-dir", "--capture-denoiser-dir",
      "--capture-denoiser-tensor", "--max-evaluations", "--verify-shards",
      "--profile-pipeline", "--h3-modulation-first-step", "--denoise-only",
      "--serve",            "--connect"};
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const int option_start = index;
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--backend")
      options.backend = value("--backend");
    else if (option == "--sampler")
      options.sampler = value("--sampler");
    else if (option == "--backend-plugin")
      options.backend_plugin = value("--backend-plugin");
    else if (option == "--denoiser-program")
      options.denoiser_program = value("--denoiser-program");
    else if (option == "--denoiser-bundle")
      options.denoiser_bundle = value("--denoiser-bundle");
    else if (option == "--vae-program")
      options.vae_program = value("--vae-program");
    else if (option == "--vae-bundle")
      options.vae_bundle = value("--vae-bundle");
    else if (option == "--text-tags")
      options.text_tags = value("--text-tags");
    else if (option == "--all-text-tokens")
      options.all_text_tokens =
          number(value("--all-text-tokens"), "all text tokens");
    else if (option == "--text")
      options.text = value("--text");
    else if (option == "--video")
      options.video = value("--video");
    else if (option == "--audio")
      options.audio = value("--audio");
    else if (option == "--video-sigmas")
      options.video_sigmas = value("--video-sigmas");
    else if (option == "--audio-sigmas")
      options.audio_sigmas = value("--audio-sigmas");
    else if (option == "--schedule-points" || option == "--steps") {
      const auto points = number(value(option.c_str()), "schedule points");
      if (points > std::numeric_limits<std::uint32_t>::max())
        dif::fail("schedule points exceed U32 range");
      options.schedule_points = static_cast<std::uint32_t>(points);
    }
    else if (option == "--simple-steps") {
      const auto evaluations =
          number(value("--simple-steps"), "simple scheduler steps");
      if (evaluations > std::numeric_limits<std::uint32_t>::max())
        dif::fail("simple scheduler steps exceed U32 range");
      options.simple_evaluations =
          static_cast<std::uint32_t>(evaluations);
    }
    else if (option == "--latent-t")
      options.latent_frames = number(value("--latent-t"), "latent frames");
    else if (option == "--latent-h")
      options.latent_height = number(value("--latent-h"), "latent height");
    else if (option == "--latent-w")
      options.latent_width = number(value("--latent-w"), "latent width");
    else if (option == "--audio-latents")
      options.audio_latents = number(value("--audio-latents"), "audio latents");
    else if (option == "--patch-h")
      options.patch_height = number(value("--patch-h"), "patch height");
    else if (option == "--patch-w")
      options.patch_width = number(value("--patch-w"), "patch width");
    else if (option == "--keyframes")
      options.keyframes = keyframes(value("--keyframes"));
    else if (option == "--reference-geometry")
      options.references.push_back(
          reference_geometry(value("--reference-geometry")));
    else if (option == "--output-latent")
      options.output_latent = value("--output-latent");
    else if (option == "--output-video-rows")
      options.output_video_rows = value("--output-video-rows");
    else if (option == "--output-audio")
      options.output_audio = value("--output-audio");
    else if (option == "--output-audio-latent")
      options.output_audio_latent = value("--output-audio-latent");
    else if (option == "--output-handoff")
      options.output_handoff = value("--output-handoff");
    else if (option == "--h3-w8a8-cache")
      options.h3_w8a8_cache = value("--h3-w8a8-cache");
    else if (option == "--h3-convrot-int8-checkpoint")
      options.h3_convrot_int8_checkpoint =
          value("--h3-convrot-int8-checkpoint");
    else if (option == "--h3-groupwise-cache")
      options.h3_groupwise_cache = value("--h3-groupwise-cache");
    else if (option == "--h3-groupwise-layers") {
      const auto layers = number(value("--h3-groupwise-layers"),
                                 "H3 groupwise layers");
      if (layers == 0U || layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 groupwise layer count must be nonzero U32");
      options.h3_groupwise_layers = static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-w8a8-resident-layers") {
      const auto layers =
          number(value("--h3-w8a8-resident-layers"), "resident layers");
      if (layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 W8A8 resident layer count exceeds U32 range");
      options.h3_w8a8_resident_layers = static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-convrot-int8-resident-layers") {
      const auto layers = number(value("--h3-convrot-int8-resident-layers"),
                                 "ConvRot INT8 resident layers");
      if (layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 ConvRot INT8 resident layer count exceeds U32 range");
      options.h3_convrot_int8_resident_layers =
          static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-convrot-int8-layers") {
      const auto layers = number(value("--h3-convrot-int8-layers"),
                                 "ConvRot INT8 layers");
      if (layers == 0U || layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 ConvRot INT8 layer count must be nonzero U32");
      options.h3_convrot_int8_attention_layers =
          static_cast<std::uint32_t>(layers);
      options.h3_convrot_int8_mlp_layers =
          static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-convrot-int8-attention-layers") {
      const auto layers = number(value("--h3-convrot-int8-attention-layers"),
                                 "ConvRot INT8 attention layers");
      if (layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 ConvRot INT8 attention layer count exceeds U32");
      options.h3_convrot_int8_attention_layers =
          static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-convrot-int8-mlp-layers") {
      const auto layers = number(value("--h3-convrot-int8-mlp-layers"),
                                 "ConvRot INT8 MLP layers");
      if (layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 ConvRot INT8 MLP layer count exceeds U32");
      options.h3_convrot_int8_mlp_layers =
          static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-int8-mlp-chunk-rows") {
      const auto rows = number(value("--h3-int8-mlp-chunk-rows"),
                               "H3 INT8 MLP chunk rows");
      if (rows == 0U || rows > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 INT8 MLP chunk rows must be in U32 range and nonzero");
      options.h3_int8_mlp_chunk_rows = static_cast<std::uint32_t>(rows);
    }
    else if (option == "--h3-int8-cublaslt")
      options.h3_int8_cublaslt = true;
    else if (option == "--h3-int8-cublaslt-rank") {
      const auto rank = number(value("--h3-int8-cublaslt-rank"),
                               "H3 INT8 cuBLASLt heuristic rank");
      if (rank > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 INT8 cuBLASLt heuristic rank exceeds U32 range");
      options.h3_int8_cublaslt_heuristic_rank =
          static_cast<std::uint32_t>(rank);
    }
    else if (option == "--h3-int8-cublaslt-tune")
      options.h3_int8_cublaslt_tune = true;
    else if (option == "--h3-int8-cutlass-scaled-fc1")
      options.h3_int8_cutlass_scaled_fc1 = true;
    else if (option == "--h3-int8-cutlass-scaled-all")
      options.h3_int8_cutlass_scaled_all = true;
    else if (option == "--h3-int8-convrot-scale-chunk") {
      const auto columns = number(value("--h3-int8-convrot-scale-chunk"),
                                  "H3 ConvRot scale chunk");
      if (columns < 256U || columns > 2048U || columns % 256U != 0U)
        dif::fail("H3 ConvRot scale chunk must be a multiple of 256 in [256,2048]");
      options.h3_int8_convrot_scale_chunk =
          static_cast<std::uint32_t>(columns);
    }
    else if (option == "--h3-int8-convrot-global-activation-scale")
      options.h3_int8_convrot_global_activation_scale = true;
    else if (option == "--h3-convrot-bf16-audio-rows")
      options.h3_convrot_bf16_audio_rows = true;
    else if (option == "--h3-int8-compact-adaln")
      options.h3_int8_compact_adaln = true;
    else if (option == "--h3-cache-text-refiner")
      options.h3_cache_text_refiner = true;
    else if (option == "--resident-streamed-constant") {
      const auto id = number(value("--resident-streamed-constant"),
                             "resident streamed constant id");
      if (id > std::numeric_limits<std::uint32_t>::max())
        dif::fail("resident streamed constant id exceeds U32 range");
      options.resident_streamed_constants.push_back(
          static_cast<std::uint32_t>(id));
    }
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
      else
        dif::fail("cuDNN attention heuristic must be a, b, fallback, or autotune");
    }
    else if (option == "--h3-ck-attention-dso")
      options.h3_ck_attention_dso = value("--h3-ck-attention-dso");
    else if (option == "--h3-owned-attention")
      options.h3_owned_attention = true;
    else if (option == "--h3-owned-attention-center-k")
      options.h3_owned_attention_center_k = true;
    else if (option == "--h3-resident-mapped-copy")
      options.h3_resident_mapped_copy = true;
    else if (option == "--h3-int8-attention-first-step")
      options.h3_int8_attention_first_step = static_cast<std::uint32_t>(
          number(value("--h3-int8-attention-first-step"), "INT8 attention first step"));
    else if (option == "--h3-resident-readahead-mib")
      options.h3_resident_readahead_mib =
          number(value("--h3-resident-readahead-mib"), "resident read-ahead MiB");
    else if (option == "--h3-int8-attention-first-layer") {
      const auto layer = number(value("--h3-int8-attention-first-layer"),
                                "first H3 INT8 attention layer");
      if (layer > std::numeric_limits<std::uint32_t>::max())
        dif::fail("first H3 INT8 attention layer exceeds U32 range");
      options.h3_int8_attention_first_layer =
          static_cast<std::uint32_t>(layer);
    }
    else if (option == "--h3-int8-attention-layers") {
      const auto layers = number(value("--h3-int8-attention-layers"),
                                 "H3 INT8 attention layers");
      if (layers == 0U || layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 INT8 attention layers must be in U32 range and nonzero");
      options.h3_int8_attention_layers = static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-modulation-cache")
      options.h3_modulation_cache = value("--h3-modulation-cache");
    else if (option == "--h3-modulation-source-index")
      options.h3_modulation_source_index =
          value("--h3-modulation-source-index");
    else if (option == "--h3-modulation-steps") {
      const auto steps =
          number(value("--h3-modulation-steps"), "modulation steps");
      if (steps > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 modulation step count exceeds U32 range");
      options.h3_modulation_steps = static_cast<std::uint32_t>(steps);
    }
    else if (option == "--h3-modulation-first-step") {
      const auto step = number(value("--h3-modulation-first-step"),
                               "H3 modulation first step");
      if (step > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 modulation first step exceeds U32 range");
      options.h3_modulation_first_step = static_cast<std::uint32_t>(step);
    }
    else if (option == "--output-raw")
      options.output_raw = value("--output-raw");
    else if (option == "--output-decoded")
      options.output_decoded = value("--output-decoded");
    else if (option == "--first-eval-input-dir")
      options.first_evaluation_input_directory =
          value("--first-eval-input-dir");
    else if (option == "--capture-denoiser-dir")
      options.capture_denoiser_directory = value("--capture-denoiser-dir");
    else if (option == "--capture-denoiser-tensor") {
      const auto tensor_id = number(value("--capture-denoiser-tensor"),
                                    "captured denoiser tensor id");
      if (tensor_id == 0U || tensor_id > std::numeric_limits<std::uint32_t>::max())
        dif::fail("captured denoiser tensor id must be in the U32 positive range");
      options.capture_denoiser_tensors.push_back(
          static_cast<std::uint32_t>(tensor_id));
    }
    else if (option == "--max-evaluations") {
      const auto evaluations =
          number(value("--max-evaluations"), "maximum evaluations");
      if (evaluations == 0U ||
          evaluations > std::numeric_limits<std::uint32_t>::max())
        dif::fail("maximum evaluations must be in the U32 positive range");
      options.maximum_evaluations =
          static_cast<std::uint32_t>(evaluations);
    }
    else if (option == "--cache-dir")
      options.cache_directory = value("--cache-dir");
    else if (option == "--min-free-mib")
      options.minimum_free_bytes =
          number(value("--min-free-mib"), "minimum free memory") * 1024ULL *
          1024ULL;
    else if (option == "--verify-shards")
      options.verify_shards = true;
    else if (option == "--profile-pipeline")
      options.profile_pipeline = true;
    else if (option == "--streamed-keep-pages")
      options.streamed_keep_pages = true;
    else if (option == "--pipelined-resident-upload")
      options.pipelined_resident_upload = true;
    else if (option == "--lazy-resident-upload")
      options.lazy_resident_upload = true;
    else if (option == "--keep-resident-host-pages")
      options.keep_resident_host_pages = true;
    else if (option == "--streamed-staging-buffers")
      options.streamed_staging_buffers = static_cast<std::uint32_t>(
          number(value("--streamed-staging-buffers"), "staging buffers"));
    else if (option == "--streamed-prefetch-depth")
      options.streamed_prefetch_depth = static_cast<std::uint32_t>(
          number(value("--streamed-prefetch-depth"), "prefetch depth"));
    else if (option == "--streamed-stage-threads")
      options.streamed_stage_threads = static_cast<std::uint32_t>(
          number(value("--streamed-stage-threads"), "stage threads"));
    else if (option == "--streamed-pinned-budget-mib")
      options.streamed_pinned_budget_mib =
          number(value("--streamed-pinned-budget-mib"), "pinned budget MiB");
    else if (option == "--pinned-io")
      options.pinned_io = true;
    else if (option == "--denoise-only")
      options.denoise_only = true;
    else if (option == "--serve")
      options.serve_socket = value("--serve");
    else if (option == "--connect")
      dif::fail("--connect is handled by the client and cannot reach parse");
    else {
      usage();
      dif::fail("unknown difh3infer option: " + option);
    }
    if (!per_request_options.contains(option)) {
      for (int token = option_start; token <= index; ++token) {
        options.prepare_signature += argv[token];
        options.prepare_signature += '\n';
      }
    }
  }
  for (const auto *path : {&options.denoiser_program, &options.denoiser_bundle,
                           &options.text, &options.video,
                           &options.audio, &options.output_latent,
                           &options.output_audio}) {
    if (path->empty()) {
      usage();
      dif::fail("difh3infer is missing a required path");
    }
  }
  if (options.text_tags.empty() == (options.all_text_tokens == 0U))
    dif::fail("select either --text-tags or --all-text-tokens");
  if (options.sampler != "euler" && options.sampler != "res_multistep")
    dif::fail("sampler must be euler or res_multistep");
  if (options.h3_modulation_cache.empty() !=
      options.h3_modulation_source_index.empty())
    dif::fail("H3 modulation cache and source index must be supplied together");
  if (options.h3_modulation_cache.empty() &&
      options.h3_modulation_first_step != 0U)
    dif::fail("H3 modulation first step requires a modulation cache");
  const auto precision_routes =
      static_cast<unsigned>(!options.h3_w8a8_cache.empty()) +
      static_cast<unsigned>(!options.h3_convrot_int8_checkpoint.empty()) +
      static_cast<unsigned>(!options.h3_groupwise_cache.empty());
  if (precision_routes > 1U)
    dif::fail("select at most one H3 INT8 precision route");
  if (options.h3_groupwise_cache.empty() &&
      options.h3_groupwise_layers != std::numeric_limits<std::uint32_t>::max())
    dif::fail("H3 groupwise layer count requires a groupwise cache");
  if (options.h3_convrot_int8_checkpoint.empty() &&
      (options.h3_convrot_int8_attention_layers !=
           std::numeric_limits<std::uint32_t>::max() ||
       options.h3_convrot_int8_mlp_layers !=
           std::numeric_limits<std::uint32_t>::max()))
    dif::fail("H3 ConvRot INT8 layer count requires a ConvRot checkpoint");
  if (!options.h3_convrot_int8_checkpoint.empty() &&
      options.h3_convrot_int8_attention_layers == 0U &&
      options.h3_convrot_int8_mlp_layers == 0U)
    dif::fail("H3 ConvRot INT8 requires at least one admitted projection chain");
  if (options.h3_convrot_bf16_audio_rows &&
      options.h3_convrot_int8_checkpoint.empty())
    dif::fail("H3 BF16 audio-row correction requires a ConvRot checkpoint");
  if (!options.keyframes.empty() && !options.references.empty())
    dif::fail("FL2VA keyframes and Ref2VA references are mutually exclusive");
  const auto supplied_schedule = !options.video_sigmas.empty() ||
                                 !options.audio_sigmas.empty();
  const auto schedule_sources =
      static_cast<unsigned>(options.schedule_points != 0U) +
      static_cast<unsigned>(options.simple_evaluations != 0U) +
      static_cast<unsigned>(supplied_schedule);
  if (schedule_sources != 1U ||
      (supplied_schedule && (options.video_sigmas.empty() ||
                             options.audio_sigmas.empty())) ||
      (options.schedule_points != 0U && options.schedule_points < 2U) ||
      options.simple_evaluations > 1000U)
    dif::fail("select exactly one of --simple-steps 1..1000, --schedule-points >= 2, or both sigma files");
  if (!options.denoise_only &&
      (options.vae_program.empty() || options.vae_bundle.empty() ||
       options.output_raw.empty() || options.output_decoded.empty())) {
    usage();
    dif::fail("joined decode requires VAE program, bundle, raw, and decoded paths");
  }
  if (options.latent_frames == 0U || options.latent_height == 0U ||
      options.latent_width == 0U || options.audio_latents == 0U ||
      options.patch_height == 0U || options.patch_width == 0U)
    dif::fail("H3 inference geometry must be positive");
  for (const auto *path : {&options.output_latent, &options.output_audio,
                           &options.output_video_rows,
                           &options.output_audio_latent,
                           &options.output_handoff, &options.output_raw,
                           &options.output_decoded}) {
    if (!path->empty() && std::filesystem::exists(*path))
      dif::fail("refusing to overwrite output: " + path->string());
  }
  return options;
}

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::F32, std::move(dims),
                              std::vector<std::uint8_t>(values.size() *
                                                        sizeof(float))};
  if (!values.empty())
    std::memcpy(tensor.mutable_data(), values.data(), tensor.byte_size());
  tensor.validate();
  return tensor;
}

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, std::move(dims),
                              std::vector<std::uint8_t>(values.size() *
                                                        sizeof(std::int32_t))};
  if (!values.empty())
    std::memcpy(tensor.mutable_data(), values.data(), tensor.byte_size());
  tensor.validate();
  return tensor;
}

void validate(const dif::ir::Program &program, std::uint32_t id,
              const dif::runtime::Tensor &tensor, const char *label) {
  tensor.validate();
  const auto *description = program.tensor(id);
  if (!description || !description->has_role(dif::ir::TensorRole::Input) ||
      tensor.dtype != description->dtype || tensor.dims != description->dims)
    dif::fail(std::string(label) + " does not match tensor " +
              std::to_string(id));
}

std::unique_ptr<dif::runtime::Executor> executor(const Options &options) {
  if (!options.backend_plugin.empty())
    return dif::backend::make_plugin_executor(options.backend_plugin);
  if (options.backend == "cpu")
    return dif::runtime::make_cpu_executor();
  if (options.backend == "cuda")
    return dif::runtime::make_cuda_executor();
  dif::fail("unknown backend: " + options.backend);
}

// Everything a served request reuses: the executor, the prepared denoiser
// (resident weights, plans, scratch), the loaded bundle bindings, and the
// signature of the flags that produced them.
struct ServerState {
  std::unique_ptr<dif::runtime::Executor> backend;
  std::unique_ptr<dif::runtime::PreparedExecution> prepared;
  dif::ir::Program program;
  dif::runtime::TensorMap inputs;
  std::string signature;
  double bundle_map_ms{};
  double prepare_ms{};
  std::uint64_t resident{};
  std::size_t requests{};
};

std::pair<std::uint32_t, std::uint32_t>
denoiser_outputs(const dif::ir::Program &program) {
  std::uint32_t video = 0U;
  std::uint32_t audio = 0U;
  for (const auto &tensor : program.tensors) {
    if (!tensor.has_role(dif::ir::TensorRole::Output) ||
        tensor.dtype != dif::ir::DType::F32 || tensor.dims.size() != 2U)
      continue;
    if (tensor.dims.back() == 96U)
      video = tensor.id;
    else if (tensor.dims.back() == 32U)
      audio = tensor.id;
  }
  if (video == 0U || audio == 0U)
    dif::fail("denoiser program lacks unique F32 video/audio outputs");
  return {video, audio};
}

std::pair<std::uint32_t, std::uint32_t>
vae_outputs(const dif::ir::Program &program) {
  std::uint32_t raw = 0U;
  std::uint32_t decoded = 0U;
  for (const auto &operation : program.operations) {
    if (operation.opcode == dif::ir::Opcode::Clamp)
      decoded = operation.outputs.at(0);
  }
  for (const auto &tensor : program.tensors) {
    if (tensor.has_role(dif::ir::TensorRole::Output) && tensor.id != decoded)
      raw = tensor.id;
  }
  if (raw == 0U || decoded == 0U ||
      !program.tensor(decoded)->has_role(dif::ir::TensorRole::Output))
    dif::fail("VAE program lacks raw and decoded outputs");
  return {raw, decoded};
}

double elapsed_milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool is_layout_operation(dif::ir::Opcode opcode) {
  using dif::ir::Opcode;
  switch (opcode) {
  case Opcode::Fill:
  case Opcode::GatherRows:
  case Opcode::IndexedUpdateRows:
  case Opcode::Cast:
  case Opcode::SelectRowChunks:
  case Opcode::SinusoidalTimestep:
  case Opcode::RotaryPosition:
  case Opcode::Patchify3D:
  case Opcode::Unpatchify3D:
  case Opcode::H3AdaLNSelect:
  case Opcode::H3DeinterleaveQkv:
  case Opcode::H3DeinterleaveQkvWeight:
    return true;
  default:
    return false;
  }
}

bool is_dequant_operation(dif::ir::Opcode opcode) {
  return opcode == dif::ir::Opcode::DequantizeInt4 ||
         opcode == dif::ir::Opcode::DequantizeInt5;
}

double operation_sum(
    const std::vector<dif::runtime::OperationTiming> &timings,
    bool (*selected)(dif::ir::Opcode)) {
  double total = 0.0;
  for (const auto &timing : timings) {
    // A fused plan executed at this slot; its opcode no longer describes
    // the work (for example the INT8 QKV projection at the weight layout op).
    if (!timing.plan.empty())
      continue;
    if (selected(timing.opcode))
      total += timing.mean_milliseconds;
  }
  return total;
}

} // namespace

int run_request(const Options &options, ServerState &state,
                const std::chrono::steady_clock::time_point command_wall_start) {
  {
    if (options.latent_height % options.patch_height != 0U ||
        options.latent_width % options.patch_width != 0U)
      dif::fail("latent geometry must be patch-divisible");
    double input_io_ms = 0.0;
    auto timed_io_start = std::chrono::steady_clock::now();
    std::vector<std::int32_t> tags;
    if (options.all_text_tokens != 0U) {
      if (options.all_text_tokens >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        dif::fail("all text token count exceeds host size range");
      tags.assign(static_cast<std::size_t>(options.all_text_tokens), 1);
    } else {
      const auto tags_tensor = dif::runtime::read_tensor(options.text_tags);
      if (tags_tensor.dtype != dif::ir::DType::I32 ||
          tags_tensor.dims.size() != 1U)
        dif::fail("text tags must be I32[N]");
      tags.resize(tags_tensor.element_count());
      std::memcpy(tags.data(), tags_tensor.data(), tags_tensor.byte_size());
    }
    input_io_ms += elapsed_milliseconds(timed_io_start);
    const auto initial_layout_start = std::chrono::steady_clock::now();
    const auto layout = options.references.empty()
                            ? dif::frontend::make_h3_t2va_layout(
                                  tags, options.latent_frames,
                                  options.latent_height, options.latent_width,
                                  options.audio_latents, 1U,
                                  options.patch_height, options.patch_width,
                                  options.keyframes)
                            : dif::frontend::make_h3_ref2va_layout(
                                  tags, options.references,
                                  options.latent_frames,
                                  options.latent_height, options.latent_width,
                                  options.audio_latents, 1U,
                                  options.patch_height, options.patch_width);
    const auto initial_layout_ms = elapsed_milliseconds(initial_layout_start);
    timed_io_start = std::chrono::steady_clock::now();
    auto text = dif::runtime::read_tensor(options.text);
    auto video = dif::runtime::read_tensor(options.video);
    auto audio = dif::runtime::read_tensor(options.audio);
    std::vector<float> video_sigmas;
    std::vector<float> audio_sigmas;
    if (options.simple_evaluations != 0U) {
      auto schedule = dif::sampling::make_h3_simple_av_schedule(
          options.simple_evaluations);
      video_sigmas = std::move(schedule.video_sigmas);
      audio_sigmas = std::move(schedule.audio_sigmas);
    } else if (options.schedule_points != 0U) {
      video_sigmas = dif::sampling::make_exponential_shifted_schedule(
                         options.schedule_points, 12.0F)
                         .sigmas;
      audio_sigmas = dif::sampling::make_exponential_shifted_schedule(
                         options.schedule_points, 3.0F)
                         .sigmas;
    } else {
      const auto video_schedule =
          dif::runtime::read_tensor(options.video_sigmas);
      const auto audio_schedule =
          dif::runtime::read_tensor(options.audio_sigmas);
      if (video_schedule.dtype != dif::ir::DType::F32 ||
          audio_schedule.dtype != dif::ir::DType::F32 ||
          video_schedule.dims.size() != 1U ||
          video_schedule.dims != audio_schedule.dims ||
          video_schedule.element_count() < 2U)
        dif::fail("video/audio sigmas must be matching F32[K+1]");
      video_sigmas.assign(video_schedule.f32().begin(),
                          video_schedule.f32().end());
      audio_sigmas.assign(audio_schedule.f32().begin(),
                          audio_schedule.f32().end());
    }
    input_io_ms += elapsed_milliseconds(timed_io_start);
    if (video_sigmas.size() != audio_sigmas.size())
      dif::fail("video/audio schedules diverged after float32 deduplication");
    const auto steps = video_sigmas.size() - 1U;
    const auto evaluations =
        options.maximum_evaluations == 0U
            ? steps
            : std::min<std::size_t>(steps, options.maximum_evaluations);
    if (options.maximum_evaluations > steps)
      dif::fail("maximum evaluations exceed the released schedule");
    if (options.capture_denoiser_tensors.empty() !=
        options.capture_denoiser_directory.empty())
      dif::fail("denoiser capture requires both a directory and tensor ids");
    if (!options.capture_denoiser_directory.empty()) {
      if (std::filesystem::exists(options.capture_denoiser_directory) &&
          !std::filesystem::is_empty(options.capture_denoiser_directory))
        dif::fail("denoiser capture directory is not empty: " +
                  options.capture_denoiser_directory.string());
      std::filesystem::create_directories(
          options.capture_denoiser_directory);
    }
    if (!options.first_evaluation_input_directory.empty()) {
      if (std::filesystem::exists(options.first_evaluation_input_directory) &&
          !std::filesystem::is_empty(
              options.first_evaluation_input_directory))
        dif::fail("first-evaluation input directory is not empty: " +
                  options.first_evaluation_input_directory.string());
      std::filesystem::create_directories(
          options.first_evaluation_input_directory);
      dif::runtime::write_tensor(
          f32_tensor({video_sigmas.size()}, video_sigmas),
          options.first_evaluation_input_directory / "video_sigmas.diftensor");
      dif::runtime::write_tensor(
          f32_tensor({audio_sigmas.size()}, audio_sigmas),
          options.first_evaluation_input_directory / "audio_sigmas.diftensor");
    }

    dif::runtime::RunOptions run_options;
    run_options.warmups = 0U;
    run_options.iterations = 1U;
    run_options.minimum_free_bytes = options.minimum_free_bytes;
    run_options.cache_directory = options.cache_directory;
    run_options.profile_pipeline = options.profile_pipeline;
    if (options.streamed_keep_pages)
      run_options.streamed_release_mapped_pages_per_copy = false;
    run_options.pipelined_resident_upload =
        options.pipelined_resident_upload;
    run_options.lazy_resident_upload = options.lazy_resident_upload;
    run_options.resident_evict_host_pages = !options.keep_resident_host_pages;
    if (options.streamed_staging_buffers != 0U)
      run_options.streamed_staging_buffers = options.streamed_staging_buffers;
    if (options.streamed_prefetch_depth != 0U)
      run_options.streamed_prefetch_depth = options.streamed_prefetch_depth;
    if (options.streamed_stage_threads != 0U)
      run_options.streamed_stage_threads = options.streamed_stage_threads;
    if (options.streamed_pinned_budget_mib != 0U)
      run_options.streamed_pinned_budget_bytes =
          options.streamed_pinned_budget_mib * 1024ULL * 1024ULL;
    if (options.pinned_io)
      run_options.pinned_io_staging = true;
    auto denoiser_run_options = run_options;
    denoiser_run_options.capture_intermediate_tensors =
        options.capture_denoiser_tensors;
    denoiser_run_options.h3_w8a8_cache = options.h3_w8a8_cache;
    denoiser_run_options.h3_w8a8_resident_layers =
        options.h3_w8a8_resident_layers;
    denoiser_run_options.h3_convrot_int8_checkpoint =
        options.h3_convrot_int8_checkpoint;
    denoiser_run_options.h3_convrot_int8_attention_layers =
        options.h3_convrot_int8_attention_layers;
    denoiser_run_options.h3_convrot_int8_mlp_layers =
        options.h3_convrot_int8_mlp_layers;
    denoiser_run_options.h3_convrot_int8_resident_layers =
        options.h3_convrot_int8_resident_layers;
    denoiser_run_options.h3_groupwise_cache = options.h3_groupwise_cache;
    denoiser_run_options.h3_groupwise_layers = options.h3_groupwise_layers;
    denoiser_run_options.h3_int8_mlp_chunk_rows =
        options.h3_int8_mlp_chunk_rows;
    denoiser_run_options.h3_int8_cublaslt = options.h3_int8_cublaslt;
    denoiser_run_options.h3_int8_cublaslt_heuristic_rank =
        options.h3_int8_cublaslt_heuristic_rank;
    denoiser_run_options.h3_int8_cublaslt_tune =
        options.h3_int8_cublaslt_tune;
    denoiser_run_options.h3_int8_cutlass_scaled_fc1 =
        options.h3_int8_cutlass_scaled_fc1;
    denoiser_run_options.h3_int8_cutlass_scaled_all =
        options.h3_int8_cutlass_scaled_all;
    denoiser_run_options.h3_int8_convrot_scale_chunk =
        options.h3_int8_convrot_scale_chunk;
    denoiser_run_options.h3_int8_convrot_global_activation_scale =
        options.h3_int8_convrot_global_activation_scale;
    if (options.h3_convrot_bf16_audio_rows) {
      denoiser_run_options.h3_convrot_bf16_correction_rows.reserve(
          layout.audio_indices.size());
      for (const auto row : layout.audio_indices) {
        if (row < 0)
          dif::fail("H3 audio correction row is negative");
        denoiser_run_options.h3_convrot_bf16_correction_rows.push_back(
            static_cast<std::uint32_t>(row));
      }
    }
    denoiser_run_options.h3_int8_compact_adaln =
        options.h3_int8_compact_adaln;
    denoiser_run_options.resident_streamed_constants =
        options.resident_streamed_constants;
    denoiser_run_options.cudnn_attention_heuristic =
        options.cudnn_attention_heuristic;
    denoiser_run_options.h3_ck_attention_dso = options.h3_ck_attention_dso;
    denoiser_run_options.h3_owned_attention = options.h3_owned_attention;
    denoiser_run_options.h3_owned_attention_center_k =
        options.h3_owned_attention_center_k;
    denoiser_run_options.h3_resident_direct_io = !options.h3_resident_mapped_copy;
    denoiser_run_options.streamed_direct_io = !options.h3_resident_mapped_copy;
    denoiser_run_options.direct_io_warm_page_cache =
        !options.h3_resident_mapped_copy;
    if (options.h3_resident_readahead_mib !=
        std::numeric_limits<std::uint64_t>::max())
      denoiser_run_options.h3_resident_readahead_bytes =
          options.h3_resident_readahead_mib * 1024ULL * 1024ULL;
    denoiser_run_options.h3_int8_attention_first_layer =
        options.h3_int8_attention_first_layer;
    denoiser_run_options.h3_int8_attention_layers =
        options.h3_int8_attention_layers;
    denoiser_run_options.h3_int8_attention_hybrid =
        options.h3_int8_attention_first_step != 0U;
    denoiser_run_options.h3_modulation_cache = options.h3_modulation_cache;
    denoiser_run_options.h3_modulation_source_index =
        options.h3_modulation_source_index;
    denoiser_run_options.h3_modulation_steps =
        options.h3_modulation_cache.empty()
            ? 0U
            : (options.h3_modulation_steps != 0U
                   ? options.h3_modulation_steps
                   : options.schedule_points);
    double denoiser_prepare_ms = 0.0;
    bool denoiser_reused = false;
    double denoiser_kernel_ms = 0.0;
    double denoiser_bundle_map_ms = 0.0;
    double denoiser_step_layout_ms = 0.0;
    double scheduler_update_ms = 0.0;
    double denoiser_gpu_layout_ms = 0.0;
    double denoiser_gpu_dequant_ms = 0.0;
    dif::runtime::PipelineProfile denoiser_profile;
    std::unordered_map<std::uint32_t, double> denoiser_operation_ms;
    std::unordered_map<std::uint32_t, double> denoiser_operation_cold_ms;
    std::unordered_map<std::uint32_t, double> denoiser_operation_hot_ms;
    std::unordered_map<std::uint32_t, dif::ir::Opcode>
        denoiser_operation_opcodes;
    std::uint64_t denoiser_resident = 0U;
    std::uint64_t denoiser_free_before = 0U;
    std::uint64_t denoiser_free_after = 0U;
    std::size_t h3_w8a8_mlp_count = 0U;
    std::size_t h3_w8a8_attention_count = 0U;
    std::size_t h3_w8a8_resident_mlp_count = 0U;
    std::size_t h3_w8a8_resident_attention_count = 0U;
    std::size_t h3_groupwise_count = 0U;
    std::size_t h3_ck_attention_count = 0U;
    std::size_t h3_modulation_cache_count = 0U;
    std::uint32_t repeated_invariant_operation_count = 0U;
    std::uint64_t repeated_invariant_persistent_bytes = 0U;
    std::size_t repeated_invariant_cache_hits = 0U;
    std::string attention_class{"exact_program_declared"};
    std::string backend_name;
    std::string device_name;
    const auto denoiser_wall_start = std::chrono::steady_clock::now();
    {
      const auto bundle_map_start = std::chrono::steady_clock::now();
      const bool reuse = static_cast<bool>(state.prepared);
      if (reuse && state.signature != options.prepare_signature)
        dif::fail("persistent denoiser is prepared for a different "
                  "configuration; restart the server with the new flags");
      if (!reuse)
        state.program = dif::ir::read_file(options.denoiser_program);
      const auto &program = state.program;
      validate(program, 1U, video, "video input");
      validate(program, 2U, audio, "audio input");
      validate(program, 3U, text, "text conditioning");
      if (program.tensor(1U)->dims[0] != layout.video_indices.size() ||
          program.tensor(2U)->dims[0] != layout.audio_indices.size() ||
          program.tensor(3U)->dims[0] != tags.size() ||
          program.tensor(5U)->dims[0] != layout.sequence_length)
        dif::fail("native H3 layout disagrees with denoiser geometry");
      if (options.h3_cache_text_refiner)
        denoiser_run_options.repeated_invariant_operations =
            h3_text_refiner_invariant_operations(program);
      const auto [video_output, audio_output] = denoiser_outputs(program);
      if (!reuse) {
        state.inputs = dif::weights::load_weight_bundle(
            dif::weights::read_weight_bundle(options.denoiser_bundle), program,
            options.verify_shards);
        state.bundle_map_ms = elapsed_milliseconds(bundle_map_start);
      }
      auto &inputs = state.inputs;
      denoiser_bundle_map_ms = reuse ? 0.0 : state.bundle_map_ms;
      inputs.insert_or_assign(1U, video);
      inputs.insert_or_assign(2U, audio);
      inputs.insert_or_assign(3U, text);
      inputs.insert_or_assign(
          5U, i32_tensor({layout.sequence_length}, layout.text_map));
      inputs.insert_or_assign(
          6U, i32_tensor({layout.sequence_length}, layout.video_map));
      inputs.insert_or_assign(
          7U, i32_tensor({layout.sequence_length}, layout.audio_map));
      inputs.insert_or_assign(
          10U, i32_tensor({layout.video_indices.size()}, layout.video_indices));
      inputs.insert_or_assign(
          11U, i32_tensor({layout.audio_indices.size()}, layout.audio_indices));
      inputs.insert_or_assign(
          12U, f32_tensor({layout.sequence_length, 3U}, layout.position_ids));
      if (!reuse) {
        state.backend = executor(options);
        state.prepared =
            state.backend->prepare(program, inputs, denoiser_run_options);
        state.prepare_ms = state.prepared->preparation_milliseconds();
        state.resident = state.prepared->resident_bytes();
        state.signature = options.prepare_signature;
      }
      auto &prepared = state.prepared;
      denoiser_prepare_ms = reuse ? 0.0 : state.prepare_ms;
      denoiser_resident = state.resident;
      denoiser_reused = reuse;
      ++state.requests;
      dif::sampling::H3ResMultistepState video_res_multistep;
      dif::sampling::H3ResMultistepState audio_res_multistep;
      for (std::size_t step = 0U; step < evaluations; ++step) {
        const auto video_timestep = 1.0F - video_sigmas[step];
        const auto audio_timestep = 1.0F - audio_sigmas[step];
        const auto step_layout_start = std::chrono::steady_clock::now();
        auto plan = dif::frontend::make_h3_row_timestep_plan(
            layout, video_timestep, audio_timestep,
            std::max(video_timestep, 0.999F), 1.0F);
        const auto *timestep_description = program.tensor(4U);
        if (!timestep_description || timestep_description->dims.size() != 1U)
          dif::fail("denoiser timestep table must be rank-one tensor 4");
        const auto timestep_slots = timestep_description->dims[0];
        if (plan.timesteps.size() > timestep_slots)
          dif::fail("runtime row timesteps exceed compiled denoiser slots");
        // A compiled H3 graph fixes the number of timestep-embedding rows.
        // Runtime video/audio schedules can coincide on later steps, reducing
        // the number of unique values.  Pad unused rows without changing any
        // of the source-faithful row indices.
        while (plan.timesteps.size() < timestep_slots)
          plan.timesteps.push_back(plan.timesteps.back());
        validate(program, 4U, f32_tensor({plan.timesteps.size()}, plan.timesteps),
                 "timestep plan");
        inputs.insert_or_assign(1U, video);
        auto audio_model_input = audio;
        if (options.sampler == "res_multistep")
          dif::sampling::h3_av_audio_carry_to_model_input(
              audio_model_input.f32(), audio.f32(), audio.dims[1],
              layout.num_condition_audio_rows, video_sigmas[step],
              audio_sigmas[step]);
        inputs.insert_or_assign(2U, audio_model_input);
        inputs.insert_or_assign(
            4U, f32_tensor({plan.timesteps.size()}, plan.timesteps));
        inputs.insert_or_assign(
            8U, i32_tensor({layout.sequence_length}, plan.adaln_indices));
        inputs.insert_or_assign(
            9U, i32_tensor({layout.sequence_length}, plan.timestep_indices));
        if (step == 0U &&
            !options.first_evaluation_input_directory.empty()) {
          constexpr std::array<const char *, 12> names = {
              "input_01_video_state_rows.diftensor",
              "input_02_audio_state_rows.diftensor",
              "input_03_text_conditioning.diftensor",
              "input_04_timesteps.diftensor",
              "input_05_text_map.diftensor",
              "input_06_video_map.diftensor",
              "input_07_audio_map.diftensor",
              "input_08_adaln_indices.diftensor",
              "input_09_timestep_indices.diftensor",
              "input_10_video_indices.diftensor",
              "input_11_audio_indices.diftensor",
              "input_12_position_ids.diftensor",
          };
          for (std::size_t input_index = 0U; input_index < names.size();
               ++input_index) {
            const auto tensor_id =
                static_cast<std::uint32_t>(input_index + 1U);
            dif::runtime::write_tensor(
                inputs.at(tensor_id),
                options.first_evaluation_input_directory /
                    names[input_index]);
          }
        }
        denoiser_step_layout_ms += elapsed_milliseconds(step_layout_start);
        if (step > std::numeric_limits<std::uint32_t>::max() -
                       options.h3_modulation_first_step)
          dif::fail("H3 modulation slice exceeds U32 range");
        denoiser_run_options.h3_modulation_slice =
            options.h3_modulation_first_step +
            static_cast<std::uint32_t>(step);
        denoiser_run_options.h3_int8_attention_active =
            step >= options.h3_int8_attention_first_step;
        auto result = prepared->run(inputs, denoiser_run_options);
        for (const auto tensor_id : options.capture_denoiser_tensors) {
          const auto captured = result.captured_intermediates.find(tensor_id);
          if (captured == result.captured_intermediates.end())
            dif::fail("requested denoiser capture was not returned for tensor " +
                      std::to_string(tensor_id));
          dif::runtime::write_tensor(
              captured->second,
              options.capture_denoiser_directory /
                  ("step-" + std::to_string(step) + "-tensor-" +
                   std::to_string(tensor_id) + ".diftensor"));
        }
        if (step == 0U) {
          h3_w8a8_mlp_count = result.h3_w8a8_mlps.size();
          h3_w8a8_attention_count = result.h3_w8a8_attentions.size();
          h3_w8a8_resident_mlp_count = static_cast<std::size_t>(
              std::count_if(result.h3_w8a8_mlps.begin(),
                            result.h3_w8a8_mlps.end(),
                            [](const auto &plan) { return plan.resident; }));
          h3_w8a8_resident_attention_count = static_cast<std::size_t>(
              std::count_if(result.h3_w8a8_attentions.begin(),
                            result.h3_w8a8_attentions.end(),
                            [](const auto &plan) { return plan.resident; }));
          h3_ck_attention_count = result.h3_ck_attentions.size();
          h3_groupwise_count = result.h3_groupwise_int8.size();
          if (!result.h3_ck_attentions.empty())
            attention_class = result.h3_ck_attentions.front().classification;
          h3_modulation_cache_count = result.h3_modulation_caches.size();
          repeated_invariant_operation_count =
              result.repeated_invariant_operation_count;
          repeated_invariant_persistent_bytes =
              result.repeated_invariant_persistent_bytes;
          if (!options.h3_w8a8_cache.empty() &&
              (h3_w8a8_mlp_count == 0U ||
               h3_w8a8_attention_count == 0U))
            dif::fail("requested H3 direct INT8 checkpoint did not cover both projection and MLP chains");
          if (!options.h3_convrot_int8_checkpoint.empty()) {
            const auto expected_attention =
                options.h3_convrot_int8_attention_layers;
            const auto expected_mlp = options.h3_convrot_int8_mlp_layers;
            if ((expected_attention == 0U && h3_w8a8_attention_count != 0U) ||
                (expected_attention != 0U &&
                 h3_w8a8_attention_count == 0U) ||
                (expected_attention !=
                     std::numeric_limits<std::uint32_t>::max() &&
                 h3_w8a8_attention_count != expected_attention) ||
                (expected_mlp == 0U && h3_w8a8_mlp_count != 0U) ||
                (expected_mlp != 0U && h3_w8a8_mlp_count == 0U) ||
                (expected_mlp !=
                     std::numeric_limits<std::uint32_t>::max() &&
                 h3_w8a8_mlp_count != expected_mlp))
              dif::fail("requested H3 ConvRot projection-chain count was not admitted exactly");
          }
          if (!options.h3_groupwise_cache.empty() &&
              h3_groupwise_count == 0U)
            dif::fail("requested H3 groupwise cache was not admitted");
          if ((!options.h3_ck_attention_dso.empty() ||
               options.h3_owned_attention) &&
              h3_ck_attention_count == 0U)
            dif::fail("requested H3 INT8 attention route was not admitted");
          if (!options.h3_modulation_cache.empty() &&
              h3_modulation_cache_count == 0U)
            dif::fail("requested H3 modulation schedule cache was not admitted");
        }
        if (result.repeated_invariant_cache_hit)
          ++repeated_invariant_cache_hits;
        denoiser_kernel_ms += result.mean_milliseconds;
        if (step == 0U)
          denoiser_free_before = result.free_bytes_before;
        denoiser_free_after = result.free_bytes_after;
        backend_name = result.backend_name;
        device_name = result.device_name;
        if (result.pipeline_profile.enabled) {
          denoiser_profile.enabled = true;
          denoiser_profile.measured_iterations +=
              result.pipeline_profile.measured_iterations;
          denoiser_profile.resident_weight_bytes =
              result.pipeline_profile.resident_weight_bytes;
          denoiser_profile.resident_host_prefault_milliseconds =
              result.pipeline_profile.resident_host_prefault_milliseconds;
          denoiser_profile.resident_minor_page_faults =
              result.pipeline_profile.resident_minor_page_faults;
          denoiser_profile.resident_major_page_faults =
              result.pipeline_profile.resident_major_page_faults;
          denoiser_profile.resident_h2d_milliseconds =
              result.pipeline_profile.resident_h2d_milliseconds;
          denoiser_profile.resident_upload_milliseconds =
              result.pipeline_profile.resident_upload_milliseconds;
          denoiser_profile.resident_readahead_bytes +=
              result.pipeline_profile.resident_readahead_bytes;
          denoiser_profile.resident_direct_read_bytes +=
              result.pipeline_profile.resident_direct_read_bytes;
          denoiser_profile.streamed_direct_read_bytes +=
              result.pipeline_profile.streamed_direct_read_bytes;
          denoiser_profile.streamed_weight_bytes +=
              result.pipeline_profile.streamed_weight_bytes;
          denoiser_profile.streamed_host_stage_milliseconds +=
              result.pipeline_profile.streamed_host_stage_milliseconds;
          denoiser_profile.streamed_host_wait_milliseconds +=
              result.pipeline_profile.streamed_host_wait_milliseconds;
          denoiser_profile.streamed_h2d_milliseconds +=
              result.pipeline_profile.streamed_h2d_milliseconds;
          denoiser_profile.operation_kernel_milliseconds +=
              result.pipeline_profile.operation_kernel_milliseconds;
          denoiser_profile.attention_kernel_milliseconds +=
              result.pipeline_profile.attention_kernel_milliseconds;
          denoiser_profile.non_kernel_device_timeline_milliseconds +=
              result.pipeline_profile.non_kernel_device_timeline_milliseconds;
          denoiser_gpu_layout_ms +=
              operation_sum(result.operation_timings, is_layout_operation);
          denoiser_gpu_dequant_ms +=
              operation_sum(result.operation_timings, is_dequant_operation);
          for (const auto &timing : result.operation_timings) {
            denoiser_operation_ms[timing.operation_id] +=
                timing.mean_milliseconds;
            (step == 0U ? denoiser_operation_cold_ms
                        : denoiser_operation_hot_ms)[timing.operation_id] +=
                timing.mean_milliseconds;
            denoiser_operation_opcodes.emplace(timing.operation_id,
                                                timing.opcode);
          }
        }
        const auto scheduler_start = std::chrono::steady_clock::now();
        const auto &video_velocity = result.outputs.at(video_output);
        const auto &audio_velocity = result.outputs.at(audio_output);
        if (video_velocity.dtype != dif::ir::DType::F32 ||
            video_velocity.dims != video.dims ||
            audio_velocity.dtype != dif::ir::DType::F32 ||
            audio_velocity.dims != audio.dims)
          dif::fail("H3 denoiser output shape does not match scheduler state");
        if (options.sampler == "res_multistep") {
          dif::sampling::h3_res_multistep_step_in_place(
              video.f32(), video_velocity.f32(), video.dims[1],
              layout.num_condition_video_rows, video_timestep,
              video_sigmas[step], video_sigmas[step + 1U],
              video_res_multistep);
          dif::sampling::h3_res_multistep_av_audio_step_in_place(
              audio.f32(), audio_model_input.f32(), audio_velocity.f32(),
              audio.dims[1], layout.num_condition_audio_rows,
              video_sigmas[step], audio_sigmas[step],
              video_sigmas[step + 1U], 4.0F, audio_res_multistep);
        } else {
          dif::sampling::h3_euler_step_in_place(
              video.f32(), video_velocity.f32(), video.dims[1],
              layout.num_condition_video_rows, video_timestep,
              video_sigmas[step], video_sigmas[step + 1U]);
          dif::sampling::h3_euler_step_in_place(
              audio.f32(), audio_velocity.f32(), audio.dims[1],
              layout.num_condition_audio_rows, audio_timestep,
              audio_sigmas[step], audio_sigmas[step + 1U]);
        }
        scheduler_update_ms += elapsed_milliseconds(scheduler_start);
        std::cout << "H3_STEP index=" << step
                  << " video_t=" << video_timestep
                  << " audio_t=" << audio_timestep
                  << " denoiser_ms=" << result.mean_milliseconds
                  << " text_refiner_cache="
                  << (result.repeated_invariant_cache_hit ? "hit" : "miss")
                  << "\n";
      }
      if (options.sampler == "res_multistep")
        dif::sampling::h3_av_audio_carry_to_physical_in_place(
            audio.f32(), audio.dims[1], layout.num_condition_audio_rows,
            video_sigmas[evaluations], audio_sigmas[evaluations], 4.0F);
    }
    const auto denoiser_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - denoiser_wall_start)
            .count();

    const auto unpatchify_start = std::chrono::steady_clock::now();
    auto latent = dif::frontend::unpack_h3_video_rows(
        video, layout.num_condition_video_rows, options.latent_frames,
        options.latent_height, options.latent_width, options.patch_height,
        options.patch_width);
    auto audio_latent = dif::frontend::unpack_h3_audio_rows(
        audio, layout.num_condition_audio_rows, options.audio_latents, true);
    const auto unpatchify_ms = elapsed_milliseconds(unpatchify_start);
    const auto denoiser_output_io_start = std::chrono::steady_clock::now();
    dif::runtime::write_tensor(latent, options.output_latent);
    if (!options.output_video_rows.empty())
      dif::runtime::write_tensor(video, options.output_video_rows);
    dif::runtime::write_tensor(audio, options.output_audio);
    if (!options.output_audio_latent.empty())
      dif::runtime::write_tensor(audio_latent, options.output_audio_latent);
    if (!options.output_handoff.empty())
      dif::frontend::write_h3_latent_handoff(options.output_handoff, video,
                                             audio);
    const auto denoiser_output_io_ms =
        elapsed_milliseconds(denoiser_output_io_start);

    if (options.denoise_only) {
      const auto command_wall_ms = elapsed_milliseconds(command_wall_start);
      if (options.profile_pipeline) {
        std::vector<std::pair<std::uint32_t, double>> ranked_operations(
            denoiser_operation_ms.begin(), denoiser_operation_ms.end());
        std::sort(ranked_operations.begin(), ranked_operations.end(),
                  [&](const auto &left, const auto &right) {
                    if (evaluations > 1U)
                      return denoiser_operation_hot_ms[left.first] >
                             denoiser_operation_hot_ms[right.first];
                    return left.second > right.second;
                  });
        const auto reported_operations =
            std::min<std::size_t>(20U, ranked_operations.size());
        for (std::size_t index = 0U; index < reported_operations; ++index) {
          const auto [operation, milliseconds] = ranked_operations[index];
          std::cout << "H3_PROFILE_OP rank=" << index + 1U
                    << " operation=" << operation
                    << " opcode="
                    << dif::ir::opcode_name(
                           denoiser_operation_opcodes.at(operation))
                    << " total_ms=" << std::fixed << std::setprecision(3)
                    << milliseconds
                    << " cold_ms=" << denoiser_operation_cold_ms[operation]
                    << " hot_mean_ms="
                    << (evaluations > 1U
                            ? denoiser_operation_hot_ms[operation] /
                                  static_cast<double>(evaluations - 1U)
                            : 0.0)
                    << '\n';
        }
        const auto denoiser_other_gpu_ms = std::max(
            0.0, denoiser_profile.operation_kernel_milliseconds -
                     denoiser_profile.attention_kernel_milliseconds -
                     denoiser_gpu_layout_ms - denoiser_gpu_dequant_ms);
        std::cout << std::fixed << std::setprecision(3)
                  << "H3_PROFILE input_io_ms=" << input_io_ms
                  << " initial_layout_ms=" << initial_layout_ms
                  << " denoiser_bundle_map_ms=" << denoiser_bundle_map_ms
                  << " denoiser_prepare_ms=" << denoiser_prepare_ms
                  << " denoiser_device_ms=" << denoiser_kernel_ms
                  << " denoiser_wall_ms=" << denoiser_wall_ms
                  << " denoiser_resident_weight_bytes="
                  << denoiser_profile.resident_weight_bytes
                  << " denoiser_resident_prefault_ms="
                  << denoiser_profile.resident_host_prefault_milliseconds
                  << " denoiser_resident_minor_faults="
                  << denoiser_profile.resident_minor_page_faults
                  << " denoiser_resident_major_faults="
                  << denoiser_profile.resident_major_page_faults
                  << " denoiser_resident_h2d_ms="
                  << denoiser_profile.resident_h2d_milliseconds
                  << " denoiser_resident_upload_ms="
                  << denoiser_profile.resident_upload_milliseconds
                  << " denoiser_resident_readahead_bytes="
                  << denoiser_profile.resident_readahead_bytes
                  << " denoiser_resident_direct_read_bytes="
                  << denoiser_profile.resident_direct_read_bytes
                  << " denoiser_streamed_direct_read_bytes="
                  << denoiser_profile.streamed_direct_read_bytes
                  << " denoiser_streamed_weight_bytes="
                  << denoiser_profile.streamed_weight_bytes
                  << " denoiser_streamed_host_stage_ms="
                  << denoiser_profile.streamed_host_stage_milliseconds
                  << " denoiser_streamed_host_wait_ms="
                  << denoiser_profile.streamed_host_wait_milliseconds
                  << " denoiser_streamed_h2d_ms="
                  << denoiser_profile.streamed_h2d_milliseconds
                  << " denoiser_non_kernel_device_ms="
                  << denoiser_profile.non_kernel_device_timeline_milliseconds
                  << " denoiser_gpu_operation_ms="
                  << denoiser_profile.operation_kernel_milliseconds
                  << " denoiser_gpu_attention_ms="
                  << denoiser_profile.attention_kernel_milliseconds
                  << " denoiser_gpu_layout_ms=" << denoiser_gpu_layout_ms
                  << " denoiser_gpu_dequant_ms=" << denoiser_gpu_dequant_ms
                  << " denoiser_gpu_other_ms=" << denoiser_other_gpu_ms
                  << " denoiser_step_layout_ms=" << denoiser_step_layout_ms
                  << " scheduler_update_ms=" << scheduler_update_ms
                  << " latent_handoff_ms=" << unpatchify_ms
                  << " denoiser_output_io_ms=" << denoiser_output_io_ms
                  << " command_wall_ms=" << command_wall_ms << '\n'
                  << std::defaultfloat << std::setprecision(6);
      }
      std::cout << "H3_DENOISE PASS backend=" << backend_name << " device=\""
                << device_name << "\" schedule_points="
                << video_sigmas.size() << " model_evaluations=" << steps
                << " executed_evaluations=" << evaluations
                << " task="
                << (!options.references.empty()
                        ? "ref2va"
                        : (options.keyframes.empty() ? "t2va" : "fl2va"))
                << " sampler=" << options.sampler
                << " sequence=" << layout.sequence_length
                << " int8_attention_first_step="
                << options.h3_int8_attention_first_step
                << " persistent_reuse=" << (denoiser_reused ? 1 : 0)
                << " persistent_request=" << state.requests
                << " attention_class=" << attention_class
                << " projection_class="
                << (!options.h3_groupwise_cache.empty()
                        ? "approximate_groupwise_int8_quality_gate"
                        : (h3_w8a8_mlp_count == 0U &&
                                   h3_w8a8_attention_count == 0U
                               ? "exact_checkpoint_dtype"
                               : (!options.h3_convrot_int8_checkpoint.empty()
                                      ? "approximate_native_h256_convrot_int8_gate"
                                      : "approximate_w8a8_established_h3_gate")))
                << " h3_w8a8_mlps=" << h3_w8a8_mlp_count
                << " h3_w8a8_attentions=" << h3_w8a8_attention_count
                << " h3_w8a8_resident_mlps="
                << h3_w8a8_resident_mlp_count
                << " h3_w8a8_resident_attentions="
                << h3_w8a8_resident_attention_count
                << " h3_groupwise_blocks=" << h3_groupwise_count
                << " h3_ck_attentions=" << h3_ck_attention_count
                << " h3_modulation_caches=" << h3_modulation_cache_count
                << " repeated_invariant_ops="
                << repeated_invariant_operation_count
                << " repeated_invariant_persistent_bytes="
                << repeated_invariant_persistent_bytes
                << " repeated_invariant_cache_hits="
                << repeated_invariant_cache_hits
                << " resident_upload="
                << (options.lazy_resident_upload
                        ? "lazy_first_use"
                        : (options.pipelined_resident_upload
                               ? "pinned_pipeline"
                               : "pageable_direct"))
                << " resident_host_pages="
                << (options.keep_resident_host_pages ? "keep" : "evict")
                << " stage_threads="
                << (options.streamed_stage_threads == 0U
                        ? 1U
                        : options.streamed_stage_threads)
                << " denoiser_kernel_sum_ms=" << denoiser_kernel_ms
                << " denoiser_wall_ms=" << denoiser_wall_ms
                << " denoiser_resident_bytes=" << denoiser_resident
                << " denoiser_free_before_bytes=" << denoiser_free_before
                << " denoiser_free_after_bytes=" << denoiser_free_after
                << "\n";
      return 0;
    }

    double vae_prepare_ms = 0.0;
    double vae_kernel_ms = 0.0;
    double vae_bundle_map_ms = 0.0;
    double vae_output_io_ms = 0.0;
    double vae_gpu_layout_ms = 0.0;
    double vae_gpu_dequant_ms = 0.0;
    dif::runtime::PipelineProfile vae_profile;
    std::uint64_t vae_resident = 0U;
    const auto vae_wall_start = std::chrono::steady_clock::now();
    {
      const auto bundle_map_start = std::chrono::steady_clock::now();
      const auto program = dif::ir::read_file(options.vae_program);
      std::uint32_t latent_id = 0U;
      for (const auto &description : program.tensors) {
        if (description.has_role(dif::ir::TensorRole::Input)) {
          if (latent_id != 0U)
            dif::fail("VAE integration requires one dynamic input");
          latent_id = description.id;
        }
      }
      if (latent_id == 0U)
        dif::fail("VAE integration found no latent input");
      validate(program, latent_id, latent, "final latent");
      const auto [raw_id, decoded_id] = vae_outputs(program);
      auto inputs = dif::weights::load_weight_bundle(
          dif::weights::read_weight_bundle(options.vae_bundle), program,
          options.verify_shards);
      vae_bundle_map_ms = elapsed_milliseconds(bundle_map_start);
      inputs.insert_or_assign(latent_id, latent);
      auto backend = executor(options);
      auto prepared = backend->prepare(program, inputs, run_options);
      auto result = prepared->run(inputs, run_options);
      vae_prepare_ms = prepared->preparation_milliseconds();
      vae_resident = prepared->resident_bytes();
      vae_kernel_ms = result.mean_milliseconds;
      vae_profile = result.pipeline_profile;
      if (result.pipeline_profile.enabled) {
        vae_gpu_layout_ms =
            operation_sum(result.operation_timings, is_layout_operation);
        vae_gpu_dequant_ms =
            operation_sum(result.operation_timings, is_dequant_operation);
      }
      if (backend_name.empty()) {
        backend_name = result.backend_name;
        device_name = result.device_name;
      }
      const auto &raw = result.outputs.at(raw_id);
      const auto &decoded = result.outputs.at(decoded_id);
      std::uint64_t nonfinite = 0U;
      float minimum = std::numeric_limits<float>::infinity();
      float maximum = -std::numeric_limits<float>::infinity();
      for (const auto value : decoded.f32()) {
        if (!std::isfinite(value))
          ++nonfinite;
        else {
          minimum = std::min(minimum, value);
          maximum = std::max(maximum, value);
        }
      }
      if (nonfinite != 0U || minimum < 0.0F || maximum > 1.0F)
        dif::fail("integrated VAE decoded output failed finite/range checks");
      const auto vae_output_io_start = std::chrono::steady_clock::now();
      dif::runtime::write_tensor(raw, options.output_raw);
      dif::runtime::write_tensor(decoded, options.output_decoded);
      vae_output_io_ms = elapsed_milliseconds(vae_output_io_start);
      std::cout << "H3_DECODE range=[" << minimum << ',' << maximum
                << "] shape=[1,3," << decoded.dims[2] << ','
                << decoded.dims[3] << ',' << decoded.dims[4] << "]\n";
    }
    const auto vae_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - vae_wall_start)
            .count();
    const auto command_wall_ms = elapsed_milliseconds(command_wall_start);
    if (options.profile_pipeline) {
      const auto denoiser_other_gpu_ms = std::max(
          0.0, denoiser_profile.operation_kernel_milliseconds -
                   denoiser_profile.attention_kernel_milliseconds -
                   denoiser_gpu_layout_ms - denoiser_gpu_dequant_ms);
      const auto vae_other_gpu_ms =
          std::max(0.0, vae_profile.operation_kernel_milliseconds -
                            vae_profile.attention_kernel_milliseconds -
                            vae_gpu_layout_ms - vae_gpu_dequant_ms);
      std::cout << std::fixed << std::setprecision(3)
                << "H3_PROFILE input_io_ms=" << input_io_ms
                << " initial_layout_ms=" << initial_layout_ms
                << " denoiser_bundle_map_ms=" << denoiser_bundle_map_ms
                << " denoiser_prepare_ms=" << denoiser_prepare_ms
                << " denoiser_device_ms=" << denoiser_kernel_ms
                << " denoiser_wall_ms=" << denoiser_wall_ms
                << " denoiser_resident_weight_bytes="
                << denoiser_profile.resident_weight_bytes
                << " denoiser_resident_prefault_ms="
                << denoiser_profile.resident_host_prefault_milliseconds
                << " denoiser_resident_minor_faults="
                << denoiser_profile.resident_minor_page_faults
                << " denoiser_resident_major_faults="
                << denoiser_profile.resident_major_page_faults
                << " denoiser_resident_h2d_ms="
                << denoiser_profile.resident_h2d_milliseconds
                << " denoiser_resident_upload_ms="
                << denoiser_profile.resident_upload_milliseconds
                << " denoiser_streamed_host_stage_ms="
                << denoiser_profile.streamed_host_stage_milliseconds
                << " denoiser_streamed_weight_bytes="
                << denoiser_profile.streamed_weight_bytes
                << " denoiser_streamed_h2d_ms="
                << denoiser_profile.streamed_h2d_milliseconds
                << " denoiser_streamed_host_wait_ms="
                << denoiser_profile.streamed_host_wait_milliseconds
                << " denoiser_non_kernel_device_ms="
                << denoiser_profile.non_kernel_device_timeline_milliseconds
                << " denoiser_gpu_operation_ms="
                << denoiser_profile.operation_kernel_milliseconds
                << " denoiser_gpu_attention_ms="
                << denoiser_profile.attention_kernel_milliseconds
                << " denoiser_gpu_layout_ms=" << denoiser_gpu_layout_ms
                << " denoiser_gpu_dequant_ms=" << denoiser_gpu_dequant_ms
                << " denoiser_gpu_other_ms=" << denoiser_other_gpu_ms
                << " denoiser_step_layout_ms=" << denoiser_step_layout_ms
                << " scheduler_update_ms=" << scheduler_update_ms
                << " unpatchify_ms=" << unpatchify_ms
                << " denoiser_output_io_ms=" << denoiser_output_io_ms
                << " vae_bundle_map_ms=" << vae_bundle_map_ms
                << " vae_prepare_ms=" << vae_prepare_ms
                << " vae_resident_weight_bytes="
                << vae_profile.resident_weight_bytes
                << " vae_resident_upload_ms="
                << vae_profile.resident_upload_milliseconds
                << " vae_gpu_operation_ms="
                << vae_profile.operation_kernel_milliseconds
                << " vae_gpu_attention_ms="
                << vae_profile.attention_kernel_milliseconds
                << " vae_gpu_layout_ms=" << vae_gpu_layout_ms
                << " vae_gpu_dequant_ms=" << vae_gpu_dequant_ms
                << " vae_gpu_other_ms=" << vae_other_gpu_ms
                << " vae_output_io_ms=" << vae_output_io_ms
                << " vae_device_ms=" << vae_kernel_ms
                << " vae_wall_ms=" << vae_wall_ms
                << " command_wall_ms=" << command_wall_ms << '\n'
                << std::defaultfloat << std::setprecision(6);
    }
    std::cout << "H3_INFER PASS backend=" << backend_name << " device=\""
              << device_name << "\" steps=" << steps
              << " sequence=" << layout.sequence_length
              << " video_rows=" << layout.video_indices.size()
              << " condition_rows=" << layout.num_condition_video_rows
              << " condition_audio_rows="
              << layout.num_condition_audio_rows
              << " audio_rows=" << layout.audio_indices.size()
              << " denoiser_prepare_ms=" << denoiser_prepare_ms
              << " denoiser_kernel_sum_ms=" << denoiser_kernel_ms
              << " denoiser_wall_ms=" << denoiser_wall_ms
              << " denoiser_resident_bytes=" << denoiser_resident
              << " denoiser_free_before_bytes=" << denoiser_free_before
              << " denoiser_free_after_bytes=" << denoiser_free_after
              << " vae_prepare_ms=" << vae_prepare_ms
              << " vae_kernel_ms=" << vae_kernel_ms
              << " vae_wall_ms=" << vae_wall_ms
              << " vae_resident_bytes=" << vae_resident << "\n";
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Persistent denoiser protocol over a Unix stream socket. Request: u32 token
// count, then per token u32 length + bytes (the difh3infer argv without the
// program name). Reply: i32 status, u32 length, the request's stdout text.
// A request consisting of the single token --shutdown stops the server.
// ---------------------------------------------------------------------------
void write_all(int descriptor, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const std::uint8_t *>(data);
  while (bytes != 0U) {
    const auto written = ::write(descriptor, cursor, bytes);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      dif::fail(std::string("socket write failed: ") + std::strerror(errno));
    }
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
}

void read_all(int descriptor, void *data, std::size_t bytes) {
  auto *cursor = static_cast<std::uint8_t *>(data);
  while (bytes != 0U) {
    const auto received = ::read(descriptor, cursor, bytes);
    if (received < 0) {
      if (errno == EINTR)
        continue;
      dif::fail(std::string("socket read failed: ") + std::strerror(errno));
    }
    if (received == 0)
      dif::fail("socket closed before the message was complete");
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

void write_u32(int descriptor, std::uint32_t value) {
  write_all(descriptor, &value, sizeof(value));
}

std::uint32_t read_u32(int descriptor) {
  std::uint32_t value = 0;
  read_all(descriptor, &value, sizeof(value));
  return value;
}

void write_blob(int descriptor, const std::string &text) {
  if (text.size() > std::numeric_limits<std::uint32_t>::max())
    dif::fail("socket message exceeds the protocol size");
  write_u32(descriptor, static_cast<std::uint32_t>(text.size()));
  write_all(descriptor, text.data(), text.size());
}

std::string read_blob(int descriptor) {
  const auto length = read_u32(descriptor);
  if (length > 64U * 1024U * 1024U)
    dif::fail("socket message exceeds 64 MiB");
  std::string text(length, '\0');
  read_all(descriptor, text.data(), length);
  return text;
}

int connect_socket(const std::filesystem::path &socket_path) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const auto path = socket_path.string();
  if (path.size() >= sizeof(address.sun_path))
    dif::fail("socket path is too long for AF_UNIX: " + path);
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1U);
  const int descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (descriptor < 0)
    dif::fail(std::string("cannot create socket: ") + std::strerror(errno));
  if (::connect(descriptor, reinterpret_cast<const sockaddr *>(&address),
                sizeof(address)) != 0) {
    const auto reason = std::string(std::strerror(errno));
    ::close(descriptor);
    dif::fail("cannot connect to persistent denoiser " + path + ": " + reason);
  }
  return descriptor;
}

int run_client(const std::filesystem::path &socket_path,
               const std::vector<std::string> &arguments) {
  const int descriptor = connect_socket(socket_path);
  write_u32(descriptor, static_cast<std::uint32_t>(arguments.size()));
  for (const auto &argument : arguments)
    write_blob(descriptor, argument);
  std::int32_t status = 0;
  read_all(descriptor, &status, sizeof(status));
  const auto text = read_blob(descriptor);
  ::close(descriptor);
  std::cout << text << std::flush;
  return static_cast<int>(status);
}

int serve(const std::filesystem::path &socket_path, ServerState &state) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const auto path = socket_path.string();
  if (path.size() >= sizeof(address.sun_path))
    dif::fail("socket path is too long for AF_UNIX: " + path);
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1U);
  ::unlink(path.c_str());
  const int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listener < 0)
    dif::fail(std::string("cannot create socket: ") + std::strerror(errno));
  if (::bind(listener, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 4) != 0) {
    const auto reason = std::string(std::strerror(errno));
    ::close(listener);
    dif::fail("cannot listen on " + path + ": " + reason);
  }
  std::cout << "H3_SERVE READY socket=" << path << " pid=" << ::getpid()
            << " prepared_requests=" << state.requests
            << " resident_bytes=" << state.resident << "\n"
            << std::flush;
  std::size_t served = 0U;
  for (;;) {
    const int connection = ::accept(listener, nullptr, nullptr);
    if (connection < 0) {
      if (errno == EINTR)
        continue;
      ::close(listener);
      dif::fail(std::string("accept failed: ") + std::strerror(errno));
    }
    std::vector<std::string> arguments;
    std::int32_t status = 0;
    std::string reply;
    try {
      const auto count = read_u32(connection);
      if (count > 4096U)
        dif::fail("request carries too many tokens");
      arguments.reserve(count);
      for (std::uint32_t token = 0U; token < count; ++token)
        arguments.push_back(read_blob(connection));
    } catch (const std::exception &error) {
      std::cerr << "difh3infer: bad request: " << error.what() << "\n";
      ::close(connection);
      continue;
    }
    if (arguments.size() == 1U && arguments.front() == "--shutdown") {
      reply = "H3_SERVE SHUTDOWN served=" + std::to_string(served) + "\n";
      write_all(connection, &status, sizeof(status));
      write_blob(connection, reply);
      ::close(connection);
      break;
    }
    const auto request_start = std::chrono::steady_clock::now();
    std::ostringstream captured;
    auto *previous = std::cout.rdbuf(captured.rdbuf());
    try {
      std::vector<std::string> storage;
      storage.reserve(arguments.size() + 1U);
      storage.emplace_back("difh3infer");
      for (const auto &argument : arguments)
        storage.push_back(argument);
      std::vector<char *> pointers;
      pointers.reserve(storage.size());
      for (auto &token : storage)
        pointers.push_back(token.data());
      const auto options = parse(static_cast<int>(pointers.size()),
                                 pointers.data());
      if (!options.serve_socket.empty())
        dif::fail("a served request cannot carry --serve");
      status = run_request(options, state, request_start);
    } catch (const std::exception &error) {
      captured << "difh3infer: " << error.what() << "\n";
      status = 1;
    }
    std::cout.rdbuf(previous);
    ++served;
    reply = captured.str();
    try {
      write_all(connection, &status, sizeof(status));
      write_blob(connection, reply);
    } catch (const std::exception &error) {
      std::cerr << "difh3infer: reply failed: " << error.what() << "\n";
    }
    ::close(connection);
    std::cout << "H3_SERVE REQUEST index=" << served << " status=" << status
              << " wall_ms=" << elapsed_milliseconds(request_start) << "\n"
              << std::flush;
  }
  ::close(listener);
  ::unlink(path.c_str());
  std::cout << "H3_SERVE STOPPED served=" << served << "\n" << std::flush;
  return 0;
}

int main(int argc, char **argv) {
  try {
    std::vector<std::string> arguments(argv + 1, argv + argc);
    for (std::size_t index = 0U; index < arguments.size(); ++index) {
      if (arguments[index] != "--connect")
        continue;
      if (index + 1U >= arguments.size())
        dif::fail("missing value for --connect");
      const std::filesystem::path socket_path = arguments[index + 1U];
      arguments.erase(arguments.begin() + static_cast<std::ptrdiff_t>(index),
                      arguments.begin() + static_cast<std::ptrdiff_t>(index) + 2);
      return run_client(socket_path, arguments);
    }
    const auto command_wall_start = std::chrono::steady_clock::now();
    const auto options = parse(argc, argv);
    ServerState state;
    const auto status = run_request(options, state, command_wall_start);
    if (status != 0 || options.serve_socket.empty())
      return status;
    return serve(options.serve_socket, state);
  } catch (const std::exception &error) {
    std::cerr << "difh3infer: " << error.what() << "\n";
    return 1;
  }
}
