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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string backend{"cuda"};
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
  std::filesystem::path output_audio;
  std::filesystem::path output_audio_latent;
  std::filesystem::path output_handoff;
  std::filesystem::path output_raw;
  std::filesystem::path output_decoded;
  std::filesystem::path cache_directory;
  std::filesystem::path h3_w8a8_cache;
  std::filesystem::path h3_ck_attention_dso;
  std::filesystem::path h3_modulation_cache;
  std::filesystem::path h3_modulation_source_index;
  std::uint32_t h3_w8a8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_modulation_steps{};
  std::uint64_t all_text_tokens{};
  std::uint64_t latent_frames{};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t audio_latents{};
  std::uint64_t patch_height{2U};
  std::uint64_t patch_width{2U};
  std::uint32_t schedule_points{};
  std::vector<dif::frontend::H3KeyframeAnchor> keyframes;
  std::uint64_t minimum_free_bytes{4096ULL * 1024ULL * 1024ULL};
  bool verify_shards{false};
  bool profile_pipeline{false};
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
      << "usage: difh3infer --backend cpu|cuda --denoiser-program FILE.difir --denoiser-bundle FILE.difbind (--text-tags FILE.diftensor | --all-text-tokens N) --text FILE.diftensor --video FILE.diftensor --audio FILE.diftensor (--schedule-points N | --video-sigmas FILE.diftensor --audio-sigmas FILE.diftensor) --latent-t N --latent-h N --latent-w N --audio-latents N --keyframes none|first|last|first-last --output-latent FILE.diftensor --output-audio FILE.diftensor [--output-audio-latent FILE.diftensor] [--output-handoff latents.safetensors] [--h3-w8a8-cache FILE.safetensors] [--h3-w8a8-resident-layers N] [--h3-modulation-cache FILE.safetensors --h3-modulation-source-index FILE.index.json [--h3-modulation-steps N]] [--h3-ck-attention-dso FILE.so] [--denoise-only | --vae-program FILE.difir --vae-bundle FILE.difbind --output-raw FILE.diftensor --output-decoded FILE.diftensor] [--patch-h N] [--patch-w N] [--backend-plugin FILE.so] [--verify-shards] [--profile-pipeline] [--cache-dir DIR] [--min-free-mib N]\n";
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

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--backend")
      options.backend = value("--backend");
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
    else if (option == "--output-latent")
      options.output_latent = value("--output-latent");
    else if (option == "--output-audio")
      options.output_audio = value("--output-audio");
    else if (option == "--output-audio-latent")
      options.output_audio_latent = value("--output-audio-latent");
    else if (option == "--output-handoff")
      options.output_handoff = value("--output-handoff");
    else if (option == "--h3-w8a8-cache")
      options.h3_w8a8_cache = value("--h3-w8a8-cache");
    else if (option == "--h3-w8a8-resident-layers") {
      const auto layers =
          number(value("--h3-w8a8-resident-layers"), "resident layers");
      if (layers > std::numeric_limits<std::uint32_t>::max())
        dif::fail("H3 W8A8 resident layer count exceeds U32 range");
      options.h3_w8a8_resident_layers = static_cast<std::uint32_t>(layers);
    }
    else if (option == "--h3-ck-attention-dso")
      options.h3_ck_attention_dso = value("--h3-ck-attention-dso");
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
    else if (option == "--output-raw")
      options.output_raw = value("--output-raw");
    else if (option == "--output-decoded")
      options.output_decoded = value("--output-decoded");
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
    else if (option == "--denoise-only")
      options.denoise_only = true;
    else {
      usage();
      dif::fail("unknown difh3infer option: " + option);
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
  if (options.h3_modulation_cache.empty() !=
      options.h3_modulation_source_index.empty())
    dif::fail("H3 modulation cache and source index must be supplied together");
  const auto supplied_schedule = !options.video_sigmas.empty() ||
                                 !options.audio_sigmas.empty();
  if ((options.schedule_points != 0U) == supplied_schedule ||
      (supplied_schedule && (options.video_sigmas.empty() ||
                             options.audio_sigmas.empty())) ||
      (options.schedule_points != 0U && options.schedule_points < 2U))
    dif::fail("select either --schedule-points >= 2 or both sigma files");
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
    if (selected(timing.opcode))
      total += timing.mean_milliseconds;
  }
  return total;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto command_wall_start = std::chrono::steady_clock::now();
    const auto options = parse(argc, argv);
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
    const auto layout = dif::frontend::make_h3_t2va_layout(
        tags, options.latent_frames, options.latent_height,
        options.latent_width, options.audio_latents, 1U, options.patch_height,
        options.patch_width, options.keyframes);
    const auto initial_layout_ms = elapsed_milliseconds(initial_layout_start);
    timed_io_start = std::chrono::steady_clock::now();
    auto text = dif::runtime::read_tensor(options.text);
    auto video = dif::runtime::read_tensor(options.video);
    auto audio = dif::runtime::read_tensor(options.audio);
    std::vector<float> video_sigmas;
    std::vector<float> audio_sigmas;
    if (options.schedule_points != 0U) {
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

    dif::runtime::RunOptions run_options;
    run_options.warmups = 0U;
    run_options.iterations = 1U;
    run_options.minimum_free_bytes = options.minimum_free_bytes;
    run_options.cache_directory = options.cache_directory;
    run_options.profile_pipeline = options.profile_pipeline;
    auto denoiser_run_options = run_options;
    denoiser_run_options.h3_w8a8_cache = options.h3_w8a8_cache;
    denoiser_run_options.h3_w8a8_resident_layers =
        options.h3_w8a8_resident_layers;
    denoiser_run_options.h3_ck_attention_dso = options.h3_ck_attention_dso;
    denoiser_run_options.h3_modulation_cache = options.h3_modulation_cache;
    denoiser_run_options.h3_modulation_source_index =
        options.h3_modulation_source_index;
    denoiser_run_options.h3_modulation_steps =
        options.h3_modulation_steps != 0U ? options.h3_modulation_steps
                                         : options.schedule_points;
    double denoiser_prepare_ms = 0.0;
    double denoiser_kernel_ms = 0.0;
    double denoiser_bundle_map_ms = 0.0;
    double denoiser_step_layout_ms = 0.0;
    double scheduler_update_ms = 0.0;
    double denoiser_gpu_layout_ms = 0.0;
    double denoiser_gpu_dequant_ms = 0.0;
    dif::runtime::PipelineProfile denoiser_profile;
    std::uint64_t denoiser_resident = 0U;
    std::uint64_t denoiser_free_before = 0U;
    std::uint64_t denoiser_free_after = 0U;
    std::size_t h3_w8a8_mlp_count = 0U;
    std::size_t h3_w8a8_attention_count = 0U;
    std::size_t h3_w8a8_resident_mlp_count = 0U;
    std::size_t h3_w8a8_resident_attention_count = 0U;
    std::size_t h3_ck_attention_count = 0U;
    std::size_t h3_modulation_cache_count = 0U;
    std::string backend_name;
    std::string device_name;
    const auto denoiser_wall_start = std::chrono::steady_clock::now();
    {
      const auto bundle_map_start = std::chrono::steady_clock::now();
      const auto program = dif::ir::read_file(options.denoiser_program);
      validate(program, 1U, video, "video input");
      validate(program, 2U, audio, "audio input");
      validate(program, 3U, text, "text conditioning");
      if (program.tensor(1U)->dims[0] != layout.video_indices.size() ||
          program.tensor(2U)->dims[0] != layout.audio_indices.size() ||
          program.tensor(3U)->dims[0] != tags.size() ||
          program.tensor(5U)->dims[0] != layout.sequence_length)
        dif::fail("native H3 layout disagrees with denoiser geometry");
      const auto [video_output, audio_output] = denoiser_outputs(program);
      auto inputs = dif::weights::load_weight_bundle(
          dif::weights::read_weight_bundle(options.denoiser_bundle), program,
          options.verify_shards);
      denoiser_bundle_map_ms = elapsed_milliseconds(bundle_map_start);
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
      auto backend = executor(options);
      auto prepared = backend->prepare(program, inputs, denoiser_run_options);
      denoiser_prepare_ms = prepared->preparation_milliseconds();
      denoiser_resident = prepared->resident_bytes();
      for (std::size_t step = 0U; step < steps; ++step) {
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
        inputs.insert_or_assign(
            1U, video);
        inputs.insert_or_assign(2U, audio);
        inputs.insert_or_assign(
            4U, f32_tensor({plan.timesteps.size()}, plan.timesteps));
        inputs.insert_or_assign(
            8U, i32_tensor({layout.sequence_length}, plan.adaln_indices));
        inputs.insert_or_assign(
            9U, i32_tensor({layout.sequence_length}, plan.timestep_indices));
        denoiser_step_layout_ms += elapsed_milliseconds(step_layout_start);
        denoiser_run_options.h3_modulation_slice =
            static_cast<std::uint32_t>(step);
        auto result = prepared->run(inputs, denoiser_run_options);
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
          h3_modulation_cache_count = result.h3_modulation_caches.size();
          if (!options.h3_w8a8_cache.empty() &&
              (h3_w8a8_mlp_count == 0U ||
               h3_w8a8_attention_count == 0U))
            dif::fail("requested H3 W8A8 cache did not cover both projection and MLP chains");
          if (!options.h3_ck_attention_dso.empty() &&
              h3_ck_attention_count == 0U)
            dif::fail("requested H3 CK attention DSO was not admitted");
          if (!options.h3_modulation_cache.empty() &&
              h3_modulation_cache_count == 0U)
            dif::fail("requested H3 modulation schedule cache was not admitted");
        }
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
        }
        const auto scheduler_start = std::chrono::steady_clock::now();
        const auto &video_velocity = result.outputs.at(video_output);
        const auto &audio_velocity = result.outputs.at(audio_output);
        if (video_velocity.dtype != dif::ir::DType::F32 ||
            video_velocity.dims != video.dims ||
            audio_velocity.dtype != dif::ir::DType::F32 ||
            audio_velocity.dims != audio.dims)
          dif::fail("H3 denoiser output shape does not match scheduler state");
        dif::sampling::h3_euler_step_in_place(
            video.f32(), video_velocity.f32(), video.dims[1],
            layout.num_condition_video_rows, video_timestep,
            video_sigmas[step], video_sigmas[step + 1U]);
        dif::sampling::h3_euler_step_in_place(
            audio.f32(), audio_velocity.f32(), audio.dims[1],
            layout.num_condition_audio_rows, audio_timestep,
            audio_sigmas[step], audio_sigmas[step + 1U]);
        scheduler_update_ms += elapsed_milliseconds(scheduler_start);
        std::cout << "H3_STEP index=" << step
                  << " video_t=" << video_timestep
                  << " audio_t=" << audio_timestep
                  << " denoiser_ms=" << result.mean_milliseconds << "\n";
      }
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
                << " sequence=" << layout.sequence_length
                << " attention_class="
                << (h3_ck_attention_count == 0U
                        ? "exact_program_declared"
                        : "approximate_ck_int8_established_h3_gate")
                << " projection_class="
                << (h3_w8a8_mlp_count == 0U &&
                            h3_w8a8_attention_count == 0U
                        ? "exact_checkpoint_dtype"
                        : "approximate_w8a8_established_h3_gate")
                << " h3_w8a8_mlps=" << h3_w8a8_mlp_count
                << " h3_w8a8_attentions=" << h3_w8a8_attention_count
                << " h3_w8a8_resident_mlps="
                << h3_w8a8_resident_mlp_count
                << " h3_w8a8_resident_attentions="
                << h3_w8a8_resident_attention_count
                << " h3_ck_attentions=" << h3_ck_attention_count
                << " h3_modulation_caches=" << h3_modulation_cache_count
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
  } catch (const std::exception &error) {
    std::cerr << "difh3infer: " << error.what() << "\n";
    return 1;
  }
}
