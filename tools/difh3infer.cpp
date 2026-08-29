#include "dif/backend/plugin.hpp"
#include "dif/frontend/h3_conditioning.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
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
  std::filesystem::path output_raw;
  std::filesystem::path output_decoded;
  std::filesystem::path cache_directory;
  std::uint64_t latent_frames{};
  std::uint64_t latent_height{};
  std::uint64_t latent_width{};
  std::uint64_t audio_latents{};
  std::uint64_t patch_height{2U};
  std::uint64_t patch_width{2U};
  float condition_video_timestep{};
  std::vector<dif::frontend::H3KeyframeAnchor> keyframes;
  std::uint64_t minimum_free_bytes{4096ULL * 1024ULL * 1024ULL};
  bool verify_shards{false};
  bool profile_pipeline{false};
};

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (text.empty() || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

float finite_float(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtof(text.c_str(), &end);
  if (text.empty() || !end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

void usage() {
  std::cerr
      << "usage: difh3infer --backend cpu|cuda --denoiser-program FILE.difir --denoiser-bundle FILE.difbind --vae-program FILE.difir --vae-bundle FILE.difbind --text-tags FILE.diftensor --text FILE.diftensor --video FILE.diftensor --audio FILE.diftensor --video-sigmas FILE.diftensor --audio-sigmas FILE.diftensor --latent-t N --latent-h N --latent-w N --audio-latents N --keyframes none|first|last|first-last --condition-video-timestep F --output-latent FILE.diftensor --output-audio FILE.diftensor --output-raw FILE.diftensor --output-decoded FILE.diftensor [--patch-h N] [--patch-w N] [--backend-plugin FILE.so] [--verify-shards] [--profile-pipeline] [--cache-dir DIR] [--min-free-mib N]\n";
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
    else if (option == "--condition-video-timestep")
      options.condition_video_timestep =
          finite_float(value("--condition-video-timestep"),
                       "condition video timestep");
    else if (option == "--output-latent")
      options.output_latent = value("--output-latent");
    else if (option == "--output-audio")
      options.output_audio = value("--output-audio");
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
    else {
      usage();
      dif::fail("unknown difh3infer option: " + option);
    }
  }
  for (const auto *path : {&options.denoiser_program, &options.denoiser_bundle,
                           &options.vae_program, &options.vae_bundle,
                           &options.text_tags, &options.text, &options.video,
                           &options.audio, &options.video_sigmas,
                           &options.audio_sigmas, &options.output_latent,
                           &options.output_audio, &options.output_raw,
                           &options.output_decoded}) {
    if (path->empty()) {
      usage();
      dif::fail("difh3infer is missing a required path");
    }
  }
  if (options.latent_frames == 0U || options.latent_height == 0U ||
      options.latent_width == 0U || options.audio_latents == 0U ||
      options.patch_height == 0U || options.patch_width == 0U)
    dif::fail("H3 inference geometry must be positive");
  for (const auto *path : {&options.output_latent, &options.output_audio,
                           &options.output_raw, &options.output_decoded}) {
    if (std::filesystem::exists(*path))
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

void euler_update(dif::runtime::Tensor &sample,
                  const dif::runtime::Tensor &velocity, float timestep,
                  float sigma, float sigma_next, std::uint64_t first_row) {
  if (!(sigma > 0.0F) || sigma_next < 0.0F || sigma_next > sigma ||
      !std::isfinite(timestep))
    dif::fail("invalid H3 scheduler step");
  if (sample.dtype != dif::ir::DType::F32 || velocity.dtype != sample.dtype ||
      velocity.dims != sample.dims || sample.dims.size() != 2U ||
      first_row > sample.dims[0])
    dif::fail("H3 scheduler sample/velocity mismatch");
  const auto width = sample.dims[1];
  const auto ratio = sigma_next / sigma;
  auto values = sample.f32();
  const auto velocities = velocity.f32();
  for (std::uint64_t row = first_row; row < sample.dims[0]; ++row) {
    for (std::uint64_t column = 0U; column < width; ++column) {
      const auto index = row * width + column;
      const auto sample_value = values[static_cast<std::size_t>(index)];
      const auto velocity_value =
          velocities[static_cast<std::size_t>(index)];
      volatile float velocity_delta = (1.0F - timestep) * velocity_value;
      volatile float denoised = sample_value + velocity_delta;
      volatile float weighted_sample = ratio * sample_value;
      volatile float weighted_denoised = (1.0F - ratio) * denoised;
      volatile float updated = weighted_sample + weighted_denoised;
      values[static_cast<std::size_t>(index)] = updated;
    }
  }
}

dif::runtime::Tensor target_latent(const dif::runtime::Tensor &video,
                                   std::uint64_t condition_rows,
                                   const Options &options) {
  constexpr std::uint64_t channels = 24U;
  const auto rows_per_frame =
      (options.latent_height / options.patch_height) *
      (options.latent_width / options.patch_width);
  const auto target_rows = options.latent_frames * rows_per_frame;
  const auto patch_values =
      channels * options.patch_height * options.patch_width;
  if (video.dtype != dif::ir::DType::F32 || video.dims.size() != 2U ||
      video.dims[0] != condition_rows + target_rows ||
      video.dims[1] != patch_values)
    dif::fail("final video rows do not match requested latent geometry");
  dif::runtime::Tensor latent{
      dif::ir::DType::F32,
      {1U, channels, options.latent_frames, options.latent_height,
       options.latent_width},
      std::vector<std::uint8_t>(static_cast<std::size_t>(
          channels * options.latent_frames * options.latent_height *
          options.latent_width * sizeof(float)))};
  const auto rows = video.f32();
  auto output = latent.f32();
  for (std::uint64_t frame = 0U; frame < options.latent_frames; ++frame) {
    for (std::uint64_t patch_y = 0U;
         patch_y < options.latent_height / options.patch_height; ++patch_y) {
      for (std::uint64_t patch_x = 0U;
           patch_x < options.latent_width / options.patch_width; ++patch_x) {
        const auto row = condition_rows +
                         (frame * (options.latent_height / options.patch_height) +
                          patch_y) *
                             (options.latent_width / options.patch_width) +
                         patch_x;
        for (std::uint64_t channel = 0U; channel < channels; ++channel) {
          for (std::uint64_t y = 0U; y < options.patch_height; ++y) {
            for (std::uint64_t x = 0U; x < options.patch_width; ++x) {
              const auto column =
                  (channel * options.patch_height + y) * options.patch_width +
                  x;
              const auto at =
                  ((channel * options.latent_frames + frame) *
                       options.latent_height +
                   patch_y * options.patch_height + y) *
                      options.latent_width +
                  patch_x * options.patch_width + x;
              output[static_cast<std::size_t>(at)] =
                  rows[static_cast<std::size_t>(row * patch_values + column)];
            }
          }
        }
      }
    }
  }
  return latent;
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
    const auto tags_tensor = dif::runtime::read_tensor(options.text_tags);
    input_io_ms += elapsed_milliseconds(timed_io_start);
    if (tags_tensor.dtype != dif::ir::DType::I32 ||
        tags_tensor.dims.size() != 1U)
      dif::fail("text tags must be I32[N]");
    std::vector<std::int32_t> tags(tags_tensor.element_count());
    std::memcpy(tags.data(), tags_tensor.data(), tags_tensor.byte_size());
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
    const auto video_schedule =
        dif::runtime::read_tensor(options.video_sigmas);
    const auto audio_schedule =
        dif::runtime::read_tensor(options.audio_sigmas);
    input_io_ms += elapsed_milliseconds(timed_io_start);
    if (video_schedule.dtype != dif::ir::DType::F32 ||
        audio_schedule.dtype != dif::ir::DType::F32 ||
        video_schedule.dims.size() != 1U ||
        video_schedule.dims != audio_schedule.dims ||
        video_schedule.element_count() < 2U)
      dif::fail("video/audio sigmas must be matching F32[K+1]");
    const auto video_sigmas = video_schedule.f32();
    const auto audio_sigmas = audio_schedule.f32();
    const auto steps = video_sigmas.size() - 1U;

    dif::runtime::RunOptions run_options;
    run_options.warmups = 0U;
    run_options.iterations = 1U;
    run_options.minimum_free_bytes = options.minimum_free_bytes;
    run_options.cache_directory = options.cache_directory;
    run_options.profile_pipeline = options.profile_pipeline;
    double denoiser_prepare_ms = 0.0;
    double denoiser_kernel_ms = 0.0;
    double denoiser_bundle_map_ms = 0.0;
    double denoiser_step_layout_ms = 0.0;
    double scheduler_update_ms = 0.0;
    double denoiser_gpu_layout_ms = 0.0;
    double denoiser_gpu_dequant_ms = 0.0;
    dif::runtime::PipelineProfile denoiser_profile;
    std::uint64_t denoiser_resident = 0U;
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
      auto prepared = backend->prepare(program, inputs, run_options);
      denoiser_prepare_ms = prepared->preparation_milliseconds();
      denoiser_resident = prepared->resident_bytes();
      for (std::size_t step = 0U; step < steps; ++step) {
        const auto video_timestep = 1.0F - video_sigmas[step];
        const auto audio_timestep = 1.0F - audio_sigmas[step];
        const auto step_layout_start = std::chrono::steady_clock::now();
        auto plan = dif::frontend::make_h3_row_timestep_plan(
            layout, video_timestep, audio_timestep,
            options.condition_video_timestep, 0.0F);
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
        auto result = prepared->run(inputs, run_options);
        denoiser_kernel_ms += result.mean_milliseconds;
        backend_name = result.backend_name;
        device_name = result.device_name;
        if (result.pipeline_profile.enabled) {
          denoiser_profile.enabled = true;
          denoiser_profile.measured_iterations +=
              result.pipeline_profile.measured_iterations;
          denoiser_profile.resident_weight_bytes =
              result.pipeline_profile.resident_weight_bytes;
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
        euler_update(video, result.outputs.at(video_output), video_timestep,
                     video_sigmas[step], video_sigmas[step + 1U],
                     layout.num_condition_video_rows);
        euler_update(audio, result.outputs.at(audio_output), audio_timestep,
                     audio_sigmas[step], audio_sigmas[step + 1U], 0U);
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
    auto latent =
        target_latent(video, layout.num_condition_video_rows, options);
    const auto unpatchify_ms = elapsed_milliseconds(unpatchify_start);
    const auto denoiser_output_io_start = std::chrono::steady_clock::now();
    dif::runtime::write_tensor(latent, options.output_latent);
    dif::runtime::write_tensor(audio, options.output_audio);
    const auto denoiser_output_io_ms =
        elapsed_milliseconds(denoiser_output_io_start);

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
                << " denoiser_checkpoint_host_stage_ms="
                << denoiser_profile.streamed_host_stage_milliseconds
                << " denoiser_streamed_weight_bytes="
                << denoiser_profile.streamed_weight_bytes
                << " denoiser_h2d_active_ms="
                << denoiser_profile.streamed_h2d_milliseconds
                << " denoiser_stream_wait_ms="
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
