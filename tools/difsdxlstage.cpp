// Run one SDXL stage (VAE decoder, CLIP-L, OpenCLIP-G, UNet) of the compiler
// on the inputs recorded by its creator oracle and write every captured
// boundary, so tools/gate_sdxl_stage.py can compare the two.
//
// usage: difsdxlstage --stage vae|clip-l|clip-g|unet --checkpoint FILE
//            --fixture reference.safetensors --output actual.safetensors
//            [--backend cuda|cpu] [--dtype f16|bf16|f32] [--cache-dir DIR]

#include "dif/frontend/sdxl_clip.hpp"
#include "dif/frontend/sdxl_unet.hpp"
#include "dif/frontend/sdxl_vae.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Arguments {
  std::string stage;
  fs::path checkpoint;
  fs::path fixture;
  fs::path output;
  std::string backend{"cuda"};
  std::string dtype;
  fs::path cache_directory;
  // Timed replay: the executor's own mean/minimum over these iterations is
  // the stage's compute cost, with preparation and readback excluded.
  std::uint32_t warmups{};
  std::uint32_t iterations{1};
  bool capture{true};
  // Print this many of the costliest operations, grouped by opcode and
  // output shape, so a slow stage names its own hot spots.
  std::uint32_t profile{};
  // Benchmark every Linear at prepare and keep the fastest cuBLASLt
  // algorithm. Same dtype, same math: an autotune, not a precision change.
  bool tune_linears{};
};

Arguments parse(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value for " + flag);
      return argv[++index];
    };
    if (flag == "--stage")
      arguments.stage = value();
    else if (flag == "--checkpoint")
      arguments.checkpoint = value();
    else if (flag == "--fixture")
      arguments.fixture = value();
    else if (flag == "--output")
      arguments.output = value();
    else if (flag == "--backend")
      arguments.backend = value();
    else if (flag == "--dtype")
      arguments.dtype = value();
    else if (flag == "--cache-dir")
      arguments.cache_directory = value();
    else if (flag == "--warmups")
      arguments.warmups = static_cast<std::uint32_t>(std::stoul(value()));
    else if (flag == "--iterations")
      arguments.iterations = static_cast<std::uint32_t>(std::stoul(value()));
    else if (flag == "--no-capture")
      arguments.capture = false;
    else if (flag == "--profile")
      arguments.profile = static_cast<std::uint32_t>(std::stoul(value()));
    else if (flag == "--tune-linears")
      arguments.tune_linears = true;
    else
      dif::fail("unknown argument " + flag);
  }
  if (arguments.stage.empty() || arguments.checkpoint.empty() ||
      arguments.fixture.empty() || arguments.output.empty())
    dif::fail("--stage, --checkpoint, --fixture, and --output are required");
  return arguments;
}

dif::ir::DType parse_dtype(const std::string &name, dif::ir::DType fallback) {
  if (name.empty())
    return fallback;
  if (name == "f16")
    return dif::ir::DType::F16;
  if (name == "bf16")
    return dif::ir::DType::BF16;
  if (name == "f32")
    return dif::ir::DType::F32;
  dif::fail("unknown dtype " + name);
}

dif::runtime::Tensor as_dtype(dif::runtime::Tensor tensor, dif::ir::DType dtype) {
  if (tensor.dtype == dtype)
    return tensor;
  return dif::runtime::convert_float_tensor(tensor, dtype);
}

dif::runtime::Tensor owned_copy(const dif::runtime::Tensor &source) {
  dif::runtime::Tensor result{source.dtype, source.dims, {}};
  result.bytes.assign(source.data(), source.data() + source.byte_size());
  result.validate();
  return result;
}

// Rows [index*rows, (index+1)*rows) of a fused [3*rows, ...] tensor.
dif::runtime::Tensor fused_rows(const dif::runtime::Tensor &source,
                                std::uint64_t index, std::uint64_t rows) {
  if (source.dims.empty() || source.dims[0] != 3U * rows)
    dif::fail("fused q/k/v tensor does not hold three row blocks");
  auto dims = source.dims;
  dims[0] = rows;
  const auto row_bytes =
      source.byte_size() / static_cast<std::size_t>(source.dims[0]);
  dif::runtime::Tensor result{source.dtype, dims, {}};
  result.bytes.assign(source.data() + index * rows * row_bytes,
                      source.data() + (index + 1U) * rows * row_bytes);
  result.validate();
  return result;
}

dif::runtime::Tensor transposed(const dif::runtime::Tensor &source) {
  if (source.dims.size() != 2U)
    dif::fail("transpose expects a rank-2 tensor");
  const auto rows = source.dims[0];
  const auto columns = source.dims[1];
  dif::runtime::Tensor result{source.dtype, {columns, rows}, {}};
  result.bytes.resize(source.byte_size());
  result.validate();
  for (std::uint64_t row = 0U; row < rows; ++row)
    for (std::uint64_t column = 0U; column < columns; ++column)
      dif::runtime::store_float(
          result, column * rows + row,
          dif::runtime::load_float(source, row * columns + column));
  return result;
}

void check_binding(const dif::ir::Program &program, std::uint32_t id,
                   const dif::runtime::Tensor &tensor,
                   const std::string &name) {
  const auto *description = program.tensor(id);
  if (!description)
    dif::fail("program lost binding tensor " + name);
  if (tensor.dtype != description->dtype || tensor.dims != description->dims)
    dif::fail("checkpoint tensor disagrees with the program: " + name);
}

dif::runtime::Tensor fixture_input(const dif::weights::SafeTensorFile &file,
                                   const std::string &name,
                                   dif::ir::DType dtype) {
  if (!file.find(name))
    dif::fail("fixture has no tensor " + name);
  auto tensor = owned_copy(dif::weights::map_safetensor(file, name));
  if (tensor.dtype != dtype) {
    if (dtype == dif::ir::DType::I32 || tensor.dtype == dif::ir::DType::I32)
      dif::fail("fixture tensor " + name + " has the wrong integer dtype");
    tensor = dif::runtime::convert_float_tensor(tensor, dtype);
  }
  return tensor;
}

struct Stage {
  dif::ir::Program program;
  dif::runtime::TensorMap bindings;
  std::vector<std::pair<std::string, std::uint32_t>> outputs;
};

Stage build_vae(const Arguments &arguments,
                const dif::weights::SafeTensorFile &checkpoint,
                const dif::weights::SafeTensorFile &fixture) {
  Stage stage;
  dif::frontend::SdxlVaeConfig config;
  config.capture_boundaries = arguments.capture;
  config.dtype = parse_dtype(arguments.dtype, dif::ir::DType::BF16);
  const auto *latent = fixture.find("latent_input");
  if (!latent || latent->dims.size() != 4U)
    dif::fail("fixture latent_input must be [B,4,h,w]");
  config.batch = latent->dims[0];
  config.latent_height = latent->dims[2];
  config.latent_width = latent->dims[3];
  auto build = dif::frontend::make_sdxl_vae_decoder(config);
  for (const auto &weight : build.weights) {
    // Map, then convert once: an owned copy first would double the traffic
    // for a multi-gigabyte tower.
    auto tensor = as_dtype(
        dif::weights::map_safetensor(checkpoint, weight.source_name),
        config.dtype);
    check_binding(build.program, weight.tensor, tensor, weight.source_name);
    stage.bindings.emplace(weight.tensor, std::move(tensor));
  }
  stage.bindings.emplace(build.latent_input,
                         fixture_input(fixture, "latent_input", config.dtype));
  stage.outputs = build.boundaries;
  stage.program = std::move(build.program);
  return stage;
}

Stage build_clip(const Arguments &arguments,
                 const dif::weights::SafeTensorFile &checkpoint,
                 const dif::weights::SafeTensorFile &fixture, bool g) {
  Stage stage;
  auto config = g ? dif::frontend::sdxl_clip_g_config()
                  : dif::frontend::sdxl_clip_l_config();
  config.capture_boundaries = arguments.capture;
  config.dtype = parse_dtype(arguments.dtype, dif::ir::DType::F32);
  if (config.dtype != dif::ir::DType::F32)
    config.attention_implementation = 2U;
  auto build = dif::frontend::make_clip_text_tower(config);
  const auto hidden = config.hidden_size;
  for (const auto &weight : build.weights) {
    auto source = dif::weights::map_safetensor(checkpoint, weight.source_name);
    using Transform = dif::frontend::ClipWeightTransform;
    switch (weight.transform) {
    case Transform::Direct:
      break;
    case Transform::FusedRowsQ:
      source = fused_rows(source, 0U, hidden);
      break;
    case Transform::FusedRowsK:
      source = fused_rows(source, 1U, hidden);
      break;
    case Transform::FusedRowsV:
      source = fused_rows(source, 2U, hidden);
      break;
    case Transform::Transpose:
      source = transposed(source);
      break;
    }
    auto tensor = as_dtype(std::move(source), config.dtype);
    check_binding(build.program, weight.tensor, tensor, weight.source_name);
    stage.bindings.emplace(weight.tensor, std::move(tensor));
  }
  const std::string tag = g ? "g_" : "l_";
  stage.bindings.emplace(
      build.token_ids_input,
      fixture_input(fixture, tag + "token_ids", dif::ir::DType::I32));
  if (config.pooled_output)
    stage.bindings.emplace(
        build.pooled_row_input,
        fixture_input(fixture, tag + "pooled_row", dif::ir::DType::I32));
  for (const auto &[name, id] : build.boundaries)
    stage.outputs.emplace_back(tag + name, id);
  stage.program = std::move(build.program);
  return stage;
}

Stage build_unet(const Arguments &arguments,
                 const dif::weights::SafeTensorFile &checkpoint,
                 const dif::weights::SafeTensorFile &fixture) {
  Stage stage;
  dif::frontend::SdxlUnetConfig config;
  config.capture_boundaries = arguments.capture;
  config.dtype = parse_dtype(arguments.dtype, dif::ir::DType::F16);
  const auto *latent = fixture.find("latent_input");
  const auto *context = fixture.find("context");
  if (!latent || latent->dims.size() != 4U || !context ||
      context->dims.size() != 3U)
    dif::fail("fixture needs latent_input [B,4,h,w] and context [B,T,2048]");
  config.batch = latent->dims[0];
  config.latent_height = latent->dims[2];
  config.latent_width = latent->dims[3];
  config.context_tokens = context->dims[1];
  auto build = dif::frontend::make_sdxl_unet(config);
  for (const auto &weight : build.weights) {
    auto tensor = dif::weights::map_safetensor(checkpoint, weight.source_name);
    if (tensor.dtype != config.dtype)
      tensor = dif::runtime::convert_float_tensor(tensor, config.dtype);
    check_binding(build.program, weight.tensor, tensor, weight.source_name);
    stage.bindings.emplace(weight.tensor, std::move(tensor));
  }
  stage.bindings.emplace(build.latent_input,
                         fixture_input(fixture, "latent_input", config.dtype));
  stage.bindings.emplace(build.timestep_input,
                         fixture_input(fixture, "timesteps", dif::ir::DType::F32));
  stage.bindings.emplace(build.context_input,
                         fixture_input(fixture, "context", config.dtype));
  stage.bindings.emplace(build.vector_input,
                         fixture_input(fixture, "vector", config.dtype));
  stage.outputs = build.boundaries;
  stage.program = std::move(build.program);
  return stage;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    const auto checkpoint = dif::weights::read_safetensors(arguments.checkpoint);
    const auto fixture = dif::weights::read_safetensors(arguments.fixture);
    auto started = Clock::now();
    Stage stage;
    if (arguments.stage == "vae")
      stage = build_vae(arguments, checkpoint, fixture);
    else if (arguments.stage == "clip-l")
      stage = build_clip(arguments, checkpoint, fixture, false);
    else if (arguments.stage == "clip-g")
      stage = build_clip(arguments, checkpoint, fixture, true);
    else if (arguments.stage == "unet")
      stage = build_unet(arguments, checkpoint, fixture);
    else
      dif::fail("unknown stage " + arguments.stage);
    const auto build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();

    dif::runtime::RunOptions options;
    options.warmups = arguments.warmups;
    options.iterations = arguments.iterations;
    options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
    options.cache_directory = arguments.cache_directory;
    options.profile_pipeline = arguments.profile != 0U;
    if (arguments.tune_linears)
      for (const auto &operation : stage.program.operations)
        if (operation.opcode == dif::ir::Opcode::Linear)
          options.tune_linear_operations.push_back(operation.id);
    auto backend = arguments.backend == "cpu"
                       ? dif::runtime::make_cpu_executor()
                       : dif::runtime::make_cuda_executor();
    started = Clock::now();
    auto prepared = backend->prepare(stage.program, stage.bindings, options);
    const auto prepare_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    started = Clock::now();
    auto result = prepared->run(stage.bindings, options);
    const auto run_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();

    if (arguments.profile != 0U) {
      struct Bucket {
        double milliseconds{};
        std::uint64_t count{};
        std::string label;
      };
      std::map<std::string, Bucket> buckets;
      double total = 0.0;
      for (const auto &timing : result.operation_timings) {
        const auto *op = std::find_if(
            stage.program.operations.begin(), stage.program.operations.end(),
            [&](const dif::ir::Operation &candidate) {
              return candidate.id == timing.operation_id;
            }).base();
        std::string shape;
        for (const auto &operation : stage.program.operations) {
          if (operation.id != timing.operation_id)
            continue;
          for (const auto output : operation.outputs) {
            const auto *description = stage.program.tensor(output);
            shape += "[";
            for (std::size_t index = 0; index < description->dims.size();
                 ++index)
              shape += (index ? "," : "") +
                       std::to_string(description->dims[index]);
            shape += "]";
          }
          break;
        }
        (void)op;
        const auto key =
            std::string(dif::ir::opcode_name(timing.opcode)) + " " + shape +
            (timing.plan.empty() ? "" : " plan=" + timing.plan);
        auto &bucket = buckets[key];
        bucket.milliseconds += timing.minimum_milliseconds;
        bucket.count += 1U;
        bucket.label = key;
        total += timing.minimum_milliseconds;
      }
      std::vector<Bucket> ordered;
      ordered.reserve(buckets.size());
      for (auto &[key, bucket] : buckets)
        ordered.push_back(bucket);
      std::sort(ordered.begin(), ordered.end(),
                [](const Bucket &a, const Bucket &b) {
                  return a.milliseconds > b.milliseconds;
                });
      std::cout << "SDXL_PROFILE stage=" << arguments.stage
                << " accounted_ms=" << total << "\n";
      for (std::size_t index = 0;
           index < ordered.size() && index < arguments.profile; ++index)
        std::cout << "  " << ordered[index].milliseconds << " ms  x"
                  << ordered[index].count << "  " << ordered[index].label
                  << "\n";
    }

    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    std::vector<dif::runtime::Tensor> values;
    for (const auto &[name, id] : stage.outputs) {
      auto found = result.outputs.find(id);
      if (found == result.outputs.end())
        dif::fail("run produced no output for boundary " + name);
      auto value = as_dtype(found->second, dif::ir::DType::F32);
      specs.push_back({name, value.dtype, value.dims});
      values.push_back(std::move(value));
    }
    dif::weights::SafeTensorWriter writer(arguments.output, specs);
    for (std::size_t index = 0U; index < values.size(); ++index)
      writer.append(specs[index].name,
                    std::span<const std::uint8_t>(values[index].data(),
                                                  values[index].byte_size()));
    (void)writer.finish();
    std::cout << "SDXL_STAGE stage=" << arguments.stage
              << " backend=" << result.backend_name
              << " device=" << result.device_name
              << " operations=" << stage.program.operations.size()
              << " tensors=" << stage.program.tensors.size()
              << " weights=" << stage.bindings.size()
              << " build_ms=" << build_ms << " prepare_ms=" << prepare_ms
              << " run_ms=" << run_ms
              << " gpu_mean_ms=" << result.mean_milliseconds
              << " gpu_min_ms=" << result.minimum_milliseconds
              << " resident_bytes_h2d=" << result.pipeline_profile.resident_weight_bytes
              << " resident_h2d_ms=" << result.pipeline_profile.resident_h2d_milliseconds
              << " resident_upload_ms="
              << result.pipeline_profile.resident_upload_milliseconds
              << " major_faults=" << result.pipeline_profile.resident_major_page_faults
              << " resident_bytes=" << result.resident_bytes
              << " boundaries=" << values.size() << " -> " << arguments.output
              << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difsdxlstage: " << error.what() << "\n";
    return 1;
  }
}
