#include "dif/backend/plugin.hpp"
#include "dif/frontend/lora.hpp"
#include "dif/frontend/training.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/training/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <unordered_set>
#include <vector>

namespace {

struct RunArguments {
  std::string backend{"cpu"};
  std::filesystem::path backend_plugin;
  std::filesystem::path program;
  std::filesystem::path features;
  std::filesystem::path targets;
  std::array<std::filesystem::path, 4> parameters;
  std::filesystem::path resume;
  std::filesystem::path checkpoint;
  std::filesystem::path losses;
  std::filesystem::path prediction;
  std::filesystem::path gradients_directory;
  std::filesystem::path cache_directory;
  std::uint64_t steps{};
  std::uint64_t minimum_free_bytes{256ULL * 1024ULL * 1024ULL};
};

struct FlowRunArguments {
  std::string backend{"cpu"};
  std::filesystem::path backend_plugin;
  std::filesystem::path program;
  std::filesystem::path fixture;
  std::filesystem::path resume;
  std::filesystem::path checkpoint;
  std::filesystem::path losses;
  std::filesystem::path predictions_directory;
  std::filesystem::path gradients_directory;
  std::filesystem::path cache_directory;
  std::uint64_t steps{};
  std::uint64_t minimum_free_bytes{256ULL * 1024ULL * 1024ULL};
};

struct LoraRunArguments {
  std::string backend{"cpu"};
  std::filesystem::path backend_plugin;
  std::filesystem::path program;
  std::filesystem::path fixture;
  std::filesystem::path resume;
  std::filesystem::path checkpoint;
  std::filesystem::path losses;
  std::filesystem::path prediction;
  std::filesystem::path gradients_directory;
  std::filesystem::path cache_directory;
  std::uint64_t steps{};
  std::uint64_t minimum_free_bytes{256ULL * 1024ULL * 1024ULL};
  bool has_init_seed{};
  std::uint64_t init_seed{};
};

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (text.empty() || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

double real_number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text.c_str(), &end);
  if (text.empty() || !end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

void usage() {
  std::cerr
      << "usage:\n"
      << "  diftrain run-mlp --backend cpu|cuda --program FILE.difir --features FILE.diftensor --targets FILE.diftensor (--w1 FILE --b1 FILE --w2 FILE --b2 FILE | --resume FILE.diftrain) --steps N --checkpoint OUT.diftrain --losses OUT.diftensor [--prediction OUT.diftensor] [--gradients-dir DIR] [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]\n"
      << "  diftrain run-flow --backend cpu|cuda --program FILE.difir --fixture DIR [--resume FILE.diftrain] --steps N --checkpoint OUT.diftrain --losses OUT.diftensor [--predictions-dir DIR] [--gradients-dir DIR] [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]\n"
      << "  diftrain make-lora OUT.difir ROWS LATENT_WIDTH TIMESTEP_WIDTH HIDDEN_WIDTH RANK ALPHA [LR BETA1 BETA2 EPS WEIGHT_DECAY]\n"
      << "  diftrain run-lora --backend cpu|cuda --program FILE.difir --fixture DIR [--resume FILE.diftrain | --init-seed N] --steps N --checkpoint OUT.diftrain --losses OUT.diftensor [--prediction OUT.diftensor] [--gradients-dir DIR] [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]\n"
      << "  diftrain export-lora --program FILE.difir --checkpoint FILE.diftrain --output OUT.safetensors\n"
      << "  diftrain inspect FILE.diftrain\n"
      << "  diftrain export FILE.diftrain OUT_DIR\n";
}

void require_new(const std::filesystem::path &path, const char *label) {
  if (path.empty())
    dif::fail(std::string("missing ") + label + " path");
  if (std::filesystem::exists(path))
    dif::fail(std::string(label) + " already exists: " + path.string());
}

dif::runtime::Tensor i32_scalar(std::int32_t value) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, {1U},
                              std::vector<std::uint8_t>(sizeof(value))};
  std::memcpy(tensor.mutable_data(), &value, sizeof(value));
  return tensor;
}

dif::runtime::Tensor f32_vector(const std::vector<float> &values) {
  dif::runtime::Tensor tensor{
      dif::ir::DType::F32, {values.size()},
      std::vector<std::uint8_t>(values.size() * sizeof(float))};
  if (!values.empty())
    std::memcpy(tensor.mutable_data(), values.data(), tensor.byte_size());
  return tensor;
}

void validate_binding(const dif::ir::Program &program, std::uint32_t id,
                      const dif::runtime::Tensor &tensor, const char *label) {
  tensor.validate();
  const auto *description = program.tensor(id);
  if (!description || tensor.dtype != description->dtype ||
      tensor.dims != description->dims)
    dif::fail(std::string(label) + " shape/dtype mismatch for tensor " +
              std::to_string(id));
}

dif::frontend::MlpTrainingBuild
recover_mlp_build(const dif::ir::Program &program) {
  const auto *features = program.tensor(1U);
  const auto *targets = program.tensor(2U);
  const auto *weight1 = program.tensor(3U);
  const auto *weight2 = program.tensor(5U);
  if (!features || !targets || !weight1 || !weight2 ||
      features->dims.size() != 2U || targets->dims.size() != 2U ||
      weight1->dims.size() != 2U || weight2->dims.size() != 2U)
    dif::fail("program is not a supported make-mlp-training graph");
  const dif::ir::Operation *optimizer = nullptr;
  for (const auto &operation : program.operations) {
    if (operation.opcode == dif::ir::Opcode::AdamWUpdate) {
      optimizer = &operation;
      break;
    }
  }
  if (!optimizer)
    dif::fail("MLP training program has no AdamW update");
  dif::frontend::MlpTrainingConfig config;
  config.rows = features->dims[0];
  config.input_width = features->dims[1];
  config.hidden_width = weight1->dims[0];
  config.output_width = weight2->dims[0];
  // The features dtype carries the builder's compute dtype (F32 or BF16
  // mixed precision); the fingerprint comparison below stays the authority.
  config.compute_dtype = features->dtype;
  config.learning_rate =
      optimizer->f64(dif::ir::AttrKey::LearningRate, 1.0e-3);
  config.beta1 = optimizer->f64(dif::ir::AttrKey::Beta1, 0.9);
  config.beta2 = optimizer->f64(dif::ir::AttrKey::Beta2, 0.999);
  config.epsilon = optimizer->f64(dif::ir::AttrKey::Epsilon, 1.0e-8);
  config.weight_decay =
      optimizer->f64(dif::ir::AttrKey::WeightDecay, 0.0);
  auto build = dif::frontend::make_mlp_training(config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("program is not the canonical MLP training graph described by "
              "its dimensions and optimizer attributes");
  return build;
}

dif::frontend::RectifiedFlowTrainingBuild
recover_rectified_flow_build(const dif::ir::Program &program) {
  const auto *clean = program.tensor(1U);
  if (!clean || clean->dtype != dif::ir::DType::F32 ||
      clean->dims.size() != 2U)
    dif::fail("program is not a supported rectified-flow training graph");
  std::vector<std::uint32_t> parameter_ids;
  for (const auto &tensor : program.tensors) {
    if (tensor.has_role(dif::ir::TensorRole::Parameter) &&
        tensor.has_role(dif::ir::TensorRole::Input))
      parameter_ids.push_back(tensor.id);
  }
  std::sort(parameter_ids.begin(), parameter_ids.end());
  if (parameter_ids.size() != 5U || parameter_ids.front() < 8U)
    dif::fail("rectified-flow training graph must have five parameters");
  const auto loss_scale_id = parameter_ids.front() - 1U;
  if ((loss_scale_id - 1U) % 6U != 0U)
    dif::fail("rectified-flow training input layout is not canonical");
  const auto accumulation_steps = (loss_scale_id - 1U) / 6U;
  const auto *time_features = program.tensor(5U);
  const auto *latent_weight = program.tensor(parameter_ids[0]);
  if (!time_features || time_features->dims.size() != 2U ||
      !latent_weight || latent_weight->dims.size() != 2U)
    dif::fail("rectified-flow training dimensions are malformed");
  const dif::ir::Operation *optimizer = nullptr;
  for (const auto &operation : program.operations) {
    if (operation.opcode == dif::ir::Opcode::AdamWUpdate) {
      optimizer = &operation;
      break;
    }
  }
  if (!optimizer)
    dif::fail("rectified-flow training program has no AdamW update");
  dif::frontend::RectifiedFlowTrainingConfig config;
  config.rows = clean->dims[0];
  config.latent_width = clean->dims[1];
  config.timestep_width = time_features->dims[1];
  config.hidden_width = latent_weight->dims[0];
  config.accumulation_steps = accumulation_steps;
  config.learning_rate =
      optimizer->f64(dif::ir::AttrKey::LearningRate, 1.0e-3);
  config.beta1 = optimizer->f64(dif::ir::AttrKey::Beta1, 0.9);
  config.beta2 = optimizer->f64(dif::ir::AttrKey::Beta2, 0.999);
  config.epsilon = optimizer->f64(dif::ir::AttrKey::Epsilon, 1.0e-8);
  config.weight_decay =
      optimizer->f64(dif::ir::AttrKey::WeightDecay, 0.0);
  auto build = dif::frontend::make_rectified_flow_training(config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("program is not the canonical rectified-flow training graph "
              "described by its dimensions and optimizer attributes");
  return build;
}

void atomic_write_checkpoint(const dif::training::Checkpoint &checkpoint,
                             const std::filesystem::path &path) {
  auto temporary = path;
  temporary += ".partial";
  require_new(path, "checkpoint output");
  require_new(temporary, "checkpoint temporary output");
  try {
    dif::training::write_checkpoint(checkpoint, temporary);
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

RunArguments parse_run(int argc, char **argv) {
  RunArguments arguments;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--backend")
      arguments.backend = value("--backend");
    else if (option == "--backend-plugin")
      arguments.backend_plugin = value("--backend-plugin");
    else if (option == "--program")
      arguments.program = value("--program");
    else if (option == "--features")
      arguments.features = value("--features");
    else if (option == "--targets")
      arguments.targets = value("--targets");
    else if (option == "--w1")
      arguments.parameters[0] = value("--w1");
    else if (option == "--b1")
      arguments.parameters[1] = value("--b1");
    else if (option == "--w2")
      arguments.parameters[2] = value("--w2");
    else if (option == "--b2")
      arguments.parameters[3] = value("--b2");
    else if (option == "--resume")
      arguments.resume = value("--resume");
    else if (option == "--steps")
      arguments.steps = number(value("--steps"), "steps");
    else if (option == "--checkpoint")
      arguments.checkpoint = value("--checkpoint");
    else if (option == "--losses")
      arguments.losses = value("--losses");
    else if (option == "--prediction")
      arguments.prediction = value("--prediction");
    else if (option == "--gradients-dir")
      arguments.gradients_directory = value("--gradients-dir");
    else if (option == "--cache-dir")
      arguments.cache_directory = value("--cache-dir");
    else if (option == "--min-free-mib")
      arguments.minimum_free_bytes =
          number(value("--min-free-mib"), "minimum free memory") * 1024ULL *
          1024ULL;
    else
      dif::fail("unknown run-mlp option: " + option);
  }
  if (arguments.program.empty() || arguments.features.empty() ||
      arguments.targets.empty() || arguments.checkpoint.empty() ||
      arguments.losses.empty() || arguments.steps == 0U)
    dif::fail("run-mlp requires program, features, targets, positive steps, "
              "checkpoint, and losses");
  const auto parameter_count =
      std::count_if(arguments.parameters.begin(), arguments.parameters.end(),
                    [](const auto &path) { return !path.empty(); });
  if (arguments.resume.empty() && parameter_count != 4)
    dif::fail("fresh training requires --w1, --b1, --w2, and --b2");
  if (!arguments.resume.empty() && parameter_count != 0)
    dif::fail("resume cannot also specify initial parameters");
  return arguments;
}

FlowRunArguments parse_flow_run(int argc, char **argv) {
  FlowRunArguments arguments;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--backend")
      arguments.backend = value("--backend");
    else if (option == "--backend-plugin")
      arguments.backend_plugin = value("--backend-plugin");
    else if (option == "--program")
      arguments.program = value("--program");
    else if (option == "--fixture")
      arguments.fixture = value("--fixture");
    else if (option == "--resume")
      arguments.resume = value("--resume");
    else if (option == "--steps")
      arguments.steps = number(value("--steps"), "steps");
    else if (option == "--checkpoint")
      arguments.checkpoint = value("--checkpoint");
    else if (option == "--losses")
      arguments.losses = value("--losses");
    else if (option == "--predictions-dir")
      arguments.predictions_directory = value("--predictions-dir");
    else if (option == "--gradients-dir")
      arguments.gradients_directory = value("--gradients-dir");
    else if (option == "--cache-dir")
      arguments.cache_directory = value("--cache-dir");
    else if (option == "--min-free-mib")
      arguments.minimum_free_bytes =
          number(value("--min-free-mib"), "minimum free memory") * 1024ULL *
          1024ULL;
    else
      dif::fail("unknown run-flow option: " + option);
  }
  if (arguments.program.empty() || arguments.fixture.empty() ||
      arguments.checkpoint.empty() || arguments.losses.empty() ||
      arguments.steps == 0U)
    dif::fail("run-flow requires program, fixture, positive steps, checkpoint, "
              "and losses");
  if (!std::filesystem::is_directory(arguments.fixture))
    dif::fail("run-flow fixture is not a directory: " +
              arguments.fixture.string());
  return arguments;
}

int run_mlp(int argc, char **argv) {
  const auto arguments = parse_run(argc, argv);
  require_new(arguments.checkpoint, "checkpoint output");
  require_new(arguments.losses, "loss output");
  if (!arguments.prediction.empty())
    require_new(arguments.prediction, "prediction output");
  if (!arguments.gradients_directory.empty() &&
      std::filesystem::exists(arguments.gradients_directory))
    dif::fail("gradients directory already exists: " +
              arguments.gradients_directory.string());

  const auto program = dif::ir::read_file(arguments.program);
  const auto build = recover_mlp_build(program);
  const auto fingerprint = dif::ir::fingerprint(program);
  dif::runtime::TensorMap inputs;
  inputs.emplace(build.features_input,
                 dif::runtime::read_tensor(arguments.features));
  inputs.emplace(build.target_input, dif::runtime::read_tensor(arguments.targets));
  validate_binding(program, build.features_input, inputs.at(build.features_input),
                   "features");
  validate_binding(program, build.target_input, inputs.at(build.target_input),
                   "targets");

  std::uint64_t completed_steps = 0U;
  if (!arguments.resume.empty()) {
    auto checkpoint = dif::training::read_checkpoint(arguments.resume);
    if (checkpoint.program_fingerprint != fingerprint)
      dif::fail("resume checkpoint targets a different program fingerprint");
    completed_steps = checkpoint.completed_steps;
    std::unordered_set<std::uint32_t> expected;
    for (const auto &binding : build.optimizer_bindings) {
      for (const auto id : {binding.parameter_input, binding.first_moment_input,
                            binding.second_moment_input}) {
        expected.insert(id);
        const auto found = checkpoint.state.find(id);
        if (found == checkpoint.state.end())
          dif::fail("resume checkpoint is missing state tensor " +
                    std::to_string(id));
        validate_binding(program, id, found->second, "checkpoint");
        inputs.emplace(id, std::move(found->second));
      }
    }
    if (checkpoint.state.size() != expected.size())
      dif::fail("resume checkpoint contains unexpected state tensors");
  } else {
    for (std::size_t index = 0U; index < build.optimizer_bindings.size();
         ++index) {
      const auto &binding = build.optimizer_bindings[index];
      auto parameter = dif::runtime::read_tensor(arguments.parameters[index]);
      validate_binding(program, binding.parameter_input, parameter,
                       "initial parameter");
      inputs.emplace(binding.parameter_input, std::move(parameter));
      inputs.emplace(binding.first_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.first_moment_input)));
      inputs.emplace(binding.second_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.second_moment_input)));
    }
  }
  if (completed_steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      arguments.steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) -
              completed_steps)
    dif::fail("training step is outside the I32 optimizer-state range");

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = arguments.minimum_free_bytes;
  options.cache_directory = arguments.cache_directory;
  std::unique_ptr<dif::runtime::Executor> executor;
  if (!arguments.backend_plugin.empty())
    executor = dif::backend::make_plugin_executor(arguments.backend_plugin);
  else if (arguments.backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (arguments.backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("unknown backend: " + arguments.backend);

  inputs.emplace(build.step_input,
                 i32_scalar(static_cast<std::int32_t>(completed_steps)));
  const auto wall_start = std::chrono::steady_clock::now();
  auto prepared = executor->prepare(program, inputs, options);
  std::vector<float> losses;
  losses.reserve(static_cast<std::size_t>(arguments.steps));
  std::vector<double> step_milliseconds;
  step_milliseconds.reserve(static_cast<std::size_t>(arguments.steps));
  dif::runtime::RunResult final_result;
  for (std::uint64_t local_step = 0U; local_step < arguments.steps;
       ++local_step) {
    inputs.insert_or_assign(
        build.step_input,
        i32_scalar(static_cast<std::int32_t>(completed_steps + local_step)));
    auto result = prepared->run(inputs, options);
    const auto loss = result.outputs.at(build.loss_output).f32()[0];
    if (!std::isfinite(loss))
      dif::fail("training produced a non-finite loss at local step " +
                std::to_string(local_step));
    losses.push_back(loss);
    step_milliseconds.push_back(result.mean_milliseconds);
    for (const auto &binding : build.optimizer_bindings) {
      inputs.insert_or_assign(
          binding.parameter_input,
          std::move(result.outputs.at(binding.parameter_output)));
      inputs.insert_or_assign(
          binding.first_moment_input,
          std::move(result.outputs.at(binding.first_moment_output)));
      inputs.insert_or_assign(
          binding.second_moment_input,
          std::move(result.outputs.at(binding.second_moment_output)));
    }
    final_result = std::move(result);
  }
  completed_steps += arguments.steps;
  const auto wall_stop = std::chrono::steady_clock::now();

  dif::training::Checkpoint checkpoint;
  checkpoint.program_fingerprint = fingerprint;
  checkpoint.completed_steps = completed_steps;
  for (const auto &binding : build.optimizer_bindings) {
    checkpoint.state.emplace(binding.parameter_input,
                             inputs.at(binding.parameter_input));
    checkpoint.state.emplace(binding.first_moment_input,
                             inputs.at(binding.first_moment_input));
    checkpoint.state.emplace(binding.second_moment_input,
                             inputs.at(binding.second_moment_input));
  }
  atomic_write_checkpoint(checkpoint, arguments.checkpoint);
  dif::runtime::write_tensor(f32_vector(losses), arguments.losses);
  if (!arguments.prediction.empty())
    dif::runtime::write_tensor(final_result.outputs.at(build.prediction_output),
                               arguments.prediction);
  if (!arguments.gradients_directory.empty()) {
    std::filesystem::create_directories(arguments.gradients_directory);
    for (const auto &binding : build.optimizer_bindings)
      dif::runtime::write_tensor(
          final_result.outputs.at(binding.gradient_output),
          arguments.gradients_directory /
              ("gradient-" + std::to_string(binding.parameter_input) +
               ".diftensor"));
  }

  const auto step_mean = std::accumulate(step_milliseconds.begin(),
                                         step_milliseconds.end(), 0.0) /
                         static_cast<double>(step_milliseconds.size());
  struct rusage resource_usage {};
  if (getrusage(RUSAGE_SELF, &resource_usage) != 0)
    dif::fail("getrusage failed");
  std::cout << "TRAIN PASS backend=" << final_result.backend_name
            << " device=\"" << final_result.device_name << "\""
            << " steps=" << arguments.steps
            << " completed_steps=" << completed_steps
            << " initial_loss=" << losses.front()
            << " final_loss=" << losses.back()
            << " prepare_ms=" << prepared->preparation_milliseconds()
            << " step_mean_ms=" << step_mean
            << " wall_ms="
            << std::chrono::duration<double, std::milli>(wall_stop - wall_start)
                   .count()
            << " resident_bytes=" << prepared->resident_bytes()
            << " free_before=" << final_result.free_bytes_before
            << " free_after=" << final_result.free_bytes_after
            << " host_max_rss_kib=" << resource_usage.ru_maxrss
            << " fingerprint=" << dif::hex_digest(fingerprint) << "\n";
  return 0;
}

int run_flow(int argc, char **argv) {
  const auto arguments = parse_flow_run(argc, argv);
  require_new(arguments.checkpoint, "checkpoint output");
  require_new(arguments.losses, "loss output");
  if (!arguments.predictions_directory.empty() &&
      std::filesystem::exists(arguments.predictions_directory))
    dif::fail("predictions directory already exists: " +
              arguments.predictions_directory.string());
  if (!arguments.gradients_directory.empty() &&
      std::filesystem::exists(arguments.gradients_directory))
    dif::fail("gradients directory already exists: " +
              arguments.gradients_directory.string());

  const auto program = dif::ir::read_file(arguments.program);
  const auto build = recover_rectified_flow_build(program);
  const auto fingerprint = dif::ir::fingerprint(program);
  dif::runtime::TensorMap inputs;
  const std::array<const char *, 6> names = {
      "clean", "noise", "clean-scale", "noise-scale", "time-features",
      "target-velocity"};
  for (std::size_t index = 0U; index < build.microbatches.size(); ++index) {
    const auto &binding = build.microbatches[index];
    const std::array<std::uint32_t, 6> ids = {
        binding.clean_input, binding.noise_input, binding.clean_scale_input,
        binding.noise_scale_input, binding.timestep_features_input,
        binding.target_velocity_input};
    for (std::size_t field = 0U; field < ids.size(); ++field) {
      const auto path =
          arguments.fixture /
          ("microbatch-" + std::to_string(index) + "-" + names[field] +
           ".diftensor");
      auto tensor = dif::runtime::read_tensor(path);
      validate_binding(program, ids[field], tensor, names[field]);
      inputs.emplace(ids[field], std::move(tensor));
    }
  }
  std::uint64_t completed_steps = 0U;
  if (!arguments.resume.empty()) {
    auto checkpoint = dif::training::read_checkpoint(arguments.resume);
    if (checkpoint.program_fingerprint != fingerprint)
      dif::fail("resume checkpoint targets a different program fingerprint");
    completed_steps = checkpoint.completed_steps;
    std::unordered_set<std::uint32_t> expected;
    for (const auto &binding : build.optimizer_bindings) {
      for (const auto id : {binding.parameter_input, binding.first_moment_input,
                            binding.second_moment_input}) {
        expected.insert(id);
        const auto found = checkpoint.state.find(id);
        if (found == checkpoint.state.end())
          dif::fail("resume checkpoint is missing state tensor " +
                    std::to_string(id));
        validate_binding(program, id, found->second, "checkpoint");
        inputs.emplace(id, std::move(found->second));
      }
    }
    if (checkpoint.state.size() != expected.size())
      dif::fail("resume checkpoint contains unexpected state tensors");
  } else {
    for (const auto &binding : build.optimizer_bindings) {
      const auto path =
          arguments.fixture /
          ("initial-parameter-" + std::to_string(binding.parameter_input) +
           ".diftensor");
      auto parameter = dif::runtime::read_tensor(path);
      validate_binding(program, binding.parameter_input, parameter,
                       "initial parameter");
      inputs.emplace(binding.parameter_input, std::move(parameter));
      inputs.emplace(binding.first_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.first_moment_input)));
      inputs.emplace(binding.second_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.second_moment_input)));
    }
  }
  if (completed_steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      arguments.steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) -
              completed_steps)
    dif::fail("training step is outside the I32 optimizer-state range");

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = arguments.minimum_free_bytes;
  options.cache_directory = arguments.cache_directory;
  std::unique_ptr<dif::runtime::Executor> executor;
  if (!arguments.backend_plugin.empty())
    executor = dif::backend::make_plugin_executor(arguments.backend_plugin);
  else if (arguments.backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (arguments.backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("unknown backend: " + arguments.backend);

  inputs.emplace(build.step_input,
                 i32_scalar(static_cast<std::int32_t>(completed_steps)));
  const auto wall_start = std::chrono::steady_clock::now();
  auto prepared = executor->prepare(program, inputs, options);
  std::vector<float> losses;
  losses.reserve(static_cast<std::size_t>(arguments.steps));
  std::vector<double> step_milliseconds;
  step_milliseconds.reserve(static_cast<std::size_t>(arguments.steps));
  dif::runtime::RunResult final_result;
  for (std::uint64_t local_step = 0U; local_step < arguments.steps;
       ++local_step) {
    inputs.insert_or_assign(
        build.step_input,
        i32_scalar(static_cast<std::int32_t>(completed_steps + local_step)));
    auto result = prepared->run(inputs, options);
    const auto loss = result.outputs.at(build.loss_output).f32()[0];
    if (!std::isfinite(loss))
      dif::fail("rectified-flow training produced a non-finite loss at step " +
                std::to_string(local_step));
    losses.push_back(loss);
    step_milliseconds.push_back(result.mean_milliseconds);
    for (const auto &binding : build.optimizer_bindings) {
      inputs.insert_or_assign(
          binding.parameter_input,
          std::move(result.outputs.at(binding.parameter_output)));
      inputs.insert_or_assign(
          binding.first_moment_input,
          std::move(result.outputs.at(binding.first_moment_output)));
      inputs.insert_or_assign(
          binding.second_moment_input,
          std::move(result.outputs.at(binding.second_moment_output)));
    }
    final_result = std::move(result);
  }
  completed_steps += arguments.steps;
  const auto wall_stop = std::chrono::steady_clock::now();

  dif::training::Checkpoint checkpoint;
  checkpoint.program_fingerprint = fingerprint;
  checkpoint.completed_steps = completed_steps;
  for (const auto &binding : build.optimizer_bindings) {
    checkpoint.state.emplace(binding.parameter_input,
                             inputs.at(binding.parameter_input));
    checkpoint.state.emplace(binding.first_moment_input,
                             inputs.at(binding.first_moment_input));
    checkpoint.state.emplace(binding.second_moment_input,
                             inputs.at(binding.second_moment_input));
  }
  atomic_write_checkpoint(checkpoint, arguments.checkpoint);
  dif::runtime::write_tensor(f32_vector(losses), arguments.losses);
  if (!arguments.predictions_directory.empty()) {
    std::filesystem::create_directories(arguments.predictions_directory);
    for (std::size_t index = 0U; index < build.microbatches.size(); ++index)
      dif::runtime::write_tensor(
          final_result.outputs.at(
              build.microbatches[index].prediction_output),
          arguments.predictions_directory /
              ("prediction-" + std::to_string(index) + ".diftensor"));
  }
  if (!arguments.gradients_directory.empty()) {
    std::filesystem::create_directories(arguments.gradients_directory);
    for (const auto &binding : build.optimizer_bindings)
      dif::runtime::write_tensor(
          final_result.outputs.at(binding.gradient_output),
          arguments.gradients_directory /
              ("gradient-" + std::to_string(binding.parameter_input) +
               ".diftensor"));
  }

  const auto step_mean = std::accumulate(step_milliseconds.begin(),
                                         step_milliseconds.end(), 0.0) /
                         static_cast<double>(step_milliseconds.size());
  struct rusage resource_usage {};
  if (getrusage(RUSAGE_SELF, &resource_usage) != 0)
    dif::fail("getrusage failed");
  std::cout << "FLOW_TRAIN PASS backend=" << final_result.backend_name
            << " device=\"" << final_result.device_name << "\""
            << " steps=" << arguments.steps
            << " completed_steps=" << completed_steps
            << " microbatches=" << build.microbatches.size()
            << " initial_loss=" << losses.front()
            << " final_loss=" << losses.back()
            << " prepare_ms=" << prepared->preparation_milliseconds()
            << " step_mean_ms=" << step_mean
            << " wall_ms="
            << std::chrono::duration<double, std::milli>(wall_stop - wall_start)
                   .count()
            << " resident_bytes=" << prepared->resident_bytes()
            << " free_before=" << final_result.free_bytes_before
            << " free_after=" << final_result.free_bytes_after
            << " host_max_rss_kib=" << resource_usage.ru_maxrss
            << " fingerprint=" << dif::hex_digest(fingerprint) << "\n";
  return 0;
}

dif::frontend::LoraFlowTrainingBuild
recover_lora_build(const dif::ir::Program &program) {
  const auto *clean = program.tensor(1U);
  const auto *time_features = program.tensor(5U);
  const auto *latent_weight = program.tensor(7U);
  const auto *first_adapter = program.tensor(12U);
  if (!clean || clean->dtype != dif::ir::DType::F32 ||
      clean->dims.size() != 2U || !time_features ||
      time_features->dims.size() != 2U || !latent_weight ||
      latent_weight->dims.size() != 2U || !first_adapter ||
      first_adapter->dims.size() != 2U)
    dif::fail("program is not a supported LoRA training graph");
  const auto fill = std::find_if(
      program.operations.begin(), program.operations.end(),
      [](const auto &operation) {
        return operation.opcode == dif::ir::Opcode::Fill;
      });
  if (fill == program.operations.end())
    dif::fail("LoRA training graph has no alpha/rank scale Fill");
  const dif::ir::Operation *optimizer = nullptr;
  for (const auto &operation : program.operations) {
    if (operation.opcode == dif::ir::Opcode::AdamWUpdate) {
      optimizer = &operation;
      break;
    }
  }
  if (!optimizer)
    dif::fail("LoRA training program has no AdamW update");
  dif::frontend::LoraFlowTrainingConfig config;
  config.rows = clean->dims[0];
  config.latent_width = clean->dims[1];
  config.timestep_width = time_features->dims[1];
  config.hidden_width = latent_weight->dims[0];
  config.rank = first_adapter->dims[0];
  config.alpha = fill->f64(dif::ir::AttrKey::Value, 0.0) *
                 static_cast<double>(config.rank);
  config.learning_rate =
      optimizer->f64(dif::ir::AttrKey::LearningRate, 1.0e-3);
  config.beta1 = optimizer->f64(dif::ir::AttrKey::Beta1, 0.9);
  config.beta2 = optimizer->f64(dif::ir::AttrKey::Beta2, 0.999);
  config.epsilon = optimizer->f64(dif::ir::AttrKey::Epsilon, 1.0e-8);
  config.weight_decay = optimizer->f64(dif::ir::AttrKey::WeightDecay, 0.0);
  auto build = dif::frontend::make_lora_flow_training(config);
  if (dif::ir::fingerprint(build.program) != dif::ir::fingerprint(program))
    dif::fail("program is not the canonical LoRA training graph described "
              "by its dimensions, scale, and optimizer attributes");
  return build;
}

LoraRunArguments parse_lora_run(int argc, char **argv) {
  LoraRunArguments arguments;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--backend")
      arguments.backend = value("--backend");
    else if (option == "--backend-plugin")
      arguments.backend_plugin = value("--backend-plugin");
    else if (option == "--program")
      arguments.program = value("--program");
    else if (option == "--fixture")
      arguments.fixture = value("--fixture");
    else if (option == "--resume")
      arguments.resume = value("--resume");
    else if (option == "--init-seed") {
      arguments.init_seed = number(value("--init-seed"), "init seed");
      arguments.has_init_seed = true;
    } else if (option == "--steps")
      arguments.steps = number(value("--steps"), "steps");
    else if (option == "--checkpoint")
      arguments.checkpoint = value("--checkpoint");
    else if (option == "--losses")
      arguments.losses = value("--losses");
    else if (option == "--prediction")
      arguments.prediction = value("--prediction");
    else if (option == "--gradients-dir")
      arguments.gradients_directory = value("--gradients-dir");
    else if (option == "--cache-dir")
      arguments.cache_directory = value("--cache-dir");
    else if (option == "--min-free-mib")
      arguments.minimum_free_bytes =
          number(value("--min-free-mib"), "minimum free memory") * 1024ULL *
          1024ULL;
    else
      dif::fail("unknown run-lora option: " + option);
  }
  if (arguments.program.empty() || arguments.fixture.empty() ||
      arguments.checkpoint.empty() || arguments.losses.empty() ||
      arguments.steps == 0U)
    dif::fail("run-lora requires program, fixture, positive steps, "
              "checkpoint, and losses");
  if (!arguments.resume.empty() && arguments.has_init_seed)
    dif::fail("resume cannot also request adapter initialization");
  if (!std::filesystem::is_directory(arguments.fixture))
    dif::fail("run-lora fixture is not a directory: " +
              arguments.fixture.string());
  return arguments;
}

int run_lora(int argc, char **argv) {
  const auto arguments = parse_lora_run(argc, argv);
  require_new(arguments.checkpoint, "checkpoint output");
  require_new(arguments.losses, "loss output");
  if (!arguments.prediction.empty())
    require_new(arguments.prediction, "prediction output");
  if (!arguments.gradients_directory.empty() &&
      std::filesystem::exists(arguments.gradients_directory))
    dif::fail("gradients directory already exists: " +
              arguments.gradients_directory.string());

  const auto program = dif::ir::read_file(arguments.program);
  const auto build = recover_lora_build(program);
  const auto fingerprint = dif::ir::fingerprint(program);
  dif::runtime::TensorMap inputs;
  const std::array<const char *, 6> data_names = {
      "clean", "noise", "clean-scale", "noise-scale", "time-features",
      "target-velocity"};
  const std::array<std::uint32_t, 6> data_ids = {
      build.clean_input, build.noise_input, build.clean_scale_input,
      build.noise_scale_input, build.timestep_features_input,
      build.target_velocity_input};
  for (std::size_t field = 0U; field < data_ids.size(); ++field) {
    auto tensor = dif::runtime::read_tensor(
        arguments.fixture / (std::string(data_names[field]) + ".diftensor"));
    validate_binding(program, data_ids[field], tensor, data_names[field]);
    inputs.emplace(data_ids[field], std::move(tensor));
  }
  for (const auto &frozen : build.frozen_constants) {
    auto tensor = dif::runtime::read_tensor(
        arguments.fixture /
        ("constant-" + std::to_string(frozen.tensor_id) + ".diftensor"));
    validate_binding(program, frozen.tensor_id, tensor, frozen.name.c_str());
    inputs.emplace(frozen.tensor_id, std::move(tensor));
  }
  std::uint64_t completed_steps = 0U;
  if (!arguments.resume.empty()) {
    auto checkpoint = dif::training::read_checkpoint(arguments.resume);
    if (checkpoint.program_fingerprint != fingerprint)
      dif::fail("resume checkpoint targets a different program fingerprint");
    completed_steps = checkpoint.completed_steps;
    std::unordered_set<std::uint32_t> expected;
    for (const auto &binding : build.optimizer_bindings) {
      for (const auto id : {binding.parameter_input, binding.first_moment_input,
                            binding.second_moment_input}) {
        expected.insert(id);
        const auto found = checkpoint.state.find(id);
        if (found == checkpoint.state.end())
          dif::fail("resume checkpoint is missing state tensor " +
                    std::to_string(id));
        validate_binding(program, id, found->second, "checkpoint");
        inputs.emplace(id, std::move(found->second));
      }
    }
    if (checkpoint.state.size() != expected.size())
      dif::fail("resume checkpoint contains unexpected state tensors");
  } else {
    std::size_t adapter_files = 0U;
    for (const auto &binding : build.optimizer_bindings) {
      if (std::filesystem::exists(
              arguments.fixture /
              ("initial-adapter-" + std::to_string(binding.parameter_input) +
               ".diftensor")))
        ++adapter_files;
    }
    if (adapter_files == build.optimizer_bindings.size()) {
      if (arguments.has_init_seed)
        dif::fail("fixture provides initial adapters; --init-seed would be "
                  "silently ignored");
      for (const auto &binding : build.optimizer_bindings) {
        auto parameter = dif::runtime::read_tensor(
            arguments.fixture /
            ("initial-adapter-" + std::to_string(binding.parameter_input) +
             ".diftensor"));
        validate_binding(program, binding.parameter_input, parameter,
                         "initial adapter");
        inputs.emplace(binding.parameter_input, std::move(parameter));
      }
    } else if (adapter_files == 0U && arguments.has_init_seed) {
      auto initial =
          dif::frontend::default_lora_adapter_init(build, arguments.init_seed);
      for (auto &[id, tensor] : initial)
        inputs.emplace(id, std::move(tensor));
    } else {
      dif::fail("fresh LoRA training requires either every "
                "initial-adapter-<id>.diftensor in the fixture or "
                "--init-seed; found " +
                std::to_string(adapter_files) + " of " +
                std::to_string(build.optimizer_bindings.size()) +
                " adapter files");
    }
    for (const auto &binding : build.optimizer_bindings) {
      inputs.emplace(binding.first_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.first_moment_input)));
      inputs.emplace(binding.second_moment_input,
                     dif::runtime::zeros(
                         *program.tensor(binding.second_moment_input)));
    }
  }
  if (completed_steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      arguments.steps >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) -
              completed_steps)
    dif::fail("training step is outside the I32 optimizer-state range");

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = arguments.minimum_free_bytes;
  options.cache_directory = arguments.cache_directory;
  std::unique_ptr<dif::runtime::Executor> executor;
  if (!arguments.backend_plugin.empty())
    executor = dif::backend::make_plugin_executor(arguments.backend_plugin);
  else if (arguments.backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (arguments.backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("unknown backend: " + arguments.backend);

  inputs.emplace(build.step_input,
                 i32_scalar(static_cast<std::int32_t>(completed_steps)));
  const auto wall_start = std::chrono::steady_clock::now();
  auto prepared = executor->prepare(program, inputs, options);
  std::vector<float> losses;
  losses.reserve(static_cast<std::size_t>(arguments.steps));
  std::vector<double> step_milliseconds;
  step_milliseconds.reserve(static_cast<std::size_t>(arguments.steps));
  dif::runtime::RunResult final_result;
  for (std::uint64_t local_step = 0U; local_step < arguments.steps;
       ++local_step) {
    inputs.insert_or_assign(
        build.step_input,
        i32_scalar(static_cast<std::int32_t>(completed_steps + local_step)));
    auto result = prepared->run(inputs, options);
    const auto loss = result.outputs.at(build.loss_output).f32()[0];
    if (!std::isfinite(loss))
      dif::fail("LoRA training produced a non-finite loss at local step " +
                std::to_string(local_step));
    losses.push_back(loss);
    step_milliseconds.push_back(result.mean_milliseconds);
    for (const auto &binding : build.optimizer_bindings) {
      inputs.insert_or_assign(
          binding.parameter_input,
          std::move(result.outputs.at(binding.parameter_output)));
      inputs.insert_or_assign(
          binding.first_moment_input,
          std::move(result.outputs.at(binding.first_moment_output)));
      inputs.insert_or_assign(
          binding.second_moment_input,
          std::move(result.outputs.at(binding.second_moment_output)));
    }
    final_result = std::move(result);
  }
  completed_steps += arguments.steps;
  const auto wall_stop = std::chrono::steady_clock::now();

  dif::training::Checkpoint checkpoint;
  checkpoint.program_fingerprint = fingerprint;
  checkpoint.completed_steps = completed_steps;
  for (const auto &binding : build.optimizer_bindings) {
    checkpoint.state.emplace(binding.parameter_input,
                             inputs.at(binding.parameter_input));
    checkpoint.state.emplace(binding.first_moment_input,
                             inputs.at(binding.first_moment_input));
    checkpoint.state.emplace(binding.second_moment_input,
                             inputs.at(binding.second_moment_input));
  }
  atomic_write_checkpoint(checkpoint, arguments.checkpoint);
  dif::runtime::write_tensor(f32_vector(losses), arguments.losses);
  if (!arguments.prediction.empty())
    dif::runtime::write_tensor(
        final_result.outputs.at(build.prediction_output),
        arguments.prediction);
  if (!arguments.gradients_directory.empty()) {
    std::filesystem::create_directories(arguments.gradients_directory);
    for (const auto &binding : build.optimizer_bindings)
      dif::runtime::write_tensor(
          final_result.outputs.at(binding.gradient_output),
          arguments.gradients_directory /
              ("gradient-" + std::to_string(binding.parameter_input) +
               ".diftensor"));
  }

  const auto step_mean = std::accumulate(step_milliseconds.begin(),
                                         step_milliseconds.end(), 0.0) /
                         static_cast<double>(step_milliseconds.size());
  struct rusage resource_usage {};
  if (getrusage(RUSAGE_SELF, &resource_usage) != 0)
    dif::fail("getrusage failed");
  std::cout << "LORA_TRAIN PASS backend=" << final_result.backend_name
            << " device=\"" << final_result.device_name << "\""
            << " steps=" << arguments.steps
            << " completed_steps=" << completed_steps
            << " adapters=" << build.adapters.size()
            << " rank=" << build.adapters.front().rank
            << " alpha=" << build.adapters.front().alpha
            << " initial_loss=" << losses.front()
            << " final_loss=" << losses.back()
            << " prepare_ms=" << prepared->preparation_milliseconds()
            << " step_mean_ms=" << step_mean
            << " wall_ms="
            << std::chrono::duration<double, std::milli>(wall_stop - wall_start)
                   .count()
            << " resident_bytes=" << prepared->resident_bytes()
            << " free_before=" << final_result.free_bytes_before
            << " free_after=" << final_result.free_bytes_after
            << " host_max_rss_kib=" << resource_usage.ru_maxrss
            << " fingerprint=" << dif::hex_digest(fingerprint) << "\n";
  return 0;
}

int make_lora(int argc, char **argv) {
  if (argc != 9 && argc != 14) {
    usage();
    return 2;
  }
  const std::filesystem::path output = argv[2];
  require_new(output, "program output");
  dif::frontend::LoraFlowTrainingConfig config;
  config.rows = number(argv[3], "rows");
  config.latent_width = number(argv[4], "latent width");
  config.timestep_width = number(argv[5], "timestep width");
  config.hidden_width = number(argv[6], "hidden width");
  config.rank = number(argv[7], "rank");
  config.alpha = real_number(argv[8], "alpha");
  if (argc == 14) {
    config.learning_rate = real_number(argv[9], "learning rate");
    config.beta1 = real_number(argv[10], "beta1");
    config.beta2 = real_number(argv[11], "beta2");
    config.epsilon = real_number(argv[12], "epsilon");
    config.weight_decay = real_number(argv[13], "weight decay");
  }
  const auto build = dif::frontend::make_lora_flow_training(config);
  dif::ir::write_file(build.program, output);
  std::cout << "LORA_PROGRAM PASS path=" << output
            << " tensors=" << build.program.tensors.size()
            << " operations=" << build.program.operations.size()
            << " adapters=" << build.adapters.size()
            << " rank=" << config.rank << " alpha=" << config.alpha
            << " fingerprint="
            << dif::hex_digest(dif::ir::fingerprint(build.program)) << "\n";
  return 0;
}

int export_lora(int argc, char **argv) {
  std::filesystem::path program_path;
  std::filesystem::path checkpoint_path;
  std::filesystem::path output_path;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--program")
      program_path = value("--program");
    else if (option == "--checkpoint")
      checkpoint_path = value("--checkpoint");
    else if (option == "--output")
      output_path = value("--output");
    else
      dif::fail("unknown export-lora option: " + option);
  }
  if (program_path.empty() || checkpoint_path.empty() || output_path.empty())
    dif::fail("export-lora requires --program, --checkpoint, and --output");
  require_new(output_path, "export output");
  const auto program = dif::ir::read_file(program_path);
  const auto build = recover_lora_build(program);
  const auto checkpoint = dif::training::read_checkpoint(checkpoint_path);
  dif::frontend::export_lora_adapters(build, checkpoint, output_path);
  dif::frontend::validate_lora_export(output_path, build.adapters);
  std::cout << "LORA_EXPORT PASS path=" << output_path
            << " adapters=" << build.adapters.size()
            << " rank=" << build.adapters.front().rank
            << " alpha=" << build.adapters.front().alpha
            << " completed_steps=" << checkpoint.completed_steps
            << " sha256=" << dif::hex_digest(dif::sha256_file(output_path))
            << " fingerprint="
            << dif::hex_digest(checkpoint.program_fingerprint) << "\n";
  return 0;
}

int inspect_checkpoint(const std::filesystem::path &path) {
  const auto checkpoint = dif::training::read_checkpoint(path);
  std::vector<std::uint32_t> ids;
  ids.reserve(checkpoint.state.size());
  for (const auto &[id, tensor] : checkpoint.state) {
    static_cast<void>(tensor);
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  std::cout << "CHECKPOINT version=1 completed_steps="
            << checkpoint.completed_steps
            << " tensors=" << checkpoint.state.size()
            << " fingerprint="
            << dif::hex_digest(checkpoint.program_fingerprint)
            << " sha256=" << dif::hex_digest(dif::sha256_file(path)) << "\n";
  for (const auto id : ids) {
    const auto &tensor = checkpoint.state.at(id);
    std::cout << "STATE id=" << id << " dtype="
              << static_cast<std::uint32_t>(tensor.dtype) << " shape=";
    for (std::size_t axis = 0U; axis < tensor.dims.size(); ++axis) {
      if (axis)
        std::cout << "x";
      std::cout << tensor.dims[axis];
    }
    std::cout << " bytes=" << tensor.byte_size() << "\n";
  }
  return 0;
}

int export_checkpoint(const std::filesystem::path &path,
                      const std::filesystem::path &directory) {
  if (std::filesystem::exists(directory))
    dif::fail("export directory already exists: " + directory.string());
  const auto checkpoint = dif::training::read_checkpoint(path);
  std::filesystem::create_directories(directory);
  for (const auto &[id, tensor] : checkpoint.state)
    dif::runtime::write_tensor(
        tensor, directory / ("tensor-" + std::to_string(id) + ".diftensor"));
  std::cout << "EXPORT PASS directory=" << directory
            << " tensors=" << checkpoint.state.size()
            << " completed_steps=" << checkpoint.completed_steps << "\n";
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
    if (command == "run-mlp")
      return run_mlp(argc, argv);
    if (command == "run-flow")
      return run_flow(argc, argv);
    if (command == "make-lora")
      return make_lora(argc, argv);
    if (command == "run-lora")
      return run_lora(argc, argv);
    if (command == "export-lora")
      return export_lora(argc, argv);
    if (command == "inspect" && argc == 3)
      return inspect_checkpoint(argv[2]);
    if (command == "export" && argc == 4)
      return export_checkpoint(argv[2], argv[3]);
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "diftrain: " << error.what() << "\n";
    return 1;
  }
}
