// DiT-block training runner for the composed backward gate.  Builds the
// dif::frontend::make_dit_block_training graph from the fixture's
// config.json, binds the exported torch fixture tensors, runs N optimizer
// steps on the requested backend, and writes losses, step-1 gradients,
// final gradients, final parameters, and final moments for difcompare.

#include "dif/frontend/dit_block.hpp"
#include "dif/frontend/dit_lora.hpp"
#include "dif/frontend/lora.hpp"
#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/training/checkpoint.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void usage() {
  std::cerr << "usage: difdittrain --backend cpu|cuda --fixture DIR "
               "--output DIR [--steps N] [--cache-dir DIR]\n"
               "       LoRA fixtures (config.json with lora_rank) also "
               "accept [--checkpoint OUT.diftrain] [--resume IN.diftrain] "
               "[--export-adapters OUT.safetensors]\n";
}

dif::runtime::Tensor i32_scalar(std::int32_t value) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, {1U}, {}};
  tensor.bytes.resize(sizeof(value));
  std::memcpy(tensor.bytes.data(), &value, sizeof(value));
  return tensor;
}

dif::runtime::Tensor f32_vector(const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::F32,
                              {static_cast<std::uint64_t>(values.size())},
                              {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  return tensor;
}

void atomic_write_checkpoint(const dif::training::Checkpoint &checkpoint,
                             const std::filesystem::path &path) {
  if (std::filesystem::exists(path))
    dif::fail("checkpoint output already exists: " + path.string());
  auto temporary = path;
  temporary += ".partial";
  if (std::filesystem::exists(temporary))
    dif::fail("checkpoint temporary already exists: " + temporary.string());
  try {
    dif::training::write_checkpoint(checkpoint, temporary);
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

int run_dit_lora(const dif::json::Value &configuration,
                 const std::filesystem::path &fixture,
                 const std::filesystem::path &output,
                 const std::filesystem::path &cache_directory,
                 const std::string &backend, std::uint64_t steps_override,
                 const std::filesystem::path &checkpoint_path,
                 const std::filesystem::path &resume_path,
                 const std::filesystem::path &export_path, bool persistent) {
  const auto field = [&](const char *key) -> const dif::json::Value & {
    const auto *value = configuration.find(key);
    if (!value)
      dif::fail(std::string("fixture config.json is missing ") + key);
    return *value;
  };
  dif::frontend::DitLoraTrainingConfig config;
  config.sequence = static_cast<std::uint64_t>(field("sequence").number());
  config.heads = static_cast<std::uint64_t>(field("heads").number());
  config.head_dim = static_cast<std::uint64_t>(field("head_dim").number());
  config.mlp_width = static_cast<std::uint64_t>(field("mlp_width").number());
  config.blocks = static_cast<std::uint64_t>(field("blocks").number());
  config.rotary_dim =
      static_cast<std::uint64_t>(field("rotary_dim").number());
  config.full_rope_table = field("full_rope_table").boolean();
  config.causal = field("causal").boolean();
  config.rank = static_cast<std::uint64_t>(field("lora_rank").number());
  config.alpha = field("lora_alpha").number();
  config.compute_dtype = field("lora_bf16").boolean() ? dif::ir::DType::BF16
                                                      : dif::ir::DType::F32;
  config.learning_rate = field("learning_rate").number();
  config.beta1 = field("beta1").number();
  config.beta2 = field("beta2").number();
  config.epsilon_adam = field("epsilon_adam").number();
  config.weight_decay = field("weight_decay").number();
  config.epsilon_norm = field("epsilon_norm").number();
  const auto steps =
      steps_override != 0U
          ? steps_override
          : static_cast<std::uint64_t>(field("steps").number());

  const auto build = dif::frontend::make_dit_lora_training(config);
  const auto fingerprint = dif::ir::fingerprint(build.program);

  dif::runtime::TensorMap inputs;
  const auto bind = [&](std::uint32_t id, const std::string &name) {
    auto tensor = dif::runtime::read_tensor(fixture / (name + ".diftensor"));
    const auto *description = build.program.tensor(id);
    if (tensor.dtype != description->dtype ||
        tensor.dims != description->dims)
      dif::fail("fixture tensor " + name +
                " does not match the program declaration");
    inputs.insert_or_assign(id, std::move(tensor));
  };
  bind(build.x_input, "x");
  bind(build.cos_input, "cos");
  bind(build.sin_input, "sin");
  bind(build.target_input, "target");
  for (std::uint64_t block = 0U; block < config.blocks; ++block) {
    const auto &modulation = build.modulation_inputs[block];
    const auto prefix = "block" + std::to_string(block) + "-";
    bind(modulation.scale1, prefix + "scale1");
    bind(modulation.shift1, prefix + "shift1");
    bind(modulation.gate1, prefix + "gate1");
    bind(modulation.scale2, prefix + "scale2");
    bind(modulation.shift2, prefix + "shift2");
    bind(modulation.gate2, prefix + "gate2");
  }
  for (std::size_t index = 0U; index < build.frozen_constants.size();
       ++index)
    bind(build.frozen_constants[index].tensor_id,
         "frozen-" + std::to_string(index));

  std::uint64_t completed_steps = 0U;
  if (!resume_path.empty()) {
    auto checkpoint = dif::training::read_checkpoint(resume_path);
    if (checkpoint.program_fingerprint != fingerprint)
      dif::fail("resume checkpoint targets a different program fingerprint");
    completed_steps = checkpoint.completed_steps;
    std::size_t expected = 0U;
    for (const auto &binding : build.optimizer_bindings) {
      for (const auto id :
           {binding.parameter_input, binding.first_moment_input,
            binding.second_moment_input}) {
        ++expected;
        const auto found = checkpoint.state.find(id);
        if (found == checkpoint.state.end())
          dif::fail("resume checkpoint is missing state tensor " +
                    std::to_string(id));
        const auto *description = build.program.tensor(id);
        if (found->second.dtype != description->dtype ||
            found->second.dims != description->dims)
          dif::fail("resume checkpoint state mismatch for tensor " +
                    std::to_string(id));
        inputs.insert_or_assign(id, std::move(found->second));
      }
    }
    if (checkpoint.state.size() != expected)
      dif::fail("resume checkpoint contains unexpected state tensors");
  } else {
    for (std::size_t index = 0U; index < build.optimizer_bindings.size();
         ++index)
      bind(build.optimizer_bindings[index].parameter_input,
           "adapter-" + std::to_string(index));
    for (const auto &binding : build.optimizer_bindings) {
      inputs.emplace(binding.first_moment_input,
                     dif::runtime::zeros(
                         *build.program.tensor(binding.first_moment_input)));
      inputs.emplace(binding.second_moment_input,
                     dif::runtime::zeros(
                         *build.program.tensor(binding.second_moment_input)));
    }
  }

  // Snapshot the frozen base bits: training must not touch them.
  std::vector<std::vector<std::uint8_t>> base_snapshot;
  base_snapshot.reserve(build.frozen_constants.size());
  for (const auto &constant : build.frozen_constants) {
    const auto &tensor = inputs.at(constant.tensor_id);
    base_snapshot.emplace_back(tensor.data(),
                               tensor.data() + tensor.byte_size());
  }

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  options.cache_directory = cache_directory;
  std::unique_ptr<dif::runtime::Executor> executor;
  if (backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("unknown backend: " + backend);

  inputs.emplace(build.step_input,
                 i32_scalar(static_cast<std::int32_t>(completed_steps)));
  // "This output is that input next time", stated once per tensor the step
  // advances. The runtime is told nothing about what an optimizer is.
  if (persistent)
    for (const auto &binding : build.optimizer_bindings) {
      options.persistent_state.push_back(
          {binding.parameter_input, binding.parameter_output});
      options.persistent_state.push_back(
          {binding.first_moment_input, binding.first_moment_output});
      options.persistent_state.push_back(
          {binding.second_moment_input, binding.second_moment_output});
    }
  const auto wall_start = std::chrono::steady_clock::now();
  auto prepared = executor->prepare(build.program, inputs, options);
  std::filesystem::create_directories(output);
  std::vector<float> losses;
  std::uint64_t state_host_to_device = 0U;
  std::uint64_t state_device_to_host = 0U;
  dif::runtime::RunResult result;
  for (std::uint64_t step = 0U; step < steps; ++step) {
    inputs.insert_or_assign(
        build.step_input,
        i32_scalar(static_cast<std::int32_t>(completed_steps + step)));
    result = prepared->run(inputs, options);
    const auto loss = result.outputs.at(build.loss_output).f32()[0];
    if (!std::isfinite(loss))
      dif::fail("DiT LoRA training produced a non-finite loss at step " +
                std::to_string(step));
    losses.push_back(loss);
    if (step == 0U && resume_path.empty())
      for (std::size_t index = 0U; index < build.optimizer_bindings.size();
           ++index)
        dif::runtime::write_tensor(
            result.outputs.at(
                build.optimizer_bindings[index].gradient_output),
            output / ("grad1-" + std::to_string(index) + ".diftensor"));
    if (!persistent)
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
    state_host_to_device += result.persistent_state_host_to_device_bytes;
    state_device_to_host += result.persistent_state_device_to_host_bytes;
  }
  // Read the state back once, on purpose, so the frozen-base check, the
  // written tensors, the checkpoint and the adapter export are unchanged.
  if (persistent)
    for (auto &[id, tensor] : prepared->capture_persistent_state())
      inputs.insert_or_assign(id, std::move(tensor));
  completed_steps += steps;
  const auto wall_stop = std::chrono::steady_clock::now();

  // Base-bits invariant: every frozen tensor is byte-identical.
  std::size_t unchanged = 0U;
  for (std::size_t index = 0U; index < build.frozen_constants.size();
       ++index) {
    const auto &tensor = inputs.at(build.frozen_constants[index].tensor_id);
    const auto &snapshot = base_snapshot[index];
    if (tensor.byte_size() != snapshot.size() ||
        !std::equal(snapshot.begin(), snapshot.end(), tensor.data()))
      dif::fail("frozen base tensor " +
                build.frozen_constants[index].name + " changed bits");
    ++unchanged;
  }
  std::cout << "BASE_BITS_UNCHANGED tensors=" << unchanged << "\n";

  dif::runtime::write_tensor(f32_vector(losses),
                             output / "losses.diftensor");
  dif::runtime::write_tensor(result.outputs.at(build.prediction_output),
                             output / "prediction.diftensor");
  for (std::size_t index = 0U; index < build.optimizer_bindings.size();
       ++index) {
    const auto &binding = build.optimizer_bindings[index];
    const auto suffix = std::to_string(index) + ".diftensor";
    dif::runtime::write_tensor(inputs.at(binding.parameter_input),
                               output / ("param-" + suffix));
    dif::runtime::write_tensor(inputs.at(binding.first_moment_input),
                               output / ("moment1-" + suffix));
    dif::runtime::write_tensor(inputs.at(binding.second_moment_input),
                               output / ("moment2-" + suffix));
    dif::runtime::write_tensor(result.outputs.at(binding.gradient_output),
                               output / ("grad-" + suffix));
  }

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
  if (!checkpoint_path.empty())
    atomic_write_checkpoint(checkpoint, checkpoint_path);
  if (!export_path.empty()) {
    dif::frontend::export_lora_adapters(build.program, build.adapters,
                                        checkpoint, export_path);
    dif::frontend::validate_lora_export(export_path, build.adapters);
  }

  std::cout << "PERSISTENT_STATE persistent=" << (persistent ? 1 : 0)
            << " resident_bytes=" << result.persistent_state_bytes
            << " step_h2d_bytes=" << state_host_to_device
            << " step_d2h_bytes=" << state_device_to_host << "\n";
  std::cout << "DIT_LORA_TRAIN PASS backend=" << result.backend_name
            << " blocks=" << config.blocks << " steps=" << steps
            << " completed_steps=" << completed_steps
            << " sites=" << build.adapters.size()
            << " adapters=" << build.optimizer_bindings.size()
            << " rank=" << config.rank << " alpha=" << config.alpha
            << " dtype="
            << (config.compute_dtype == dif::ir::DType::BF16 ? "bf16"
                                                             : "f32")
            << " initial_loss=" << losses.front()
            << " final_loss=" << losses.back() << " wall_ms="
            << std::chrono::duration<double, std::milli>(wall_stop -
                                                         wall_start)
                   .count()
            << " fingerprint=" << dif::hex_digest(fingerprint) << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path fixture;
    std::filesystem::path output;
    std::filesystem::path cache_directory;
    std::filesystem::path checkpoint_path;
    // Ask the runtime to own the optimizer state instead of carrying it back
    // through host memory between steps.
    bool persistent = false;
    std::filesystem::path resume_path;
    std::filesystem::path export_path;
    std::string backend{"cpu"};
    std::uint64_t steps_override = 0U;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      auto value = [&](const char *label) -> std::string {
        if (++index >= argc)
          dif::fail(std::string("missing value for ") + label);
        return argv[index];
      };
      if (option == "--backend")
        backend = value("--backend");
      else if (option == "--fixture")
        fixture = value("--fixture");
      else if (option == "--output")
        output = value("--output");
      else if (option == "--cache-dir")
        cache_directory = value("--cache-dir");
      else if (option == "--persistent")
        persistent = true;
      else if (option == "--checkpoint")
        checkpoint_path = value("--checkpoint");
      else if (option == "--resume")
        resume_path = value("--resume");
      else if (option == "--export-adapters")
        export_path = value("--export-adapters");
      else if (option == "--steps")
        steps_override = std::stoull(value("--steps"));
      else {
        usage();
        return 2;
      }
    }
    if (fixture.empty() || output.empty()) {
      usage();
      return 2;
    }

    std::ifstream config_stream(fixture / "config.json");
    if (!config_stream)
      dif::fail("missing fixture config.json");
    std::stringstream config_buffer;
    config_buffer << config_stream.rdbuf();
    const auto configuration = dif::json::parse(config_buffer.str());
    if (configuration.find("lora_rank"))
      return run_dit_lora(configuration, fixture, output, cache_directory,
                          backend, steps_override, checkpoint_path,
                          resume_path, export_path, persistent);
    if (!checkpoint_path.empty() || !resume_path.empty() ||
        !export_path.empty())
      dif::fail("--checkpoint/--resume/--export-adapters require a LoRA "
                "fixture (config.json with lora_rank)");
    const auto field = [&](const char *key) -> const dif::json::Value & {
      const auto *value = configuration.find(key);
      if (!value)
        dif::fail(std::string("fixture config.json is missing ") + key);
      return *value;
    };

    dif::frontend::DitBlockTrainingConfig config;
    config.sequence = static_cast<std::uint64_t>(field("sequence").number());
    config.heads = static_cast<std::uint64_t>(field("heads").number());
    config.head_dim = static_cast<std::uint64_t>(field("head_dim").number());
    config.mlp_width =
        static_cast<std::uint64_t>(field("mlp_width").number());
    config.blocks = static_cast<std::uint64_t>(field("blocks").number());
    config.rotary_dim =
        static_cast<std::uint64_t>(field("rotary_dim").number());
    config.full_rope_table = field("full_rope_table").boolean();
    config.causal = field("causal").boolean();
    config.learning_rate = field("learning_rate").number();
    config.beta1 = field("beta1").number();
    config.beta2 = field("beta2").number();
    config.epsilon_adam = field("epsilon_adam").number();
    config.weight_decay = field("weight_decay").number();
    config.epsilon_norm = field("epsilon_norm").number();
    const auto steps =
        steps_override != 0U
            ? steps_override
            : static_cast<std::uint64_t>(field("steps").number());

    const auto build = dif::frontend::make_dit_block_training(config);
    const auto fingerprint = dif::ir::fingerprint(build.program);

    dif::runtime::TensorMap inputs;
    const auto bind = [&](std::uint32_t id, const std::string &name) {
      auto tensor = dif::runtime::read_tensor(fixture / (name + ".diftensor"));
      const auto *description = build.program.tensor(id);
      if (tensor.dtype != description->dtype ||
          tensor.dims != description->dims)
        dif::fail("fixture tensor " + name +
                  " does not match the program declaration");
      inputs.insert_or_assign(id, std::move(tensor));
    };
    bind(build.x_input, "x");
    bind(build.cos_input, "cos");
    bind(build.sin_input, "sin");
    bind(build.target_input, "target");
    for (std::uint64_t block = 0U; block < config.blocks; ++block) {
      const auto &modulation = build.modulation_inputs[block];
      const auto prefix = "block" + std::to_string(block) + "-";
      bind(modulation.scale1, prefix + "scale1");
      bind(modulation.shift1, prefix + "shift1");
      bind(modulation.gate1, prefix + "gate1");
      bind(modulation.scale2, prefix + "scale2");
      bind(modulation.shift2, prefix + "shift2");
      bind(modulation.gate2, prefix + "gate2");
    }
    for (std::size_t index = 0U; index < build.parameters.size(); ++index)
      bind(build.parameters[index], "param-" + std::to_string(index));
    for (const auto &binding : build.optimizer_bindings) {
      inputs.emplace(binding.first_moment_input,
                     dif::runtime::zeros(
                         *build.program.tensor(binding.first_moment_input)));
      inputs.emplace(binding.second_moment_input,
                     dif::runtime::zeros(
                         *build.program.tensor(binding.second_moment_input)));
    }

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    options.cache_directory = cache_directory;
    std::unique_ptr<dif::runtime::Executor> executor;
    if (backend == "cpu")
      executor = dif::runtime::make_cpu_executor();
    else if (backend == "cuda")
      executor = dif::runtime::make_cuda_executor();
    else
      dif::fail("unknown backend: " + backend);

    inputs.emplace(build.step_input, i32_scalar(0));
    // "This output is that input next time", stated once per tensor the step
    // advances. The runtime is told nothing about what an optimizer is.
    if (persistent)
      for (const auto &binding : build.optimizer_bindings) {
        options.persistent_state.push_back(
            {binding.parameter_input, binding.parameter_output});
        options.persistent_state.push_back(
            {binding.first_moment_input, binding.first_moment_output});
        options.persistent_state.push_back(
            {binding.second_moment_input, binding.second_moment_output});
      }
    const auto wall_start = std::chrono::steady_clock::now();
    auto prepared = executor->prepare(build.program, inputs, options);
    std::filesystem::create_directories(output);
    std::vector<float> losses;
    std::uint64_t state_host_to_device = 0U;
    std::uint64_t state_device_to_host = 0U;
    dif::runtime::RunResult result;
    for (std::uint64_t step = 0U; step < steps; ++step) {
      inputs.insert_or_assign(build.step_input,
                              i32_scalar(static_cast<std::int32_t>(step)));
      result = prepared->run(inputs, options);
      const auto loss = result.outputs.at(build.loss_output).f32()[0];
      if (!std::isfinite(loss))
        dif::fail("DiT block training produced a non-finite loss at step " +
                  std::to_string(step));
      losses.push_back(loss);
      if (step == 0U)
        for (std::size_t index = 0U; index < build.optimizer_bindings.size();
             ++index)
          dif::runtime::write_tensor(
              result.outputs.at(
                  build.optimizer_bindings[index].gradient_output),
              output / ("grad1-" + std::to_string(index) + ".diftensor"));
      if (!persistent)
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
      state_host_to_device += result.persistent_state_host_to_device_bytes;
      state_device_to_host += result.persistent_state_device_to_host_bytes;
    }
    // Read the state back once, on purpose, so everything written below is
    // unchanged.
    if (persistent)
      for (auto &[id, tensor] : prepared->capture_persistent_state())
        inputs.insert_or_assign(id, std::move(tensor));
    const auto wall_stop = std::chrono::steady_clock::now();

    dif::runtime::write_tensor(f32_vector(losses),
                               output / "losses.diftensor");
    for (std::size_t index = 0U; index < build.optimizer_bindings.size();
         ++index) {
      const auto &binding = build.optimizer_bindings[index];
      const auto suffix = std::to_string(index) + ".diftensor";
      dif::runtime::write_tensor(inputs.at(binding.parameter_input),
                                 output / ("param-" + suffix));
      dif::runtime::write_tensor(inputs.at(binding.first_moment_input),
                                 output / ("moment1-" + suffix));
      dif::runtime::write_tensor(inputs.at(binding.second_moment_input),
                                 output / ("moment2-" + suffix));
      dif::runtime::write_tensor(result.outputs.at(binding.gradient_output),
                                 output / ("grad-" + suffix));
    }
    std::cout << "PERSISTENT_STATE persistent=" << (persistent ? 1 : 0)
              << " resident_bytes=" << result.persistent_state_bytes
              << " step_h2d_bytes=" << state_host_to_device
              << " step_d2h_bytes=" << state_device_to_host << "\n";
    std::cout << "DIT_TRAIN PASS backend=" << result.backend_name
              << " blocks=" << config.blocks << " steps=" << steps
              << " parameters=" << build.parameters.size()
              << " initial_loss=" << losses.front()
              << " final_loss=" << losses.back() << " wall_ms="
              << std::chrono::duration<double, std::milli>(wall_stop -
                                                           wall_start)
                     .count()
              << " fingerprint=" << dif::hex_digest(fingerprint) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difdittrain: " << error.what() << "\n";
    return 1;
  }
}
