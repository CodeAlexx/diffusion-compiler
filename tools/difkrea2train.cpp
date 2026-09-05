// Train a Krea 2 LoRA.
//
// This tool is deliberately thin. It reads two files, asks the frontend for
// a training plan, and drives difcore's session. It contains no optimizer, no
// autograd, no memory policy and no tensor type -- if something here looks
// like infrastructure, it belongs in difcore instead, and the next model's
// trainer would have had to write it again.
//
//   --config PATH     the run's parameters (required)
//   --prompts PATH    sample prompts; defaults to the config's own
//   --steps N         override the config's step count, for a smoke run
//   --backend NAME    cpu or cuda
//   --plan-only       report what the run would cost and stop
//
// Everything a run produces goes under the config's workspace_dir and
// nowhere else: adapters, samples, checkpoints and the per-step record.

#include "dif/compiler/int8.hpp"
#include "dif/frontend/krea2_training.hpp"
#include "dif/support/error.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/target/profile.hpp"
#include "dif/training/config.hpp"
#include "dif/training/memory.hpp"
#include "dif/training/report.hpp"
#include "dif/training/session.hpp"

#include "dif/runtime/tensor.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <iomanip>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

const double kGiB = 1024.0 * 1024.0 * 1024.0;

[[noreturn]] void usage(const std::string &problem) {
  std::cerr << "difkrea2train: " << problem
            << "\nusage: difkrea2train --config PATH [--prompts PATH] "
               "[--steps N] [--backend cpu|cuda] [--plan-only]\n";
  std::exit(2);
}

std::string argument(int argc, char **argv, int &index) {
  if (index + 1 >= argc)
    usage(std::string(argv[index]) + " needs a value");
  return argv[++index];
}

// What a run would cost, before it is started. A trainer that discovers at
// step 0 that it does not fit has already spent the checkpoint load.
void report_cost(const dif::frontend::Krea2TrainingBuild &build,
                 const dif::training::TrainingRun &run,
                 const std::string &backend) {
  const auto memory = dif::training::analyze_memory(build.plan);
  std::uint64_t trainable = 0U;
  for (const auto &binding : build.plan.bindings)
    trainable +=
        build.plan.program.tensor(binding.parameter_input)->element_count();
  std::uint64_t frozen = 0U;
  for (const auto &[id, name] : build.frozen)
    frozen += build.plan.program.tensor(id)->element_count();

  std::cout << "KREA2_PLAN sites=" << build.sites.size()
            << " adapters=" << build.plan.bindings.size()
            << " trainable_parameters=" << trainable
            << " frozen_parameters=" << frozen
            << " operations=" << build.plan.program.operations.size()
            << " resident_state="
            << build.plan.persistent_state().size() << "\n";
  std::cout << "KREA2_MEMORY frozen_gib=" << memory.frozen_weights / kGiB
            << " trainable_gib=" << memory.trainable_weights / kGiB
            << " optimizer_gib=" << memory.optimizer_state / kGiB
            << " gradients_gib=" << memory.gradients / kGiB
            << " saved_activations_gib=" << memory.saved_activations / kGiB
            << " transient_gib=" << memory.transient_activations / kGiB
            << " planned_gib=" << memory.planned_bytes / kGiB << "\n";

  // What recompute could do about it, on the device actually present.
  dif::training::RecomputePolicy policy;
  const auto probe_backend = backend == "cuda"
                                 ? dif::runtime::ProbeBackend::Cuda
                                 : dif::runtime::ProbeBackend::Host;
  policy.budget = dif::runtime::probe_runtime_budget(
      dif::runtime::probe_target(probe_backend));
  const auto choice = dif::training::choose_recompute(build.plan, policy);
  std::cout << "KREA2_RECOMPUTE budget_gib=" << policy.device_bytes() / kGiB
            << " segments=" << choice.segments
            << " planned_gib=" << choice.planned_bytes / kGiB
            << " without_recompute_gib="
            << choice.bytes_without_recompute / kGiB
            << " replayed_operations=" << choice.replayed_operations
            << " within_budget=" << (choice.within_budget ? 1 : 0) << "\n";
  if (!choice.within_budget)
    std::cout << "KREA2_NOTE the plan does not fit this device even with "
                 "recompute; the frozen base is "
              << memory.frozen_weights / kGiB
              << " GiB in its checkpoint dtype and the config asks for '"
              << (run.resident_format.empty() ? std::string("none")
                                              : run.resident_format)
              << "' resident\n";
}

} // namespace

int main(int argc, char **argv) {
  std::filesystem::path config_path;
  std::filesystem::path prompts_path;
  std::string backend = "cuda";
  std::uint64_t step_override = 0U;
  bool plan_only = false;
  bool trace = false;

  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--config")
      config_path = argument(argc, argv, index);
    else if (flag == "--prompts")
      prompts_path = argument(argc, argv, index);
    else if (flag == "--backend")
      backend = argument(argc, argv, index);
    else if (flag == "--steps")
      step_override = std::stoull(argument(argc, argv, index));
    else if (flag == "--plan-only")
      plan_only = true;
    else if (flag == "--trace")
      trace = true;
    else
      usage("unknown argument " + flag);
  }
  if (config_path.empty())
    usage("--config is required: this trainer has no defaults to fall back "
          "on, because a run nobody can read back is a run nobody can repeat");

  try {
    const auto config = dif::training::TrainingConfig::read(config_path);
    auto run = dif::training::read_run(config);
    if (step_override != 0U)
      run.max_steps = step_override;
    if (run.model_type != "krea2")
      dif::fail("this trainer trains krea2; the config says '" +
                run.model_type + "'");

    // The sampler reads the same checkpoint the trainer trains. A model
    // shipped as a base and a distilled variant trains on the base, and a
    // sampler quietly using the other one would report quality that has
    // nothing to do with what was trained.
    std::cout << "KREA2_RUN checkpoint=" << run.checkpoint
              << " steps=" << run.max_steps << " batch=" << run.batch
              << " rank=" << run.lora_rank << " alpha=" << run.lora_alpha
              << " learning_rate=" << run.optimizer.learning_rate << "\n";

    if (prompts_path.empty())
      prompts_path = run.prompts_file;
    if (!prompts_path.empty() && std::filesystem::exists(prompts_path)) {
      const auto prompts = dif::training::read_sample_prompts(prompts_path);
      std::cout << "KREA2_PROMPTS file=" << prompts_path
                << " count=" << prompts.prompts.size()
                << " sample_every=" << prompts.sample_every << "\n";
    } else if (run.sample_every != 0U) {
      std::cout << "KREA2_PROMPTS none; the config asks to sample every "
                << run.sample_every
                << " steps but names no prompts file, so nothing will be "
                   "sampled\n";
    }

    auto build = dif::frontend::build_krea2_training(config);
    report_cost(build, run, backend);

    // Apply the recompute the report just described. Reporting a
    // segmentation that fits and then running the one that does not is worse
    // than not reporting it at all.
    if (!plan_only && backend != "cpu") {
      dif::training::RecomputePolicy policy;
      policy.budget = dif::runtime::probe_runtime_budget(
          dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda));
      dif::ir::Program segmented;
      const auto choice =
          dif::training::choose_recompute(build.plan, policy, &segmented);
      if (choice.segments > 1U) {
        if (!choice.within_budget)
          std::cout << "KREA2_NOTE running anyway at the closest "
                       "segmentation; it is expected to exceed the budget\n";
        build.plan = dif::training::plan_from_composed(
            std::move(segmented), build.plan.step_input, build.plan.loss_tensor,
            build.plan.bindings);
        std::cout << "KREA2_RECOMPUTE_APPLIED segments=" << choice.segments
                  << " operations=" << build.plan.program.operations.size()
                  << " planned_gib=" << choice.planned_bytes / kGiB << "\n";
      }
    }

    // A misspelled knob does nothing and says nothing, so say it here --
    // after everything that reads has read, or every architecture key would
    // be reported as a mistake.
    const auto unread = config.unread_keys();
    if (!unread.empty()) {
      std::cout << "KREA2_UNREAD";
      for (const auto &key : unread)
        std::cout << " " << key;
      std::cout << "\n";
    }

    if (plan_only) {
      std::cout << "KREA2_PLAN_ONLY ok\n";
      return 0;
    }

    if (run.workspace.empty())
      dif::fail("'workspace_dir' is required for a run that produces "
                "anything: adapters, samples and the step record all go "
                "there and nowhere else");
    std::filesystem::create_directories(run.workspace);
    std::cout << "KREA2_WORKSPACE " << run.workspace << "\n";

    // ---- the weights ----------------------------------------------------
    const auto load_start = std::chrono::steady_clock::now();
    const auto checkpoint = dif::weights::read_safetensors(run.checkpoint);
    dif::runtime::TensorMap bindings;
    const auto bind_checkpoint = [&](const std::string &name,
                                     dif::ir::DType wanted) {
      auto tensor = dif::weights::map_safetensor(checkpoint, name);
      if (tensor.dtype != wanted)
        tensor = dif::runtime::convert_float_tensor(tensor, wanted);
      return tensor;
    };
    for (const auto &[id, name] : build.frozen) {
      const auto *description = build.plan.program.tensor(id);
      auto tensor = bind_checkpoint(name, description->dtype);
      if (description->dims != tensor.dims)
        dif::fail("checkpoint tensor " + name + " does not have the shape "
                  "the graph gives it");
      bindings.insert_or_assign(id, std::move(tensor));
    }
    // Quantized ONCE, here, on the way in. Nothing converts them again.
    double worst_relative_error = 0.0;
    for (const auto &weight : build.resident) {
      const auto source = bind_checkpoint(weight.name, dif::ir::DType::BF16);
      auto quantized = dif::compiler::quantize_int8_weight(source);
      if (quantized.squared_reference > 0.0)
        worst_relative_error =
            std::max(worst_relative_error,
                     std::sqrt(quantized.squared_error /
                               quantized.squared_reference));
      bindings.insert_or_assign(weight.weight, std::move(quantized.weight));
      bindings.insert_or_assign(weight.scales, std::move(quantized.scales));
    }
    const auto load_stop = std::chrono::steady_clock::now();
    std::cout << "KREA2_WEIGHTS frozen=" << build.frozen.size()
              << " resident=" << build.resident.size()
              << " format=" << (build.resident_format.empty()
                                    ? std::string("bf16")
                                    : build.resident_format)
              << " bytes_before=" << build.resident_bytes_before
              << " bytes_after=" << build.resident_bytes_after
              << " worst_relative_quantization_error=" << worst_relative_error
              << " load_seconds="
              << std::chrono::duration<double>(load_stop - load_start).count()
              << "\n";

    // The rotary tables: derived from the geometry, not read from the
    // checkpoint, exactly as the sampler derives them.
    {
      const auto i32 = [](std::vector<std::int32_t> values) {
        dif::runtime::Tensor tensor;
        tensor.dtype = dif::ir::DType::I32;
        tensor.dims = {static_cast<std::uint64_t>(values.size())};
        tensor.bytes.resize(values.size() * sizeof(std::int32_t));
        std::memcpy(tensor.mutable_data(), values.data(),
                    tensor.bytes.size());
        return tensor;
      };
      std::vector<std::int32_t> pair_axes;
      std::vector<std::int32_t> pair_indices;
      for (std::int32_t axis = 0; axis < 3; ++axis) {
        const std::int32_t dimension = axis == 0 ? 32 : 48;
        for (std::int32_t pair = 0; pair < dimension / 2; ++pair) {
          pair_axes.push_back(axis);
          pair_indices.push_back(pair);
        }
      }
      bindings.insert_or_assign(build.rotary_pair_axes,
                                i32(std::move(pair_axes)));
      bindings.insert_or_assign(build.rotary_pair_indices,
                                i32(std::move(pair_indices)));
      bindings.insert_or_assign(build.rotary_axis_dims, i32({32, 48, 48}));
    }

    // Anything else the graph declares constant and nothing has filled is a
    // mistake worth naming now rather than at prepare time.
    for (const auto &tensor : build.plan.program.tensors)
      if (tensor.has_role(dif::ir::TensorRole::Constant) &&
          !bindings.contains(tensor.id))
        dif::fail("constant tensor " + std::to_string(tensor.id) +
                  " is filled by nothing; the loader does not know what it "
                  "holds");

    // ---- the adapters and their optimizer state -------------------------
    // A is small and random, B is zero, so the adapted model starts exactly
    // equal to the frozen one. Starting anywhere else means step 0 is
    // already a different model than the checkpoint.
    std::mt19937 generator(static_cast<std::uint32_t>(run.seed));
    std::normal_distribution<float> normal(
        0.0F, 1.0F / std::sqrt(static_cast<float>(run.lora_rank)));
    std::set<std::uint32_t> down_tensors;
    for (const auto &site : build.sites)
      down_tensors.insert(site.down);
    const auto fill = [&](std::uint32_t id, bool random) {
      const auto *description = build.plan.program.tensor(id);
      dif::runtime::Tensor tensor;
      tensor.dtype = description->dtype;
      tensor.dims = description->dims;
      tensor.bytes.assign(description->byte_count(), static_cast<unsigned char>(0));
      if (random && description->dtype == dif::ir::DType::F32) {
        auto *values = reinterpret_cast<float *>(tensor.mutable_data());
        for (std::uint64_t index = 0U; index < description->element_count();
             ++index)
          values[index] = normal(generator);
      }
      return tensor;
    };
    for (const auto &binding : build.plan.bindings) {
      bindings.insert_or_assign(
          binding.parameter_input,
          fill(binding.parameter_input,
               down_tensors.contains(binding.parameter_input)));
      bindings.insert_or_assign(binding.first_moment_input,
                                fill(binding.first_moment_input, false));
      bindings.insert_or_assign(binding.second_moment_input,
                                fill(binding.second_moment_input, false));
    }

    // ---- the batch ------------------------------------------------------
    // Synthetic, and labelled as such. No dataset is wired yet, so this
    // exercises the step, not the data: a falling loss here says the
    // machinery works, and says nothing whatever about image quality.
    const auto random_tensor = [&](std::uint32_t id) {
      const auto *description = build.plan.program.tensor(id);
      dif::runtime::Tensor tensor;
      tensor.dtype = description->dtype;
      tensor.dims = description->dims;
      tensor.bytes.assign(description->byte_count(), static_cast<unsigned char>(0));
      std::normal_distribution<float> unit(0.0F, 1.0F);
      if (description->dtype == dif::ir::DType::BF16) {
        auto *values =
            reinterpret_cast<std::uint16_t *>(tensor.mutable_data());
        for (std::uint64_t index = 0U; index < description->element_count();
             ++index) {
          const float value = unit(generator);
          std::uint32_t bits = 0U;
          std::memcpy(&bits, &value, sizeof(bits));
          values[index] = static_cast<std::uint16_t>(bits >> 16U);
        }
      } else if (description->dtype == dif::ir::DType::F32) {
        auto *values = reinterpret_cast<float *>(tensor.mutable_data());
        for (std::uint64_t index = 0U; index < description->element_count();
             ++index)
          values[index] = unit(generator);
      }
      return tensor;
    };
    dif::runtime::TensorMap batch;
    batch.emplace(build.clean_latents, random_tensor(build.clean_latents));
    batch.emplace(build.noise, random_tensor(build.noise));
    batch.emplace(build.context, random_tensor(build.context));
    // A timestep in (0,1), and positions and validity that describe the
    // sequence the graph was built for.
    {
      const auto *description = build.plan.program.tensor(build.timestep);
      dif::runtime::Tensor timestep;
      timestep.dtype = description->dtype;
      timestep.dims = description->dims;
      timestep.bytes.assign(description->byte_count(), static_cast<unsigned char>(0));
      auto *values =
          reinterpret_cast<std::uint16_t *>(timestep.mutable_data());
      for (std::uint64_t index = 0U; index < description->element_count();
           ++index) {
        const float value = 0.5F;
        std::uint32_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        values[index] = static_cast<std::uint16_t>(bits >> 16U);
      }
      batch.emplace(build.timestep, std::move(timestep));
    }
    for (const auto &tensor : build.plan.program.tensors)
      if (tensor.has_role(dif::ir::TensorRole::Input) &&
          !bindings.contains(tensor.id) && !batch.contains(tensor.id) &&
          tensor.id != build.plan.step_input)
        batch.emplace(tensor.id, random_tensor(tensor.id));

    // ---- the run --------------------------------------------------------
    auto executor = backend == "cpu" ? dif::runtime::make_cpu_executor()
                                     : dif::runtime::make_cuda_executor();
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.trace_events = trace;
    options.profile_pipeline = trace;
    // One map, not a copy of one. The weights are eleven gigabytes.
    for (const auto &[id, tensor] : batch)
      bindings.insert_or_assign(id, tensor);
    const auto prepare_start = std::chrono::steady_clock::now();
    dif::training::TrainingSession session(build.plan, *executor,
                                           std::move(bindings), options);
    const auto prepare_stop = std::chrono::steady_clock::now();
    std::cout << "KREA2_PREPARE seconds="
              << std::chrono::duration<double>(prepare_stop - prepare_start)
                     .count()
              << " resident_state_bytes=" << session.persistent_state_bytes()
              << "\n";

    std::ofstream record(run.workspace / "training-report.jsonl",
                         std::ios::binary | std::ios::trunc);
    float first_loss = 0.0F;
    float last_loss = 0.0F;
    double total_milliseconds = 0.0;
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint64_t step = 0U; step < run.max_steps; ++step) {
      const auto result = session.step(batch);
      auto report = session.report(result);
      report.model = "krea2";
      report.checkpoint = run.checkpoint.string();
      report.physical_formats = build.resident_format;
      record << report.json() << "\n";
      record.flush();
      total_milliseconds += result.step_milliseconds;
      if (step == 0U)
        first_loss = result.loss;
      last_loss = result.loss;
      if (!result.operation_timings.empty()) {
        // Device time per opcode. This is the measurement that counts: the
        // trace records when the HOST submitted work, which for an
        // asynchronous launch says nothing about what the GPU spent.
        std::map<std::string, std::pair<double, std::uint64_t>> by_opcode;
        double total = 0.0;
        for (const auto &timing : result.operation_timings) {
          auto &entry =
              by_opcode[std::string(dif::ir::opcode_name(timing.opcode))];
          entry.first += timing.mean_milliseconds;
          ++entry.second;
          total += timing.mean_milliseconds;
        }
        std::vector<std::pair<std::string, std::pair<double, std::uint64_t>>>
            ranked(by_opcode.begin(), by_opcode.end());
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto &left, const auto &right) {
                    return left.second.first > right.second.first;
                  });
        std::cout << "KREA2_DEVICE_BY_OPCODE total_ms=" << total << "\n";
        for (std::size_t index = 0U;
             index < ranked.size() && index < 12U; ++index)
          std::cout << "  " << std::left << std::setw(42)
                    << ranked[index].first << " ms=" << std::setw(10)
                    << ranked[index].second.first
                    << " count=" << std::setw(6) << ranked[index].second.second
                    << " share=" << 100.0 * ranked[index].second.first / total
                    << "%\n";
        std::cout << std::flush;
      }
      if (!result.trace_events.empty()) {
        // Where the time actually went, by opcode. Guessing twice was
        // expensive; this is the measurement.
        std::map<std::string, std::pair<double, std::uint64_t>> by_opcode;
        double traced = 0.0;
        for (const auto &event : result.trace_events) {
          const auto elapsed = event.host_end_ms - event.host_start_ms;
          if (elapsed <= 0.0)
            continue;
          auto &entry = by_opcode[event.opcode.empty() ? event.category
                                                       : event.opcode];
          entry.first += elapsed;
          ++entry.second;
          traced += elapsed;
        }
        std::vector<std::pair<std::string, std::pair<double, std::uint64_t>>>
            ranked(by_opcode.begin(), by_opcode.end());
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto &left, const auto &right) {
                    return left.second.first > right.second.first;
                  });
        std::cout << "KREA2_HOT traced_ms=" << traced << "\n";
        for (std::size_t index = 0U;
             index < ranked.size() && index < 12U; ++index)
          std::cout << "  " << ranked[index].first << " ms="
                    << ranked[index].second.first
                    << " launches=" << ranked[index].second.second
                    << " share="
                    << 100.0 * ranked[index].second.first / traced << "%\n";
        std::cout << std::flush;
      }
      if (result.profile.enabled)
        std::cout << "KREA2_DEVICE operation_kernels_ms="
                  << result.profile.operation_kernel_milliseconds
                  << " attention_kernels_ms="
                  << result.profile.attention_kernel_milliseconds
                  << " non_kernel_timeline_ms="
                  << result.profile.non_kernel_device_timeline_milliseconds
                  << " resident_weight_bytes="
                  << result.profile.resident_weight_bytes
                  << " streamed_weight_bytes="
                  << result.profile.streamed_weight_bytes
                  << " streamed_h2d_ms="
                  << result.profile.streamed_h2d_milliseconds
                  << " streamed_host_stage_ms="
                  << result.profile.streamed_host_stage_milliseconds
                  << std::endl;
      if (result.phases)
        std::cout << "KREA2_PHASES forward_ms="
                  << result.phases->forward_milliseconds
                  << " backward_ms=" << result.phases->backward_milliseconds
                  << " optimizer_ms=" << result.phases->optimizer_milliseconds
                  << " transfer_ms=" << result.phases->transfer_milliseconds
                  << std::endl;
      if (step % 10U == 0U || step + 1U == run.max_steps)
        std::cout << "KREA2_STEP " << (step + 1U) << "/" << run.max_steps
                  << " loss=" << result.loss
                  << " ms=" << result.step_milliseconds
                  << " state_h2d=" << result.persistent_state_host_to_device_bytes
                  << " state_d2h=" << result.persistent_state_device_to_host_bytes
                  << std::endl;
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    const double wall =
        std::chrono::duration<double>(wall_stop - wall_start).count();
    std::cout << "KREA2_SMOKE steps=" << run.max_steps
              << " first_loss=" << first_loss << " final_loss=" << last_loss
              << " seconds_per_step=" << wall / static_cast<double>(run.max_steps)
              << " wall_seconds=" << wall
              << " data=synthetic\n";
  } catch (const std::exception &error) {
    std::cerr << "difkrea2train: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
