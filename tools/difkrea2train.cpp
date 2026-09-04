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

#include "dif/frontend/krea2_training.hpp"
#include "dif/support/error.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/target/profile.hpp"
#include "dif/training/config.hpp"
#include "dif/training/memory.hpp"
#include "dif/training/report.hpp"
#include "dif/training/session.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    const auto build = dif::frontend::build_krea2_training(config);
    report_cost(build, run, backend);

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
    dif::fail("training steps are not wired to a dataset yet; use "
              "--plan-only, which reports exactly what a step would cost");
  } catch (const std::exception &error) {
    std::cerr << "difkrea2train: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
