// Krea 2, composed as a training step out of generic parts.
//
// This is the test the whole foundation exists to pass. Nothing here builds
// a Krea trainer: it reads a config, asks the frontend for the same forward
// graph that samples, and hands it to difcore's low-rank adaptation,
// difcore's flow-matching objective and difcore's optimizer. What is checked
// is that the result is a training step -- 224 adapted sites, one loss, a
// backward pass, an AdamW update per adapter, and state that stays on the
// device -- and that the numbers are the ones the checkpoint implies rather
// than the ones a comment claims.

#include "dif/frontend/krea2_training.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/json.hpp"
#include "dif/training/memory.hpp"

#include <filesystem>
#include <iostream>
#include <set>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

const char *kConfig = R"({
  "model_type": "krea2",
  "checkpoint": "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors",
  "inner_dim": 6144, "in_channels": 64, "joint_attention_dim": 2560,
  "out_channels": 64, "num_double": 0, "num_single": 28,
  "num_heads": 48, "head_dim": 128, "mlp_hidden": 16384,
  "timestep_dim": 256, "resolution": "512", "batch_size": 1,
  "learning_rate": 0.0001, "lora_rank": 16, "lora_alpha": 16,
  "max_steps": 10, "optimizer": {"optimizer": "ADAMW"}})";

dif::training::TrainingConfig config_from(std::string text) {
  return dif::training::TrainingConfig::from_json(dif::json::parse(text),
                                                  "krea2-test.json");
}

void krea_composes_into_one_training_step() {
  const auto build = dif::frontend::build_krea2_training(config_from(kConfig));

  // 28 blocks, 8 sites each. Both numbers come from the checkpoint: 28 is
  // how many blocks it has, and 8 is how many 2-D tensors each block holds.
  expect(build.sites.size() == 224U,
         "a LoRA adapts 224 sites: eight in each of 28 blocks");
  expect(build.site_names.size() == build.sites.size(),
         "and every site knows which checkpoint tensor it adapts");
  // Two tensors per site, and the optimizer binds each of them.
  expect(build.plan.bindings.size() == 448U,
         "which is 448 adapter tensors the optimizer updates");

  std::set<std::uint32_t> adapted;
  for (const auto &site : build.sites) {
    expect(adapted.insert(site.operation).second,
           "no site is adapted twice");
    expect(site.down != 0U && site.up != 0U && site.down != site.up,
           "each site has its own two adapters");
  }

  // The step is one program: forward, backward and the optimizer, in that
  // order, with nothing between them.
  expect(build.plan.forward_operations > 0U,
         "the step has a forward pass");
  expect(build.plan.optimizer_operations > build.plan.forward_operations,
         "and a backward pass before its optimizer");
  expect(build.plan.program.operations.size() >
             build.plan.optimizer_operations,
         "and an optimizer after that");

  // Every adapter carries state that stays on the device between steps: the
  // parameter and its two moments. Nothing is copied to a host.
  expect(build.plan.persistent_state().size() == 448U * 3U,
         "each adapter carries its parameter and both moments resident");

  // The frozen base is named, all of it. A trainer that guesses this list
  // trains against zeros and reports a falling loss while doing it. The
  // denoiser is 28 blocks of 13 tensors plus the embedding and projection
  // towers; the text fusion tower is a separate graph and not counted here.
  std::cout << "  frozen tensors named: " << build.frozen.size() << "\n";
  expect(build.frozen.size() >= 28U * 13U,
         "every checkpoint tensor the graph needs is named");
  std::set<std::string> named;
  for (const auto &[id, name] : build.frozen)
    named.insert(name);
  for (const auto &site : build.site_names)
    expect(named.contains(site),
           "the frozen weight behind " + site + " is named for the loader");

  // The objective's inputs are what a step supplies, and the noisy sample is
  // not among them: it is produced inside the program.
  const auto *clean = build.plan.program.tensor(build.clean_latents);
  const auto *noise = build.plan.program.tensor(build.noise);
  expect(clean != nullptr && noise != nullptr &&
             clean->dims == noise->dims,
         "a step supplies a clean latent and noise of the same shape");
  expect(build.timestep != 0U, "and one timestep");
  const auto *loss = build.plan.program.tensor(build.loss);
  expect(loss != nullptr && loss->element_count() == 1U,
         "and gets one scalar loss back");

  dif::ir::verify(build.plan.program);
}

// What it costs, reported rather than assumed. A LoRA's whole point is that
// the optimizer state is small; if it is not, something is training that
// should not be.
void the_cost_is_the_cost_of_a_lora() {
  const auto build = dif::frontend::build_krea2_training(config_from(kConfig));
  const auto memory = dif::training::analyze_memory(build.plan);
  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::cout << "  frozen " << memory.frozen_weights / gib << " GiB, trainable "
            << memory.trainable_weights / gib << " GiB, optimizer "
            << memory.optimizer_state / gib << " GiB, saved activations "
            << memory.saved_activations / gib << " GiB, planned "
            << memory.planned_bytes / gib << " GiB\n";
  // Parameters, not bytes: the base is BF16 and the adapters are F32, so
  // comparing bytes would understate the ratio by two.
  std::uint64_t trainable_parameters = 0U;
  for (const auto &binding : build.plan.bindings)
    trainable_parameters +=
        build.plan.program.tensor(binding.parameter_input)->element_count();
  std::uint64_t frozen_parameters = 0U;
  for (const auto &[id, name] : build.frozen)
    frozen_parameters += build.plan.program.tensor(id)->element_count();
  std::cout << "  trainable " << trainable_parameters << " of "
            << frozen_parameters << " parameters ("
            << 100.0 * static_cast<double>(trainable_parameters) /
                   static_cast<double>(frozen_parameters)
            << "%)\n";
  expect(trainable_parameters * 20U < frozen_parameters,
         "a rank-16 LoRA trains a small fraction of the model");
  // The moments are F32 and there are two, so the optimizer state is four
  // times the adapters' F32 bytes... which are the adapters themselves.
  expect(memory.optimizer_state >= memory.trainable_weights * 2U,
         "the optimizer holds two moments per adapter");
  expect(memory.live_peak_bytes <= memory.planned_bytes,
         "and the plan reserves at least what is live at once");
}

// The config is authoritative, and wrong values do not build a graph.
void a_wrong_config_never_builds_a_graph() {
  std::string text = kConfig;
  const auto at = text.find("\"num_heads\": 48");
  text.replace(at, 15U, "\"num_heads\": 24");
  try {
    (void)dif::frontend::build_krea2_training(config_from(text));
  } catch (const std::exception &error) {
    const std::string what = error.what();
    expect(what.find("num_heads") != std::string::npos,
           "a wrong head count is refused by name");
    return;
  }
  ++failures;
  std::cerr << "FAIL: a config claiming 24 heads built a graph\n";
}

} // namespace

int main() {
  if (!std::filesystem::exists(
          "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors")) {
    std::cout << "SKIP: the Krea 2 checkpoint is not present\n";
    return 0;
  }
  krea_composes_into_one_training_step();
  the_cost_is_the_cost_of_a_lora();
  a_wrong_config_never_builds_a_graph();
  if (failures != 0) {
    std::cerr << failures << " Krea 2 training failure(s)\n";
    return 1;
  }
  std::cout << "krea2 training tests passed\n";
  return 0;
}
