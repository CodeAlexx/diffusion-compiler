// Reading a training run from files instead of from a rebuild.
//
// The claims under test are the ones that decide whether a run is
// reproducible: every parameter comes from the file, a missing one is an
// error that names it, and a dimension the file gets wrong is caught before
// step 0 rather than discovered as a shape mismatch or, worse, not
// discovered at all.

#include "dif/frontend/krea2_training.hpp"
#include "dif/support/json.hpp"
#include "dif/training/architecture.hpp"
#include "dif/training/config.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

// Refuses for the stated reason -- not merely refuses. A test that accepts
// any failure passes when the code fails for the wrong reason.
template <typename Body>
void expect_refused(Body &&body, const std::string &fragment,
                    const std::string &message) {
  try {
    body();
  } catch (const std::exception &error) {
    const std::string text = error.what();
    if (text.find(fragment) != std::string::npos)
      return;
    ++failures;
    std::cerr << "FAIL: " << message << " -- refused, but for the wrong "
              << "reason: " << text << "\n";
    return;
  }
  ++failures;
  std::cerr << "FAIL: " << message << " -- was accepted\n";
}

using dif::training::TrainingConfig;

TrainingConfig config_from(const std::string &text,
                           const std::string &name = "test-config.json") {
  return TrainingConfig::from_json(dif::json::parse(text), name);
}

const char *kKreaConfig = R"({
  "model_type": "krea2",
  "checkpoint": "models/krea2/raw.safetensors",
  "inner_dim": 6144, "in_channels": 64, "joint_attention_dim": 2560,
  "out_channels": 64, "num_double": 0, "num_single": 28,
  "num_heads": 48, "head_dim": 128, "mlp_hidden": 16384,
  "timestep_dim": 256, "resolution": "512", "batch_size": 1,
  "learning_rate": 0.0001, "lora_rank": 16, "lora_alpha": 16,
  "max_steps": 2000, "save_every": 500, "sample_every": 500,
  "quantized_resident": "fp8_e4m3",
  "optimizer": {"optimizer": "ADAMW", "weight_decay": 0.01, "beta1": 0.9}
})";

void a_run_comes_from_the_file() {
  const auto config = config_from(kKreaConfig);
  const auto run = dif::training::read_run(config);
  expect(run.model_type == "krea2", "the model type is read");
  expect(run.max_steps == 2000U, "the step count is read");
  expect(run.optimizer.learning_rate == 0.0001, "the learning rate is read");
  expect(run.optimizer.weight_decay == 0.01, "the optimizer block is read");
  expect(run.optimizer.beta2 == 0.999,
         "and an omitted optimizer field keeps its default");
  expect(run.lora_rank == 16U && run.lora_alpha == 16U, "the LoRA is read");
  expect(run.resident_format == "fp8_e4m3",
         "the resident format the config asks for is read, not assumed");
  // "resolution": "512" -- a number written as a string, which the reference
  // configs do.
  expect(run.resolutions.size() == 1U && run.resolutions.front() == 512U,
         "a numeric string is read as a number");
  // A relative path resolves against the config, so a tree can be moved.
  expect(run.checkpoint == std::filesystem::path("models/krea2/raw.safetensors"),
         "a relative path resolves against the config's directory");
}

void a_missing_parameter_names_itself() {
  expect_refused([] { config_from(R"({"model_type": "krea2"})").u64("max_steps"); },
                 "expected 'max_steps'",
                 "a missing key names the key");
  expect_refused([] { config_from(R"({"a": 1})", "krea2.json").u64("num_heads"); },
                 "krea2.json", "and names the file it was wanted from");
  // A default is for a field nobody wrote, not one written wrongly.
  expect_refused(
      [] { config_from(R"({"max_steps": "soon"})").u64_or("max_steps", 10U); },
      "is not a number", "a present-but-wrong value is not defaulted away");
  expect_refused([] { config_from(R"({"max_steps": -5})").u64("max_steps"); },
                 "non-negative whole number", "a negative count is refused");
  expect_refused([] { config_from(R"({"max_steps": 1.5})").u64("max_steps"); },
                 "non-negative whole number", "a fractional count is refused");
  expect_refused(
      [] {
        dif::training::read_run(config_from(
            R"({"model_type":"m","checkpoint":"c","max_steps":10,)"
            R"("learning_rate":1e-4,"optimizer":{"optimizer":"LION"}})"));
      },
      "is not implemented",
      "an optimizer nothing implements is refused, not silently swapped");
}

void an_unread_key_is_visible() {
  const auto config = config_from(R"({"model_type":"m","checkpoint":"c",)"
                                  R"("max_steps":10,"learning_rate":1e-4,)"
                                  R"("lora_rnak":16})");
  (void)dif::training::read_run(config);
  const auto unread = config.unread_keys();
  expect(std::find(unread.begin(), unread.end(), "lora_rnak") != unread.end(),
         "a misspelled knob that silently did nothing is reported");
  expect(std::find(unread.begin(), unread.end(), "max_steps") == unread.end(),
         "and a key that was read is not");
}

// The architecture check, against the checkpoint the config names. This is
// what makes reading dimensions from a file safe: the file is authoritative,
// and the checkpoint is what proves the file right.
void a_wrong_dimension_is_caught_before_step_zero() {
  const std::filesystem::path checkpoint =
      "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors";
  if (!std::filesystem::exists(checkpoint)) {
    std::cerr << "SKIP: the Krea 2 checkpoint is not present\n";
    return;
  }
  const auto file = dif::weights::read_safetensors(checkpoint);
  {
    const auto architecture =
        dif::frontend::krea2_training_architecture(config_from(kKreaConfig));
    const auto disagreements =
        dif::training::check_architecture(file, architecture.claims);
    for (const auto &problem : disagreements)
      std::cerr << "  unexpected: " << problem.key << " " << problem.detail
                << "\n";
    expect(disagreements.empty(),
           "the released config describes the released checkpoint");
    expect(architecture.claims.size() >= 10U,
           "and every dimension it states is actually checked");
  }
  // Now the teeth. Each of these is a plausible typo, and each has to be
  // caught by the checkpoint rather than by a shape mismatch much later.
  const std::pair<const char *, const char *> typos[] = {
      {"\"num_heads\": 48", "\"num_heads\": 32"},
      {"\"head_dim\": 128", "\"head_dim\": 64"},
      {"\"mlp_hidden\": 16384", "\"mlp_hidden\": 12288"},
      {"\"inner_dim\": 6144", "\"inner_dim\": 4096"},
      {"\"timestep_dim\": 256", "\"timestep_dim\": 320"},
      {"\"joint_attention_dim\": 2560", "\"joint_attention_dim\": 4096"},
      {"\"num_single\": 28", "\"num_single\": 32"}};
  for (const auto &[original, broken] : typos) {
    std::string text = kKreaConfig;
    const auto at = text.find(original);
    expect(at != std::string::npos,
           std::string("the fixture contains ") + original);
    if (at == std::string::npos)
      continue;
    text.replace(at, std::string(original).size(), broken);
    const auto architecture =
        dif::frontend::krea2_training_architecture(config_from(text));
    const auto disagreements =
        dif::training::check_architecture(file, architecture.claims);
    expect(!disagreements.empty(),
           std::string("a config claiming ") + broken + " is refused");
  }
  expect_refused(
      [&] {
        std::string text = kKreaConfig;
        const auto at = text.find("\"num_double\": 0");
        text.replace(at, 15U, "\"num_double\": 19");
        (void)dif::frontend::krea2_training_architecture(config_from(text));
      },
      "single-stream", "a double-stream config is refused by name");
  // The eight sites a LoRA adapts are the checkpoint's own per-block 2-D
  // tensors -- checked against the file, not against a list kept by hand.
  const auto sites = dif::frontend::krea2_lora_sites();
  expect(sites.size() == 8U, "there are eight LoRA sites in a block");
  for (const auto &site : sites)
    for (std::uint64_t block = 0U; block < 28U; ++block) {
      const auto name = "blocks." + std::to_string(block) + "." + site;
      const auto *entry = file.find(name);
      expect(entry != nullptr && entry->dims.size() == 2U,
             "the checkpoint has a 2-D tensor at " + name);
    }
}

void sample_prompts_are_the_second_file() {
  const auto path =
      std::filesystem::temp_directory_path() / "dif-sample-prompts.json";
  {
    std::ofstream out(path);
    out << R"({"defaults": {"sample_every": 500, "sample_at_start": true,
      "width": 1024, "height": 1024, "steps": 30, "cfg": 3.5, "seed": 42,
      "negative": "blurry"},
      "prompts": [{"id": "garden", "prompt": "a woman in a sunlit garden"},
                  {"prompt": "a cat", "seed": 7, "steps": 4}]})";
  }
  const auto prompts = dif::training::read_sample_prompts(path);
  expect(prompts.sample_every == 500U, "the sample cadence is read");
  expect(prompts.sample_at_start, "sampling before training is read");
  expect(prompts.prompts.size() == 2U, "both prompts are read");
  expect(prompts.prompts[0].width == 1024U && prompts.prompts[0].seed == 42U,
         "a prompt inherits the defaults");
  expect(prompts.prompts[0].negative == "blurry",
         "including the negative, which is a default worth having");
  expect(prompts.prompts[1].seed == 7U && prompts.prompts[1].steps == 4U,
         "and overrides the ones it states");
  expect(prompts.prompts[1].id == "a cat",
         "a prompt with no id is identified by its own text");
  std::filesystem::remove(path);
  expect_refused(
      [&] {
        const auto empty =
            std::filesystem::temp_directory_path() / "dif-empty-prompts.json";
        { std::ofstream out(empty); out << R"({"prompts": []})"; }
        try {
          dif::training::read_sample_prompts(empty);
        } catch (...) {
          std::filesystem::remove(empty);
          throw;
        }
      },
      "is empty", "a prompts file with no prompts is refused");
}

} // namespace

int main() {
  a_run_comes_from_the_file();
  a_missing_parameter_names_itself();
  an_unread_key_is_visible();
  a_wrong_dimension_is_caught_before_step_zero();
  sample_prompts_are_the_second_file();
  if (failures != 0) {
    std::cerr << failures << " training config failure(s)\n";
    return 1;
  }
  std::cout << "training config tests passed\n";
  return 0;
}
