#pragma once

#include "dif/support/json.hpp"
#include "dif/training/step.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dif::training {

// A training run described by a file, not by a rebuild.
//
// Two files, because they change on different schedules: the run's parameters
// and the prompts it samples. Neither is a header, so changing a learning
// rate does not recompile a compiler.
//
// This is deliberately generic. difcore parses the fields EVERY trainer has
// and nothing else; a model's architecture is read by that model's frontend
// through the same accessors, because difcore does not know what
// "joint_attention_dim" means and should not pretend to. The next model adds
// a frontend reader, not a config class.
class TrainingConfig {
public:
  static TrainingConfig read(const std::filesystem::path &path);
  // For a config assembled in memory, and for tests.
  static TrainingConfig from_json(json::Value document,
                                  std::filesystem::path source = {});

  // Typed, fail-closed. A missing key names the key AND the file it was
  // wanted from: the most common configuration mistake is a typo that nobody
  // sees until step 0, and "expected 'num_heads' in krea2.json" ends that
  // hunt immediately. A number that arrived as a string is accepted -- the
  // reference configs write "resolution": "512" -- but a value that is
  // neither is an error rather than a silent zero.
  std::uint64_t u64(std::string_view key) const;
  double f64(std::string_view key) const;
  bool boolean(std::string_view key) const;
  std::string text(std::string_view key) const;

  // The same with a fallback, for a knob that has a defensible default. A key
  // that is PRESENT but the wrong type is still an error: a default is for a
  // field nobody wrote, not for one somebody wrote wrongly.
  std::uint64_t u64_or(std::string_view key, std::uint64_t fallback) const;
  double f64_or(std::string_view key, double fallback) const;
  bool boolean_or(std::string_view key, bool fallback) const;
  std::string text_or(std::string_view key, std::string fallback) const;

  // A relative path resolves against the config's own directory, so a config
  // can be moved with the tree it describes. An absolute one is left alone.
  std::filesystem::path path(std::string_view key) const;
  std::filesystem::path path_or(std::string_view key,
                                std::filesystem::path fallback) const;

  bool has(std::string_view key) const;
  // A nested object, for blocks like "optimizer". Fails closed if the key
  // exists but is not an object.
  std::optional<TrainingConfig> section(std::string_view key) const;

  const std::filesystem::path &source() const { return source_; }
  const json::Value &document() const { return document_; }

  // Every key the config carries that nothing has read. Not an error -- a
  // config may legitimately carry fields for another tool -- but a trainer
  // that reports them catches the misspelled knob that silently did nothing.
  std::vector<std::string> unread_keys() const;

private:
  const json::Value *lookup(std::string_view key) const;
  const json::Value &require(std::string_view key) const;

  json::Value document_;
  std::filesystem::path source_;
  // Which keys were asked for. Mutable because reading a config is a const
  // operation from every caller's point of view.
  mutable std::vector<std::string> read_;
};

// What every trainer needs, whatever it trains.
//
// The architecture is NOT here. A trainer reads its dims from the same
// TrainingConfig through its own frontend, which is the only thing that knows
// what its dims are called or which of them the checkpoint must agree with.
struct TrainingRun {
  std::string model_type;
  // Where the weights are. `checkpoint` is what trains AND what samples: a
  // model shipped as a base and a distilled variant trains on the base, and a
  // sampler that quietly used the other one would report quality that has
  // nothing to do with what was trained.
  std::filesystem::path checkpoint;
  std::filesystem::path vae;
  std::filesystem::path text_encoder;
  std::filesystem::path dataset;
  std::filesystem::path dataset_cache;
  // Everything a run produces -- adapters, samples, checkpoints, the step
  // record -- goes here, and nothing goes anywhere else.
  std::filesystem::path workspace;
  std::filesystem::path prompts_file;
  std::string prefix;

  std::uint64_t max_steps{};
  std::uint64_t save_every{};
  std::uint64_t sample_every{};
  std::uint64_t batch{1U};
  std::uint64_t gradient_accumulation{1U};
  std::uint64_t seed{};
  std::vector<std::uint64_t> resolutions;

  OptimizerHyperparameters optimizer;
  std::string optimizer_name;
  std::string schedule;
  std::uint64_t warmup_steps{};
  double max_grad_norm{};
  double caption_dropout{};
  double timestep_shift{1.0};

  std::string network_algorithm;
  std::uint64_t lora_rank{};
  std::uint64_t lora_alpha{};
  // The physical format the FROZEN base is expected to be resident in. The
  // reference trainer names one here and then dequantizes it to bf16 before
  // every GEMM of every step, which is where its 18x went. Naming it is not
  // the same as consuming it, so a trainer that cannot feed this format to a
  // GEMM directly must say so rather than quietly dequantize.
  std::string resident_format;

  // Where each of the above came from, for the run's own record.
  std::filesystem::path source;
};

TrainingRun read_run(const TrainingConfig &config);

// One prompt to sample, and the defaults it inherits. The second file.
struct SamplePrompt {
  std::string id;
  std::string prompt;
  std::string negative;
  std::uint64_t width{};
  std::uint64_t height{};
  std::uint64_t steps{};
  std::uint64_t seed{};
  double guidance{};
};

struct SamplePrompts {
  std::uint64_t sample_every{};
  bool sample_at_start{};
  std::vector<SamplePrompt> prompts;
  std::filesystem::path source;
};

SamplePrompts read_sample_prompts(const std::filesystem::path &path);

} // namespace dif::training
