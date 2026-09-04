#include "dif/training/config.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>

namespace dif::training {
namespace {

std::string read_whole_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open the configuration file " + path.string());
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

// A number that arrived as a string. The reference configs write
// "resolution": "512", and refusing that would be pedantry rather than
// safety -- but "512px" is a mistake, not a number, and is refused.
double number_from_text(const std::string &text, std::string_view key,
                        const std::filesystem::path &source) {
  try {
    std::size_t consumed = 0U;
    const double value = std::stod(text, &consumed);
    while (consumed < text.size() && std::isspace(static_cast<unsigned char>(
                                         text[consumed])))
      ++consumed;
    if (consumed == text.size())
      return value;
  } catch (const std::exception &) {
  }
  fail("'" + std::string(key) + "' in " + source.string() +
       " is not a number: \"" + text + "\"");
}

std::uint64_t checked_u64(double value, std::string_view key,
                          const std::filesystem::path &source) {
  if (!(value >= 0.0) || value != std::floor(value) ||
      value > 9.007199254740992e15)
    fail("'" + std::string(key) + "' in " + source.string() +
         " must be a non-negative whole number");
  return static_cast<std::uint64_t>(value);
}

} // namespace

TrainingConfig TrainingConfig::read(const std::filesystem::path &path) {
  return from_json(json::parse(read_whole_file(path)), path);
}

TrainingConfig TrainingConfig::from_json(json::Value document,
                                         std::filesystem::path source) {
  TrainingConfig config;
  if (!document.is_object())
    fail("the configuration in " + source.string() +
         " is not a JSON object");
  config.document_ = std::move(document);
  config.source_ = std::move(source);
  return config;
}

const json::Value *TrainingConfig::lookup(std::string_view key) const {
  read_.emplace_back(key);
  return document_.find(key);
}

const json::Value &TrainingConfig::require(std::string_view key) const {
  const auto *found = lookup(key);
  if (found == nullptr)
    fail("expected '" + std::string(key) + "' in " + source_.string());
  return *found;
}

bool TrainingConfig::has(std::string_view key) const {
  return lookup(key) != nullptr;
}

double TrainingConfig::f64(std::string_view key) const {
  const auto &value = require(key);
  if (std::holds_alternative<std::string>(value.storage))
    return number_from_text(value.string(), key, source_);
  if (!std::holds_alternative<double>(value.storage))
    fail("'" + std::string(key) + "' in " + source_.string() +
         " must be a number");
  return value.number();
}

std::uint64_t TrainingConfig::u64(std::string_view key) const {
  return checked_u64(f64(key), key, source_);
}

bool TrainingConfig::boolean(std::string_view key) const {
  const auto &value = require(key);
  if (std::holds_alternative<bool>(value.storage))
    return value.boolean();
  if (std::holds_alternative<double>(value.storage))
    return value.number() != 0.0;
  fail("'" + std::string(key) + "' in " + source_.string() +
       " must be true or false");
}

std::string TrainingConfig::text(std::string_view key) const {
  const auto &value = require(key);
  if (!std::holds_alternative<std::string>(value.storage))
    fail("'" + std::string(key) + "' in " + source_.string() +
         " must be a string");
  return value.string();
}

// The fallbacks deliberately do NOT swallow a type error. A default covers a
// field nobody wrote; a field somebody wrote wrongly still has to be found.
std::uint64_t TrainingConfig::u64_or(std::string_view key,
                                     std::uint64_t fallback) const {
  return has(key) ? u64(key) : fallback;
}

double TrainingConfig::f64_or(std::string_view key, double fallback) const {
  return has(key) ? f64(key) : fallback;
}

bool TrainingConfig::boolean_or(std::string_view key, bool fallback) const {
  return has(key) ? boolean(key) : fallback;
}

std::string TrainingConfig::text_or(std::string_view key,
                                    std::string fallback) const {
  return has(key) ? text(key) : std::move(fallback);
}

std::filesystem::path TrainingConfig::path(std::string_view key) const {
  std::filesystem::path value(text(key));
  if (value.is_absolute() || source_.empty())
    return value;
  return source_.parent_path() / value;
}

std::filesystem::path
TrainingConfig::path_or(std::string_view key,
                        std::filesystem::path fallback) const {
  return has(key) ? path(key) : std::move(fallback);
}

std::optional<TrainingConfig>
TrainingConfig::section(std::string_view key) const {
  const auto *found = lookup(key);
  if (found == nullptr)
    return std::nullopt;
  if (!found->is_object())
    fail("'" + std::string(key) + "' in " + source_.string() +
         " must be an object");
  return from_json(*found, source_);
}

std::vector<std::string> TrainingConfig::unread_keys() const {
  std::vector<std::string> unread;
  for (const auto &[key, value] : document_.object()) {
    (void)value;
    if (std::find(read_.begin(), read_.end(), key) == read_.end())
      unread.push_back(key);
  }
  return unread;
}

namespace {

// "512" or "512,768" or "512x768" -- one field, because a config that says
// one resolution should not have to say it as a list.
std::vector<std::uint64_t> parse_resolutions(const TrainingConfig &config) {
  std::vector<std::uint64_t> values;
  if (!config.has("resolution"))
    return values;
  const auto text = config.text_or("resolution", "");
  std::string current;
  const auto flush = [&] {
    if (current.empty())
      return;
    values.push_back(static_cast<std::uint64_t>(std::stoull(current)));
    current.clear();
  };
  for (const char character : text) {
    if (std::isdigit(static_cast<unsigned char>(character)))
      current.push_back(character);
    else
      flush();
  }
  flush();
  if (values.empty())
    fail("'resolution' in " + config.source().string() +
         " names no resolution");
  return values;
}

} // namespace

TrainingRun read_run(const TrainingConfig &config) {
  TrainingRun run;
  run.source = config.source();
  run.model_type = config.text("model_type");
  run.checkpoint = config.path("checkpoint");
  run.vae = config.path_or("vae", {});
  run.text_encoder = config.path_or("text_encoder", {});
  // The reference configs spell the same two things more than one way. Both
  // spellings are read, because a config that trains one trainer should not
  // have to be rewritten to train this one.
  run.dataset = config.has("dataset_path") ? config.path("dataset_path")
                                           : config.path_or("dataset_dir", {});
  run.dataset_cache = config.has("dataset_cache_dir")
                          ? config.path("dataset_cache_dir")
                          : config.path_or("train_cache_dir", {});
  run.workspace = config.path_or("workspace_dir", {});
  run.prompts_file = config.path_or("validation_prompts_file", {});
  run.prefix = config.text_or("save_filename_prefix", "adapter");

  run.max_steps = config.u64("max_steps");
  run.save_every = config.u64_or("save_every", 0U);
  run.sample_every = config.u64_or("sample_every", 0U);
  run.batch = config.u64_or("batch_size", 1U);
  run.gradient_accumulation =
      config.u64_or("gradient_accumulation_steps", 1U);
  run.seed = config.u64_or("seed", 42U);
  run.resolutions = parse_resolutions(config);

  run.optimizer.learning_rate = config.f64("learning_rate");
  run.schedule = config.text_or("learning_rate_scheduler", "constant");
  run.warmup_steps = config.u64_or("learning_rate_warmup_steps", 0U);
  run.max_grad_norm = config.f64_or("max_grad_norm", 0.0);
  run.caption_dropout = config.f64_or("caption_dropout_prob", 0.0);
  run.timestep_shift = config.f64_or("timestep_shift", 1.0);
  if (const auto optimizer = config.section("optimizer")) {
    run.optimizer_name = optimizer->text_or("optimizer", "ADAMW");
    run.optimizer.beta1 = optimizer->f64_or("beta1", run.optimizer.beta1);
    run.optimizer.beta2 = optimizer->f64_or("beta2", run.optimizer.beta2);
    run.optimizer.epsilon = optimizer->f64_or("eps", run.optimizer.epsilon);
    run.optimizer.weight_decay =
        optimizer->f64_or("weight_decay", run.optimizer.weight_decay);
  } else {
    run.optimizer_name = "ADAMW";
  }

  run.network_algorithm =
      config.text_or("network_algorithm", config.text_or("training_method",
                                                         "lora"));
  run.lora_rank = config.u64_or("lora_rank", 0U);
  // An alpha nobody named is the rank, which makes the scale 1.
  run.lora_alpha = config.u64_or("lora_alpha", run.lora_rank);
  run.resident_format = config.text_or("quantized_resident", "");

  if (run.max_steps == 0U)
    fail("'max_steps' in " + run.source.string() + " trains nothing");
  if (run.optimizer_name != "ADAMW" && run.optimizer_name != "adamw")
    fail("optimizer '" + run.optimizer_name + "' in " + run.source.string() +
         " is not implemented; this trainer applies AdamW");
  return run;
}

SamplePrompts read_sample_prompts(const std::filesystem::path &path) {
  const auto config = TrainingConfig::read(path);
  SamplePrompts prompts;
  prompts.source = path;

  SamplePrompt defaults;
  defaults.width = 1024U;
  defaults.height = 1024U;
  defaults.steps = 30U;
  defaults.guidance = 3.5;
  defaults.seed = 42U;
  if (const auto section = config.section("defaults")) {
    prompts.sample_every = section->u64_or("sample_every", 0U);
    prompts.sample_at_start = section->boolean_or("sample_at_start", false);
    defaults.width = section->u64_or("width", defaults.width);
    defaults.height = section->u64_or("height", defaults.height);
    defaults.steps = section->u64_or("steps", defaults.steps);
    defaults.guidance = section->f64_or("cfg", defaults.guidance);
    defaults.seed = section->u64_or("seed", defaults.seed);
    defaults.negative = section->text_or("negative", "");
  }

  const auto *list = config.document().find("prompts");
  if (list == nullptr || !list->is_array())
    fail("expected a 'prompts' array in " + path.string());
  for (const auto &entry : list->array()) {
    if (!entry.is_object())
      fail("every entry of 'prompts' in " + path.string() +
           " must be an object");
    const auto item = TrainingConfig::from_json(entry, path);
    SamplePrompt prompt = defaults;
    prompt.prompt = item.text("prompt");
    prompt.id = item.text_or("id", prompt.prompt.substr(0U, 32U));
    prompt.negative = item.text_or("negative", prompt.negative);
    prompt.width = item.u64_or("width", prompt.width);
    prompt.height = item.u64_or("height", prompt.height);
    prompt.steps = item.u64_or("steps", prompt.steps);
    prompt.guidance = item.f64_or("cfg", prompt.guidance);
    prompt.seed = item.u64_or("seed", prompt.seed);
    prompts.prompts.push_back(std::move(prompt));
  }
  if (prompts.prompts.empty())
    fail("'prompts' in " + path.string() + " is empty");
  return prompts;
}

} // namespace dif::training
