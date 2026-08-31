#include "dif/backend/plugin.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/optimizer.hpp"
#include "dif/opt/plan.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/tune/database.hpp"
#include "dif/weights/bundle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Binding = std::pair<std::uint32_t, std::filesystem::path>;

Binding binding(const std::string &text) {
  const auto split = text.find('=');
  if (split == std::string::npos)
    dif::fail("tensor binding must be ID=PATH");
  char *end = nullptr;
  const auto id = std::strtoul(text.substr(0, split).c_str(), &end, 10);
  if (!end || *end != '\0' || id == 0)
    dif::fail("tensor binding has invalid id");
  return {static_cast<std::uint32_t>(id), text.substr(split + 1)};
}

std::vector<std::uint64_t> unsigned_list(const std::string &text,
                                         const char *label) {
  if (text == "keep")
    return {0U};
  std::vector<std::uint64_t> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    char *end = nullptr;
    const auto value = std::strtoull(item.c_str(), &end, 10);
    if (!end || *end != '\0')
      dif::fail(std::string("invalid ") + label + " list");
    result.push_back(value);
  }
  if (result.empty())
    dif::fail(std::string("empty ") + label + " list");
  return result;
}

std::vector<std::uint64_t> math_modes(const std::string &text) {
  std::vector<std::uint64_t> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item == "strict")
      result.push_back(1U);
    else if (item == "tf32")
      result.push_back(2U);
    else if (item == "direct-int5")
      result.push_back(3U);
    else
      dif::fail("linear math list accepts strict, tf32, or direct-int5");
  }
  if (result.empty())
    dif::fail("empty linear math list");
  return result;
}

std::vector<std::uint64_t> attention_implementations(
    const std::string &text) {
  std::vector<std::uint64_t> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item == "generated")
      result.push_back(1U);
    else if (item == "cudnn")
      result.push_back(2U);
    else
      dif::fail("attention implementation list accepts generated or cudnn");
  }
  if (result.empty())
    dif::fail("empty attention implementation list");
  return result;
}

std::vector<dif::ir::DType> storage_dtypes(const std::string &text) {
  std::vector<dif::ir::DType> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item == "f32")
      result.push_back(dif::ir::DType::F32);
    else if (item == "bf16")
      result.push_back(dif::ir::DType::BF16);
    else if (item == "f16")
      result.push_back(dif::ir::DType::F16);
    else
      dif::fail("storage dtype list accepts f32, bf16, or f16");
  }
  if (result.empty())
    dif::fail("empty storage dtype list");
  return result;
}

struct Metrics {
  double max_abs{};
  double cosine{};
  double norm_ratio{};
  std::uint64_t nonfinite{};
};

Metrics compare(const dif::runtime::TensorMap &reference,
                const dif::runtime::RunResult &candidate) {
  long double dot = 0.0;
  long double norm_a = 0.0;
  long double norm_b = 0.0;
  double max_abs = 0.0;
  std::uint64_t nonfinite = 0;
  for (const auto &[id, expected_tensor] : reference) {
    const auto actual_it = candidate.outputs.find(id);
    if (actual_it == candidate.outputs.end())
      dif::fail("candidate omitted reference output tensor " + std::to_string(id));
    const auto expected = expected_tensor.f32();
    const auto actual = actual_it->second.f32();
    if (expected.size() != actual.size() ||
        expected_tensor.dims != actual_it->second.dims)
      dif::fail("candidate reference output shape mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (!std::isfinite(actual[i]))
        ++nonfinite;
      max_abs = std::max(max_abs,
                         std::abs(static_cast<double>(expected[i]) - actual[i]));
      dot += static_cast<long double>(expected[i]) * actual[i];
      norm_a += static_cast<long double>(expected[i]) * expected[i];
      norm_b += static_cast<long double>(actual[i]) * actual[i];
    }
  }
  const auto denominator = std::sqrt(norm_a * norm_b);
  const auto cosine = denominator == 0.0L
                          ? (norm_a == norm_b ? 1.0 : 0.0)
                          : static_cast<double>(dot / denominator);
  const auto ratio = norm_a == 0.0L
                         ? (norm_b == 0.0L
                                ? 1.0
                                : std::numeric_limits<double>::infinity())
                         : std::sqrt(static_cast<double>(norm_b / norm_a));
  return {max_abs, cosine, ratio, nonfinite};
}

bool has_linear(const dif::ir::Program &program) {
  return std::any_of(program.operations.begin(), program.operations.end(),
                     [](const dif::ir::Operation &operation) {
                       return operation.opcode == dif::ir::Opcode::Linear;
                     });
}

bool tf32_legal(const dif::ir::Program &program) {
  bool found = false;
  for (const auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Linear)
      continue;
    found = true;
    const auto *input = program.tensor(operation.inputs.front());
    if (!input || input->dtype != dif::ir::DType::F32)
      return false;
  }
  return found;
}

bool has_opcode(const dif::ir::Program &program, dif::ir::Opcode opcode) {
  return std::any_of(program.operations.begin(), program.operations.end(),
                     [opcode](const dif::ir::Operation &operation) {
                       return operation.opcode == opcode;
                     });
}

bool cudnn_attention_legal(const dif::ir::Program &program) {
  bool found = false;
  for (const auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Attention)
      continue;
    found = true;
    const auto *query = program.tensor(operation.inputs.front());
    if (!query || (query->dtype != dif::ir::DType::BF16 &&
                   query->dtype != dif::ir::DType::F16))
      return false;
  }
  return found;
}

double average(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U)
    return values[middle];
  return (values[middle - 1U] + values[middle]) / 2.0;
}

std::uint64_t mib(const std::string &text, const char *label) {
  const auto value = std::stoull(text);
  constexpr std::uint64_t scale = 1024ULL * 1024ULL;
  if (value > std::numeric_limits<std::uint64_t>::max() / scale)
    dif::fail(std::string(label) + " overflows bytes");
  return value * scale;
}

std::vector<std::uint64_t> mib_list(const std::string &text,
                                    const char *label) {
  auto values = unsigned_list(text, label);
  constexpr std::uint64_t scale = 1024ULL * 1024ULL;
  for (auto &value : values) {
    if (value > std::numeric_limits<std::uint64_t>::max() / scale)
      dif::fail(std::string(label) + " overflows bytes");
    value *= scale;
  }
  return values;
}

void refuse_overwrite(const std::filesystem::path &path) {
  if (path.empty())
    return;
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory)
    return;
  if (error)
    dif::fail("cannot inspect output path: " + error.message());
  if (status.type() != std::filesystem::file_type::not_found)
    dif::fail("refusing to overwrite existing output: " + path.string());
}

void usage() {
  std::cerr
      << "usage: diftune --program FILE --db FILE"
         " [--backend cpu|cuda] [--backend-plugin FILE.so]"
         " [--weight-bundle FILE.difbind] [--verify-shards]"
         " --input ID=FILE [--input ...]"
         " [--reference ID=FILE ...] [--blocks keep|64,128,256,512]"
         " [--block-scope all|pointwise]"
         " [--linear-math strict,tf32,direct-int5]"
         " [--attention-implementations generated,cudnn]"
         " [--cast-storage-dtypes f32,bf16,f16]"
         " [--warmups N] [--iterations N]"
         " [--linear-algorithms keep|0,1,2,...]"
         " [--split-residual-gate]"
         " [--prefetch-distances 0,1]"
         " [--weight-budgets-mib N,N] [--expected-evaluations N]"
         " [--mark-recompute-candidates]"
         " [--min-free-mib N] [--max-abs N] [--min-cos N]"
         " [--min-norm-ratio N] [--max-norm-ratio N]"
         " [--max-device-mib N] [--max-slowdown-ratio N]"
         " [--orders 1|2] [--candidate-limit N]"
         " [--winner-program FILE] [--winner-plan FILE]\n";
}

struct EvaluatedCandidate {
  const dif::opt::Candidate *candidate{};
  dif::tune::Measurement measurement;
  double minimum_norm_ratio{std::numeric_limits<double>::infinity()};
  double maximum_norm_ratio{-std::numeric_limits<double>::infinity()};
  bool runtime_failed{};
  bool has_native_iteration_samples{true};
};

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path program_path;
    std::filesystem::path database_path;
    std::string backend = "cuda";
    std::filesystem::path backend_plugin;
    std::filesystem::path weight_bundle_path;
    bool verify_shards = false;
    std::vector<Binding> input_paths;
    std::vector<Binding> reference_paths;
    std::vector<std::uint64_t> block_sizes = {64, 128, 256};
    std::vector<std::uint64_t> linear_modes = {1U};
    std::vector<std::uint64_t> attention_modes;
    std::vector<dif::ir::DType> cast_storage_modes;
    std::vector<std::uint64_t> linear_algorithms;
    std::vector<std::uint64_t> prefetch_distances;
    std::vector<std::uint64_t> weight_budgets;
    std::uint64_t expected_evaluations = 1U;
    bool split_residual_gate = false;
    bool mark_recompute_candidates = false;
    std::string block_scope = "all";
    std::filesystem::path winner_program_path;
    std::filesystem::path winner_plan_path;
    dif::runtime::RunOptions options;
    options.warmups = 2;
    options.iterations = 10;
    double max_abs_bar = 1.0e-4;
    double min_cos_bar = 0.999999;
    double min_norm_ratio_bar = 0.9999;
    double max_norm_ratio_bar = 1.0001;
    std::uint64_t maximum_device_bytes = 0U;
    double maximum_slowdown_ratio =
        std::numeric_limits<double>::infinity();
    std::uint32_t orders = 1U;
    std::size_t candidate_limit = 1024U;

    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--program" && i + 1 < argc)
        program_path = argv[++i];
      else if (option == "--db" && i + 1 < argc)
        database_path = argv[++i];
      else if (option == "--backend" && i + 1 < argc)
        backend = argv[++i];
      else if (option == "--backend-plugin" && i + 1 < argc)
        backend_plugin = argv[++i];
      else if (option == "--weight-bundle" && i + 1 < argc)
        weight_bundle_path = argv[++i];
      else if (option == "--verify-shards")
        verify_shards = true;
      else if (option == "--input" && i + 1 < argc)
        input_paths.push_back(binding(argv[++i]));
      else if (option == "--reference" && i + 1 < argc)
        reference_paths.push_back(binding(argv[++i]));
      else if (option == "--blocks" && i + 1 < argc)
        block_sizes = unsigned_list(argv[++i], "block-size");
      else if (option == "--block-scope" && i + 1 < argc)
        block_scope = argv[++i];
      else if (option == "--linear-math" && i + 1 < argc)
        linear_modes = math_modes(argv[++i]);
      else if (option == "--attention-implementations" && i + 1 < argc)
        attention_modes = attention_implementations(argv[++i]);
      else if (option == "--cast-storage-dtypes" && i + 1 < argc)
        cast_storage_modes = storage_dtypes(argv[++i]);
      else if (option == "--linear-algorithms" && i + 1 < argc)
        linear_algorithms = unsigned_list(argv[++i], "linear-algorithm");
      else if (option == "--split-residual-gate")
        split_residual_gate = true;
      else if (option == "--prefetch-distances" && i + 1 < argc)
        prefetch_distances = unsigned_list(argv[++i], "prefetch-distance");
      else if (option == "--weight-budgets-mib" && i + 1 < argc)
        weight_budgets = mib_list(argv[++i], "weight budget MiB");
      else if (option == "--expected-evaluations" && i + 1 < argc)
        expected_evaluations = std::stoull(argv[++i]);
      else if (option == "--mark-recompute-candidates")
        mark_recompute_candidates = true;
      else if (option == "--warmups" && i + 1 < argc)
        options.warmups = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (option == "--iterations" && i + 1 < argc)
        options.iterations = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (option == "--min-free-mib" && i + 1 < argc)
        options.minimum_free_bytes = mib(argv[++i], "minimum free MiB");
      else if (option == "--max-abs" && i + 1 < argc)
        max_abs_bar = std::stod(argv[++i]);
      else if (option == "--min-cos" && i + 1 < argc)
        min_cos_bar = std::stod(argv[++i]);
      else if (option == "--min-norm-ratio" && i + 1 < argc)
        min_norm_ratio_bar = std::stod(argv[++i]);
      else if (option == "--max-norm-ratio" && i + 1 < argc)
        max_norm_ratio_bar = std::stod(argv[++i]);
      else if (option == "--max-device-mib" && i + 1 < argc)
        maximum_device_bytes = mib(argv[++i], "maximum device MiB");
      else if (option == "--max-slowdown-ratio" && i + 1 < argc)
        maximum_slowdown_ratio = std::stod(argv[++i]);
      else if (option == "--orders" && i + 1 < argc)
        orders = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (option == "--candidate-limit" && i + 1 < argc)
        candidate_limit = static_cast<std::size_t>(std::stoull(argv[++i]));
      else if (option == "--winner-program" && i + 1 < argc)
        winner_program_path = argv[++i];
      else if (option == "--winner-plan" && i + 1 < argc)
        winner_plan_path = argv[++i];
      else {
        usage();
        return 2;
      }
    }
    if (program_path.empty() || database_path.empty() || input_paths.empty() ||
        min_norm_ratio_bar > max_norm_ratio_bar || max_abs_bar < 0.0 ||
        min_cos_bar > 1.0 || min_cos_bar < -1.0 ||
        maximum_slowdown_ratio < 1.0 || (orders != 1U && orders != 2U) ||
        candidate_limit == 0U || expected_evaluations == 0U ||
        (backend_plugin.empty() && backend != "cpu" && backend != "cuda") ||
        (block_scope != "all" && block_scope != "pointwise")) {
      usage();
      return 2;
    }
    refuse_overwrite(winner_program_path);
    refuse_overwrite(winner_plan_path);

    const auto base = dif::ir::read_file(program_path);
    dif::runtime::TensorMap inputs;
    if (!weight_bundle_path.empty()) {
      const auto bundle = dif::weights::read_weight_bundle(weight_bundle_path);
      inputs = dif::weights::load_weight_bundle(bundle, base, verify_shards);
    }
    for (const auto &[id, path] : input_paths)
      if (!inputs.emplace(id, dif::runtime::read_tensor(path)).second)
        dif::fail("manual input duplicates weight-bundle tensor " +
                  std::to_string(id));
    const auto base_hash = dif::hex_digest(dif::ir::fingerprint(base));

    dif::runtime::TensorMap reference;
    if (reference_paths.empty()) {
      auto cpu = dif::runtime::make_cpu_executor();
      auto reference_options = options;
      reference_options.warmups = 0;
      reference_options.iterations = 1;
      reference = cpu->run(base, inputs, reference_options).outputs;
    } else {
      for (const auto &[id, path] : reference_paths)
        reference.emplace(id, dif::runtime::read_tensor(path));
    }

    std::vector<std::unique_ptr<dif::opt::Pass>> owned_passes;
    std::vector<std::uint64_t> changed_blocks;
    std::copy_if(block_sizes.begin(), block_sizes.end(),
                 std::back_inserter(changed_blocks),
                 [](std::uint64_t value) { return value != 0U; });
    if (!changed_blocks.empty())
      if (block_scope == "pointwise") {
        owned_passes.push_back(
            std::make_unique<dif::opt::MatchingOperationU64AttributePass>(
                "schedule.pointwise-block-size",
                std::vector<dif::ir::Opcode>{
                    dif::ir::Opcode::Add,
                    dif::ir::Opcode::Multiply,
                    dif::ir::Opcode::AffineLastDim,
                    dif::ir::Opcode::BiasAdd,
                    dif::ir::Opcode::SwiGlu,
                    dif::ir::Opcode::ResidualGate,
                    dif::ir::Opcode::SiLU,
                    dif::ir::Opcode::Cast,
                    dif::ir::Opcode::Clamp,
                    dif::ir::Opcode::LinearBlend,
                    dif::ir::Opcode::FlowEulerStep},
                dif::ir::AttrKey::BlockSize, changed_blocks, true));
      } else {
        owned_passes.push_back(
            std::make_unique<dif::opt::MatchingOperationU64AttributePass>(
                "schedule.block-size", std::nullopt,
                dif::ir::AttrKey::BlockSize, changed_blocks, true));
      }

    if (has_linear(base)) {
      std::vector<std::uint64_t> legal_linear_modes;
      for (const auto mode : linear_modes) {
        if (mode != 2U || tf32_legal(base))
          legal_linear_modes.push_back(mode);
        else
          std::cout << "DISCOVERY SKIP transform=linear-tf32"
                       " reason=storage-dtype-not-f32\n";
      }
      if (!legal_linear_modes.empty())
        owned_passes.push_back(
            std::make_unique<dif::opt::MatchingOperationU64AttributePass>(
                "numeric.linear-implementation", dif::ir::Opcode::Linear,
                dif::ir::AttrKey::Implementation, legal_linear_modes, false));
      if (!linear_algorithms.empty())
        owned_passes.push_back(
            std::make_unique<dif::opt::MatchingOperationU64AttributePass>(
                "schedule.linear-algorithm", dif::ir::Opcode::Linear,
                dif::ir::AttrKey::Algorithm, linear_algorithms, false));
    }
    if (!attention_modes.empty() &&
        has_opcode(base, dif::ir::Opcode::Attention)) {
      std::vector<std::uint64_t> legal_attention_modes;
      for (const auto mode : attention_modes) {
        if (mode != 2U || cudnn_attention_legal(base))
          legal_attention_modes.push_back(mode);
        else
          std::cout << "DISCOVERY SKIP transform=attention-cudnn"
                       " reason=storage-dtype-not-bf16-or-f16\n";
      }
      if (!legal_attention_modes.empty())
        owned_passes.push_back(
            std::make_unique<dif::opt::MatchingOperationU64AttributePass>(
                "numeric.attention-implementation",
                dif::ir::Opcode::Attention,
                dif::ir::AttrKey::Implementation, legal_attention_modes,
                false));
    }
    if (!cast_storage_modes.empty())
      owned_passes.push_back(
          std::make_unique<dif::opt::CastStoragePrecisionPass>(
              cast_storage_modes));
    if (split_residual_gate)
      owned_passes.push_back(
          std::make_unique<dif::opt::SplitResidualGatePass>());
    if (!weight_budgets.empty()) {
      std::vector<dif::opt::WeightPlacementOptions> placements;
      placements.reserve(weight_budgets.size());
      for (const auto budget : weight_budgets)
        placements.push_back(
            {budget, expected_evaluations, 256U, 1U});
      owned_passes.push_back(
          std::make_unique<dif::opt::WeightPlacementPass>(
              std::move(placements)));
    }
    if (!prefetch_distances.empty())
      owned_passes.push_back(
          std::make_unique<dif::opt::StreamPrefetchPass>(
              prefetch_distances));
    if (mark_recompute_candidates)
      owned_passes.push_back(
          std::make_unique<dif::opt::RecomputeCandidatePass>());
    std::vector<const dif::opt::Pass *> passes;
    passes.reserve(owned_passes.size());
    for (const auto &pass : owned_passes)
      passes.push_back(pass.get());
    const auto candidates = dif::opt::compose_candidates(
        base, passes, {.maximum_candidates = candidate_limit});

    const auto make_executor = [&]() -> std::unique_ptr<dif::runtime::Executor> {
      if (!backend_plugin.empty())
        return dif::backend::make_plugin_executor(backend_plugin);
      if (backend == "cpu")
        return dif::runtime::make_cpu_executor();
      if (backend == "cuda")
        return dif::runtime::make_cuda_executor();
      dif::fail("unknown tuning backend: " + backend);
    };

    std::cout << "SEARCH base_program_fingerprint=" << base_hash
              << " candidates=" << candidates.size()
              << " orders=" << orders << " warmups=" << options.warmups
              << " iterations=" << options.iterations
              << " max_abs=" << max_abs_bar
              << " min_cos=" << min_cos_bar
              << " min_norm_ratio=" << min_norm_ratio_bar
              << " max_norm_ratio=" << max_norm_ratio_bar
              << " max_device_bytes=" << maximum_device_bytes
              << " max_slowdown_ratio=" << maximum_slowdown_ratio << "\n";

    std::vector<EvaluatedCandidate> evaluated;
    evaluated.reserve(candidates.size());
    const auto created_unix =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    for (const auto &candidate : candidates) {
      dif::ir::verify(candidate.program);
      EvaluatedCandidate value;
      value.candidate = &candidate;
      auto &measurement = value.measurement;
      measurement.candidate_program_hash =
          dif::hex_digest(candidate.program_fingerprint);
      measurement.candidate_hash =
          dif::hex_digest(candidate.candidate_fingerprint);
      measurement.program_hash = base_hash;
      measurement.recipe_hash =
          dif::hex_digest(candidate.recipe.fingerprint());
      measurement.recipe_text = candidate.recipe.canonical_text();
      measurement.planned_device_bytes =
          dif::compiler::plan_memory(candidate.program, 256U,
                                     candidate.policy.stream_prefetch_distance)
              .total_bytes;
      measurement.memory_limit_bytes = maximum_device_bytes;
      measurement.minimum_milliseconds =
          std::numeric_limits<double>::infinity();
      measurement.cosine_similarity =
          std::numeric_limits<double>::infinity();
      measurement.norm_ratio = 1.0;
      measurement.created_unix = created_unix;
      evaluated.push_back(std::move(value));
    }

    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0U);
    for (std::uint32_t round = 0U; round < orders; ++round) {
      if (round == 1U)
        std::reverse(order.begin(), order.end());
      for (const auto candidate_index : order) {
        auto &value = evaluated[candidate_index];
        if (value.runtime_failed)
          continue;
        try {
          auto executor = make_executor();
          auto candidate_options = options;
          candidate_options.overlap_streaming =
              value.candidate->policy.stream_prefetch_distance != 0U;
          const auto result = executor->run(value.candidate->program, inputs,
                                            candidate_options);
          const auto metrics = compare(reference, result);
          auto &measurement = value.measurement;
          if (!measurement.backend.empty() &&
              (measurement.backend != result.backend_name ||
               measurement.device != result.device_name))
            dif::fail("candidate backend/device changed between A/B orders");
          measurement.backend = result.backend_name;
          measurement.device = result.device_name;
          measurement.trial_mean_milliseconds.push_back(
              result.mean_milliseconds);
          if (result.iteration_milliseconds.empty()) {
            value.has_native_iteration_samples = false;
            measurement.iteration_milliseconds.push_back(
                result.mean_milliseconds);
          } else {
            measurement.iteration_milliseconds.insert(
                measurement.iteration_milliseconds.end(),
                result.iteration_milliseconds.begin(),
                result.iteration_milliseconds.end());
          }
          measurement.preparation_milliseconds +=
              result.preparation_milliseconds;
          measurement.minimum_milliseconds = std::min(
              measurement.minimum_milliseconds, result.minimum_milliseconds);
          measurement.maximum_milliseconds = std::max(
              measurement.maximum_milliseconds, result.maximum_milliseconds);
          measurement.max_absolute_error = std::max(
              measurement.max_absolute_error, metrics.max_abs);
          measurement.cosine_similarity = std::min(
              measurement.cosine_similarity, metrics.cosine);
          value.minimum_norm_ratio =
              std::min(value.minimum_norm_ratio, metrics.norm_ratio);
          value.maximum_norm_ratio =
              std::max(value.maximum_norm_ratio, metrics.norm_ratio);
          if (std::abs(metrics.norm_ratio - 1.0) >
              std::abs(measurement.norm_ratio - 1.0))
            measurement.norm_ratio = metrics.norm_ratio;
          measurement.nonfinite_count = std::max(
              measurement.nonfinite_count, metrics.nonfinite);
          measurement.measured_resident_bytes = std::max(
              measurement.measured_resident_bytes, result.resident_bytes);
        } catch (const std::exception &error) {
          value.runtime_failed = true;
          value.measurement.status = "rejected-runtime";
          value.measurement.rejection_reason = error.what();
          if (value.measurement.backend.empty())
            value.measurement.backend =
                backend_plugin.empty() ? backend : backend_plugin.string();
        }
      }
    }

    for (auto &value : evaluated) {
      auto &measurement = value.measurement;
      if (!measurement.trial_mean_milliseconds.empty()) {
        measurement.mean_milliseconds =
            average(measurement.trial_mean_milliseconds);
        measurement.preparation_milliseconds /=
            static_cast<double>(measurement.trial_mean_milliseconds.size());
      }
      measurement.objective_name = value.has_native_iteration_samples
                                       ? "median-iteration"
                                       : "median-trial-mean";
      measurement.objective_milliseconds =
          median(measurement.iteration_milliseconds);
      if (value.runtime_failed)
        continue;
      const auto numerical = measurement.nonfinite_count == 0U &&
                             measurement.max_absolute_error <= max_abs_bar &&
                             measurement.cosine_similarity >= min_cos_bar &&
                             value.minimum_norm_ratio >= min_norm_ratio_bar &&
                             value.maximum_norm_ratio <= max_norm_ratio_bar;
      if (!numerical) {
        measurement.status = "rejected-numerical";
        measurement.rejection_reason = "fixed numerical gate failed";
      } else if (maximum_device_bytes != 0U &&
                 std::max(measurement.planned_device_bytes,
                          measurement.measured_resident_bytes) >
                     maximum_device_bytes) {
        measurement.status = "rejected-memory";
        measurement.rejection_reason =
            "planned/measured device-memory gate failed";
      } else {
        measurement.status = "accepted";
      }
    }

    const auto base_benchmark_available =
        !evaluated.empty() && !evaluated.front().runtime_failed;
    const auto base_objective =
        evaluated.empty()
            ? 0.0
            : evaluated.front().measurement.objective_milliseconds;
    if (base_benchmark_available) {
      for (std::size_t index = 1U; index < evaluated.size(); ++index) {
        auto &measurement = evaluated[index].measurement;
        if (measurement.status == "accepted" &&
            measurement.objective_milliseconds >
                base_objective * maximum_slowdown_ratio) {
          measurement.status = "rejected-performance";
          measurement.rejection_reason = "fixed slowdown gate failed";
        }
      }
    }

    dif::tune::Database database(database_path);
    double best = std::numeric_limits<double>::infinity();
    std::optional<std::size_t> best_index;
    for (std::size_t index = 0U; index < evaluated.size(); ++index) {
      const auto &value = evaluated[index];
      database.record(value.measurement);
      const auto &measurement = value.measurement;
      std::cout << "CANDIDATE index=" << index
                << " candidate_hash=" << measurement.candidate_hash
                << " candidate_program_fingerprint="
                << measurement.candidate_program_hash
                << " recipe_fingerprint=" << measurement.recipe_hash
                << " transformations="
                << value.candidate->recipe.transformations.size()
                << " prefetch_distance="
                << value.candidate->policy.stream_prefetch_distance
                << " status=" << measurement.status
                << " rejection_reason="
                << (measurement.rejection_reason.empty()
                        ? "none"
                        : measurement.rejection_reason)
                << " mean_ms=" << measurement.mean_milliseconds
                << " objective=" << measurement.objective_name
                << " objective_ms=" << measurement.objective_milliseconds
                << " timing_samples="
                << measurement.iteration_milliseconds.size()
                << " min_ms=" << measurement.minimum_milliseconds
                << " max_ms=" << measurement.maximum_milliseconds
                << " preparation_ms="
                << measurement.preparation_milliseconds
                << " planned_device_bytes="
                << measurement.planned_device_bytes
                << " measured_resident_bytes="
                << measurement.measured_resident_bytes
                << " max_abs=" << measurement.max_absolute_error
                << " cosine=" << measurement.cosine_similarity
                << " norm_ratio=" << measurement.norm_ratio
                << " nonfinite=" << measurement.nonfinite_count << "\n";
      if (measurement.status == "accepted" &&
          measurement.objective_milliseconds < best) {
        best = measurement.objective_milliseconds;
        best_index = index;
      }
    }
    if (!best_index)
      dif::fail("no tuning candidate passed ordered admission");
    const auto &winner = candidates[*best_index];
    if (!winner_program_path.empty())
      dif::ir::write_file(winner.program, winner_program_path);
    if (!winner_plan_path.empty())
      dif::opt::write_plan(dif::opt::make_plan(base, winner),
                           winner_plan_path);
    std::cout << "BEST index=" << *best_index
              << " candidate_hash="
              << evaluated[*best_index].measurement.candidate_hash
              << " candidate_program_fingerprint="
              << dif::hex_digest(winner.program_fingerprint)
              << " recipe_fingerprint="
              << dif::hex_digest(winner.recipe.fingerprint())
              << " objective="
              << evaluated[*best_index].measurement.objective_name
              << " objective_ms=" << best
              << " base_objective_ms=" << base_objective
              << " mean_ms="
              << evaluated[*best_index].measurement.mean_milliseconds
              << " base_mean_ms="
              << evaluated.front().measurement.mean_milliseconds
              << " speedup=" << (base_objective / best)
              << " planned_memory_delta_bytes="
              << static_cast<std::int64_t>(
                     evaluated.front().measurement.planned_device_bytes) -
                     static_cast<std::int64_t>(
                         evaluated[*best_index]
                             .measurement.planned_device_bytes)
              << " winner_program="
              << (winner_program_path.empty() ? "none"
                                              : winner_program_path.string())
              << " winner_plan="
              << (winner_plan_path.empty() ? "none"
                                           : winner_plan_path.string())
              << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "diftune: " << error.what() << "\n";
    return 1;
  }
}
