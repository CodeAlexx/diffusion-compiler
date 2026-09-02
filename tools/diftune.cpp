#include "dif/ir/codec.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/tune/database.hpp"
#include "dif/telemetry/schema.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    else
      dif::fail("linear math list accepts strict or tf32");
  }
  if (result.empty())
    dif::fail("empty linear math list");
  return result;
}

const char *math_name(std::uint64_t mode) { return mode == 2U ? "tf32" : "strict"; }

std::string block_name(std::uint64_t block) {
  return block == 0U ? "keep" : std::to_string(block);
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

void mutate_block_size(dif::ir::Program &program, std::uint64_t block_size) {
  for (auto &operation : program.operations) {
    auto *attribute = const_cast<dif::ir::Attribute *>(
        operation.find(dif::ir::AttrKey::BlockSize));
    if (attribute)
      *attribute = dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, block_size);
  }
}

void mutate_linear_math(dif::ir::Program &program, std::uint64_t mode) {
  for (auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Linear)
      continue;
    auto *attribute = const_cast<dif::ir::Attribute *>(
        operation.find(dif::ir::AttrKey::Implementation));
    if (attribute)
      *attribute =
          dif::ir::Attribute::u64(dif::ir::AttrKey::Implementation, mode);
    else
      operation.attributes.push_back(
          dif::ir::Attribute::u64(dif::ir::AttrKey::Implementation, mode));
  }
}

bool has_linear(const dif::ir::Program &program) {
  return std::any_of(program.operations.begin(), program.operations.end(),
                     [](const dif::ir::Operation &operation) {
                       return operation.opcode == dif::ir::Opcode::Linear;
                     });
}

void usage() {
  std::cerr
      << "usage: diftune --program FILE --db FILE --input ID=FILE [--input ...]"
         " [--reference ID=FILE ...] [--blocks keep|64,128,256,512]"
         " [--linear-math strict,tf32] [--warmups N] [--iterations N]"
         " [--min-free-mib N] [--max-abs N] [--min-cos N]"
         " [--min-norm-ratio N] [--max-norm-ratio N]"
         " [--json] [--report FILE]\n"
         "Order per candidate: verify -> execute -> numerical admission ->"
         " timing; only admitted candidates rank on time.\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path program_path;
    std::filesystem::path database_path;
    bool json_output = false;
    std::filesystem::path report_path;
    std::vector<Binding> input_paths;
    std::vector<Binding> reference_paths;
    std::vector<std::uint64_t> block_sizes = {64, 128, 256, 512};
    std::vector<std::uint64_t> linear_modes = {1U};
    dif::runtime::RunOptions options;
    options.warmups = 2;
    options.iterations = 10;
    double max_abs_bar = 1.0e-4;
    double min_cos_bar = 0.999999;
    double min_norm_ratio_bar = 0.9999;
    double max_norm_ratio_bar = 1.0001;

    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--program" && i + 1 < argc)
        program_path = argv[++i];
      else if (option == "--db" && i + 1 < argc)
        database_path = argv[++i];
      else if (option == "--input" && i + 1 < argc)
        input_paths.push_back(binding(argv[++i]));
      else if (option == "--reference" && i + 1 < argc)
        reference_paths.push_back(binding(argv[++i]));
      else if (option == "--blocks" && i + 1 < argc)
        block_sizes = unsigned_list(argv[++i], "block-size");
      else if (option == "--linear-math" && i + 1 < argc)
        linear_modes = math_modes(argv[++i]);
      else if (option == "--warmups" && i + 1 < argc)
        options.warmups = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (option == "--iterations" && i + 1 < argc)
        options.iterations = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (option == "--min-free-mib" && i + 1 < argc)
        options.minimum_free_bytes = std::stoull(argv[++i]) * 1024ULL * 1024ULL;
      else if (option == "--max-abs" && i + 1 < argc)
        max_abs_bar = std::stod(argv[++i]);
      else if (option == "--min-cos" && i + 1 < argc)
        min_cos_bar = std::stod(argv[++i]);
      else if (option == "--min-norm-ratio" && i + 1 < argc)
        min_norm_ratio_bar = std::stod(argv[++i]);
      else if (option == "--max-norm-ratio" && i + 1 < argc)
        max_norm_ratio_bar = std::stod(argv[++i]);
      else if (option == "--json")
        json_output = true;
      else if (option == "--report" && i + 1 < argc)
        report_path = argv[++i];
      else {
        usage();
        return 2;
      }
    }
    if (program_path.empty() || database_path.empty() || input_paths.empty() ||
        min_norm_ratio_bar > max_norm_ratio_bar) {
      usage();
      return 2;
    }

    dif::runtime::TensorMap inputs;
    for (const auto &[id, path] : input_paths)
      inputs.emplace(id, dif::runtime::read_tensor(path));
    const auto base = dif::ir::read_file(program_path);
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

    if (!has_linear(base))
      linear_modes = {1U};
    auto cuda = dif::runtime::make_cuda_executor();
    dif::tune::Database database(database_path);
    double best = std::numeric_limits<double>::infinity();
    std::string best_hash;
    dif::telemetry::Array candidate_entries;

    for (const auto block_size : block_sizes) {
      for (const auto math_mode : linear_modes) {
        auto candidate_program = base;
        if (block_size != 0U)
          mutate_block_size(candidate_program, block_size);
        mutate_linear_math(candidate_program, math_mode);
        const auto candidate_hash =
            dif::hex_digest(dif::ir::fingerprint(candidate_program));
        const auto result = cuda->run(candidate_program, inputs, options);
        const auto metrics = compare(reference, result);
        const bool accepted = metrics.nonfinite == 0U &&
                              metrics.max_abs <= max_abs_bar &&
                              metrics.cosine >= min_cos_bar &&
                              metrics.norm_ratio >= min_norm_ratio_bar &&
                              metrics.norm_ratio <= max_norm_ratio_bar;
        dif::tune::Measurement measurement;
        measurement.candidate_hash = candidate_hash;
        measurement.program_hash = base_hash;
        measurement.backend = result.backend_name;
        measurement.device = result.device_name;
        measurement.mean_milliseconds = result.mean_milliseconds;
        measurement.minimum_milliseconds = result.minimum_milliseconds;
        measurement.maximum_milliseconds = result.maximum_milliseconds;
        measurement.max_absolute_error = metrics.max_abs;
        measurement.cosine_similarity = metrics.cosine;
        measurement.norm_ratio = metrics.norm_ratio;
        measurement.nonfinite_count = metrics.nonfinite;
        measurement.status = accepted ? "accepted" : "rejected";
        measurement.created_unix =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        database.record(measurement);
        if (!json_output)
          std::cout << "CANDIDATE block=" << block_name(block_size)
                  << " linear_math=" << math_name(math_mode)
                  << " hash=" << candidate_hash << " status=" << measurement.status
                  << " mean_ms=" << result.mean_milliseconds
                  << " min_ms=" << result.minimum_milliseconds
                  << " max_abs=" << metrics.max_abs
                  << " cosine=" << metrics.cosine
                  << " norm_ratio=" << metrics.norm_ratio
                  << " nonfinite=" << metrics.nonfinite << "\n";
        if (accepted && result.mean_milliseconds < best) {
          best = result.mean_milliseconds;
          best_hash = candidate_hash;
        }
        dif::telemetry::Object entry;
        entry.set("block_size", block_size);
        entry.set("linear_math", math_name(math_mode));
        entry.set("candidate_fingerprint", candidate_hash);
        entry.set("status", measurement.status);
        entry.set("mean_ms", result.mean_milliseconds);
        entry.set("min_ms", result.minimum_milliseconds);
        entry.set("max_ms", result.maximum_milliseconds);
        entry.set("preparation_ms", result.preparation_milliseconds);
        entry.set("max_absolute_error", metrics.max_abs);
        entry.set("cosine", metrics.cosine);
        entry.set("norm_ratio", metrics.norm_ratio);
        entry.set("nonfinite", metrics.nonfinite);
        candidate_entries.push_back(std::move(entry));
      }
    }
    if (json_output || !report_path.empty()) {
      auto document = dif::telemetry::make_document("tune-report");
      document.set("program_fingerprint", base_hash);
      document.set("backend", "cuda");
      document.set("selection_order",
                   "verify -> execute -> numerical admission -> timing");
      dif::telemetry::Object bars;
      bars.set("max_absolute", max_abs_bar);
      bars.set("min_cosine", min_cos_bar);
      bars.set("min_norm_ratio", min_norm_ratio_bar);
      bars.set("max_norm_ratio", max_norm_ratio_bar);
      document.set("bars", std::move(bars));
      document.set("candidates", candidate_entries);
      document.set("best_candidate_fingerprint",
                   best_hash.empty() ? dif::telemetry::Value(nullptr)
                                     : dif::telemetry::Value(best_hash));
      document.set("best_mean_ms", best_hash.empty()
                                       ? dif::telemetry::Value(nullptr)
                                       : dif::telemetry::Value(best));
      document.set("status", best_hash.empty() ? "no-candidate-admitted"
                                               : "admitted");
      const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
      if (!report_path.empty()) {
        std::ofstream stream(report_path, std::ios::binary | std::ios::trunc);
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
      }
      if (json_output)
        std::cout << text;
    }
    if (best_hash.empty())
      dif::fail("no tuning candidate passed numerical admission");
    if (!json_output)
      std::cout << "BEST hash=" << best_hash << " mean_ms=" << best << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "diftune: " << error.what() << "\n";
    return 1;
  }
}
