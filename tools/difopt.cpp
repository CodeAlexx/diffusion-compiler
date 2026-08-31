#include "dif/ir/codec.hpp"
#include "dif/opt/optimizer.hpp"
#include "dif/opt/plan.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace {

std::uint64_t number(const std::string &text, const char *label) {
  std::size_t consumed = 0U;
  std::uint64_t value = 0U;
  try {
    value = std::stoull(text, &consumed, 10);
  } catch (const std::exception &) {
    dif::fail(std::string("invalid ") + label + ": " + text);
  }
  if (text.empty() || consumed != text.size())
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

std::uint64_t mebibytes(std::uint64_t value) {
  constexpr std::uint64_t scale = 1024U * 1024U;
  if (value > std::numeric_limits<std::uint64_t>::max() / scale)
    dif::fail("device budget MiB overflows bytes");
  return value * scale;
}

void usage() {
  std::cerr
      << "usage: difopt weight-placement --program IN.difir"
         " --output OUT.difir --device-budget-mib N [--evaluations N]"
         " [--alignment N] [--prefetch-distance N]"
         " [--plan-output OUT.difplan]\n"
      << "       difopt replay --program BASE.difir --plan FILE.difplan"
         " --output OUT.difir\n"
      << "       difopt set-u64 --program IN.difir --output OUT.difir"
         " --operation ID [--operation ID ...]"
         " --attribute block-size|tile-m|tile-n|tile-k|implementation|accumulator-dtype|algorithm"
         " --value N [--plan-output OUT.difplan]\n";
}

void refuse_overwrite(const std::filesystem::path &path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory)
    return;
  if (error)
    dif::fail("cannot inspect output path: " + error.message());
  if (status.type() != std::filesystem::file_type::not_found)
    dif::fail("refusing to overwrite existing output: " + path.string());
}

dif::ir::AttrKey attribute_key(const std::string &name) {
  if (name == "block-size")
    return dif::ir::AttrKey::BlockSize;
  if (name == "tile-m")
    return dif::ir::AttrKey::TileM;
  if (name == "tile-n")
    return dif::ir::AttrKey::TileN;
  if (name == "tile-k")
    return dif::ir::AttrKey::TileK;
  if (name == "implementation")
    return dif::ir::AttrKey::Implementation;
  if (name == "accumulator-dtype")
    return dif::ir::AttrKey::AccumulatorDType;
  if (name == "algorithm")
    return dif::ir::AttrKey::Algorithm;
  dif::fail("unsupported U64 attribute name: " + name);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }

    const std::string command = argv[1];
    if (command == "set-u64") {
      std::filesystem::path program_path;
      std::filesystem::path output_path;
      std::filesystem::path plan_output_path;
      std::vector<std::uint32_t> operation_ids;
      std::optional<dif::ir::AttrKey> key;
      std::uint64_t attribute_value = 0U;
      bool has_value = false;
      for (int argument = 2; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--program" && argument + 1 < argc)
          program_path = argv[++argument];
        else if (option == "--output" && argument + 1 < argc)
          output_path = argv[++argument];
        else if (option == "--plan-output" && argument + 1 < argc)
          plan_output_path = argv[++argument];
        else if (option == "--operation" && argument + 1 < argc) {
          const auto id = number(argv[++argument], "operation id");
          if (id == 0U || id > UINT32_MAX)
            dif::fail("operation id is outside the DiffIR range");
          operation_ids.push_back(static_cast<std::uint32_t>(id));
        } else if (option == "--attribute" && argument + 1 < argc) {
          key = attribute_key(argv[++argument]);
        } else if (option == "--value" && argument + 1 < argc) {
          attribute_value = number(argv[++argument], "attribute value");
          has_value = true;
        } else {
          usage();
          return 2;
        }
      }
      if (program_path.empty() || output_path.empty() || operation_ids.empty() ||
          !key || !has_value) {
        usage();
        return 2;
      }
      std::sort(operation_ids.begin(), operation_ids.end());
      if (std::adjacent_find(operation_ids.begin(), operation_ids.end()) !=
          operation_ids.end())
        dif::fail("duplicate --operation target");
      refuse_overwrite(output_path);
      if (!plan_output_path.empty())
        refuse_overwrite(plan_output_path);
      const auto base = dif::ir::read_file(program_path);
      dif::opt::Recipe recipe{{"manual.set-u64"}, {}};
      for (const auto operation_id : operation_ids)
        recipe.transformations.push_back(dif::opt::Transformation::set_u64(
            operation_id, *key, attribute_value));
      auto program = dif::opt::apply_recipe(base, recipe);
      auto candidate =
          dif::opt::make_candidate(std::move(program), std::move(recipe));
      dif::ir::write_file(candidate.program, output_path);
      if (!plan_output_path.empty())
        dif::opt::write_plan(dif::opt::make_plan(base, candidate),
                             plan_output_path);
      std::cout << "TRANSFORM PASS output=" << output_path.string()
                << " operations=" << operation_ids.size()
                << " program_fingerprint="
                << dif::hex_digest(candidate.program_fingerprint)
                << " recipe_fingerprint="
                << dif::hex_digest(candidate.recipe.fingerprint())
                << " plan_output="
                << (plan_output_path.empty() ? "none"
                                             : plan_output_path.string())
                << "\n";
      return 0;
    }
    if (command == "replay") {
      std::filesystem::path program_path;
      std::filesystem::path plan_path;
      std::filesystem::path output_path;
      for (int argument = 2; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--program" && argument + 1 < argc)
          program_path = argv[++argument];
        else if (option == "--plan" && argument + 1 < argc)
          plan_path = argv[++argument];
        else if (option == "--output" && argument + 1 < argc)
          output_path = argv[++argument];
        else {
          usage();
          return 2;
        }
      }
      if (program_path.empty() || plan_path.empty() || output_path.empty()) {
        usage();
        return 2;
      }
      refuse_overwrite(output_path);
      const auto base = dif::ir::read_file(program_path);
      const auto plan = dif::opt::read_plan(plan_path);
      const auto candidate = dif::opt::replay_candidate(base, plan);
      dif::ir::write_file(candidate.program, output_path);
      std::cout << "REPLAY PASS output=" << output_path.string()
                << " program_fingerprint="
                << dif::hex_digest(candidate.program_fingerprint)
                << " candidate_fingerprint="
                << dif::hex_digest(candidate.candidate_fingerprint)
                << " recipe_fingerprint="
                << dif::hex_digest(plan.recipe.fingerprint())
                << " prefetch_distance="
                << candidate.policy.stream_prefetch_distance << "\n";
      return 0;
    }
    if (command != "weight-placement") {
      usage();
      return 2;
    }

    std::filesystem::path program_path;
    std::filesystem::path output_path;
    std::filesystem::path plan_output_path;
    dif::opt::WeightPlacementOptions options;
    bool has_budget = false;
    for (int argument = 2; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--program" && argument + 1 < argc) {
        program_path = argv[++argument];
      } else if (option == "--output" && argument + 1 < argc) {
        output_path = argv[++argument];
      } else if (option == "--device-budget-mib" && argument + 1 < argc) {
        options.device_budget_bytes =
            mebibytes(number(argv[++argument], "device budget"));
        has_budget = true;
      } else if (option == "--evaluations" && argument + 1 < argc) {
        options.expected_evaluations =
            number(argv[++argument], "expected evaluations");
      } else if (option == "--alignment" && argument + 1 < argc) {
        options.alignment = number(argv[++argument], "alignment");
      } else if (option == "--prefetch-distance" &&
                 argument + 1 < argc) {
        options.stream_prefetch_distance =
            number(argv[++argument], "prefetch distance");
      } else if (option == "--plan-output" && argument + 1 < argc) {
        plan_output_path = argv[++argument];
      } else {
        usage();
        return 2;
      }
    }
    if (program_path.empty() || output_path.empty() || !has_budget) {
      usage();
      return 2;
    }

    refuse_overwrite(output_path);
    if (!plan_output_path.empty())
      refuse_overwrite(plan_output_path);
    const auto program = dif::ir::read_file(program_path);
    const auto result = dif::opt::place_weights(program, options);
    dif::ir::write_file(result.candidate.program, output_path);
    if (!plan_output_path.empty())
      dif::opt::write_plan(dif::opt::make_plan(program, result.candidate),
                           plan_output_path);
    std::cout
        << "OPTIMIZE PASS pass=weight-placement.greedy-v1"
        << " policy=deterministic-greedy"
        << " output=" << output_path.string()
        << " program_fingerprint="
        << dif::hex_digest(result.candidate.program_fingerprint)
        << " recipe_fingerprint="
        << dif::hex_digest(result.candidate.recipe.fingerprint())
        << " plan_output="
        << (plan_output_path.empty() ? "none" : plan_output_path.string())
        << " transformations="
        << result.candidate.recipe.transformations.size()
        << " device_budget_bytes=" << result.stats.device_budget_bytes
        << " all_streamed_planned_bytes="
        << result.stats.all_streamed_planned_bytes
        << " planned_device_bytes=" << result.stats.planned_device_bytes
        << " total_weight_bytes=" << result.stats.total_weight_bytes
        << " resident_weight_bytes=" << result.stats.resident_weight_bytes
        << " streamed_weight_bytes=" << result.stats.streamed_weight_bytes
        << " resident_weights=" << result.stats.resident_weights
        << " streamed_weights=" << result.stats.streamed_weights
        << " evaluations=" << result.stats.expected_evaluations
        << " estimated_repeated_transfer_bytes_saved="
        << result.stats.estimated_repeated_transfer_bytes_saved << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difopt: " << error.what() << "\n";
    return 1;
  }
}
