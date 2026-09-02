#include "dif/frontend/h3.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/bindings.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/opt/plan.hpp"
#include "dif/opt/search.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/tune/database.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (text.empty() || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

double floating(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text.c_str(), &end);
  if (text.empty() || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

std::vector<std::uint64_t> number_list(const std::string &text,
                                       const char *label) {
  std::vector<std::uint64_t> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
    values.push_back(number(item, label));
  if (values.empty())
    dif::fail(std::string("empty ") + label + " list");
  return values;
}

std::pair<std::uint32_t, std::filesystem::path> binding(const std::string &text) {
  const auto split = text.find('=');
  if (split == std::string::npos)
    dif::fail("tensor binding must be ID=PATH");
  const auto id = number(text.substr(0, split), "tensor id");
  if (id == 0U || id > 0xffffffffULL)
    dif::fail("tensor binding has an invalid id");
  return {static_cast<std::uint32_t>(id), text.substr(split + 1)};
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    dif::fail("cannot open " + path.string() + " for writing");
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!stream)
    dif::fail("cannot write " + path.string());
}

void usage() {
  std::cerr <<
      "usage: difopt [program source] [bindings] [options]\n"
      "\n"
      "program source (exactly one):\n"
      "  --program FILE              verified DiffIR program to optimize\n"
      "  --h3-denoiser               build the MiniMax-H3 denoiser frontend\n"
      "\n"
      "H3 geometry (with --h3-denoiser):\n"
      "  --video-tokens N --audio-tokens N --text-tokens N --timestep-tables N\n"
      "  --hidden N --heads N --head-dim N --ffn N --rotary N\n"
      "  --layers N --refiner-layers N --block-size N\n"
      "  --video-input-dim N --audio-input-dim N --text-input-dim N\n"
      "  --time-input-dim N --time-hidden-dim N --time-embed-dim N\n"
      "  --attention-implementation 1|2\n"
      "  --streamed-constants\n"
      "\n"
      "bindings (a sealed bundle and/or explicit tensors, or a fixture):\n"
      "  --weight-bundle FILE.difbind sealed checkpoint bindings\n"
      "  --verify-shards              re-digest every bundle shard on load\n"
      "  --bind ID=FILE [--bind ...]  bind inputs and constants from tensors\n"
      "  --reference ID=FILE           trusted source output for the gate\n"
      "  --synthetic-bindings SEED    deterministic experiment fixture\n"
      "\n"
      "search:\n"
      "  --objective latency|memory   default latency\n"
      "  --backend cpu|cuda           default cpu\n"
      "  --precision-policy NAME      plan compatibility policy identity\n"
      "  --warmups N --iterations N   measurement shape\n"
      "  --min-free-mib N              CUDA pressure guard\n"
      "  --beam N --depth N --max-candidates N --margin F\n"
      "  --latency-tolerance F        memory objective latency ceiling\n"
      "  --no-structural --no-schedule --no-numeric --no-memory\n"
      "  --arithmetic-folding         fold arithmetic constants (backend rounding)\n"
      "  --blocks 64,128,256 --prefetch 1,2,4 --quant-bits 4,5 --quant-groups 64\n"
      "\n"
      "acceptance bars (trusted; the search cannot change them):\n"
      "  --max-abs F --min-cos F --min-norm-ratio F --max-norm-ratio F\n"
      "  --max-rel-l2 F --memory-budget-mib N\n"
      "\n"
      "outputs:\n"
      "  --plan FILE --journal FILE --out FILE --db FILE\n"
      "\n"
      "replay:\n"
      "  --replay FILE                rebuild a recorded plan and verify it\n"
      "  --replay-global-strategy FILE apply each transform to all legal sites\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path program_path;
    std::filesystem::path replay_path;
    std::filesystem::path global_strategy_path;
    std::filesystem::path plan_path;
    std::filesystem::path journal_path;
    std::filesystem::path output_path;
    std::filesystem::path database_path;
    bool build_h3 = false;
    std::optional<std::uint64_t> synthetic_seed;
    std::vector<std::pair<std::uint32_t, std::filesystem::path>> bindings;
    std::vector<std::pair<std::uint32_t, std::filesystem::path>> references;
    std::filesystem::path weight_bundle;
    bool verify_shards = false;
    std::string backend = "cpu";
    std::string precision_policy = "diffir-declared-v1";
    dif::frontend::H3DenoiserConfig h3;
    h3.video_tokens = 2;
    h3.audio_tokens = 1;
    h3.text_tokens = 2;
    h3.timestep_tables = 2;
    h3.hidden = 512;
    h3.heads = 8;
    h3.head_dim = 64;
    h3.ffn = 1024;
    h3.rotary = 24;
    h3.layers = 1;
    h3.refiner_layers = 1;
    h3.video_input_dim = 16;
    h3.audio_input_dim = 8;
    h3.text_input_dim = 32;
    h3.time_input_dim = 32;
    h3.time_hidden_dim = 64;
    h3.time_embed_dim = 64;
    h3.block_size = 256;
    h3.attention_implementation = 1;
    dif::opt::SearchOptions options;
    dif::opt::AcceptanceBars bars;

    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      const auto value = [&]() -> std::string {
        if (index + 1 >= argc)
          dif::fail(option + " requires a value");
        return argv[++index];
      };
      if (option == "--program")
        program_path = value();
      else if (option == "--h3-denoiser")
        build_h3 = true;
      else if (option == "--video-tokens")
        h3.video_tokens = number(value(), "video token count");
      else if (option == "--audio-tokens")
        h3.audio_tokens = number(value(), "audio token count");
      else if (option == "--text-tokens")
        h3.text_tokens = number(value(), "text token count");
      else if (option == "--timestep-tables")
        h3.timestep_tables = number(value(), "timestep table count");
      else if (option == "--hidden")
        h3.hidden = number(value(), "hidden width");
      else if (option == "--heads")
        h3.heads = number(value(), "head count");
      else if (option == "--head-dim")
        h3.head_dim = number(value(), "head dimension");
      else if (option == "--ffn")
        h3.ffn = number(value(), "feed-forward width");
      else if (option == "--rotary")
        h3.rotary = number(value(), "rotary width");
      else if (option == "--layers")
        h3.layers = number(value(), "layer count");
      else if (option == "--refiner-layers")
        h3.refiner_layers = number(value(), "refiner layer count");
      else if (option == "--video-input-dim")
        h3.video_input_dim = number(value(), "video input dimension");
      else if (option == "--audio-input-dim")
        h3.audio_input_dim = number(value(), "audio input dimension");
      else if (option == "--text-input-dim")
        h3.text_input_dim = number(value(), "text input dimension");
      else if (option == "--time-input-dim")
        h3.time_input_dim = number(value(), "time input dimension");
      else if (option == "--time-hidden-dim")
        h3.time_hidden_dim = number(value(), "time hidden dimension");
      else if (option == "--time-embed-dim")
        h3.time_embed_dim = number(value(), "time embedding dimension");
      else if (option == "--attention-implementation")
        h3.attention_implementation =
            number(value(), "attention implementation");
      else if (option == "--block-size")
        h3.block_size = number(value(), "block size");
      else if (option == "--streamed-constants")
        h3.streamed_constants = true;
      else if (option == "--bind")
        bindings.push_back(binding(value()));
      else if (option == "--reference")
        references.push_back(binding(value()));
      else if (option == "--weight-bundle")
        weight_bundle = value();
      else if (option == "--verify-shards")
        verify_shards = true;
      else if (option == "--synthetic-bindings")
        synthetic_seed = number(value(), "binding seed");
      else if (option == "--objective") {
        const auto name = value();
        if (name == "latency")
          options.objective = dif::opt::Objective::Latency;
        else if (name == "memory")
          options.objective = dif::opt::Objective::PlannedMemory;
        else
          dif::fail("objective must be latency or memory");
      } else if (option == "--backend")
        backend = value();
      else if (option == "--precision-policy")
        precision_policy = value();
      else if (option == "--warmups")
        options.warmups = static_cast<std::uint32_t>(number(value(), "warmups"));
      else if (option == "--iterations")
        options.iterations =
            static_cast<std::uint32_t>(number(value(), "iterations"));
      else if (option == "--min-free-mib")
        options.minimum_free_bytes =
            number(value(), "minimum free memory") * 1024ULL * 1024ULL;
      else if (option == "--beam")
        options.beam_width =
            static_cast<std::uint32_t>(number(value(), "beam width"));
      else if (option == "--depth")
        options.max_depth = static_cast<std::uint32_t>(number(value(), "depth"));
      else if (option == "--max-candidates")
        options.max_candidates =
            static_cast<std::uint32_t>(number(value(), "candidate ceiling"));
      else if (option == "--margin")
        options.improvement_margin = floating(value(), "improvement margin");
      else if (option == "--latency-tolerance")
        options.latency_regression_tolerance =
            floating(value(), "latency tolerance");
      else if (option == "--no-structural")
        options.discovery.structural = false;
      else if (option == "--no-schedule")
        options.discovery.schedule = false;
      else if (option == "--no-numeric")
        options.discovery.numeric = false;
      else if (option == "--no-memory")
        options.discovery.memory = false;
      else if (option == "--arithmetic-folding")
        options.discovery.arithmetic_constant_folding = true;
      else if (option == "--blocks")
        options.discovery.block_sizes = number_list(value(), "block size");
      else if (option == "--prefetch")
        options.discovery.prefetch_distances =
            number_list(value(), "prefetch distance");
      else if (option == "--quant-bits")
        options.discovery.quantization_bits = number_list(value(), "bit width");
      else if (option == "--quant-groups")
        options.discovery.quantization_group_sizes =
            number_list(value(), "group size");
      else if (option == "--max-abs")
        bars.max_absolute_error = floating(value(), "absolute error bar");
      else if (option == "--min-cos")
        bars.min_cosine_similarity = floating(value(), "cosine bar");
      else if (option == "--min-norm-ratio")
        bars.min_norm_ratio = floating(value(), "norm ratio bar");
      else if (option == "--max-norm-ratio")
        bars.max_norm_ratio = floating(value(), "norm ratio bar");
      else if (option == "--max-rel-l2")
        bars.max_relative_l2 = floating(value(), "relative L2 bar");
      else if (option == "--memory-budget-mib")
        bars.memory_budget_bytes =
            number(value(), "memory budget") * 1024ULL * 1024ULL;
      else if (option == "--plan")
        plan_path = value();
      else if (option == "--journal")
        journal_path = value();
      else if (option == "--out")
        output_path = value();
      else if (option == "--db")
        database_path = value();
      else if (option == "--replay")
        replay_path = value();
      else if (option == "--replay-global-strategy")
        global_strategy_path = value();
      else {
        usage();
        return 2;
      }
    }

    if (program_path.empty() == !build_h3) {
      usage();
      return 2;
    }
    if (!replay_path.empty() && !global_strategy_path.empty())
      dif::fail("exact replay and global-strategy replay are mutually exclusive");
    // A sealed bundle and explicit tensors compose: the bundle carries the
    // checkpoint constants and --bind supplies captured inputs on top. The
    // synthetic fixture is the alternative to both, never a supplement to
    // either, so a real run can never silently fall back to invented values.
    const bool explicit_bindings = !bindings.empty() || !weight_bundle.empty();
    if (explicit_bindings == synthetic_seed.has_value()) {
      usage();
      return 2;
    }
    if (!weight_bundle.empty() && !verify_shards)
      std::cerr << "difopt: warning: binding " << weight_bundle
                << " without --verify-shards\n";

    dif::opt::RewriteContext base;
    base.program = build_h3 ? dif::frontend::make_h3_denoiser(h3)
                            : dif::ir::read_file(program_path);
    dif::ir::verify(base.program);
    if (synthetic_seed) {
      base.bindings =
          dif::opt::synthesize_bindings(base.program, *synthetic_seed);
    } else {
      if (!weight_bundle.empty()) {
        const auto bundle = dif::weights::read_weight_bundle(weight_bundle);
        // load_weight_bundle verifies before it maps anything: the program
        // fingerprint, every binding against its descriptor, and each shard's
        // size, plus the shard digests when asked and the SafeTensors metadata
        // as it maps. Verifying separately here would hash every shard a second
        // time, so the load is the single verification point.
        base.bindings =
            dif::weights::load_weight_bundle(bundle, base.program, verify_shards);
        std::cout << "BUNDLE path=" << weight_bundle.string()
                  << " shards=" << bundle.shards.size()
                  << " bindings=" << bundle.bindings.size()
                  << " index=" << dif::hex_digest(bundle.index_fingerprint)
                  << "\n";
      }
      // Explicit tensors are layered last so a captured input overrides a
      // bundle entry rather than being silently dropped.
      for (const auto &[id, path] : bindings)
        base.bindings.insert_or_assign(id, dif::runtime::read_tensor(path));
    }

    std::cout << "BASE program=" << dif::opt::program_fingerprint(base.program)
              << " candidate=" << dif::opt::candidate_fingerprint(base)
              << " operations=" << base.program.operations.size()
              << " tensors=" << base.program.tensors.size() << "\n";

    if (!replay_path.empty()) {
      const auto plan = dif::opt::read_plan(replay_path);
      dif::opt::RewriteContext rebuilt;
      if (plan.compatibility) {
        dif::runtime::BudgetRequest request;
        request.reserved_device_memory_bytes = options.minimum_free_bytes;
        const auto probe = dif::runtime::probe_device(
            backend == "cuda" ? dif::runtime::ProbeBackend::Cuda
                              : dif::runtime::ProbeBackend::Host,
            0, request);
        rebuilt = dif::opt::replay(plan, base, probe.target, probe.budget,
                                   precision_policy);
      } else {
        rebuilt = dif::opt::replay(plan, base);
      }
      const auto footprint = dif::opt::measure_memory(rebuilt);
      std::cout << "REPLAY transforms=" << plan.transforms.size()
                << " candidate=" << dif::opt::candidate_fingerprint(rebuilt)
                << " planned_bytes=" << footprint.planned_bytes << "\n";
      for (const auto &transform : plan.transforms)
        std::cout << "  " << dif::opt::encode_transform(transform) << "\n";
      if (!output_path.empty())
        dif::ir::write_file(rebuilt.program, output_path);
      std::cout << "REPLAY reproduced the recorded candidate\n";
      return 0;
    }
    if (!global_strategy_path.empty()) {
      const auto plan = dif::opt::read_plan(global_strategy_path);
      const auto rebuilt = dif::opt::apply_global_strategy(plan, base);
      std::cout << "GLOBAL_STRATEGY transforms=" << plan.transforms.size()
                << " program="
                << dif::opt::program_fingerprint(rebuilt.program)
                << " candidate=" << dif::opt::candidate_fingerprint(rebuilt)
                << " planned_bytes="
                << dif::opt::measure_memory(rebuilt).planned_bytes << "\n";
      for (const auto &transform : plan.transforms)
        std::cout << "  global "
                  << dif::opt::transform_kind_name(transform.kind) << "\n";
      if (!output_path.empty())
        dif::ir::write_file(rebuilt.program, output_path);
      std::cout << "GLOBAL_STRATEGY applied with target-side legality checks\n";
      return 0;
    }

    auto executor = backend == "cuda" ? dif::runtime::make_cuda_executor()
                                      : dif::runtime::make_cpu_executor();
    if (backend != "cpu" && backend != "cuda")
      dif::fail("backend must be cpu or cuda");
    const dif::opt::AcceptanceGate gate(bars);
    dif::runtime::TensorMap reference_outputs;
    for (const auto &[id, path] : references)
      reference_outputs.insert_or_assign(id, dif::runtime::read_tensor(path));
    std::optional<dif::tune::Database> database;
    if (!database_path.empty())
      database.emplace(database_path);

    auto result = dif::opt::optimize(
        base, *executor, gate, options, database ? &*database : nullptr,
        reference_outputs.empty() ? nullptr : &reference_outputs,
        "source-capture");

    std::cout << std::fixed << std::setprecision(4);
    for (const auto &record : result.candidates) {
      std::cout << "CANDIDATE " << record.index << " depth=" << record.depth
                << " status=" << dif::opt::verdict_name(record.verdict)
                << " prepare_ms=" << record.preparation_milliseconds
                << " min_ms=" << record.minimum_milliseconds
                << " mean_ms=" << record.mean_milliseconds
                << " planned_bytes=" << record.memory.planned_bytes
                << " max_abs=" << record.numerics.max_absolute_error
                << " cos=" << record.numerics.cosine_similarity
                << " norm_ratio=" << record.numerics.norm_ratio
                << " rel_l2=" << record.numerics.relative_l2
                << " nonfinite=" << record.numerics.nonfinite_count
                << " exact_mismatches="
                << record.numerics.exact_mismatch_count
                << " resident_bytes=" << record.resident_bytes
                << " free_before=" << record.free_bytes_before
                << " free_after=" << record.free_bytes_after
                << " hash=" << record.candidate_fingerprint.substr(
                                   0, std::min<std::size_t>(
                                          16U,
                                          record.candidate_fingerprint.size()))
                << " plan=[" << dif::opt::encode_transform_sequence(
                                    record.transforms)
                << "]";
      if (!record.diagnostic.empty())
        std::cout << " diagnostic=\"" << record.diagnostic << "\"";
      std::cout << "\n";
    }

    const auto &baseline = result.candidates.front();
    const auto &winner = result.candidates[result.winner];
    std::cout << "WINNER index=" << result.winner
              << " improved=" << (result.improved ? "yes" : "no")
              << " objective=" << dif::opt::objective_name(result.objective)
              << "\n";
    std::cout << "AB latency baseline_min_ms=" << baseline.minimum_milliseconds
              << " winner_min_ms=" << winner.minimum_milliseconds
              << " speedup="
              << (winner.minimum_milliseconds > 0.0
                      ? baseline.minimum_milliseconds /
                            winner.minimum_milliseconds
                      : 0.0)
              << " baseline_recheck_ms="
              << result.baseline_recheck_minimum_milliseconds
              << " drift_ratio=" << result.baseline_drift_ratio << "\n";
    std::cout << "AB memory baseline_planned_bytes="
              << baseline.memory.planned_bytes
              << " winner_planned_bytes=" << winner.memory.planned_bytes
              << " ratio="
              << (baseline.memory.planned_bytes > 0U
                      ? static_cast<double>(winner.memory.planned_bytes) /
                            static_cast<double>(baseline.memory.planned_bytes)
                      : 0.0)
              << "\n";
    std::cout << "NOISE bound=" << result.latency_noise_bound
              << " effective_margin=" << result.effective_improvement_margin
              << " effective_latency_tolerance="
              << result.effective_latency_tolerance
              << " reference_backend=" << result.reference_backend << "\n";

    if (!plan_path.empty()) {
      dif::runtime::BudgetRequest request;
      request.reserved_device_memory_bytes = options.minimum_free_bytes;
      const auto probe = dif::runtime::probe_device(
          backend == "cuda" ? dif::runtime::ProbeBackend::Cuda
                            : dif::runtime::ProbeBackend::Host,
          0, request);
      const auto required_device_bytes =
          backend == "cuda"
              ? std::max(winner.resident_bytes, winner.memory.planned_bytes)
              : 0U;
      dif::opt::bind_plan_compatibility(
          result.plan, probe.target, probe.budget, precision_policy,
          required_device_bytes, 0U);
      {
        dif::opt::PlanDecision policy;
        policy.subject = "plan";
        policy.decision = "precision-policy";
        policy.reason =
            "declared on the difopt command line (--precision-policy, default "
            "diffir-declared-v1); numeric-class candidates were admitted only "
            "under the fixed acceptance bars recorded here";
        policy.evidence.emplace_back("precision_policy", precision_policy);
        policy.evidence.emplace_back(
            "max_absolute_error", dif::opt::json_number(bars.max_absolute_error));
        policy.evidence.emplace_back(
            "min_cosine_similarity",
            dif::opt::json_number(bars.min_cosine_similarity));
        policy.evidence.emplace_back(
            "max_relative_l2", dif::opt::json_number(bars.max_relative_l2));
        policy.evidence.emplace_back("min_norm_ratio",
                                     dif::opt::json_number(bars.min_norm_ratio));
        policy.evidence.emplace_back("max_norm_ratio",
                                     dif::opt::json_number(bars.max_norm_ratio));
        result.plan.decisions.push_back(std::move(policy));
        dif::opt::PlanDecision target;
        target.subject = "plan";
        target.decision = "required";
        target.reason =
            "bound to the probed target capability and runtime budget class; "
            "replay fails closed when the compiler revision, capability "
            "fingerprint, precision policy, budget class, usable VRAM, or "
            "workspace budget differs";
        target.evidence.emplace_back(
            "target_fingerprint", dif::target::target_fingerprint(probe.target));
        target.evidence.emplace_back(
            "architecture",
            std::string(dif::target::architecture_name(probe.target.architecture)));
        target.evidence.emplace_back(
            "budget_class", dif::target::runtime_budget_class(probe.budget));
        target.evidence.emplace_back("minimum_usable_device_bytes",
                                     std::to_string(required_device_bytes));
        target.evidence.emplace_back("measurement_backend", backend);
        result.plan.decisions.push_back(std::move(target));
      }
      dif::opt::write_plan(result.plan, plan_path);
      std::cout << "PLAN fingerprint="
                << dif::opt::plan_fingerprint(result.plan)
                << " target=" << dif::target::target_fingerprint(probe.target)
                << " budget_class="
                << dif::target::runtime_budget_class(probe.budget) << "\n";
    }
    if (!journal_path.empty())
      write_text(journal_path, dif::opt::serialize_journal(result));
    if (!output_path.empty())
      dif::ir::write_file(result.optimized.program, output_path);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difopt: " << error.what() << "\n";
    return 1;
  }
}
