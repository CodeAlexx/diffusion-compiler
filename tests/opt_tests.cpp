#include "dif/frontend/h3.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/bindings.hpp"
#include "dif/opt/gate.hpp"
#include "dif/opt/plan.hpp"
#include "dif/opt/rewrite.hpp"
#include "dif/opt/search.hpp"
#include "dif/opt/semantics.hpp"
#include "dif/opt/transform.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

using namespace dif;

runtime::RunOptions single_run() {
  runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return options;
}

runtime::TensorMap run_once(const opt::RewriteContext &context) {
  return runtime::make_cpu_executor()
      ->run(context.program, context.bindings, single_run())
      .outputs;
}

// Byte-for-byte comparison. Structural rewrites claim exact preservation, so the
// tests hold them to exactly that rather than to a tolerance.
bool identical_outputs(const runtime::TensorMap &left,
                       const runtime::TensorMap &right) {
  if (left.size() != right.size())
    return false;
  for (const auto &[id, tensor] : left) {
    const auto found = right.find(id);
    if (found == right.end() || found->second.dtype != tensor.dtype ||
        found->second.dims != tensor.dims ||
        found->second.byte_size() != tensor.byte_size())
      return false;
    if (std::memcmp(found->second.data(), tensor.data(), tensor.byte_size()) !=
        0)
      return false;
  }
  return true;
}

std::size_t count_opcode(const ir::Program &program, ir::Opcode opcode) {
  return static_cast<std::size_t>(
      std::count_if(program.operations.begin(), program.operations.end(),
                    [&](const ir::Operation &operation) {
                      return operation.opcode == opcode;
                    }));
}

opt::Transform whole(opt::TransformKind kind,
                     std::vector<std::uint64_t> parameters = {}) {
  opt::Transform transform;
  transform.kind = kind;
  transform.parameters = std::move(parameters);
  return transform;
}

bool applies(const opt::Transform &transform,
             const opt::RewriteContext &context) {
  auto copy = context;
  try {
    opt::apply(transform, copy);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

// A program that carries one of every structural rewrite opportunity: a
// bias epilogue, a dead operation, a duplicated subexpression, and an exact
// widen/narrow cast pair.
opt::RewriteContext structural_fixture() {
  using namespace ir;
  ir::Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {4, 3}},
      {2, DType::F32, TensorRole::Constant, {5, 3}},
      {3, DType::F32, TensorRole::Constant, {5}},
      {4, DType::F32, TensorRole::Internal, {4, 5}},
      {5, DType::F32, TensorRole::Output, {4, 5}},
      {6, DType::F32, TensorRole::Internal, {4, 3}},
      {7, DType::F32, TensorRole::Internal, {4, 3}},
      {8, DType::F32, TensorRole::Internal, {4, 3}},
      {9, DType::F32, TensorRole::Output, {4, 3}},
      {10, DType::BF16, TensorRole::Input, {4, 3}},
      {11, DType::F32, TensorRole::Internal, {4, 3}},
      {12, DType::BF16, TensorRole::Internal, {4, 3}},
      {13, DType::BF16, TensorRole::Output, {4, 3}},
  };
  program.operations = {
      {1, Opcode::Linear, {1, 2}, {4}, {}},
      {2, Opcode::BiasAdd, {4, 3}, {5}, {}},
      {3, Opcode::Add, {1, 1}, {6}, {}},
      {4, Opcode::Multiply, {1, 1}, {7}, {}},
      {5, Opcode::Multiply, {1, 1}, {8}, {}},
      {6, Opcode::Add, {7, 8}, {9}, {}},
      {7, Opcode::Cast, {10}, {11}, {}},
      {8, Opcode::Cast, {11}, {12}, {}},
      {9, Opcode::Add, {12, 10}, {13}, {}},
  };
  ir::verify(program);
  opt::RewriteContext context;
  context.program = std::move(program);
  context.bindings = opt::synthesize_bindings(context.program, 7U);
  return context;
}

opt::RewriteContext raw_h3_block_fixture() {
  opt::RewriteContext context;
  context.program =
      frontend::make_h3_block_raw_bf16(6U, 32U, 2U, 16U, 64U, 8U, 32U, false);
  ir::verify(context.program);
  context.bindings = opt::synthesize_bindings(context.program, 11U);
  return context;
}

frontend::H3DenoiserConfig proving_config(std::uint64_t layers,
                                          std::uint64_t hidden) {
  frontend::H3DenoiserConfig config;
  config.video_tokens = 2;
  config.audio_tokens = 1;
  config.text_tokens = 2;
  config.timestep_tables = 2;
  config.hidden = hidden;
  config.heads = 8;
  config.head_dim = hidden / 8U;
  config.ffn = 2U * hidden;
  // Partial rope needs an even width that divides into three axes and fits the
  // head, so it tracks the head dimension rather than being fixed.
  config.rotary = std::min<std::uint64_t>(24U, (hidden / 8U) / 6U * 6U);
  config.layers = layers;
  config.refiner_layers = 1;
  config.video_input_dim = 16;
  config.audio_input_dim = 16;
  config.text_input_dim = 32;
  config.time_input_dim = 32;
  config.time_hidden_dim = 64;
  config.time_embed_dim = 64;
  config.block_size = 256;
  config.attention_implementation = 1;
  return config;
}

opt::RewriteContext h3_denoiser_fixture(std::uint64_t layers,
                                        std::uint64_t hidden) {
  opt::RewriteContext context;
  context.program = frontend::make_h3_denoiser(proving_config(layers, hidden));
  ir::verify(context.program);
  context.bindings = opt::synthesize_bindings(context.program, 13U);
  return context;
}

// ---------------------------------------------------------------------------

void test_transform_text_round_trip() {
  const std::vector<opt::Transform> samples = {
      whole(opt::TransformKind::FoldConstantSubgraph, {0U}),
      whole(opt::TransformKind::EliminateDeadOperations),
      whole(opt::TransformKind::SetPrefetchDistance, {4U}),
      {opt::TransformKind::SetBlockSize, {3U, 9U}, {}, {128U}},
      {opt::TransformKind::SetConstantResidency, {}, {5U, 6U}, {1U}},
      {opt::TransformKind::SetTileShape, {2U}, {}, {64U, 64U, 32U}},
  };
  for (const auto &transform : samples) {
    const auto text = opt::encode_transform(transform);
    expect(opt::decode_transform(text) == transform,
           "transform " + text + " survives a text round trip");
  }
  bool rejected = false;
  try {
    opt::decode_transform("not_a_transform ops= tensors= params=");
  } catch (const std::exception &) {
    rejected = true;
  }
  expect(rejected, "unknown transform names are rejected rather than guessed");
  expect(opt::changes_numerics(opt::TransformKind::QuantizeConstantWeights) &&
             !opt::changes_numerics(opt::TransformKind::FoldConstantSubgraph) &&
             !opt::changes_numerics(opt::TransformKind::SetConstantResidency),
         "transform classes separate value-preserving from numeric rewrites");
}

void test_acceptance_gate_is_a_fixed_oracle() {
  opt::AcceptanceBars bars;
  bars.max_absolute_error = 1.0e-4;
  bars.min_cosine_similarity = 0.999999;
  bars.memory_budget_bytes = 1024U;
  const opt::AcceptanceGate gate(bars);
  expect(gate.bars().max_absolute_error == 1.0e-4 &&
             gate.bars().memory_budget_bytes == 1024U,
         "the gate reports the bars it was constructed with");

  opt::NumericalMeasurement exact;
  exact.compared_elements = 16U;
  expect(gate.judge(exact, 1024U) == opt::Verdict::Accepted,
         "an exact candidate inside the budget is accepted");
  expect(gate.judge(exact, 1025U) == opt::Verdict::RejectedMemory,
         "the memory budget is a hard constraint");

  auto noisy = exact;
  noisy.max_absolute_error = 1.0e-3;
  expect(gate.judge(noisy, 1024U) == opt::Verdict::RejectedNumerical,
         "a candidate outside the absolute-error bar is rejected");

  auto broken = noisy;
  broken.nonfinite_count = 1U;
  expect(gate.judge(broken, 4096U) == opt::Verdict::RejectedNonFinite,
         "non-finite output is reported before the numerical bars");

  auto skewed = exact;
  skewed.cosine_similarity = 0.99;
  expect(gate.judge(skewed, 1024U) == opt::Verdict::RejectedNumerical,
         "the cosine bar is enforced independently of absolute error");

  // A candidate that drops or reshapes an output is a failure, never a
  // tolerance question.
  runtime::TensorMap reference;
  runtime::Tensor value{ir::DType::F32, {2}, {}};
  value.bytes.resize(8U);
  value.validate();
  runtime::store_float(value, 0U, 1.0F);
  runtime::store_float(value, 1U, 2.0F);
  reference.emplace(1U, value);
  bool refused = false;
  try {
    gate.measure(reference, {});
  } catch (const std::exception &) {
    refused = true;
  }
  expect(refused, "the gate refuses a candidate that omits a reference output");
  const auto measured = gate.measure(reference, reference);
  expect(measured.max_absolute_error == 0.0 && measured.relative_l2 == 0.0 &&
             measured.compared_elements == 2U,
         "identical outputs measure as exactly identical");
}

void test_structural_rewrites_preserve_semantics() {
  const auto base = structural_fixture();
  const auto reference = run_once(base);

  struct Case {
    opt::TransformKind kind;
    ir::Opcode removed;
    std::size_t expected_remaining;
    const char *label;
  };
  const std::vector<Case> cases = {
      {opt::TransformKind::EliminateDeadOperations, ir::Opcode::Add, 2U,
       "dead operation elimination"},
      {opt::TransformKind::CommonSubexpression, ir::Opcode::Multiply, 1U,
       "common subexpression elimination"},
      {opt::TransformKind::ElideCastRoundTrip, ir::Opcode::Cast, 0U,
       "exact cast round-trip elimination"},
  };
  for (const auto &entry : cases) {
    auto candidate = base;
    opt::apply(whole(entry.kind), candidate);
    ir::verify(candidate.program);
    expect(count_opcode(candidate.program, entry.removed) ==
               entry.expected_remaining,
           std::string(entry.label) + " removes the redundant operations");
    expect(identical_outputs(reference, run_once(candidate)),
           std::string(entry.label) + " is bit-exact");
  }

  // Every value-preserving rewrite together still reproduces the base program.
  auto combined = base;
  for (const auto &entry : cases)
    opt::apply(whole(entry.kind), combined);
  ir::verify(combined.program);
  expect(combined.program.operations.size() == 5U,
         "the structural rewrites compose down to five operations");
  expect(identical_outputs(reference, run_once(combined)),
         "composed structural rewrites are bit-exact");

  // Bias epilogue fusion is value-preserving in exact arithmetic only: it seeds
  // the accumulator with the bias instead of adding it afterwards. The
  // optimizer classifies it as a numeric transform for exactly that reason, and
  // the gate, not the rewrite, decides whether the difference is admissible.
  expect(opt::changes_numerics(opt::TransformKind::FuseLinearBias),
         "bias epilogue fusion is classified as a numeric transform");
  auto biased = base;
  opt::apply(whole(opt::TransformKind::FuseLinearBias), biased);
  ir::verify(biased.program);
  expect(count_opcode(biased.program, ir::Opcode::BiasAdd) == 0U,
         "bias epilogue fusion removes the separate bias pass");
  const opt::AcceptanceGate epilogue_gate{opt::AcceptanceBars{}};
  const auto epilogue = epilogue_gate.measure(reference, run_once(biased));
  expect(epilogue_gate.judge(epilogue, 0U) == opt::Verdict::Accepted,
         "the fused bias epilogue stays inside the default numerical bars");

  const auto discovered = opt::discover(base, {});
  expect(std::any_of(discovered.begin(), discovered.end(),
                     [](const opt::Transform &transform) {
                       return transform.kind ==
                              opt::TransformKind::FuseLinearBias;
                     }),
         "discovery finds the bias epilogue without being told where it is");
}

void test_qkv_projection_fusion_is_exact_and_invertible() {
  const auto base = raw_h3_block_fixture();
  const auto reference = run_once(base);
  expect(count_opcode(base.program, ir::Opcode::H3DeinterleaveQkv) == 1U,
         "the raw H3 block projects QKV as one packed linear");

  auto split = base;
  opt::apply(whole(opt::TransformKind::SplitQkvProjection), split);
  ir::verify(split.program);
  expect(count_opcode(split.program, ir::Opcode::Linear) ==
                 count_opcode(base.program, ir::Opcode::Linear) + 2U &&
             count_opcode(split.program,
                          ir::Opcode::H3DeinterleaveQkvWeight) == 1U,
         "splitting the packed projection produces three weight-side linears");
  expect(identical_outputs(reference, run_once(split)),
         "splitting the QKV projection is bit-exact");

  auto fused = split;
  opt::apply(whole(opt::TransformKind::FuseQkvProjection), fused);
  ir::verify(fused.program);
  expect(identical_outputs(reference, run_once(fused)),
         "fusing the QKV projection back is bit-exact");
  expect(count_opcode(fused.program, ir::Opcode::H3DeinterleaveQkvWeight) == 0U &&
             count_opcode(fused.program, ir::Opcode::H3DeinterleaveQkv) == 1U,
         "fusion restores the packed projection shape");

  // Folding the weight-side split turns per-step data movement into constants.
  auto folded = split;
  opt::apply(whole(opt::TransformKind::FoldConstantSubgraph, {0U}), folded);
  ir::verify(folded.program);
  expect(count_opcode(folded.program, ir::Opcode::H3DeinterleaveQkvWeight) ==
             0U,
         "constant folding removes the weight deinterleave from the graph");
  expect(identical_outputs(reference, run_once(folded)),
         "constant folding of a data-movement operation is bit-exact");
}

void test_numeric_transforms_respect_pinned_semantics() {
  auto base = raw_h3_block_fixture();
  const ir::Operation *swiglu = nullptr;
  const ir::Operation *linear = nullptr;
  for (const auto &operation : base.program.operations) {
    if (operation.opcode == ir::Opcode::SwiGlu)
      swiglu = &operation;
    if (operation.opcode == ir::Opcode::Linear && !linear)
      linear = &operation;
  }
  expect(swiglu != nullptr && linear != nullptr,
         "the raw H3 block contains a SwiGLU and a linear projection");
  expect(opt::pinned_numeric_semantics(*swiglu),
         "an explicit SwiGLU gate ordering pins the operation's semantics");

  opt::Transform demote;
  demote.kind = opt::TransformKind::SetOperationPrecision;
  demote.operations = {swiglu->id};
  demote.parameters = {static_cast<std::uint64_t>(ir::DType::F32)};
  expect(!applies(demote, base),
         "precision selection refuses an operation with pinned semantics");

  opt::Transform promote = demote;
  promote.operations = {linear->id};
  auto promoted = base;
  opt::apply(promote, promoted);
  ir::verify(promoted.program);
  expect(count_opcode(promoted.program, ir::Opcode::Cast) == 3U,
         "precision selection brackets the operation with explicit casts");
  const auto discovered = opt::discover(base, {});
  expect(std::none_of(discovered.begin(), discovered.end(),
                      [&](const opt::Transform &transform) {
                        return transform.kind ==
                                   opt::TransformKind::SetOperationPrecision &&
                               transform.operations ==
                                   std::vector<std::uint32_t>{swiglu->id};
                      }),
         "discovery never proposes changing a pinned operation's precision");
}

void test_plan_serialization_round_trip() {
  opt::OptimizationPlan plan;
  plan.base_program_fingerprint = std::string(64U, 'a');
  plan.base_fingerprint = std::string(64U, 'b');
  plan.candidate_program_fingerprint = std::string(64U, 'c');
  plan.candidate_fingerprint = std::string(64U, 'd');
  plan.transforms = {whole(opt::TransformKind::FoldConstantSubgraph, {0U}),
                     {opt::TransformKind::SetBlockSize, {4U, 7U}, {}, {128U}},
                     whole(opt::TransformKind::SetPrefetchDistance, {2U})};
  const auto parsed = opt::parse_plan(opt::serialize_plan(plan));
  expect(parsed.base_fingerprint == plan.base_fingerprint &&
             parsed.candidate_fingerprint == plan.candidate_fingerprint &&
             parsed.transforms == plan.transforms,
         "an optimization plan survives a JSON round trip");

  const auto empty = opt::parse_plan(
      opt::serialize_plan({"a", "b", "c", "d", {}}));
  expect(empty.transforms.empty(), "an empty plan serializes to valid JSON");
}

void test_plan_replay_rejects_a_different_base() {
  const auto base = raw_h3_block_fixture();
  opt::OptimizationPlan plan;
  plan.base_program_fingerprint = opt::program_fingerprint(base.program);
  plan.base_fingerprint = opt::candidate_fingerprint(base);
  plan.transforms = {whole(opt::TransformKind::SplitQkvProjection)};
  auto expected = base;
  opt::apply(plan.transforms[0], expected);
  plan.candidate_program_fingerprint =
      opt::program_fingerprint(expected.program);
  plan.candidate_fingerprint = opt::candidate_fingerprint(expected);
  const auto replayed = opt::replay(plan, base);
  expect(opt::candidate_fingerprint(replayed) == plan.candidate_fingerprint,
         "replaying a plan reproduces the recorded candidate fingerprint");

  auto other = base;
  other.bindings = opt::synthesize_bindings(other.program, 99U);
  bool refused = false;
  try {
    opt::replay(plan, other);
  } catch (const std::exception &) {
    refused = true;
  }
  expect(refused, "replay refuses a base whose constants differ");

  auto tampered = plan;
  tampered.candidate_fingerprint = std::string(64U, '0');
  refused = false;
  try {
    opt::replay(tampered, base);
  } catch (const std::exception &) {
    refused = true;
  }
  expect(refused, "replay refuses a plan that does not reproduce its candidate");
}

} // namespace

namespace {

std::size_t count_verdict(const opt::SearchResult &result,
                          opt::Verdict verdict) {
  return static_cast<std::size_t>(
      std::count_if(result.candidates.begin(), result.candidates.end(),
                    [&](const opt::CandidateRecord &record) {
                      return record.verdict == verdict;
                    }));
}

void report(const opt::SearchResult &result, const char *label) {
  const auto &baseline = result.candidates.front();
  const auto &winner = result.candidates[result.winner];
  std::cout << "SEARCH " << label
            << " objective=" << opt::objective_name(result.objective)
            << " backend=" << result.backend
            << " candidates=" << result.candidates.size()
            << " discovered=" << result.discovered_transforms
            << " improved=" << (result.improved ? "yes" : "no") << "\n";
  for (const auto &record : result.candidates) {
    std::cout << "  candidate " << record.index << " depth=" << record.depth
              << " status=" << opt::verdict_name(record.verdict)
              << " min_ms=" << record.minimum_milliseconds
              << " planned_bytes=" << record.memory.planned_bytes
              << " max_abs=" << record.numerics.max_absolute_error
              << " rel_l2=" << record.numerics.relative_l2
              << (record.winner ? " WINNER" : "")
              << " plan=[" << opt::encode_transform_sequence(record.transforms)
              << "]";
    if (!record.diagnostic.empty())
      std::cout << " diagnostic=\"" << record.diagnostic << "\"";
    std::cout << "\n";
  }
  std::cout << "  AB latency baseline_min_ms=" << baseline.minimum_milliseconds
            << " winner_min_ms=" << winner.minimum_milliseconds << " speedup="
            << (winner.minimum_milliseconds > 0.0
                    ? baseline.minimum_milliseconds / winner.minimum_milliseconds
                    : 0.0)
            << " baseline_recheck_ms="
            << result.baseline_recheck_minimum_milliseconds
            << " drift_ratio=" << result.baseline_drift_ratio << "\n";
  std::cout << "  AB memory baseline_planned_bytes="
            << baseline.memory.planned_bytes
            << " winner_planned_bytes=" << winner.memory.planned_bytes
            << " ratio="
            << (baseline.memory.planned_bytes > 0U
                    ? static_cast<double>(winner.memory.planned_bytes) /
                          static_cast<double>(baseline.memory.planned_bytes)
                    : 0.0)
            << "\n";
}

opt::SearchOptions proving_options(std::uint64_t layers) {
  opt::SearchOptions options;
  options.objective = opt::Objective::PlannedMemory;
  options.warmups = 1U;
  options.iterations = 2U;
  options.beam_width = 2U;
  options.max_depth = 3U;
  options.max_candidates = layers > 4U ? 24U : 48U;
  options.improvement_margin = 0.02;
  // The portable CPU reference does not model streaming cost, so a residency or
  // prefetch policy would win the memory objective for free without paying the
  // bandwidth a real device would. Those knobs stay out of the proving search
  // so the winner has to come from an actual program rewrite; difopt exposes
  // them for a backend that does model the cost.
  options.discovery.memory = false;
  options.discovery.block_sizes = {64U, 512U};
  options.discovery.quantization_group_sizes = {16U};
  options.discovery.max_precision_candidates = 4U;
  return options;
}

// ---------------------------------------------------------------------------
// Phase 1: a verified base H3 program, an automatic search over legal
// transforms, measured candidates on both sides of the gate, one automatically
// selected winner, and a plan that rebuilds it.
// ---------------------------------------------------------------------------
void test_phase_one_h3_optimization_search() {
  const auto base = h3_denoiser_fixture(1U, 256U);
  const auto baseline_memory = opt::measure_memory(base);

  opt::AcceptanceBars bars;
  bars.max_absolute_error = 1.0e-4;
  bars.min_cosine_similarity = 0.999999;
  bars.min_norm_ratio = 0.9999;
  bars.max_norm_ratio = 1.0001;
  bars.max_relative_l2 = 1.0e-3;
  // A candidate may never need more device memory than the program it came
  // from.
  bars.memory_budget_bytes = baseline_memory.planned_bytes;
  const opt::AcceptanceGate gate(bars);

  const auto options = proving_options(1U);
  auto executor = runtime::make_cpu_executor();
  const auto result = opt::optimize(base, *executor, gate, options, nullptr);
  report(result, "h3_denoiser_1_layer");

  expect(result.candidates.size() >= 8U,
         "the search measures a real candidate space, not a single guess");
  expect(result.discovered_transforms >= 8U,
         "discovery finds legal transforms without a hand-written list");
  expect(count_verdict(result, opt::Verdict::RejectedNumerical) >= 1U,
         "at least one candidate is rejected by the numerical gate");
  expect(result.candidates.front().accepted &&
             result.candidates.front().numerics.max_absolute_error == 0.0,
         "the baseline reproduces the reference exactly");
  expect(result.improved && result.winner != 0U,
         "the search selects a candidate over the baseline automatically");

  const auto &winner = result.candidates[result.winner];
  expect(winner.accepted && winner.verdict == opt::Verdict::Accepted,
         "the winner cleared every acceptance stage");
  expect(!winner.transforms.empty(),
         "the winner is described by an explicit transform sequence");
  expect(winner.numerics.max_absolute_error == 0.0 &&
             winner.numerics.nonfinite_count == 0U,
         "the winning candidate reproduces the base program exactly");
  expect(winner.memory.planned_bytes <
             result.candidates.front().memory.planned_bytes,
         "the winner improves the planned working set");
  expect(static_cast<double>(winner.memory.planned_bytes) <=
             static_cast<double>(baseline_memory.planned_bytes) * 0.98,
         "the memory improvement clears the search's improvement margin");

  // The recorded plan is the deliverable: it must survive serialization and
  // rebuild the winner from the base program alone.
  const auto plan = opt::parse_plan(opt::serialize_plan(result.plan));
  expect(plan.transforms == result.plan.transforms,
         "the winning plan survives serialization");
  const auto rebuilt = opt::replay(plan, base);
  expect(opt::candidate_fingerprint(rebuilt) == winner.candidate_fingerprint,
         "replaying the plan reproduces the winning candidate fingerprint");
  expect(identical_outputs(run_once(rebuilt), run_once(result.optimized)),
         "the replayed program produces bit-identical results");
  expect(identical_outputs(run_once(base), run_once(rebuilt)),
         "the optimized program reproduces the base program's outputs exactly");

  // Repeating the search from the same inputs must reach the same winner.
  const auto repeated = opt::optimize(base, *executor, gate, options, nullptr);
  expect(repeated.plan.transforms == result.plan.transforms &&
             repeated.plan.candidate_fingerprint ==
                 result.plan.candidate_fingerprint,
         "the search is deterministic: a repeated run selects the same plan");
  std::cout << "PHASE1 winner_plan=["
            << opt::encode_transform_sequence(result.plan.transforms)
            << "] candidate=" << result.plan.candidate_fingerprint << "\n";
}

void test_search_enforces_the_memory_constraint() {
  const auto base = raw_h3_block_fixture();
  const auto baseline_memory = opt::measure_memory(base);
  opt::AcceptanceBars bars;
  bars.memory_budget_bytes = baseline_memory.planned_bytes;
  const opt::AcceptanceGate gate(bars);

  opt::SearchOptions options;
  options.objective = opt::Objective::PlannedMemory;
  options.warmups = 0U;
  options.iterations = 1U;
  options.beam_width = 1U;
  options.max_depth = 1U;
  options.max_candidates = 32U;
  options.discovery.numeric = false;
  options.discovery.memory = false;
  auto executor = runtime::make_cpu_executor();
  const auto result = opt::optimize(base, *executor, gate, options, nullptr);
  report(result, "h3_raw_block_memory_constraint");

  // Splitting the packed QKV projection is exact but materializes three
  // weight-shaped intermediates, so it is rejected by the memory constraint
  // rather than by numerics.
  const auto rejected = std::find_if(
      result.candidates.begin(), result.candidates.end(),
      [](const opt::CandidateRecord &record) {
        return record.verdict == opt::Verdict::RejectedMemory;
      });
  expect(rejected != result.candidates.end(),
         "the memory constraint rejects a candidate that grows the working set");
  if (rejected != result.candidates.end()) {
    expect(rejected->numerics.max_absolute_error == 0.0,
           "the memory-rejected candidate was numerically exact, so only the "
           "constraint stopped it");
    expect(rejected->memory.planned_bytes > baseline_memory.planned_bytes,
           "the memory-rejected candidate really does need more memory");
  }
}

void test_search_scales_to_a_full_h3_denoiser() {
  const auto base = h3_denoiser_fixture(50U, 64U);
  expect(base.program.operations.size() > 800U,
         "the fifty-block H3 denoiser is a large program");
  const auto baseline_memory = opt::measure_memory(base);
  opt::AcceptanceBars bars;
  bars.max_absolute_error = 1.0e-4;
  bars.min_cosine_similarity = 0.999999;
  bars.max_relative_l2 = 1.0e-3;
  bars.memory_budget_bytes = baseline_memory.planned_bytes;
  const opt::AcceptanceGate gate(bars);
  auto executor = runtime::make_cpu_executor();

  // The mechanism itself has to survive an eight-hundred-operation program:
  // discovery, application, verification, execution, gating and replay.
  auto options = proving_options(50U);
  options.max_depth = 2U;
  options.beam_width = 1U;
  const auto program = opt::optimize(base, *executor, gate, options, nullptr);
  report(program, "h3_denoiser_50_layer_program_rewrites");
  expect(program.candidates.size() >= 8U,
         "the search measures a candidate space on the fifty-block denoiser");
  expect(std::all_of(program.candidates.begin(), program.candidates.end(),
                     [](const opt::CandidateRecord &record) {
                       return !record.accepted ||
                              record.numerics.max_absolute_error == 0.0;
                     }),
         "every accepted fifty-block candidate reproduces the base exactly");
  expect(opt::candidate_fingerprint(opt::replay(program.plan, base)) ==
             program.candidates[program.winner].candidate_fingerprint,
         "the fifty-block plan replays reproducibly");
  // At fifty blocks the planned working set is dominated by resident per-block
  // constants, so a rewrite that only removes activation-shaped intermediates
  // moves a small fraction of it. The search reports that instead of
  // manufacturing a win, which is the behaviour being asserted here.
  expect(program.candidates[program.winner].memory.planned_bytes <=
             baseline_memory.planned_bytes,
         "the fifty-block winner never costs more memory than the baseline");

  // The memory-policy vocabulary is what moves a constant-dominated working
  // set. On this backend streaming is free, so the search takes it; a backend
  // that charges streaming bandwidth would see the latency constraint push
  // back, which is exactly the trade the search exists to measure.
  auto policy = proving_options(50U);
  policy.max_depth = 1U;
  policy.beam_width = 1U;
  policy.iterations = 3U;
  policy.discovery.structural = false;
  policy.discovery.schedule = false;
  policy.discovery.numeric = false;
  policy.discovery.memory = true;
  const auto residency = opt::optimize(base, *executor, gate, policy, nullptr);
  report(residency, "h3_denoiser_50_layer_memory_policy");
  const auto &winner = residency.candidates[residency.winner];
  expect(residency.improved && winner.numerics.max_absolute_error == 0.0,
         "the memory policy search finds an exact, smaller candidate");
  expect(winner.memory.planned_bytes * 4U < baseline_memory.planned_bytes,
         "streaming the fifty-block constants cuts the planned working set");
  expect(std::any_of(winner.transforms.begin(), winner.transforms.end(),
                     [](const opt::Transform &transform) {
                       return transform.kind ==
                              opt::TransformKind::SetConstantResidency;
                     }),
         "the winning memory plan is a residency policy the search chose");
}

} // namespace

int main() {
  try {
    test_transform_text_round_trip();
    test_acceptance_gate_is_a_fixed_oracle();
    test_structural_rewrites_preserve_semantics();
    test_qkv_projection_fusion_is_exact_and_invertible();
    test_numeric_transforms_respect_pinned_semantics();
    test_plan_serialization_round_trip();
    test_plan_replay_rejects_a_different_base();
    test_search_enforces_the_memory_constraint();
    test_phase_one_h3_optimization_search();
    test_search_scales_to_a_full_h3_denoiser();
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << "\n";
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " optimizer test failure(s)\n";
    return 1;
  }
  std::cout << "all optimizer tests passed\n";
  return 0;
}
