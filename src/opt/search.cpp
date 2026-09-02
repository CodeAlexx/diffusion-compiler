#include "dif/opt/search.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dif::opt {
namespace {

std::int64_t now_unix() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

runtime::RunOptions run_options(const SearchOptions &options) {
  runtime::RunOptions run;
  run.warmups = options.warmups;
  run.iterations = options.iterations;
  run.minimum_free_bytes = options.minimum_free_bytes;
  return run;
}

runtime::TensorMap reference_outputs(const RewriteContext &base) {
  runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return runtime::make_cpu_executor()
      ->run(base.program, base.bindings, options)
      .outputs;
}

// Objective value: lower is better. Kept in one place so the winner rule cannot
// drift between selection and reporting.
double objective_value(Objective objective, const CandidateRecord &record) {
  return objective == Objective::Latency
             ? record.minimum_milliseconds
             : static_cast<double>(record.memory.planned_bytes);
}

bool objective_admissible(Objective objective, double latency_tolerance,
                          const CandidateRecord &baseline,
                          const CandidateRecord &record) {
  if (objective != Objective::PlannedMemory)
    return true;
  return record.minimum_milliseconds <=
         baseline.minimum_milliseconds * latency_tolerance;
}

void record_measurement(tune::Database &database, const SearchResult &result,
                        const CandidateRecord &record) {
  tune::Measurement measurement;
  measurement.candidate_hash = record.candidate_fingerprint;
  measurement.program_hash = result.base_program_fingerprint;
  measurement.backend = record.backend;
  measurement.device = record.device;
  measurement.mean_milliseconds = record.mean_milliseconds;
  measurement.minimum_milliseconds = record.minimum_milliseconds;
  measurement.maximum_milliseconds = record.maximum_milliseconds;
  measurement.max_absolute_error = record.numerics.max_absolute_error;
  measurement.cosine_similarity = record.numerics.cosine_similarity;
  measurement.norm_ratio = record.numerics.norm_ratio;
  measurement.nonfinite_count = record.numerics.nonfinite_count;
  measurement.planned_memory_bytes = record.memory.planned_bytes;
  // The reproducible identity of this candidate: replaying this sequence
  // against program_hash must rebuild candidate_hash. Rejected candidates keep
  // theirs too, so a later run can see what was already tried and refused.
  measurement.plan = encode_transform_sequence(record.transforms);
  measurement.status = std::string(verdict_name(record.verdict));
  measurement.created_unix = now_unix();
  database.record(measurement);
}

std::string transform_json(const Transform &transform) {
  std::string out = "{\"kind\": ";
  out += json_quote(transform_kind_name(transform.kind));
  out += ", \"class\": ";
  out += json_quote(transform_class_name(transform_class(transform.kind)));
  out += ", \"encoded\": ";
  out += json_quote(encode_transform(transform));
  out += "}";
  return out;
}

} // namespace

std::string_view objective_name(Objective objective) {
  switch (objective) {
  case Objective::Latency:
    return "latency";
  case Objective::PlannedMemory:
    return "planned_memory";
  }
  fail("unknown optimization objective");
}

SearchResult optimize(
    const RewriteContext &base, runtime::Executor &executor,
    const AcceptanceGate &gate, const SearchOptions &options,
    tune::Database *database,
    const runtime::TensorMap *trusted_reference_outputs,
    std::string_view trusted_reference_backend) {
  if (options.iterations == 0U)
    fail("optimization search requires a nonzero iteration count");
  if (options.beam_width == 0U)
    fail("optimization search requires a nonzero beam width");
  if (!(options.improvement_margin >= 0.0) ||
      !(options.latency_regression_tolerance >= 1.0))
    fail("optimization search requires a nonnegative improvement margin and a "
         "latency tolerance of at least one");

  ir::verify(base.program);

  SearchResult result;
  result.objective = options.objective;
  result.bars = gate.bars();
  result.base_program_fingerprint = program_fingerprint(base.program);
  result.base_fingerprint = candidate_fingerprint(base);
  result.reference_backend =
      trusted_reference_outputs ? std::string(trusted_reference_backend)
                                : "cpu";
  if (trusted_reference_outputs) {
    for (const auto &description : base.program.tensors) {
      if (description.has_role(ir::TensorRole::Output) &&
          !trusted_reference_outputs->contains(description.id))
        fail("trusted reference omitted program output tensor " +
             std::to_string(description.id));
    }
    for (const auto &[id, tensor] : *trusted_reference_outputs) {
      (void)tensor;
      const auto *description = base.program.tensor(id);
      if (!description || !description->has_role(ir::TensorRole::Output))
        fail("trusted reference names non-output tensor " +
             std::to_string(id));
    }
  }

  const auto reference = trusted_reference_outputs
                             ? *trusted_reference_outputs
                             : reference_outputs(base);
  const auto run = run_options(options);
  if (options.warm_process_before_baseline) {
    runtime::RunOptions warm = run;
    warm.warmups = 0U;
    warm.iterations = 1U;
    executor.run(base.program, base.bindings, warm);
  }

  // Accepted candidate contexts, keyed by candidate index. Only beam members
  // are retained between depths: a context carries every bound constant, so
  // keeping all of them would cost the model's weight footprint per candidate.
  std::unordered_map<std::size_t, RewriteContext> contexts;
  std::unordered_set<std::string> seen;

  const auto evaluate = [&](RewriteContext candidate,
                            std::vector<Transform> transforms,
                            std::size_t parent, std::uint32_t depth,
                            bool &kept) -> std::size_t {
    kept = false;
    CandidateRecord record;
    record.index = result.candidates.size();
    record.parent = parent;
    record.depth = depth;
    record.transforms = std::move(transforms);
    record.warmups = options.warmups;
    record.iterations = options.iterations;
    record.backend = executor.name();
    record.device = "unknown";
    try {
      ir::verify(candidate.program);
      record.program_fingerprint = program_fingerprint(candidate.program);
      record.candidate_fingerprint = candidate_fingerprint(candidate);
      record.memory = measure_memory(candidate);
    } catch (const std::exception &error) {
      record.verdict = Verdict::RejectedVerify;
      record.diagnostic = error.what();
      result.candidates.push_back(record);
      return record.index;
    }
    try {
      const auto measured =
          executor.run(candidate.program, candidate.bindings, run);
      record.backend = measured.backend_name.empty() ? executor.name()
                                                     : measured.backend_name;
      record.device = measured.device_name;
      record.preparation_milliseconds = measured.preparation_milliseconds;
      record.mean_milliseconds = measured.mean_milliseconds;
      record.minimum_milliseconds = measured.minimum_milliseconds;
      record.maximum_milliseconds = measured.maximum_milliseconds;
      record.free_bytes_before = measured.free_bytes_before;
      record.free_bytes_after = measured.free_bytes_after;
      record.resident_bytes = measured.resident_bytes;
      record.numerics = gate.measure(reference, measured.outputs);
    } catch (const std::exception &error) {
      record.verdict = Verdict::RejectedExecution;
      record.diagnostic = error.what();
      result.candidates.push_back(record);
      return record.index;
    }
    record.verdict = gate.judge(record.numerics, record.memory.planned_bytes);
    record.accepted = record.verdict == Verdict::Accepted;
    result.candidates.push_back(record);
    if (record.accepted) {
      kept = true;
      contexts.emplace(record.index, std::move(candidate));
    }
    return record.index;
  };

  bool baseline_kept = false;
  const auto baseline_index =
      evaluate(base, {}, 0U, 0U, baseline_kept);
  auto &baseline = result.candidates[baseline_index];
  baseline.parent = baseline_index;
  result.backend = baseline.backend;
  result.device = baseline.device;
  if (!baseline.accepted)
    fail("the base program does not clear its own acceptance bars on this "
         "backend (" + std::string(verdict_name(baseline.verdict)) +
         (baseline.diagnostic.empty() ? "" : ": " + baseline.diagnostic) +
         "); optimization results would not be comparable");
  seen.insert(baseline.candidate_fingerprint);

  std::vector<std::size_t> beam = {baseline_index};
  for (std::uint32_t depth = 1U; depth <= options.max_depth; ++depth) {
    std::vector<std::size_t> produced;
    for (const auto parent_index : beam) {
      if (result.candidates.size() >= options.max_candidates)
        break;
      const auto parent = contexts.at(parent_index);
      const auto transforms = discover(parent, options.discovery);
      result.discovered_transforms +=
          static_cast<std::uint32_t>(transforms.size());
      for (const auto &transform : transforms) {
        if (result.candidates.size() >= options.max_candidates)
          break;
        RewriteContext candidate = parent;
        auto sequence = result.candidates[parent_index].transforms;
        sequence.push_back(transform);
        std::string diagnostic;
        try {
          apply(transform, candidate);
        } catch (const std::exception &error) {
          diagnostic = error.what();
        }
        if (!diagnostic.empty()) {
          CandidateRecord record;
          record.index = result.candidates.size();
          record.parent = parent_index;
          record.depth = depth;
          record.transforms = std::move(sequence);
          record.backend = executor.name();
          record.verdict = Verdict::RejectedVerify;
          record.diagnostic = std::move(diagnostic);
          result.candidates.push_back(std::move(record));
          continue;
        }
        std::string fingerprint;
        try {
          fingerprint = candidate_fingerprint(candidate);
        } catch (const std::exception &) {
          fingerprint.clear();
        }
        // Two transform sequences that reach the same program are the same
        // candidate; measuring it twice would only add noise.
        if (!fingerprint.empty() && !seen.insert(fingerprint).second)
          continue;
        bool kept = false;
        const auto index = evaluate(std::move(candidate), std::move(sequence),
                                    parent_index, depth, kept);
        if (kept)
          produced.push_back(index);
      }
    }
    if (produced.empty())
      break;
    std::stable_sort(produced.begin(), produced.end(),
                     [&](std::size_t left, std::size_t right) {
                       const auto &a = result.candidates[left];
                       const auto &b = result.candidates[right];
                       const auto left_value =
                           objective_value(options.objective, a);
                       const auto right_value =
                           objective_value(options.objective, b);
                       if (left_value != right_value)
                         return left_value < right_value;
                       return a.candidate_fingerprint < b.candidate_fingerprint;
                     });
    if (produced.size() > options.beam_width)
      produced.resize(options.beam_width);
    beam = std::move(produced);
    const std::unordered_set<std::size_t> retained(beam.begin(), beam.end());
    std::erase_if(contexts, [&](const auto &entry) {
      return !retained.contains(entry.first);
    });
  }

  // Measure how far the machine drifted while the candidates were running. A
  // latency comparison is only trustworthy outside this band, so the selection
  // rules below are widened to it before any winner is chosen.
  // Two sources of latency error are estimated from the base program itself:
  // the spread across a single run's timed iterations, and the drift between
  // the baseline measured before the candidates and one measured after them.
  // The bound is the larger, and every latency decision below is widened to it.
  double noise = 1.0;
  {
    const auto &measured = result.candidates[baseline_index];
    if (measured.minimum_milliseconds > 0.0)
      noise = std::max(noise, measured.maximum_milliseconds /
                                  measured.minimum_milliseconds);
  }
  if (options.measure_baseline_drift) {
    const auto recheck = executor.run(base.program, base.bindings, run);
    result.baseline_recheck_minimum_milliseconds = recheck.minimum_milliseconds;
    const auto original = result.candidates[baseline_index].minimum_milliseconds;
    result.baseline_drift_ratio =
        original > 0.0 ? recheck.minimum_milliseconds / original : 1.0;
    if (result.baseline_drift_ratio > 0.0)
      noise = std::max({noise, result.baseline_drift_ratio,
                        1.0 / result.baseline_drift_ratio});
    if (recheck.minimum_milliseconds > 0.0)
      noise = std::max(noise, recheck.maximum_milliseconds /
                                  recheck.minimum_milliseconds);
  }
  result.latency_noise_bound = noise;
  // Planned memory is computed, not timed, so its margin is not widened. A
  // latency objective is timed, so it is.
  result.effective_improvement_margin =
      options.objective == Objective::Latency
          ? std::max(options.improvement_margin,
                     result.latency_noise_bound - 1.0)
          : options.improvement_margin;
  result.effective_latency_tolerance =
      std::max(options.latency_regression_tolerance, result.latency_noise_bound);

  // Winner selection. Only accepted candidates compete, and a challenger must
  // clear the improvement margin to displace the baseline.
  const auto &baseline_record = result.candidates[baseline_index];
  const auto baseline_value = objective_value(options.objective, baseline_record);
  result.winner = baseline_index;
  double best_value = baseline_value;
  for (const auto &record : result.candidates) {
    if (!record.accepted || record.index == baseline_index)
      continue;
    if (!objective_admissible(options.objective,
                              result.effective_latency_tolerance,
                              baseline_record, record))
      continue;
    const auto value = objective_value(options.objective, record);
    if (value > baseline_value * (1.0 - result.effective_improvement_margin))
      continue;
    const auto &incumbent = result.candidates[result.winner];
    const auto better =
        value < best_value ||
        (value == best_value &&
         (record.transforms.size() < incumbent.transforms.size() ||
          (record.transforms.size() == incumbent.transforms.size() &&
           record.candidate_fingerprint < incumbent.candidate_fingerprint)));
    if (better) {
      best_value = value;
      result.winner = record.index;
    }
  }
  result.improved = result.winner != baseline_index;
  result.candidates[result.winner].winner = true;

  result.plan.base_program_fingerprint = result.base_program_fingerprint;
  result.plan.base_fingerprint = result.base_fingerprint;
  result.plan.transforms = result.candidates[result.winner].transforms;
  result.plan.candidate_program_fingerprint =
      result.candidates[result.winner].program_fingerprint;
  result.plan.candidate_fingerprint =
      result.candidates[result.winner].candidate_fingerprint;
  for (const auto &status : format_statuses(options.discovery)) {
    // Bounded physical-format competition: every requested format is
    // accounted for, competing or not, with the capability and availability
    // facts the decision was made from.
    PlanDecision decision;
    decision.subject = "physical-format";
    decision.subject_id = static_cast<std::uint64_t>(status.format);
    decision.decision = status.competes ? "competes" : "excluded";
    decision.reason = status.legality_reason + "; " + status.availability_reason;
    decision.evidence.emplace_back("format",
                                   std::string(physical_format_name(status.format)));
    decision.evidence.emplace_back("legal_on_target",
                                   status.legal_on_target ? "true" : "false");
    decision.evidence.emplace_back(
        "availability",
        std::string(format_availability_name(status.availability)));
    if (options.discovery.target)
      decision.evidence.emplace_back(
          "target_fingerprint",
          target::target_fingerprint(*options.discovery.target));
    result.plan.decisions.push_back(std::move(decision));
  }
  {
    // Record why the winner was selected and why every other measured
    // candidate was not, from the verdicts the gate actually returned.
    PlanDecision summary;
    summary.subject = "plan";
    summary.decision = "objective";
    summary.reason =
        std::string("search minimized ") +
        std::string(objective_name(result.objective)) +
        (result.improved ? "; the winner beat the baseline by more than the "
                           "effective margin"
                         : "; no candidate beat the baseline by the effective "
                           "margin, so the baseline is retained");
    summary.evidence.emplace_back("winner_index", std::to_string(result.winner));
    summary.evidence.emplace_back("improved", result.improved ? "true" : "false");
    summary.evidence.emplace_back(
        "effective_improvement_margin",
        json_number(result.effective_improvement_margin));
    summary.evidence.emplace_back("latency_noise_bound",
                                  json_number(result.latency_noise_bound));
    summary.evidence.emplace_back("baseline_drift_ratio",
                                  json_number(result.baseline_drift_ratio));
    summary.evidence.emplace_back("measured_candidates",
                                  std::to_string(result.candidates.size()));
    result.plan.decisions.push_back(std::move(summary));
    for (const auto &record : result.candidates) {
      PlanDecision decision;
      decision.subject = "candidate";
      decision.subject_id = record.index;
      if (record.winner)
        decision.decision = "selected";
      else if (record.accepted)
        decision.decision = "accepted";
      else
        decision.decision = "rejected";
      decision.reason = std::string(verdict_name(record.verdict));
      if (!record.diagnostic.empty())
        decision.reason += ": " + record.diagnostic;
      else if (record.winner)
        decision.reason += ": fastest accepted candidate under the objective";
      else if (record.accepted)
        decision.reason +=
            ": passed every gate but did not beat the incumbent by the "
            "effective margin";
      decision.evidence.emplace_back(
          "transforms", encode_transform_sequence(record.transforms));
      decision.evidence.emplace_back("parent", std::to_string(record.parent));
      decision.evidence.emplace_back("mean_ms",
                                     json_number(record.mean_milliseconds));
      decision.evidence.emplace_back("min_ms",
                                     json_number(record.minimum_milliseconds));
      decision.evidence.emplace_back(
          "planned_bytes", std::to_string(record.memory.planned_bytes));
      decision.evidence.emplace_back(
          "resident_bytes", std::to_string(record.resident_bytes));
      decision.evidence.emplace_back(
          "cosine_similarity", json_number(record.numerics.cosine_similarity));
      decision.evidence.emplace_back("relative_l2",
                                     json_number(record.numerics.relative_l2));
      decision.evidence.emplace_back(
          "max_absolute_error",
          json_number(record.numerics.max_absolute_error));
      decision.evidence.emplace_back(
          "nonfinite_count", std::to_string(record.numerics.nonfinite_count));
      result.plan.decisions.push_back(std::move(decision));
    }
  }
  // Rebuild the winner from the base so the returned program is exactly what a
  // clean replay of the recorded plan produces.
  result.optimized = replay(result.plan, base);

  if (database) {
    for (const auto &record : result.candidates) {
      if (record.candidate_fingerprint.empty())
        continue;
      record_measurement(*database, result, record);
    }
  }
  return result;
}

std::string serialize_journal(const SearchResult &result) {
  std::string out = "{\n";
  out += "  \"kind\": " +
         json_quote("diffusion-compiler-optimization-journal") + ",\n";
  out += "  \"version\": 1,\n";
  out += "  \"objective\": " + json_quote(objective_name(result.objective)) +
         ",\n";
  out += "  \"backend\": " + json_quote(result.backend) + ",\n";
  out += "  \"device\": " + json_quote(result.device) + ",\n";
  out += "  \"reference_backend\": " + json_quote(result.reference_backend) +
         ",\n";
  out += "  \"base_program_fingerprint\": " +
         json_quote(result.base_program_fingerprint) + ",\n";
  out += "  \"base_fingerprint\": " + json_quote(result.base_fingerprint) +
         ",\n";
  out += "  \"discovered_transforms\": " +
         std::to_string(result.discovered_transforms) + ",\n";
  out += "  \"winner\": " + std::to_string(result.winner) + ",\n";
  out += "  \"improved\": " + std::string(result.improved ? "true" : "false") +
         ",\n";
  out += "  \"baseline_recheck_minimum_milliseconds\": " +
         json_number(result.baseline_recheck_minimum_milliseconds) + ",\n";
  out += "  \"baseline_drift_ratio\": " +
         json_number(result.baseline_drift_ratio) + ",\n";
  out += "  \"latency_noise_bound\": " +
         json_number(result.latency_noise_bound) + ",\n";
  out += "  \"effective_improvement_margin\": " +
         json_number(result.effective_improvement_margin) + ",\n";
  out += "  \"effective_latency_tolerance\": " +
         json_number(result.effective_latency_tolerance) + ",\n";
  out += "  \"acceptance_bars\": {";
  out += "\"max_absolute_error\": " +
         json_number(result.bars.max_absolute_error);
  out += ", \"min_cosine_similarity\": " +
         json_number(result.bars.min_cosine_similarity);
  out += ", \"min_norm_ratio\": " + json_number(result.bars.min_norm_ratio);
  out += ", \"max_norm_ratio\": " + json_number(result.bars.max_norm_ratio);
  out += ", \"max_relative_l2\": " + json_number(result.bars.max_relative_l2);
  out += ", \"memory_budget_bytes\": " +
         std::to_string(result.bars.memory_budget_bytes);
  out += "},\n";
  out += "  \"candidates\": [";
  for (std::size_t index = 0; index < result.candidates.size(); ++index) {
    const auto &record = result.candidates[index];
    out += index == 0U ? "\n" : ",\n";
    out += "    {\"index\": " + std::to_string(record.index);
    out += ", \"parent\": " + std::to_string(record.parent);
    out += ", \"depth\": " + std::to_string(record.depth);
    out += ", \"program_fingerprint\": " +
           json_quote(record.program_fingerprint);
    out += ", \"candidate_fingerprint\": " +
           json_quote(record.candidate_fingerprint);
    out += ", \"backend\": " + json_quote(record.backend);
    out += ", \"device\": " + json_quote(record.device);
    out += ", \"warmups\": " + std::to_string(record.warmups);
    out += ", \"iterations\": " + std::to_string(record.iterations);
    out += ", \"preparation_milliseconds\": " +
           json_number(record.preparation_milliseconds);
    out += ", \"mean_milliseconds\": " +
           json_number(record.mean_milliseconds);
    out += ", \"minimum_milliseconds\": " +
           json_number(record.minimum_milliseconds);
    out += ", \"maximum_milliseconds\": " +
           json_number(record.maximum_milliseconds);
    out += ", \"free_bytes_before\": " +
           std::to_string(record.free_bytes_before);
    out += ", \"free_bytes_after\": " +
           std::to_string(record.free_bytes_after);
    out += ", \"resident_bytes\": " +
           std::to_string(record.resident_bytes);
    out += ", \"planned_bytes\": " +
           std::to_string(record.memory.planned_bytes);
    out += ", \"naive_bytes\": " + std::to_string(record.memory.naive_bytes);
    out += ", \"resident_constant_bytes\": " +
           std::to_string(record.memory.resident_constant_bytes);
    out += ", \"streamed_constant_bytes\": " +
           std::to_string(record.memory.streamed_constant_bytes);
    out += ", \"compared_elements\": " +
           std::to_string(record.numerics.compared_elements);
    out += ", \"exact_mismatch_count\": " +
           std::to_string(record.numerics.exact_mismatch_count);
    out += ", \"nonfinite_count\": " +
           std::to_string(record.numerics.nonfinite_count);
    out += ", \"max_absolute_error\": " +
           json_number(record.numerics.max_absolute_error);
    out += ", \"cosine_similarity\": " +
           json_number(record.numerics.cosine_similarity);
    out += ", \"norm_ratio\": " + json_number(record.numerics.norm_ratio);
    out += ", \"relative_l2\": " + json_number(record.numerics.relative_l2);
    out += ", \"status\": " + json_quote(verdict_name(record.verdict));
    out += ", \"accepted\": " +
           std::string(record.accepted ? "true" : "false");
    out += ", \"winner\": " + std::string(record.winner ? "true" : "false");
    out += ", \"diagnostic\": " + json_quote(record.diagnostic);
    out += ", \"transforms\": [";
    for (std::size_t position = 0; position < record.transforms.size();
         ++position) {
      if (position != 0U)
        out += ", ";
      out += transform_json(record.transforms[position]);
    }
    out += "]}";
  }
  out += result.candidates.empty() ? "]\n" : "\n  ]\n";
  out += "}\n";
  return out;
}

} // namespace dif::opt
