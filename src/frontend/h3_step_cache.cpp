#include "dif/frontend/h3_step_cache.hpp"

#include "dif/compiler/slice.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace dif::frontend {
namespace {
using Contract = H3MiddleCacheContract;
std::vector<float> difference(std::span<const float> before,
                              std::span<const float> after) {
  if (before.size() != after.size()) fail("H3 middle cache probe shape mismatch");
  std::vector<float> result(before.size());
  for (std::size_t i = 0; i < before.size(); ++i) {
    if (!std::isfinite(before[i]) || !std::isfinite(after[i]))
      fail("H3 middle cache requires finite probes");
    result[i] = after[i] - before[i];
    if (!std::isfinite(result[i])) fail("H3 middle cache probe overflow");
  }
  return result;
}
float relative_l1(std::span<const float> previous, std::span<const float> current) {
  if (previous.size() != current.size() || current.empty()) return -1.0F;
  double numerator = 0.0, denominator = 0.0;
  for (std::size_t i = 0; i < current.size(); ++i) {
    // Source subtracts in F32 BEFORE its sequential F64 reduction.
    const float delta = previous[i] - current[i];
    numerator += std::abs(static_cast<double>(delta));
    denominator += std::abs(static_cast<double>(previous[i]));
  }
  return static_cast<float>(numerator / std::max(1.0e-12, denominator));
}
const ir::TensorDesc &description(const ir::Program &program, std::uint32_t id) {
  const auto it = std::find_if(program.tensors.begin(), program.tensors.end(),
      [id](const auto &tensor) { return tensor.id == id; });
  if (it == program.tensors.end()) fail("H3 middle cache missing hidden tensor");
  return *it;
}
} // namespace

const char *h3_middle_cache_reason_name(H3MiddleCacheReason reason) {
  switch (reason) {
    case H3MiddleCacheReason::Disabled: return "disabled";
    case H3MiddleCacheReason::ExactTail: return "exact-tail";
    case H3MiddleCacheReason::Warmup: return "warmup";
    case H3MiddleCacheReason::MissingResidual: return "missing-residual";
    case H3MiddleCacheReason::ConsecutiveLimit: return "consecutive-limit";
    case H3MiddleCacheReason::MainProbeVeto: return "main-probe-veto";
    case H3MiddleCacheReason::AudioProbeVeto: return "audio-probe-veto";
    case H3MiddleCacheReason::Reuse: return "reuse-middle";
  }
  fail("invalid H3 middle cache reason");
}

H3MiddleCachePolicy::H3MiddleCachePolicy(bool enabled, std::uint64_t total,
                                         std::string identity) {
  reset(enabled, total, std::move(identity));
}
void H3MiddleCachePolicy::reset(bool enabled, std::uint64_t total,
                               std::string identity) {
  if (total == 0 || (enabled && identity.empty()))
    fail("H3 middle cache requires a bounded schedule and execution identity");
  enabled_ = enabled;
  total_ = total;
  identity_ = std::move(identity);
  next_ = consecutive_ = full_evaluations_ = cached_evaluations_ = 0;
  accumulated_diff_ = accumulated_audio_diff_ = 0;
  residual_ready_ = refresh_pending_ = false;
  previous_.clear();
  previous_audio_.clear();
}
H3MiddleCacheDecision H3MiddleCachePolicy::evaluate(
    std::uint64_t evaluation, std::span<const float> before,
    std::span<const float> after, std::span<const float> audio_before,
    std::span<const float> audio_after) {
  if (evaluation != next_ || evaluation >= total_)
    fail("H3 middle cache requires each scheduled evaluation exactly once");
  H3MiddleCacheDecision decision;
  if (!enabled_) {
    ++next_;
    ++full_evaluations_;
    return decision;
  }
  if (before.empty()) fail("H3 middle cache requires a nonempty main probe");
  auto current = difference(before, after);
  auto current_audio = difference(audio_before, audio_after);
  decision.residual_diff = relative_l1(previous_, current);
  decision.audio_residual_diff = relative_l1(previous_audio_, current_audio);
  ++next_;
  const auto tail_start = total_ > Contract::exact_tail_evaluations
      ? total_ - Contract::exact_tail_evaluations : 0;
  if (evaluation >= tail_start) {
    enabled_ = residual_ready_ = refresh_pending_ = false;
    consecutive_ = 0;
    ++full_evaluations_;
    decision.reason = H3MiddleCacheReason::ExactTail;
    decision.discard_residual = true;
    return decision;
  }
  const bool main_ok = decision.residual_diff >= 0.0F &&
      decision.residual_diff < Contract::residual_threshold &&
      accumulated_diff_ + decision.residual_diff < Contract::accumulated_budget;
  const bool audio_ok = current_audio.empty() ||
      (decision.audio_residual_diff >= 0.0F &&
       decision.audio_residual_diff < Contract::audio_threshold &&
       accumulated_audio_diff_ + decision.audio_residual_diff <
           Contract::audio_accumulated_budget);
  if (evaluation < Contract::warmup_evaluations)
    decision.reason = H3MiddleCacheReason::Warmup;
  else if (consecutive_ >= Contract::maximum_consecutive_reuses)
    decision.reason = H3MiddleCacheReason::ConsecutiveLimit;
  else if (!residual_ready_)
    decision.reason = H3MiddleCacheReason::MissingResidual;
  else if (!main_ok)
    decision.reason = H3MiddleCacheReason::MainProbeVeto;
  else if (!audio_ok)
    decision.reason = H3MiddleCacheReason::AudioProbeVeto;
  else {
    decision.reason = H3MiddleCacheReason::Reuse;
    decision.reuse_middle = true;
    ++consecutive_;
    ++cached_evaluations_;
    accumulated_diff_ += decision.residual_diff;
    // Preserve the source's optional-empty-audio accounting (-1 sentinel).
    // This never affects audio-bearing generation or empty-audio decisions.
    accumulated_audio_diff_ += decision.audio_residual_diff;
    return decision;
  }
  previous_ = std::move(current);
  previous_audio_ = std::move(current_audio);
  consecutive_ = 0;
  ++full_evaluations_;
  accumulated_diff_ = accumulated_audio_diff_ = 0.0F;
  residual_ready_ = false;
  refresh_pending_ = true;
  decision.refresh_residual = decision.discard_residual = true;
  return decision;
}
void H3MiddleCachePolicy::mark_residual_ready() {
  if (!enabled_ || !refresh_pending_)
    fail("H3 middle cache residual completed without a pending refresh");
  residual_ready_ = true;
  refresh_pending_ = false;
}

std::vector<std::int32_t> h3_middle_cache_probe_rows(std::uint64_t sequence) {
  if (sequence == 0 || sequence > static_cast<std::uint64_t>(
          std::numeric_limits<std::int32_t>::max()))
    fail("H3 middle cache sequence cannot be indexed by I32 probes");
  const auto count = std::min(sequence, Contract::probe_rows);
  std::vector<std::int32_t> result(count);
  for (std::uint64_t i = 0; i < count; ++i)
    result[i] = static_cast<std::int32_t>(
        count == 1 ? 0 : i * (sequence - 1) / (count - 1));
  return result;
}
std::vector<std::int32_t> h3_middle_cache_audio_probe_rows(
    std::span<const std::int32_t> indices, std::uint64_t sequence) {
  if (sequence == 0) fail("H3 middle cache audio probe needs a sequence");
  for (const auto index : indices)
    if (index < 0 || static_cast<std::uint64_t>(index) >= sequence)
      fail("H3 middle cache audio index outside packed sequence");
  if (indices.empty()) return {};
  const auto selected = h3_middle_cache_probe_rows(indices.size());
  std::vector<std::int32_t> result;
  for (const auto index : selected) result.push_back(indices[index]);
  return result;
}

H3MiddleCachePartition partition_h3_middle_cache(const ir::Program &program,
                                                 std::uint64_t expected_layers) {
  ir::verify(program);
  using ir::Opcode;
  constexpr std::array pattern{
      Opcode::Linear, Opcode::H3AdaLNSelect, Opcode::RmsNormModulate,
      Opcode::H3DeinterleaveQkvWeight, Opcode::Concat, Opcode::Linear,
      Opcode::Slice, Opcode::Slice, Opcode::Slice, Opcode::Reshape,
      Opcode::Reshape, Opcode::Reshape, Opcode::QkNormPartialRope,
      Opcode::QkNormPartialRope, Opcode::Attention, Opcode::Linear,
      Opcode::ResidualGate, Opcode::RmsNormModulate, Opcode::Slice,
      Opcode::Slice, Opcode::Concat, Opcode::Linear, Opcode::SwiGlu,
      Opcode::Linear, Opcode::ResidualGate};
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i < program.operations.size(); ++i) {
    if (i > 0 && program.operations[i].id <= program.operations[i - 1].id)
      fail("H3 middle cache needs monotonic source operation IDs");
    if (program.operations[i].opcode == Opcode::H3AdaLNSelect) {
      if (i == 0) fail("H3 middle cache missing block modulation projection");
      starts.push_back(i - 1);
    }
  }
  if (starts.size() != expected_layers ||
      starts.size() <= Contract::front_blocks + Contract::back_blocks)
    fail("H3 middle cache requires complete source-form block stack and a middle band");
  for (std::size_t layer = 0; layer < starts.size(); ++layer) {
    const auto start = starts[layer];
    if (start + pattern.size() > program.operations.size() ||
        (layer > 0 && start != starts[layer - 1] + pattern.size()))
      fail("H3 middle cache blocks are not source-contiguous");
    for (std::size_t offset = 0; offset < pattern.size(); ++offset)
      if (program.operations[start + offset].opcode != pattern[offset])
        fail("H3 middle cache cannot partition transformed block operations");
    if (layer > 0 && program.operations[start + 2].inputs.at(0) !=
        program.operations[start - 1].outputs.at(0))
      fail("H3 middle cache block hidden chain is not contiguous");
  }
  const auto front_start = starts.front();
  const auto middle_start = starts[Contract::front_blocks];
  const auto back_start = starts[starts.size() - Contract::back_blocks];
  if (front_start == 0 || starts.back() + pattern.size() >= program.operations.size())
    fail("H3 middle cache requires complete prelude and final output heads");
  H3MiddleCachePartition result;
  result.before_front_tensor = program.operations[front_start + 2].inputs.at(0);
  result.after_front_tensor = program.operations[middle_start - 1].outputs.at(0);
  result.after_middle_tensor = program.operations[back_start - 1].outputs.at(0);
  const auto &hidden = description(program, result.before_front_tensor);
  if (hidden.dtype != ir::DType::BF16 || hidden.dims.size() != 2 ||
      hidden.dims[1] % Contract::group_size != 0)
    fail("H3 middle cache needs BF16[S,H] with H divisible by32");
  for (const auto id : {result.after_front_tensor, result.after_middle_tensor}) {
    const auto &other = description(program, id);
    if (other.dtype != hidden.dtype || other.dims != hidden.dims)
      fail("H3 middle cache boundary shape or dtype changed");
  }
  const auto slice = [&](std::size_t start, std::size_t end) {
    return compiler::slice_operations(program, program.operations[start].id,
                                      program.operations[end].id);
  };
  result.prelude = slice(0, front_start - 1);
  result.front = slice(front_start, middle_start - 1);
  result.middle = slice(middle_start, back_start - 1);
  result.back = slice(back_start, program.operations.size() - 1);
  result.layers = starts.size();
  result.sequence = hidden.dims[0];
  result.hidden = hidden.dims[1];
  return result;
}
} // namespace dif::frontend
