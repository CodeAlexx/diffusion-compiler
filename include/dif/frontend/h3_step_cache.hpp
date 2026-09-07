#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace dif::frontend {

// Explicit, approximate middle-block residual reuse. This is not EasyCache:
// both edge bands execute every evaluation and final heads always execute.
// Source: serenitymojo/models/dit/minimax_h3_step_cache.mojo (2026-09-06).
struct H3MiddleCacheContract {
  static constexpr std::uint64_t front_blocks = 8;
  static constexpr std::uint64_t back_blocks = 8;
  static constexpr std::uint64_t group_size = 32;
  static constexpr std::uint64_t probe_rows = 16;
  static constexpr std::uint64_t warmup_evaluations = 4;
  static constexpr std::uint64_t exact_tail_evaluations = 3;
  static constexpr std::uint64_t maximum_consecutive_reuses = 2;
  static constexpr float residual_threshold = 0.12F;
  static constexpr float accumulated_budget = 0.24F;
  static constexpr float audio_threshold = 0.06F;
  static constexpr float audio_accumulated_budget = 0.12F;
};

enum class H3MiddleCacheReason {
  Disabled, ExactTail, Warmup, MissingResidual, ConsecutiveLimit,
  MainProbeVeto, AudioProbeVeto, Reuse
};
const char *h3_middle_cache_reason_name(H3MiddleCacheReason reason);

struct H3MiddleCacheDecision {
  bool reuse_middle{};
  bool refresh_residual{};
  // Release any existing device residual BEFORE creating a front snapshot.
  bool discard_residual{};
  H3MiddleCacheReason reason{H3MiddleCacheReason::Disabled};
  float residual_diff{-1.0F};
  float audio_residual_diff{-1.0F};
};

// One state per execution identity (model/adapter/conditioning/geometry,
// schedule, numeric plan and backend). Never share it between requests.
// Inputs are small F32 probes gathered from BF16 hidden rows on the device;
// the full hidden/residual tensors stay in the shared native device runtime.
class H3MiddleCachePolicy {
 public:
  H3MiddleCachePolicy(bool enabled, std::uint64_t total_evaluations,
                      std::string execution_identity);
  void reset(bool enabled, std::uint64_t total_evaluations,
             std::string execution_identity);
  H3MiddleCacheDecision evaluate(
      std::uint64_t evaluation, std::span<const float> before_front,
      std::span<const float> after_front,
      std::span<const float> audio_before = {},
      std::span<const float> audio_after = {});
  // Call only after a requested GPU residual refresh actually succeeded.
  void mark_residual_ready();
  bool enabled() const { return enabled_; }
  bool residual_ready() const { return residual_ready_; }
  std::uint64_t full_evaluations() const { return full_evaluations_; }
  std::uint64_t cached_evaluations() const { return cached_evaluations_; }
  float accumulated_diff() const { return accumulated_diff_; }
  float accumulated_audio_diff() const { return accumulated_audio_diff_; }
  const std::string &execution_identity() const { return identity_; }

 private:
  bool enabled_{}, residual_ready_{}, refresh_pending_{};
  std::uint64_t total_{}, next_{}, consecutive_{};
  std::uint64_t full_evaluations_{}, cached_evaluations_{};
  float accumulated_diff_{}, accumulated_audio_diff_{};
  std::string identity_;
  std::vector<float> previous_, previous_audio_;
};

// Main probe spans the WHOLE mixed sequence, not only the video rows. Audio
// probe is separate and uses the actual packed audio indices, conditions too.
std::vector<std::int32_t> h3_middle_cache_probe_rows(std::uint64_t sequence);
std::vector<std::int32_t> h3_middle_cache_audio_probe_rows(
    std::span<const std::int32_t> audio_indices, std::uint64_t sequence);

struct H3MiddleCachePartition {
  ir::Program prelude, front, middle, back;
  std::uint32_t before_front_tensor{}, after_front_tensor{}, after_middle_tensor{};
  std::uint64_t layers{}, sequence{}, hidden{};
};

// Partition source-form make_h3_denoiser DiffIR BEFORE fusion/lowering.
// Stable IDs and cross-band dependencies survive compiler::slice_operations.
// Execute prelude, probe, front, probe, middle-or-residual, back. On reuse,
// bind the reconstructed after_front value to after_middle_tensor. Keep ALL
// cross-band prelude outputs alive, not just the three named hidden tensors.
// Runtime residual contract: BF16 hidden [S,H], H%32==0; signed I8 codes and
// F32 scales per adjacent32 values, round-to-nearest-even, min scale1e-30.
// Snapshot=quantize(front); residual=quantize(BF16(middle-dequant(snapshot))).
// Apply=BF16(current_front+F32(code)*scale), WITHOUT an audio-row mask.
H3MiddleCachePartition partition_h3_middle_cache(
    const ir::Program &program, std::uint64_t expected_layers = 50);

} // namespace dif::frontend
