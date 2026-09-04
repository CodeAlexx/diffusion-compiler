#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <vector>

namespace dif::training {

// Recompute (gradient checkpointing) as a decision the compiler can make.
//
// A training step holds every forward activation the backward pass will read.
// Measured on the SDXL UNet at 64x64 latent: the forward alone plans to
// 4.84 GiB, forward-and-backward to 11.57 GiB, so 6.7 GiB is activations
// waiting for their gradient.  Recompute trades that for arithmetic -- drop
// the interior activations, keep only the segment boundaries, and replay each
// segment just before the backward pass needs it.
//
// An eager runtime has to be told where the segments are, because it cannot
// see the future.  Here the whole forward program is already known, so the
// cost of replaying any segment is known before the step runs, and the
// segmentation is a planner decision rather than a caller's guess.
//
// The replay is the same operations on the same inputs in the same order, so
// the gradients are bit-identical, not merely close.

// Splits the forward region into `segments` contiguous pieces and replays
// each one just before the backward operations that read it.  Only the
// operations actually needed to rebuild a dropped activation are replayed.
// `forward_operation_count` is the number of leading operations that form the
// forward pass; everything after it is backward and update work.
// `segments <= 1` returns the program unchanged.
ir::Program plan_recompute(const ir::Program &program,
                           std::size_t forward_operation_count,
                           std::size_t segments);

struct RecomputeChoice {
  std::size_t segments{1U};
  std::uint64_t planned_bytes{};
  std::uint64_t bytes_without_recompute{};
  std::size_t replayed_operations{};
  bool within_budget{};
};

// Chooses the fewest segments whose memory plan fits `budget_bytes`, and
// returns that choice alongside the program it implies.  Fewer segments means
// less replayed arithmetic, so the search stops at the first that fits.  When
// nothing fits, the closest is returned with `within_budget` false -- an
// honest "this is as far as recompute gets you", not a silent failure.
RecomputeChoice choose_recompute(const ir::Program &program,
                                 std::size_t forward_operation_count,
                                 std::uint64_t budget_bytes,
                                 ir::Program *chosen = nullptr);

} // namespace dif::training
