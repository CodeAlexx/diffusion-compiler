#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <vector>

namespace dif::opt {

// Low-rank adaptation as a transform over an arbitrary program.
//
// Until now every LoRA graph in this repository was hand-built: a fixture
// wrote out the base Linear, the two adapter Linears, the scale and the add,
// by hand, for the sites it happened to know about. A second model meant a
// second hand-built graph, which is the duplication the training foundation
// exists to remove.
//
// This takes any program, a list of Linear operations to adapt, a rank and an
// alpha, and returns the adapted program with its adapter tensors named. It
// knows nothing about which model it is adapting or what the Linears mean.
//
// The inserted shape matches what the hand-built graphs produce, because that
// shape is already gated against the reference:
//
//     base     = Linear(x, W, bias?)          the frozen path, untouched
//     low      = Linear(x, A)                 A is [rank, in]
//     delta    = Linear(low, B)               B is [out, rank]
//     scaled   = delta * (alpha / rank)       an in-graph Fill, not a folded
//                                             constant, so the scale is
//                                             visible to the optimizer search
//     combined = base + scaled
//
// The combined value KEEPS the tensor id the Linear used to produce, so every
// existing reader sees it without being rewired and the program's interface
// is unchanged -- output roles stay put, and a caller holding a tensor id
// still holds the right one. The frozen path moves to a fresh internal
// tensor instead. Rewiring consumers would have worked too, but it changes
// which id is the program's output whenever an adapted Linear fed one, which
// is a surprise a caller should not have to know about.

struct LoraSpec {
  std::uint64_t rank{};
  double alpha{};
  // Adapters are stored F32 by default whatever the compute dtype is; when
  // they differ the transform inserts the Cast the mixed-precision path
  // needs, rather than leaving a dtype disagreement for the verifier.
  ir::DType parameter_dtype{ir::DType::F32};
  // The Linear operations to adapt, by operation id. Naming them explicitly
  // is deliberate: which sites a model adapts is a model's decision, and a
  // predicate over shapes would silently adapt the wrong ones the first time
  // an architecture changed.
  std::vector<std::uint32_t> operations;
};

struct LoraSite {
  std::uint32_t operation{};    // the Linear that was adapted
  std::uint32_t base_weight{};  // the frozen weight it wraps
  std::uint32_t down{};         // A, [rank, in]
  std::uint32_t up{};           // B, [out, rank]
  std::uint32_t base_output{};  // the frozen result, now an internal value
  std::uint32_t output{};       // the adapted result, the Linear's original id
};

struct LoraResult {
  ir::Program program;
  std::vector<LoraSite> sites;
  // Every adapter tensor, in site order, ready to hand to a training plan as
  // the trainable set.
  std::vector<std::uint32_t> parameters;
};

// Throws on anything it cannot honour: an operation the program does not
// have, one that is not a Linear, a rank of zero, a non-positive alpha, or a
// site named twice.
LoraResult insert_lora(const ir::Program &program, const LoraSpec &spec);

} // namespace dif::opt
