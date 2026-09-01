#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dif::opt {

// A Transform is the explicit, serializable unit of optimization. Every change
// the optimizer makes to a verified DiffIR program is expressed as one of these
// values, so a candidate program is fully described by its base program plus an
// ordered transform sequence. Optimization decisions therefore live in the IR
// layer and not inside a backend emitter.
enum class TransformKind : std::uint32_t {
  // Structural rewrites. These preserve program semantics exactly and never
  // reinterpret rounding or accumulation.
  FoldConstantSubgraph = 1,
  EliminateDeadOperations = 2,
  CommonSubexpression = 3,
  // Epilogue fusion. Value-preserving in exact arithmetic but not in floating
  // point: seeding a Linear's accumulator with the bias is a different
  // accumulation order from adding it afterwards, so this is classified numeric.
  FuseLinearBias = 4,
  FuseQkvProjection = 5,
  SplitQkvProjection = 6,
  ElideCastRoundTrip = 7,
  RematerializeProducer = 8,
  // Scheduling policy. Backend-visible launch geometry, no semantic change.
  SetBlockSize = 9,
  SetTileShape = 10,
  // Numeric policy. These may change results and are gated numerically.
  SetLinearImplementation = 11,
  SetAttentionImplementation = 12,
  SetOperationPrecision = 13,
  QuantizeConstantWeights = 14,
  // Memory policy. These change residency and streaming, not arithmetic.
  SetConstantResidency = 15,
  SetPrefetchDistance = 16,
  // Packs two unbiased Linears sharing one activation, followed by
  // SiLU(gate) * value, into one wider Linear and a SwiGlu. Packing changes
  // the vendor GEMM problem shape and is therefore numerical, even though the
  // real-number equation is unchanged.
  FuseParallelLinearSwiGlu = 17,
  // Packs two or more unbiased Linears with a common activation into one
  // wider Linear followed by last-dimension slices. The wider vendor GEMM may
  // select a different accumulation schedule, so admission is numerical.
  FuseParallelLinears = 18,
};

// Broad classification used by search policy and by reporting. It is derived
// from the kind alone so a transform cannot misrepresent what it is allowed to
// change.
enum class TransformClass : std::uint32_t {
  Structural = 1,
  Schedule = 2,
  Numeric = 3,
  Memory = 4,
};

struct Transform {
  TransformKind kind{};
  // Operation ids the transform applies to. Empty means "whole program".
  std::vector<std::uint32_t> operations;
  // Tensor ids the transform applies to. Empty means "whole program".
  std::vector<std::uint32_t> tensors;
  // Kind-specific scalars. Each kind documents its own parameter list in
  // rewrite.cpp; values are bounded so they round-trip exactly through JSON.
  std::vector<std::uint64_t> parameters;
};

std::string_view transform_kind_name(TransformKind kind);
bool transform_kind_from_name(std::string_view name, TransformKind &kind);
TransformClass transform_class(TransformKind kind);
std::string_view transform_class_name(TransformClass value);

// True when applying the transform can change produced values. Structural,
// schedule, and memory transforms are value-preserving by construction; the
// numeric class is not and must clear the acceptance gate.
bool changes_numerics(TransformKind kind);

// Compact single-line canonical form, used for logs, tuning-database notes and
// deterministic candidate labels. Round-trips through decode_transform.
std::string encode_transform(const Transform &transform);
Transform decode_transform(std::string_view text);

std::string encode_transform_sequence(const std::vector<Transform> &transforms);
// Inverse of encode_transform_sequence. An empty string decodes to an empty
// sequence, which is the baseline. This is what makes the canonical one-line
// form a reproducible plan identity rather than only a label: a sequence
// recorded in the tuning database can be decoded and replayed.
std::vector<Transform> decode_transform_sequence(std::string_view text);

bool operator==(const Transform &left, const Transform &right);
inline bool operator!=(const Transform &left, const Transform &right) {
  return !(left == right);
}

} // namespace dif::opt
