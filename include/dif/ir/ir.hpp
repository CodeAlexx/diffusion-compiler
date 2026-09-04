#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace dif::ir {

constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxRank = 8;

enum class DType : std::uint32_t {
  F32 = 1,
  BF16 = 2,
  F16 = 3,
  I8 = 4,
  I32 = 5,
  Bool = 6,
  FP8E4M3 = 7,
  FP8E8M0 = 8,
};

enum TensorRole : std::uint32_t {
  Internal = 0,
  Input = 1U << 0U,
  Output = 1U << 1U,
  Constant = 1U << 2U,
  Streamed = 1U << 3U,
  Parameter = 1U << 4U,
  OptimizerState = 1U << 5U,
};

// Opcodes are declared once in opcodes.def (value, codec name, traits, doc);
// this enum, the codec names, decode validation and the optimizer's semantic
// tables all derive from that file.
enum class Opcode : std::uint32_t {
#define DIF_OPCODE(name, value, codec_name, traits) name = value,
#include "dif/ir/opcodes.def"
#undef DIF_OPCODE
};

namespace opcode_trait {
inline constexpr std::uint32_t None = 0U;
inline constexpr std::uint32_t PinnedNumerics = 1U << 0U;
inline constexpr std::uint32_t DataMovement = 1U << 1U;
inline constexpr std::uint32_t DtypeUniform = 1U << 2U;
} // namespace opcode_trait

struct OpcodeInfo {
  Opcode opcode;
  std::uint32_t value;
  std::string_view name;   // codec / telemetry name
  std::uint32_t traits;    // opcode_trait bits
};

inline constexpr std::size_t kOpcodeCount = 0U
#define DIF_OPCODE(name, value, codec_name, traits) +1U
#include "dif/ir/opcodes.def"
#undef DIF_OPCODE
    ;

constexpr std::array<OpcodeInfo, kOpcodeCount> make_opcode_registry() {
  using namespace opcode_trait;
  return {{
#define DIF_OPCODE(name, value, codec_name, traits) \
  OpcodeInfo{Opcode::name, value, codec_name, (traits)},
#include "dif/ir/opcodes.def"
#undef DIF_OPCODE
  }};
}
inline constexpr std::array<OpcodeInfo, kOpcodeCount> kOpcodeRegistry =
    make_opcode_registry();

const OpcodeInfo *opcode_info(Opcode opcode);
std::optional<Opcode> opcode_from_name(std::string_view name);
bool opcode_is_registered(std::uint32_t value);
bool opcode_has_trait(Opcode opcode, std::uint32_t trait);

enum class AttrKey : std::uint32_t {
  Epsilon = 1,
  BlockSize = 2,
  Heads = 3,
  HeadDim = 4,
  RotaryDim = 5,
  AttentionScale = 6,
  Causal = 7,
  AccumulatorDType = 8,
  Implementation = 9,
  TileM = 10,
  TileN = 11,
  TileK = 12,
  GateFirst = 13,
  GroupSize = 14,
  Value = 15,
  FlipSinToCos = 16,
  DownscaleFreqShift = 17,
  Scale = 18,
  MaxPeriod = 19,
  StepIndex = 20,
  PatchT = 21,
  PatchH = 22,
  PatchW = 23,
  Lower = 24,
  Upper = 25,
  LearningRate = 26,
  Beta1 = 27,
  Beta2 = 28,
  WeightDecay = 29,
  Stride = 30,
  Dilation = 31,
  Groups = 32,
  PadLeft = 33,
  PadRight = 34,
  PadMode = 35,
  Transposed = 36,
  TrimLeft = 37,
  TrimRight = 38,
  KvHeads = 39,
  Approximation = 40,
  Axis = 41,
  Start = 42,
  Theta = 43,
  Ntk = 44,
  WeightOffset = 45,
  RotaryLayout = 46,
  Permutation0 = 47,
  Permutation1 = 48,
  Permutation2 = 49,
  Permutation3 = 50,
  Permutation4 = 51,
  Permutation5 = 52,
  Permutation6 = 53,
  Permutation7 = 54,
  // Biased Linear lowering contract. The default vendor epilogue is a
  // different BF16 rounding boundary from creator-style addmm (prefill C with
  // bias, then GEMM with beta=1).
  LinearBiasMode = 55,
  // Optional reduction tile used when source-faithful floating-point order is
  // part of the executable contract. Zero keeps the backend default.
  ReductionTileSize = 56,
  // RmsNormModulate input contract. ExplicitScaleShift consumes materialized
  // scale/shift tensors. SharedVectorDelta consumes a per-batch vector and a
  // [2,hidden] delta table without introducing intermediate dtype stores.
  ModulationLayout = 57,
  // Backend-neutral maximum temporary workspace admitted for an operation.
  // This can also freeze source-observable vendor algorithm selection.
  WorkspaceLimitBytes = 58,
  StrideH = 59,
  StrideW = 60,
  DilationH = 61,
  DilationW = 62,
  PadTop = 63,
  PadBottom = 64,
  PadWest = 65,
  PadEast = 66,
  ScaleH = 67,
  ScaleW = 68,
  StrideT = 69,
  DilationT = 70,
  PadFront = 71,
  PadBack = 72,
  // BooleanMaskToBias vector-mask policy. True (the v1 default) masks both
  // invalid query and key rows; false masks keys only, matching causal language
  // model padding where padded query states remain observable.
  MaskQueries = 73,
  // AdamWUpdate: multiplier applied to the gradient before the moment
  // updates (gradient clipping folded into the optimizer kernel, Mojo
  // lora_adamw_plain_fused lesson: a free per-element multiply instead of a
  // host pass). Default 1.0 is a provable no-op.
  ClipScale = 74,
  // Sentinel: add new keys above this line with the next value; the verifier
  // bounds the valid range by it, so nothing else needs updating.
  EndSentinel_,
};

// Gelu carries an explicit approximation because exact-erf and tanh GELU are
// observably different source semantics. Krea 2 and Qwen vision blocks use
// the tanh form; Qwen vision mergers use the exact erf form.
// QuickSigmoid is the CLIP text tower's x * sigmoid(1.702 x) ("quick_gelu").
enum class GeluApproximation : std::uint64_t {
  Tanh = 1,
  ExactErf = 2,
  QuickSigmoid = 3
};

enum class LinearBiasMode : std::uint64_t { Epilogue = 1, Addmm = 2 };

enum class ModulationLayout : std::uint64_t {
  ExplicitScaleShift = 1,
  SharedVectorDelta = 2,
};

// QuantizeInt8Rows keeps low-precision policy explicit in DiffIR. Direct is
// ordinary dynamic symmetric row quantization. H256ConvRot applies the
// normalized H4^4 orthogonal transform independently to each 256-wide group
// before quantization, with BF16 value/scale/division boundaries. The F32
// variants retain F32 transform, scale, and division arithmetic and use the
// ordinary symmetric [-127,127] code range. The signed
// variants first apply a fixed Rademacher diagonal, which remains a generic
// orthogonal transform. H4096SignedConvRot extends the normalized Kronecker
// transform to independent 4096-wide groups. A matching transform on Linear
// weights preserves the unquantized dot product.
enum class Int8RowQuantization : std::uint64_t {
  Direct = 1,
  H256ConvRot = 2,
  H256SignedConvRot = 3,
  H4096SignedConvRot = 4,
  H256F32ConvRot = 5,
  H256F32SignedConvRot = 6,
  H4096F32SignedConvRot = 7,
  // Sylvester-ordered normalized H256 (H2^{x8}, the natural-order fast
  // Walsh-Hadamard transform), F32 arithmetic, no sign flips. This is the
  // rotation SquareQ v3 slabs (squareq_w4_v1) store their residual in, so a
  // slab-derived INT8 weight needs no extra transform on the weight path.
  // The H256 variants above use a different (H4-tuple) Hadamard ordering.
  H256F32SylvesterConvRot = 8,
};

// True for every QuantizeInt8Rows implementation that applies a Hadamard
// rotation before quantizing (shared-memory row kernel on CUDA).
inline bool is_convrot_int8_row_quantization(Int8RowQuantization mode) {
  switch (mode) {
  case Int8RowQuantization::H256ConvRot:
  case Int8RowQuantization::H256SignedConvRot:
  case Int8RowQuantization::H4096SignedConvRot:
  case Int8RowQuantization::H256F32ConvRot:
  case Int8RowQuantization::H256F32SignedConvRot:
  case Int8RowQuantization::H4096F32SignedConvRot:
  case Int8RowQuantization::H256F32SylvesterConvRot:
    return true;
  default:
    return false;
  }
}

// RotaryApply names the channel pairing explicitly.  Interleaved rotates
// adjacent pairs (2d,2d+1), which is the layout used by Krea 2 and several
// other DiT families.  HalfSplit remains a distinct semantic and must never
// be inferred from tensor shape.
enum class RotaryLayout : std::uint64_t { Interleaved = 1, HalfSplit = 2 };

enum class AttrKind : std::uint32_t {
  U64 = 1,
  I64 = 2,
  F64 = 3,
  Bool = 4,
  EndSentinel_, // add kinds above; the verifier bounds the range by it
};

struct Attribute {
  AttrKey key{};
  AttrKind kind{};
  std::uint64_t bits{};

  static Attribute u64(AttrKey key, std::uint64_t value) {
    return {key, AttrKind::U64, value};
  }
  static Attribute i64(AttrKey key, std::int64_t value) {
    return {key, AttrKind::I64, std::bit_cast<std::uint64_t>(value)};
  }
  static Attribute f64(AttrKey key, double value) {
    return {key, AttrKind::F64, std::bit_cast<std::uint64_t>(value)};
  }
  static Attribute boolean(AttrKey key, bool value) {
    return {key, AttrKind::Bool, value ? 1U : 0U};
  }

  std::uint64_t as_u64() const;
  std::int64_t as_i64() const;
  double as_f64() const;
  bool as_bool() const;
};

struct TensorDesc {
  std::uint32_t id{};
  DType dtype{};
  std::uint32_t roles{};
  std::vector<std::uint64_t> dims;

  std::uint64_t element_count() const;
  std::uint64_t byte_count() const;
  bool has_role(TensorRole role) const {
    return (roles & static_cast<std::uint32_t>(role)) != 0U;
  }
};

struct Operation {
  std::uint32_t id{};
  Opcode opcode{};
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<Attribute> attributes;

  const Attribute *find(AttrKey key) const;
  std::uint64_t u64(AttrKey key, std::uint64_t fallback) const;
  double f64(AttrKey key, double fallback) const;
  bool boolean(AttrKey key, bool fallback) const;
};

struct Program {
  std::uint32_t version{kVersion};
  std::vector<TensorDesc> tensors;
  std::vector<Operation> operations;

  const TensorDesc *tensor(std::uint32_t id) const;
};

std::size_t dtype_size(DType dtype);
std::string_view dtype_name(DType dtype);
std::string_view opcode_name(Opcode opcode);

} // namespace dif::ir
