#pragma once

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

enum class Opcode : std::uint32_t {
  Add = 1,
  Multiply = 2,
  RmsNormModulate = 3,
  SwiGlu = 4,
  ResidualGate = 5,
  Linear = 6,
  QkNormPartialRope = 7,
  Attention = 8,
  Barrier = 9,
  BiasAdd = 10,
  H3AdaLNSelect = 11,
  H3DeinterleaveQkv = 12,
  H3DeinterleaveQkvWeight = 13,
  DequantizeInt4 = 14,
  DequantizeInt5 = 15,
  SiLU = 16,
  RmsNorm = 17,
  Fill = 18,
  GatherRows = 19,
  IndexedUpdateRows = 20,
  Cast = 21,
  SelectRowChunks = 22,
  SinusoidalTimestep = 23,
  RotaryPosition = 24,
  LinearBlend = 25,
  FlowEulerStep = 26,
  Patchify3D = 27,
  Unpatchify3D = 28,
  AffineLastDim = 29,
  LayerNorm = 30,
  Clamp = 31,
  MseLoss = 32,
  MseLossBackward = 33,
  LinearBackwardInput = 34,
  LinearBackwardWeight = 35,
  BiasBackward = 36,
  SiLUBackward = 37,
  AdamWUpdate = 38,
  RmsNormBackward = 39,
  RmsNormModulateBackward = 40,
  SwiGluBackward = 41,
  ResidualGateBackward = 42,
  LayerNormBackward = 43,
  QkNormPartialRopeBackward = 44,
  AttentionLse = 45,
  AttentionBackward = 46,
  Conv1d = 47,
  SnakeBeta = 48,
  Gelu = 49,
  Sigmoid = 50,
  Reshape = 51,
  BroadcastTo = 52,
  Slice = 53,
  RotaryFrequency = 54,
  RotaryApply = 55,
  BooleanMaskToBias = 56,
  // Generic velocity-form Euler update with eager storage-dtype boundaries:
  // output = sample + round_dtype((next_t - current_t) * velocity).
  // This is deliberately distinct from H3's data-ward FlowEulerStep.
  EulerVelocityStep = 57,
  Permute = 58,
  // Concatenate equal-rank tensors along an explicit axis. The operation is
  // physical (not a view): output storage follows row-major tensor order.
  Concat = 59,
  // NCHW cross-correlation with OIHW weights and optional [C_out] bias.
  Conv2d = 60,
  // Channel-axis RMS normalization with the explicit creator storage
  // boundaries used by image/video VAEs.
  ChannelRmsNorm = 61,
  // Integer nearest-neighbor spatial expansion for NCHW tensors.
  UpsampleNearest2d = 62,
  // Constant padding for NCHW/NCDHW tensors. Spatial padding uses the shared
  // 2D attributes; rank-5 tensors additionally use front/back.
  PadConstant = 63,
  // NCDHW cross-correlation with OIDHW weights and optional [C_out] bias.
  Conv3d = 64,
  // Group normalization over contiguous channel groups. For NCHW/NCDHW,
  // each batch item is normalized independently across C/groups and every
  // trailing spatial dimension. Frontends that require an isolated temporal
  // axis express that distinction with Permute + Reshape before this op.
  GroupNorm = 65,
  // Edge-exclusive reflection padding for NCHW/NCDHW tensors. Each padded
  // extent must be smaller than the corresponding source dimension, matching
  // the common framework reflect-padding contract.
  PadReflect = 66,
  // Dynamic symmetric per-row quantization. One or more BF16 inputs are
  // logically concatenated on their last axis; an optional final F32 scalar
  // input supplies the runtime clipping ratio. The first output is I8 with
  // that combined shape and the second is the F32 dequantization scale for
  // every flattened row. An optional third/fourth output pair carries a second I8
  // code and F32 row scale for the residual left by the first code. This
  // keeps higher-fidelity activation quantization reusable and explicit
  // without coupling it to a model family. Zero rows use the smallest
  // positive guarded scale. The Implementation attribute selects the
  // explicit row transform/rounding contract; it is never inferred from a
  // model family.
  QuantizeInt8Rows = 67,
  // Unbiased scaled INT8 matrix multiplication with an explicit numerical
  // contract:
  //   BF16 Y[m,n] = BF16(I32(X[m,k] * W[n,k]^T) * xs[m] * ws[n]).
  // Quantization policy is deliberately separate from this reusable math.
  LinearInt8Scaled = 68,
  // Dynamic symmetric per-row E4M3 quantization. The first output retains the
  // input shape in FP8; the second is one F32 dequantization scale per row.
  QuantizeFp8Rows = 69,
  // FP8 E4M3 matrix multiplication with F32 accumulation, a BF16 raw GEMM
  // boundary, explicit F32 row/column dequantization scales, and BF16 output.
  LinearFp8Scaled = 70,
  // Blackwell MXFP8 quantization: E4M3 values with one positive UE8M0
  // dequantization scale per 32 adjacent K values. Scale storage uses the
  // cuBLASLt 128x4 tiled layout and pads the outer dimension to 128.
  QuantizeFp8Blocks32 = 71,
  // Unbiased MXFP8 matrix multiplication with F32 accumulation and BF16
  // output. A/B scale tensors carry the explicit tiled UE8M0 block scales.
  LinearFp8BlockScaled = 72,
  // Dequantize row-major signed INT8 weights with one F32 scale per adjacent
  // K block into BF16. The block size is explicit and model agnostic; this is
  // a storage/runtime primitive, not an approximate activation contract.
  DequantizeInt8Blocks = 73,
  // Mixed-input weight-only matrix multiplication. Activations remain BF16,
  // signed INT8 weights carry one F32 dequantization scale per output row,
  // accumulation is F32, and the explicit output storage boundary is BF16:
  //   BF16 Y[m,n] = BF16(F32(X[m,k] * I8(W[n,k])^T) * scale[n]).
  LinearInt8WeightScaled = 74,
  // Layer normalization followed by creator-style adaptive scale/shift with
  // explicit storage-dtype boundaries after normalization, scale addition,
  // multiplication, and shift addition. Scale/shift carry one or more rows
  // which broadcast over contiguous groups of input rows.
  LayerNormModulate = 75,
};

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
};

// Gelu carries an explicit approximation because exact-erf and tanh GELU are
// observably different source semantics. Krea 2 and Qwen vision blocks use
// the tanh form; Qwen vision mergers use the exact erf form.
enum class GeluApproximation : std::uint64_t { Tanh = 1, ExactErf = 2 };

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
};

// RotaryApply names the channel pairing explicitly.  Interleaved rotates
// adjacent pairs (2d,2d+1), which is the layout used by Krea 2 and several
// other DiT families.  HalfSplit remains a distinct semantic and must never
// be inferred from tensor shape.
enum class RotaryLayout : std::uint64_t { Interleaved = 1, HalfSplit = 2 };

enum class AttrKind : std::uint32_t { U64 = 1, I64 = 2, F64 = 3, Bool = 4 };

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
