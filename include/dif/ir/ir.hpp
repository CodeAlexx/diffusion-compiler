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
};

// Gelu carries an explicit approximation because exact-erf and tanh GELU are
// observably different source semantics.  Krea 2 uses the tanh form.
enum class GeluApproximation : std::uint64_t { Tanh = 1 };

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
