#include "dif/ir/codec.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace dif::ir {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'D', 'I', 'F', 'I', 'R', '0', '0', '1'};
constexpr std::size_t kDigestBytes = 32;
constexpr std::uint32_t kMaxObjects = 1U << 20U;

class Writer {
public:
  void u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void raw(std::span<const std::uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  std::uint32_t u32() {
    require(4);
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U)
      value |= static_cast<std::uint32_t>(bytes_[offset_++]) << shift;
    return value;
  }

  std::uint64_t u64() {
    require(8);
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
    return value;
  }

  std::span<const std::uint8_t> raw(std::size_t count) {
    require(count);
    const auto result = bytes_.subspan(offset_, count);
    offset_ += count;
    return result;
  }

  bool done() const { return offset_ == bytes_.size(); }

private:
  void require(std::size_t count) const {
    if (count > bytes_.size() - offset_)
      fail("truncated DiffIR file");
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_{};
};

std::vector<std::uint8_t> read_all(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    fail("cannot open DiffIR file: " + path.string());
  const auto end = input.tellg();
  if (end < 0)
    fail("cannot size DiffIR file: " + path.string());
  const auto size = static_cast<std::size_t>(end);
  std::vector<std::uint8_t> bytes(size);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
  if (!input)
    fail("cannot read DiffIR file: " + path.string());
  return bytes;
}

void check_count(std::uint32_t value, const char *what) {
  if (value > kMaxObjects)
    fail(std::string("unreasonable DiffIR ") + what + " count");
}

} // namespace

std::uint64_t Attribute::as_u64() const {
  if (kind != AttrKind::U64)
    fail("DiffIR attribute kind is not u64");
  return bits;
}

std::int64_t Attribute::as_i64() const {
  if (kind != AttrKind::I64)
    fail("DiffIR attribute kind is not i64");
  return std::bit_cast<std::int64_t>(bits);
}

double Attribute::as_f64() const {
  if (kind != AttrKind::F64)
    fail("DiffIR attribute kind is not f64");
  return std::bit_cast<double>(bits);
}

bool Attribute::as_bool() const {
  if (kind != AttrKind::Bool || bits > 1U)
    fail("DiffIR attribute kind/value is not bool");
  return bits != 0U;
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::F32:
  case DType::I32:
    return 4;
  case DType::Bool:
    return 1;
  case DType::BF16:
  case DType::F16:
    return 2;
  case DType::I8:
    return 1;
  }
  fail("unknown DiffIR dtype");
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return "f32";
  case DType::BF16:
    return "bf16";
  case DType::F16:
    return "f16";
  case DType::I8:
    return "i8";
  case DType::I32:
    return "i32";
  case DType::Bool:
    return "bool";
  }
  return "invalid";
}

std::string_view opcode_name(Opcode opcode) {
  switch (opcode) {
  case Opcode::Add:
    return "add";
  case Opcode::Multiply:
    return "multiply";
  case Opcode::RmsNormModulate:
    return "rms_norm_modulate";
  case Opcode::SwiGlu:
    return "swiglu";
  case Opcode::ResidualGate:
    return "residual_gate";
  case Opcode::Linear:
    return "linear";
  case Opcode::QkNormPartialRope:
    return "qk_norm_partial_rope";
  case Opcode::Attention:
    return "attention";
  case Opcode::Barrier:
    return "barrier";
  case Opcode::BiasAdd:
    return "bias_add";
  case Opcode::H3AdaLNSelect:
    return "h3_adaln_select";
  case Opcode::H3DeinterleaveQkv:
    return "h3_deinterleave_qkv";
  case Opcode::H3DeinterleaveQkvWeight:
    return "h3_deinterleave_qkv_weight";
  case Opcode::DequantizeInt4:
    return "dequantize_int4";
  case Opcode::DequantizeInt5:
    return "dequantize_int5";
  case Opcode::SiLU:
    return "silu";
  case Opcode::RmsNorm:
    return "rms_norm";
  case Opcode::Fill:
    return "fill";
  case Opcode::GatherRows:
    return "gather_rows";
  case Opcode::IndexedUpdateRows:
    return "indexed_update_rows";
  case Opcode::Cast:
    return "cast";
  case Opcode::SelectRowChunks:
    return "select_row_chunks";
  case Opcode::SinusoidalTimestep:
    return "sinusoidal_timestep";
  case Opcode::RotaryPosition:
    return "rotary_position";
  case Opcode::LinearBlend:
    return "linear_blend";
  case Opcode::FlowEulerStep:
    return "flow_euler_step";
  case Opcode::Patchify3D:
    return "patchify_3d";
  case Opcode::Unpatchify3D:
    return "unpatchify_3d";
  case Opcode::AffineLastDim:
    return "affine_last_dim";
  case Opcode::LayerNorm:
    return "layer_norm";
  case Opcode::Clamp:
    return "clamp";
  case Opcode::MseLoss:
    return "mse_loss";
  case Opcode::MseLossBackward:
    return "mse_loss_backward";
  case Opcode::LinearBackwardInput:
    return "linear_backward_input";
  case Opcode::LinearBackwardWeight:
    return "linear_backward_weight";
  case Opcode::BiasBackward:
    return "bias_backward";
  case Opcode::SiLUBackward:
    return "silu_backward";
  case Opcode::AdamWUpdate:
    return "adamw_update";
  case Opcode::RmsNormBackward:
    return "rms_norm_backward";
  case Opcode::RmsNormModulateBackward:
    return "rms_norm_modulate_backward";
  case Opcode::SwiGluBackward:
    return "swiglu_backward";
  case Opcode::ResidualGateBackward:
    return "residual_gate_backward";
  case Opcode::LayerNormBackward:
    return "layer_norm_backward";
  case Opcode::QkNormPartialRopeBackward:
    return "qk_norm_partial_rope_backward";
  case Opcode::AttentionLse:
    return "attention_lse";
  case Opcode::AttentionBackward:
    return "attention_backward";
  case Opcode::Conv1d:
    return "conv1d";
  case Opcode::SnakeBeta:
    return "snake_beta";
  case Opcode::Gelu:
    return "gelu";
  case Opcode::Sigmoid:
    return "sigmoid";
  case Opcode::Reshape:
    return "reshape";
  case Opcode::BroadcastTo:
    return "broadcast_to";
  case Opcode::Slice:
    return "slice";
  case Opcode::RotaryFrequency:
    return "rotary_frequency";
  case Opcode::RotaryApply:
    return "rotary_apply";
  case Opcode::BooleanMaskToBias:
    return "boolean_mask_to_bias";
  case Opcode::EulerVelocityStep:
    return "euler_velocity_step";
  case Opcode::Permute:
    return "permute";
  case Opcode::Concat:
    return "concat";
  case Opcode::Conv2d:
    return "conv2d";
  case Opcode::ChannelRmsNorm:
    return "channel_rms_norm";
  case Opcode::UpsampleNearest2d:
    return "upsample_nearest_2d";
  case Opcode::PadConstant:
    return "pad_constant";
  case Opcode::Conv3d:
    return "conv3d";
  case Opcode::GroupNorm:
    return "group_norm";
  case Opcode::PadReflect:
    return "pad_reflect";
  }
  return "invalid";
}

std::uint64_t TensorDesc::element_count() const {
  std::uint64_t count = 1;
  for (const auto dim : dims) {
    if (dim == 0 || count > std::numeric_limits<std::uint64_t>::max() / dim)
      fail("DiffIR tensor element count overflow or zero dimension");
    count *= dim;
  }
  return count;
}

std::uint64_t TensorDesc::byte_count() const {
  const auto count = element_count();
  const auto width = dtype_size(dtype);
  if (count > std::numeric_limits<std::uint64_t>::max() / width)
    fail("DiffIR tensor byte count overflow");
  return count * width;
}

const Attribute *Operation::find(AttrKey key) const {
  const auto it = std::find_if(attributes.begin(), attributes.end(),
                               [key](const Attribute &attr) { return attr.key == key; });
  return it == attributes.end() ? nullptr : &*it;
}

std::uint64_t Operation::u64(AttrKey key, std::uint64_t fallback) const {
  const auto *attr = find(key);
  return attr ? attr->as_u64() : fallback;
}

double Operation::f64(AttrKey key, double fallback) const {
  const auto *attr = find(key);
  return attr ? attr->as_f64() : fallback;
}

bool Operation::boolean(AttrKey key, bool fallback) const {
  const auto *attr = find(key);
  return attr ? attr->as_bool() : fallback;
}

const TensorDesc *Program::tensor(std::uint32_t id) const {
  const auto it = std::find_if(tensors.begin(), tensors.end(),
                               [id](const TensorDesc &desc) { return desc.id == id; });
  return it == tensors.end() ? nullptr : &*it;
}

std::vector<std::uint8_t> encode(const Program &program) {
  verify(program);
  Writer writer;
  writer.raw(kMagic);
  writer.u32(program.version);
  writer.u32(static_cast<std::uint32_t>(program.tensors.size()));
  writer.u32(static_cast<std::uint32_t>(program.operations.size()));
  for (const auto &tensor : program.tensors) {
    writer.u32(tensor.id);
    writer.u32(static_cast<std::uint32_t>(tensor.dtype));
    writer.u32(tensor.roles);
    writer.u32(static_cast<std::uint32_t>(tensor.dims.size()));
    for (const auto dim : tensor.dims)
      writer.u64(dim);
  }
  for (const auto &operation : program.operations) {
    writer.u32(operation.id);
    writer.u32(static_cast<std::uint32_t>(operation.opcode));
    writer.u32(static_cast<std::uint32_t>(operation.inputs.size()));
    writer.u32(static_cast<std::uint32_t>(operation.outputs.size()));
    writer.u32(static_cast<std::uint32_t>(operation.attributes.size()));
    for (const auto input : operation.inputs)
      writer.u32(input);
    for (const auto output : operation.outputs)
      writer.u32(output);
    for (const auto &attribute : operation.attributes) {
      writer.u32(static_cast<std::uint32_t>(attribute.key));
      writer.u32(static_cast<std::uint32_t>(attribute.kind));
      writer.u64(attribute.bits);
    }
  }
  auto bytes = writer.take();
  const auto digest = sha256(bytes);
  bytes.insert(bytes.end(), digest.begin(), digest.end());
  return bytes;
}

Program decode(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < kMagic.size() + 3U * sizeof(std::uint32_t) + kDigestBytes)
    fail("DiffIR file is too small");
  const auto payload = bytes.first(bytes.size() - kDigestBytes);
  const auto stored_digest = bytes.last(kDigestBytes);
  const auto actual_digest = sha256(payload);
  if (!std::equal(actual_digest.begin(), actual_digest.end(), stored_digest.begin()))
    fail("DiffIR SHA-256 mismatch");

  Reader reader(payload);
  const auto magic = reader.raw(kMagic.size());
  if (!std::equal(kMagic.begin(), kMagic.end(), magic.begin()))
    fail("invalid DiffIR magic");

  Program program;
  program.version = reader.u32();
  if (program.version != kVersion)
    fail("unsupported DiffIR version");
  const auto tensor_count = reader.u32();
  const auto operation_count = reader.u32();
  check_count(tensor_count, "tensor");
  check_count(operation_count, "operation");

  program.tensors.reserve(tensor_count);
  for (std::uint32_t i = 0; i < tensor_count; ++i) {
    TensorDesc tensor;
    tensor.id = reader.u32();
    tensor.dtype = static_cast<DType>(reader.u32());
    tensor.roles = reader.u32();
    const auto rank = reader.u32();
    if (rank == 0 || rank > kMaxRank)
      fail("invalid DiffIR tensor rank");
    tensor.dims.reserve(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis)
      tensor.dims.push_back(reader.u64());
    program.tensors.push_back(std::move(tensor));
  }

  program.operations.reserve(operation_count);
  for (std::uint32_t i = 0; i < operation_count; ++i) {
    Operation operation;
    operation.id = reader.u32();
    operation.opcode = static_cast<Opcode>(reader.u32());
    const auto input_count = reader.u32();
    const auto output_count = reader.u32();
    const auto attribute_count = reader.u32();
    check_count(input_count, "operation input");
    check_count(output_count, "operation output");
    check_count(attribute_count, "attribute");
    operation.inputs.reserve(input_count);
    operation.outputs.reserve(output_count);
    operation.attributes.reserve(attribute_count);
    for (std::uint32_t input = 0; input < input_count; ++input)
      operation.inputs.push_back(reader.u32());
    for (std::uint32_t output = 0; output < output_count; ++output)
      operation.outputs.push_back(reader.u32());
    for (std::uint32_t attr = 0; attr < attribute_count; ++attr) {
      Attribute value;
      value.key = static_cast<AttrKey>(reader.u32());
      value.kind = static_cast<AttrKind>(reader.u32());
      value.bits = reader.u64();
      operation.attributes.push_back(value);
    }
    program.operations.push_back(std::move(operation));
  }
  if (!reader.done())
    fail("trailing bytes in DiffIR payload");
  verify(program);
  return program;
}

void write_file(const Program &program, const std::filesystem::path &path) {
  const auto bytes = encode(program);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create DiffIR file: " + path.string());
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    fail("cannot write DiffIR file: " + path.string());
}

Program read_file(const std::filesystem::path &path) {
  const auto bytes = read_all(path);
  return decode(bytes);
}

Sha256Digest fingerprint(const Program &program) {
  const auto bytes = encode(program);
  return sha256(std::span<const std::uint8_t>(bytes).first(bytes.size() - kDigestBytes));
}

} // namespace dif::ir
