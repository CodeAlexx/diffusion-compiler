#include "dif/opt/semantics.hpp"

namespace dif::opt {

bool pinned_numeric_semantics(const ir::Operation &op) {
  switch (op.opcode) {
  // Schedule construction, optimizer state transitions, and low-bit
  // dequantization all encode an exact reference formula. Re-expressing them at
  // another precision or through another math mode is not admissible.
  case ir::Opcode::SinusoidalTimestep:
  case ir::Opcode::FlowEulerStep:
  case ir::Opcode::EulerVelocityStep:
  case ir::Opcode::AdamWUpdate:
  case ir::Opcode::DequantizeInt4:
  case ir::Opcode::DequantizeInt5:
  case ir::Opcode::RotaryPosition:
  case ir::Opcode::Gelu:
    return true;
  default:
    break;
  }
  // An explicit accumulator dtype is a recorded accumulation decision.
  if (op.find(ir::AttrKey::AccumulatorDType) != nullptr)
    return true;
  // GateFirst records the source SwiGLU gate/up ordering; FlipSinToCos and the
  // frequency-shift attributes record embedding conventions. Any of them means
  // the operation exists to reproduce a specific reference expression.
  if (op.find(ir::AttrKey::GateFirst) != nullptr ||
      op.find(ir::AttrKey::FlipSinToCos) != nullptr ||
      op.find(ir::AttrKey::DownscaleFreqShift) != nullptr ||
      op.find(ir::AttrKey::MaxPeriod) != nullptr)
    return true;
  // Implementation 3 is the direct packed INT5 Linear: its packing is part of
  // the operation's meaning.
  if (op.opcode == ir::Opcode::Linear &&
      op.u64(ir::AttrKey::Implementation, 1U) == 3U)
    return true;
  return false;
}

bool pure_operation(ir::Opcode opcode) { return opcode != ir::Opcode::Barrier; }

bool bit_exact_data_movement(ir::Opcode opcode) {
  switch (opcode) {
  case ir::Opcode::H3DeinterleaveQkv:
  case ir::Opcode::H3DeinterleaveQkvWeight:
  case ir::Opcode::H3AdaLNSelect:
  case ir::Opcode::GatherRows:
  case ir::Opcode::IndexedUpdateRows:
  case ir::Opcode::SelectRowChunks:
  case ir::Opcode::Fill:
  // Cast is admitted only when the discovery pass has established that the
  // conversion widens, which is exact; see rewrite.cpp.
  case ir::Opcode::Cast:
  case ir::Opcode::Permute:
  case ir::Opcode::Concat:
    return true;
  default:
    return false;
  }
}

bool dtype_uniform(ir::Opcode opcode) {
  switch (opcode) {
  case ir::Opcode::Add:
  case ir::Opcode::Multiply:
  case ir::Opcode::SiLU:
  case ir::Opcode::Gelu:
  case ir::Opcode::RmsNorm:
  case ir::Opcode::RmsNormModulate:
  case ir::Opcode::SwiGlu:
  case ir::Opcode::ResidualGate:
  case ir::Opcode::Linear:
  case ir::Opcode::Attention:
  case ir::Opcode::BiasAdd:
  case ir::Opcode::AffineLastDim:
  case ir::Opcode::LayerNorm:
  case ir::Opcode::Clamp:
  case ir::Opcode::QkNormPartialRope:
  case ir::Opcode::Conv2d:
  case ir::Opcode::ChannelRmsNorm:
  case ir::Opcode::UpsampleNearest2d:
  case ir::Opcode::PadConstant:
  case ir::Opcode::Conv3d:
    return true;
  default:
    return false;
  }
}

} // namespace dif::opt
