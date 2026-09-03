#include "dif/opt/semantics.hpp"

namespace dif::opt {

bool pinned_numeric_semantics(const ir::Operation &op) {
  // Exact reference formulas (schedules, optimizer updates, low-bit codecs,
  // scaled GEMMs, ...) are declared in opcodes.def as PinnedNumerics.
  if (ir::opcode_has_trait(op.opcode, ir::opcode_trait::PinnedNumerics))
    return true;
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
  return ir::opcode_has_trait(opcode, ir::opcode_trait::DataMovement);
}

bool dtype_uniform(ir::Opcode opcode) {
  return ir::opcode_has_trait(opcode, ir::opcode_trait::DtypeUniform);
}

} // namespace dif::opt
