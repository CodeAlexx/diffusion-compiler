#pragma once

#include "dif/ir/ir.hpp"

namespace dif::opt {

// Source-fidelity guard rails consulted by every rewrite.
//
// Some DiffIR operations carry attributes that record an explicit rounding,
// accumulation, ordering, or packing decision taken by the frontend to match a
// reference implementation. The optimizer is allowed to move such an operation
// around, but it must never reinterpret those decisions. `pinned_numeric_semantics`
// identifies those operations so numeric-class transforms skip them; structural
// transforms copy the attributes through verbatim.
bool pinned_numeric_semantics(const ir::Operation &op);

// Deterministic and free of side effects, so two occurrences with equal inputs
// and attributes produce equal results. Required for common-subexpression
// elimination and for dead-operation elimination.
bool pure_operation(ir::Opcode opcode);

// Data movement or a widening conversion whose result is bit-identical on every
// conforming backend. Only these opcodes are folded into constants by default:
// folding an arithmetic operation would bake the reference backend's rounding
// into the program.
bool bit_exact_data_movement(ir::Opcode opcode);

// Every floating-point operand of the operation shares one dtype, so the whole
// operation can be re-expressed at another floating-point precision by casting
// its inputs and outputs.
bool dtype_uniform(ir::Opcode opcode);

} // namespace dif::opt
