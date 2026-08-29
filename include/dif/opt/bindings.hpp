#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"

#include <cstdint>

namespace dif::opt {

// Deterministic experiment bindings for a verified program.
//
// Optimization is a measurement discipline: to compare candidates you need a
// fixed, reproducible input and constant set. This produces one from the
// program's own declarations, so an A/B experiment can be rerun from a clean
// build without distributing checkpoints. Integer index tensors are filled with
// values inside the range their consuming operation admits, so the result is
// executable rather than merely well-typed.
//
// These values are a controlled experiment fixture. They are not a model, and
// results measured against them say nothing about output quality.
runtime::TensorMap synthesize_bindings(const ir::Program &program,
                                       std::uint64_t seed);

} // namespace dif::opt
