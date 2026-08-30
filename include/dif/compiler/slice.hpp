#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>

namespace dif::compiler {

// Extract a contiguous operation-id range as a standalone, verifiable
// program. Values entering the range become inputs, referenced constants
// retain their residency, and values leaving the range become outputs.
// Tensor and operation ids stay stable for source-capture traceability.
ir::Program slice_operations(const ir::Program &program,
                             std::uint32_t first_operation,
                             std::uint32_t last_operation);


} // namespace dif::compiler
