#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <vector>

namespace dif::compiler {

// Weight-only INT8 residency for a FROZEN linear.
//
// The arithmetic is not new. It is the symmetric per-row absmax the
// reference implementation uses and that this compiler already implements as
// QuantizeInt8Rows in its Direct mode:
//
//   scale[n] = max(max_k |W[n,k]| / 127, 1e-30)
//   q[n,k]   = clamp(nearbyint(W[n,k] / scale[n]), -127, 127)
//
// It is repeated here as a host function because a frozen weight should be
// converted ONCE, when it is loaded, not by a graph operation that would run
// again on every step. There being two copies of one arithmetic is a real
// risk, so a test runs both and requires the bytes to be identical.
//
// Two deliberate differences from the reference trainer's W8A8 base, both
// worth stating because they are choices and not oversights.
//
// It scales per output row; the reference scales per tensor. A single scalar
// factors out of the backward pass's contraction, which lets an ordinary
// INT8 matmul compute the input gradient. A per-row scale cannot factor out,
// so the input gradient needs its own kernel -- which is why
// LinearInt8WeightScaledBackwardInput exists. The per-row scale is the more
// accurate of the two, and a dedicated kernel is a smaller price than a
// coarser quantization of a 12-billion-parameter base.
//
// It keeps no transposed copy. The reference caches the weight twice, [N,K]
// for the forward and [K,N] for the backward, so both are ordinary matmuls.
// Here the gradient kernel reads the [N,K] weight with a strided access
// instead. That trades coalescing for not spending a second copy of the
// entire frozen base -- 11.6 GiB, on a model that does not currently fit.
//
// What is NOT a difference: the input gradient is the only gradient. A
// weight this format holds is frozen, so there is no weight gradient to
// compute, and the reference says the same thing in its own words.
struct Int8QuantizedWeight {
  runtime::Tensor weight;  // I8 [N, K]
  runtime::Tensor scales;  // F32 [N]
  // What the rounding cost, so a caller can report it rather than assume it.
  double squared_error{};
  double squared_reference{};
  float maximum_absolute_error{};
};

Int8QuantizedWeight quantize_int8_weight(const runtime::Tensor &source);

struct Int8WeightOnlyEntry {
  std::uint32_t operation{};        // the Linear that was rewritten
  std::uint32_t source_tensor_id{}; // the float weight it used to read
  std::uint32_t weight_tensor_id{}; // the I8 weight it reads now
  std::uint32_t scales_tensor_id{};
};

struct Int8WeightOnlyRewrite {
  ir::Program program;
  std::vector<Int8WeightOnlyEntry> entries;
  std::uint64_t bytes_before{};
  std::uint64_t bytes_after{};
};

// Rewrites the named Linear operations to read an INT8 weight directly.
//
// No dequantization is inserted, which is the entire point. A graph that
// dequantizes a frozen weight before each matmul pays for the conversion
// once per linear per step forever; the reference trainer does exactly that
// and it costs it an order of magnitude. Here the GEMM consumes the INT8
// bytes, and so does the input gradient.
//
// Only unbiased Linears whose weight is a Constant used by nothing else are
// admissible. Anything else is refused by name rather than skipped, because a
// weight that silently stayed BF16 is a memory plan that silently missed.
Int8WeightOnlyRewrite
rewrite_int8_weight_only(const ir::Program &program,
                         const std::vector<std::uint32_t> &operations);

} // namespace dif::compiler
