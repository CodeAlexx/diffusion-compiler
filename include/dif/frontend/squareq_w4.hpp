// SquareQ W4 (format tag squareq_w4_v1) as a DiffIR rewrite.
//
// A SquareQ slab stores each quantized Linear weight W [out,in] as
//   qweight   I8/U8 [out, in/2]   int4 nibbles (low nibble = even column,
//                                 two's complement) of the Hadamard-rotated
//                                 residual R_rot = (W - lora_up lora_down^T) H
//   wscales   BF16 [in/64, out]   per-(group of 64 inputs, output) scales
//   lora_down BF16 [in, R]
//   lora_up   BF16 [out, R]
// with H the block-diagonal normalized Hadamard-256 along the input dim,
// so that  W_hat = dequant(qweight, wscales) H + lora_up lora_down^T.
//
// The rewrite expresses exactly that with existing DiffIR semantics, per
// consuming Linear:
//   d  = DequantizeInt4(qweight, wscales^T; group 64)   -> BF16 [out, in]
//   r  = Reshape(Linear(Reshape(d, [out*in/256, 256]), H256), [out, in])
//   l  = Linear(lora_up, lora_down)                     -> lora_up lora_down^T
//   W' = Add(r, l);  Linear(x, W', ...) replaces Linear(x, W, ...)
// The BF16 weight binding is dropped; the slab tensors become constants
// (about 0.28x the BF16 bytes), which is what lets a model larger than the
// card's BF16 budget stay resident. Fails closed on a missing key, a shape
// or rank disagreement with squareq-plan.json, or a wrong format tag.
#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace dif::frontend {

struct SquareQW4RewriteResult {
  std::string format;
  std::uint32_t rank{};
  std::uint32_t linear_count{};
  std::uint64_t quantized_bytes{};    // slab bytes bound (q + scales + low-rank)
  std::uint64_t bf16_bytes_replaced{}; // BF16 weight bytes no longer bound
  double plan_cos_w_min{};
  std::vector<std::string> names;
};

// checkpoint_tensors[i] is the program tensor bound from checkpoint key
// checkpoint_names[i]. Every key present in the slab plan must be the weight
// of exactly one Linear in `program`; keys absent from the plan are left
// untouched. Returns the receipt; throws dif::Error on any disagreement.
SquareQW4RewriteResult rewrite_linear_weights_squareq_w4(
    ir::Program &program, runtime::TensorMap &bindings,
    std::span<const std::uint32_t> checkpoint_tensors,
    std::span<const std::string> checkpoint_names,
    const std::filesystem::path &slab_directory);

} // namespace dif::frontend
