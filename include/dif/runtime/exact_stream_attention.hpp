#pragma once

#include <cstdint>

namespace dif::runtime {

// Project-owned exact dense attention lowering (Implementation 5).  Q/K/V/O
// are contiguous [S,H,128] or [B,S,H,128] BF16 tensors.  Softmax state and
// cross-tile output accumulation are FP32; the per-tile 32-term P*V partial
// uses the full-rate f16 tensor-core path before being folded into the FP32
// accumulator.  No global workspace is required.  Returns nullptr on success
// and a stable diagnostic string on failure.
const char *exact_stream_attention_bf16_forward(
    std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
    std::uintptr_t output, std::uint32_t batch, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t key_value_heads,
    std::uint32_t head_dimension, float scale, std::uintptr_t stream) noexcept;

} // namespace dif::runtime
