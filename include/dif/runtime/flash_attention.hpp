#pragma once

#include <cstdint>

namespace dif::runtime {

// Native forward-only FlashAttention lowering. Q/K/V/O are contiguous
// [B,S,H,D] BF16 tensors and workspace is F32 [B,H,S] log-sum-exp storage.
// Returns nullptr on success and a stable diagnostic string on failure.
const char *flash_attention_bf16_forward(
    std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
    std::uintptr_t output, std::uintptr_t workspace, std::uint32_t batch,
    std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t key_value_heads, std::uint32_t head_dimension, float scale,
    std::uintptr_t stream) noexcept;

} // namespace dif::runtime
