#include "dif/runtime/flash_attention.hpp"

#include <cutlass/numeric_types.h>

#include "flash.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

namespace dif::runtime {
namespace {

thread_local std::string last_error;

template <typename T> bool fits_int(T value) {
  return value <= static_cast<T>(std::numeric_limits<int>::max());
}

} // namespace

const char *flash_attention_bf16_forward(
    std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
    std::uintptr_t output, std::uintptr_t workspace, std::uint32_t batch,
    std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t key_value_heads, std::uint32_t head_dimension, float scale,
    std::uintptr_t stream) noexcept {
  try {
    if (!query || !key || !value || !output || !workspace || batch == 0U ||
        sequence == 0U || heads == 0U || key_value_heads == 0U ||
        heads % key_value_heads != 0U || head_dimension != 128U ||
        !std::isfinite(scale) || !(scale > 0.0F) || !fits_int(batch) ||
        !fits_int(sequence) || !fits_int(heads) ||
        !fits_int(key_value_heads))
      throw std::runtime_error("invalid native FlashAttention BF16 shape");

    FLASH_NAMESPACE::Flash_fwd_params params{};
    params.q_ptr = reinterpret_cast<void *>(query);
    params.k_ptr = reinterpret_cast<void *>(key);
    params.v_ptr = reinterpret_cast<void *>(value);
    params.o_ptr = reinterpret_cast<void *>(output);
    params.softmax_lse_ptr = reinterpret_cast<void *>(workspace);
    params.q_row_stride = static_cast<std::int64_t>(heads) * head_dimension;
    params.k_row_stride =
        static_cast<std::int64_t>(key_value_heads) * head_dimension;
    params.v_row_stride = params.k_row_stride;
    params.o_row_stride = params.q_row_stride;
    params.q_head_stride = head_dimension;
    params.k_head_stride = head_dimension;
    params.v_head_stride = head_dimension;
    params.o_head_stride = head_dimension;
    params.q_batch_stride =
        static_cast<std::int64_t>(sequence) * params.q_row_stride;
    params.k_batch_stride =
        static_cast<std::int64_t>(sequence) * params.k_row_stride;
    params.v_batch_stride = params.k_batch_stride;
    params.o_batch_stride = params.q_batch_stride;
    params.b = static_cast<int>(batch);
    params.h = static_cast<int>(heads);
    params.h_k = static_cast<int>(key_value_heads);
    params.h_h_k_ratio = static_cast<int>(heads / key_value_heads);
    params.seqlen_q = static_cast<int>(sequence);
    params.seqlen_k = static_cast<int>(sequence);
    params.seqlen_q_rounded = static_cast<int>((sequence + 127U) / 128U * 128U);
    params.seqlen_k_rounded = params.seqlen_q_rounded;
    params.d = static_cast<int>(head_dimension);
    params.d_rounded = static_cast<int>(head_dimension);
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * 1.4426950408889634F;
    params.p_dropout = 1.0F;
    params.p_dropout_in_uint8_t = 255U;
    params.rp_dropout = 1.0F;
    params.scale_softmax_rp_dropout = scale;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_bf16 = true;
    params.is_causal = false;
    params.is_seqlens_k_cumulative = true;
    params.num_splits = 0;

    FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, false>(
        params, reinterpret_cast<cudaStream_t>(stream));
    last_error.clear();
    return nullptr;
  } catch (const std::exception &error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown native FlashAttention failure";
  }
  return last_error.c_str();
}

} // namespace dif::runtime
