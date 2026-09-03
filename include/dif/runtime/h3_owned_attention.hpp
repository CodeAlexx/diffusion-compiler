#pragma once

#include <cstdint>

// In-tree owned MiniMax-H3 dense INT8 attention (ABI v4 shapes). Compiled only
// when the build has a CUDA compiler (DIF_HAS_H3_OWNED_ATTENTION). Layouts:
// q/k/v/out are BF16 [B,S,H,128] addressed by (stride_b, stride_h, stride_s)
// in elements; k8 is INT8 [B,H,S_pad,128] with one F32 scale per 128-key
// tile, v8 is INT8 [B,H,128,S_pad] with one F32 scale per channel per head.
// Every function returns 0 on success or a cudaError_t value.
namespace dif::runtime::h3_owned_attention {

int abi_version();
int target_sm();
const char *cuda_error(int status);

int quantize_kv_bf16(const void *k, const void *v, void *k8, void *k_scale,
                     void *v8, void *v_scale, int batch, int heads,
                     int sequence, int padded_sequence, std::int64_t k_stride_b,
                     std::int64_t k_stride_h, std::int64_t k_stride_s,
                     std::int64_t v_stride_b, std::int64_t v_stride_h,
                     std::int64_t v_stride_s, void *stream);

// K mean-centering variant: k_mean_partials is F32 [B,H,S_pad/128,128]
// scratch, k_mean is F32 [B,H,128]; both must be device memory. Softmax is
// exactly invariant to the centering; only the INT8 quantization changes.
int quantize_kv_centered_bf16(const void *k, const void *v, void *k8,
                              void *k_scale, void *v8, void *v_scale,
                              void *k_mean_partials, void *k_mean, int batch,
                              int heads, int sequence, int padded_sequence,
                              std::int64_t k_stride_b, std::int64_t k_stride_h,
                              std::int64_t k_stride_s, std::int64_t v_stride_b,
                              std::int64_t v_stride_h, std::int64_t v_stride_s,
                              void *stream);

int attention_bf16(const void *q, const void *k8, const void *k_scale,
                   const void *v8, const void *v_scale, void *output,
                   int batch, int heads, int sequence, int padded_sequence,
                   float attention_scale, std::int64_t q_stride_b,
                   std::int64_t q_stride_h, std::int64_t q_stride_s,
                   std::int64_t out_stride_b, std::int64_t out_stride_h,
                   std::int64_t out_stride_s, void *stream);

int attention_config(int *warps, int *key_tile, int *q_in_registers);
int attention_geometry(int *queries_per_cta, int *rows_per_warp, int *key_tile,
                       int *k_scale_keys, int *v_scale_keys);

}  // namespace dif::runtime::h3_owned_attention
