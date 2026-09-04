
typedef float dif_f32;
typedef unsigned short dif_bf16;
typedef unsigned short dif_f16;
extern "C" __device__ float dif_load_f32(const dif_f32* value, unsigned long long index) {
  return value[index];
}
extern "C" __device__ void dif_store_f32(dif_f32* value, unsigned long long index, float input) {
  value[index] = input;
}
extern "C" __device__ float dif_round_f32(float input) { return input; }
extern "C" __device__ float dif_load_bf16(const dif_bf16* value, unsigned long long index) {
  return __uint_as_float((unsigned int)value[index] << 16U);
}
extern "C" __device__ void dif_store_bf16(dif_bf16* value, unsigned long long index, float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  value[index] = (dif_bf16)((bits + rounding) >> 16U);
}
extern "C" __device__ float dif_round_bf16(float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return __uint_as_float(((bits + rounding) >> 16U) << 16U);
}
extern "C" __device__ float dif_load_f16(const dif_f16* value, unsigned long long index) {
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value[index]));
  return result;
}
extern "C" __device__ void dif_store_f16(dif_f16* value, unsigned long long index, float input) {
  dif_f16 result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(input));
  value[index] = result;
}
extern "C" __device__ float dif_round_f16(float input) {
  dif_f16 rounded;
  float result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(rounded) : "f"(input));
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(rounded));
  return result;
}
extern "C" __device__ float dif_silu(float x) {
  return x / (1.0f + expf(-x));
}
#define dif_scalar dif_bf16
#define dif_load dif_load_bf16
#define dif_store dif_store_bf16
#define dif_round dif_round_bf16
// Decomposed attention backward (reference form). Thread i = (s,h,d) over
// the QUERY geometry computes dq[s,h,d]; threads with h < KvHeads also own
// dk/dv[s,h,d], accumulating in F32 across every query and every query head
// of their group (grouped-KV gradient accumulation). P is recomputed from
// Q,K and the saved F32 logsumexp; delta = rowsum(dO*O) uses the forward
// output. O(S) score recomputations per thread: acceptable at gate scale,
// cuDNN SDPA backward stays future work.
extern "C" __global__ void dif_op_1(const dif_bf16* grad_output, const dif_bf16* q, const dif_bf16* k, const dif_bf16* v, const dif_bf16* forward_output, const dif_f32* lse, dif_bf16* grad_q, dif_bf16* grad_k, dif_bf16* grad_v) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64ULL) {
    unsigned long long row = i / 8ULL, d = i % 8ULL, s = row / 2ULL, h = row % 2ULL, base = row * 8ULL;
    float dq = 0.0f;
    {
      unsigned long long qb = base, kh = h / 1ULL, kend = s + 1ULL;
      float row_lse = dif_load_f32(lse, row);
      float delta = 0.0f;
      for (unsigned long long e = 0ULL; e < 8ULL; ++e) delta = fmaf(dif_load_bf16(grad_output, qb + e), dif_load_bf16(forward_output, qb + e), delta);
      for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
        unsigned long long kb = (ks * 2ULL + kh) * 8ULL;
        float score = 0.0f, projected = 0.0f;
        for (unsigned long long e = 0ULL; e < 8ULL; ++e) {
          score = fmaf(dif_load_bf16(q, qb + e), dif_load_bf16(k, kb + e), score);
          projected = fmaf(dif_load_bf16(grad_output, qb + e), dif_load_bf16(v, kb + e), projected);
        }
        float probability = expf(score * 3.535533845e-01f - row_lse);
        dq = fmaf(probability * (projected - delta) * 3.535533845e-01f, dif_load_bf16(k, kb + d), dq);
      }
    }
    dif_store_bf16(grad_q, i, dq);
    if (h < 2ULL) {
      unsigned long long kb = (s * 2ULL + h) * 8ULL + d;
      float dk = 0.0f, dv = 0.0f;
      unsigned long long kvb = (s * 2ULL + h) * 8ULL;
      for (unsigned long long g = 0ULL; g < 1ULL; ++g) {
        unsigned long long qh = h * 1ULL + g;
        for (unsigned long long qs = s; qs < 4ULL; ++qs) {
          unsigned long long qb = (qs * 2ULL + qh) * 8ULL;
          float row_lse = dif_load_f32(lse, qs * 2ULL + qh);
          float score = 0.0f, projected = 0.0f, delta = 0.0f;
          for (unsigned long long e = 0ULL; e < 8ULL; ++e) {
            score = fmaf(dif_load_bf16(q, qb + e), dif_load_bf16(k, kvb + e), score);
            projected = fmaf(dif_load_bf16(grad_output, qb + e), dif_load_bf16(v, kvb + e), projected);
            delta = fmaf(dif_load_bf16(grad_output, qb + e), dif_load_bf16(forward_output, qb + e), delta);
          }
          float probability = expf(score * 3.535533845e-01f - row_lse);
          dk = fmaf(probability * (projected - delta) * 3.535533845e-01f, dif_load_bf16(q, qb + d), dk);
          dv = fmaf(probability, dif_load_bf16(grad_output, qb + d), dv);
        }
      }
      dif_store_bf16(grad_k, kb, dk);
      dif_store_bf16(grad_v, kb, dv);
    }
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
