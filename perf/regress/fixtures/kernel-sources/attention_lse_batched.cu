
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
#define dif_scalar dif_f32
#define dif_load dif_load_f32
#define dif_store dif_store_f32
#define dif_round dif_round_f32
// Log-sum-exp of the scaled scores per (query, head), the value the
// decomposed attention backward reuses. Serial F32 reference form.
extern "C" __global__ void dif_op_1(const dif_bf16* q, const dif_bf16* k, dif_f32* lse) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 16ULL) {
    unsigned long long qs = i / 2ULL, h = i % 2ULL, qb = (qs * 2ULL + h) * 8ULL,
                       kend = 4ULL;
    float maximum = -3.402823466e+38f;
    for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
      float score = 0.0f;
      unsigned long long kb = ((qs / 4ULL * 4ULL + ks) * 2ULL + h) * 8ULL;
      for (unsigned long long d = 0ULL; d < 8ULL; ++d) score = fmaf(dif_load_bf16(q, qb + d), dif_load_bf16(k, kb + d), score);
      score *= 3.535533845e-01f;
      maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0f;
    for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
      float score = 0.0f;
      unsigned long long kb = ((qs / 4ULL * 4ULL + ks) * 2ULL + h) * 8ULL;
      for (unsigned long long d = 0ULL; d < 8ULL; ++d) score = fmaf(dif_load_bf16(q, qb + d), dif_load_bf16(k, kb + d), score);
      denominator += expf(score * 3.535533845e-01f - maximum);
    }
    dif_store_f32(lse, i, maximum + logf(denominator));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
