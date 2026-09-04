
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
// Implementation 2: BF16 128-wide heads, interleaved rotary with a
// half-width f32 table: one warp squares 4 values per lane (mul.rn), every
// thread owns one element; even/odd partners are rotated with separately
// rounded PTX ops after the BF16 norm boundary.
extern "C" __global__ void dif_op_1(const dif_bf16* x, const dif_bf16* weight, const dif_f32* cosv, const dif_f32* sinv, dif_bf16* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 8ULL) return;
  unsigned long long base = row * 128ULL;
  float sigma2 = 0.0f;
  if (tid < 32U) {
    unsigned long long lane_base = base + (unsigned long long)tid;
    float v0 = dif_load_bf16(x, lane_base), v1 = dif_load_bf16(x, lane_base + 32ULL),
          v2 = dif_load_bf16(x, lane_base + 64ULL), v3 = dif_load_bf16(x, lane_base + 96ULL), s0, s1, s2, s3;
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s0) : "f"(v0));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s1) : "f"(v1));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s2) : "f"(v2));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s3) : "f"(v3));
    sigma2 = ((s0 + s1) + s2) + s3;
    for (unsigned offset = 1U; offset < 32U; offset <<= 1U) sigma2 = sigma2 + __shfl_down_sync(0xffffffffU, sigma2, offset);
  }
  if (tid == 0U) reduction[0] = sigma2 * 0.0078125f;
  __syncthreads();
  float inv = rsqrtf(reduction[0] + 9.9999999747524271e-07f);
  if (tid < 128U) {
    unsigned pair = tid / 2U, even = pair * 2U;
    float e = dif_round_bf16(dif_load_bf16(x, base + even) * inv);
    e = dif_round_bf16(e * dif_load_bf16(weight, even));
    float o = dif_round_bf16(dif_load_bf16(x, base + even + 1U) * inv);
    o = dif_round_bf16(o * dif_load_bf16(weight, even + 1U));
    unsigned long long token = row / 2ULL,
                       table_token = (token / 4ULL) * 4ULL + 0ULL + token % 4ULL,
                       table = table_token * 64ULL + pair;
    float c = dif_load_f32(cosv, table), s = dif_load_f32(sinv, table), first, second, result;
    if (tid & 1U) {
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(e), "f"(s));
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(o), "f"(c));
      asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
    } else {
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(e), "f"(c));
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(o), "f"(s));
      asm volatile("sub.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
    }
    dif_store_bf16(y, base + tid, result);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
