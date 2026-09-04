
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
// Port of Serenity's accepted MiniMax-H3 fused Q/K RMSNorm + partial-RoPE
// primitive: one lane owns one head value of a 128-wide BF16 head, a
// 128-lane F32 reduction, the BF16 normalization boundary, then half-split
// rotation of the first `rotary` values.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 8ULL) return;
  unsigned long long base = row * 128ULL;
  float local = 0.0f;
  for (unsigned col = tid; col < 128U; col += 128U) {
    float value = dif_load(x, base + col);
    local = __fadd_rn(local, __fmul_rn(value, value));
  }
  reduction[tid] = local;
  __syncthreads();
  for (unsigned active = 64U; active > 0U; active >>= 1U) {
    if (tid < active) reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + active]);
    __syncthreads();
  }
  float inv = rsqrtf(__fadd_rn(__fdiv_rn(reduction[0], 128.0f), 9.9999999747524271e-07f));
  unsigned long long token = row / 2ULL,
                     table_token = (token / 4ULL) * 4ULL + 0ULL + token % 4ULL;
  if (tid < 64U) {
    unsigned lane = tid;
    float value0 = dif_load(x, base + lane);
    float value1 = dif_load(x, base + lane + 64ULL);
    float norm0 = dif_round(value0 * inv * dif_load(weight, lane));
    float norm1 = dif_round(value1 * inv * dif_load(weight, lane + 64ULL));
    unsigned long long table = table_token * 128ULL;
    float result0 = norm0 * dif_load(cosv, table + lane) - norm1 * dif_load(sinv, table + lane);
    float result1 = norm1 * dif_load(cosv, table + lane + 64ULL) + norm0 * dif_load(sinv, table + lane + 64ULL);
    dif_store(y, base + lane, result0);
    dif_store(y, base + lane + 64ULL, result1);
  } else if (tid >= 128U && tid < 128U) {
    float value = dif_load(x, base + tid);
    dif_store(y, base + tid, value * inv * dif_load(weight, tid));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
