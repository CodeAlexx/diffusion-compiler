
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
// RMS normalization, one block per row: sum of squares reduced into
// reduction[0] by the reduction strategy the emitter selected for the row
// width and block size, then y = x * rsqrt(mean + eps) * (weight + offset).
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  float local = 0.0f;
  if (row >= 4ULL) return;
  // 128-wide rows, 128 threads: 16 lanes take 8 values each, one warp folds.
  if (threadIdx.x < 16U) {
    unsigned long long base = row * 128ULL + (unsigned long long)threadIdx.x * 8ULL;
    float v0 = dif_load(x, base), v1 = dif_load(x, base + 1ULL),
          v2 = dif_load(x, base + 2ULL), v3 = dif_load(x, base + 3ULL),
          v4 = dif_load(x, base + 4ULL), v5 = dif_load(x, base + 5ULL),
          v6 = dif_load(x, base + 6ULL), v7 = dif_load(x, base + 7ULL);
    local = v1 * v1; local = fmaf(v0, v0, local);
    local = fmaf(v2, v2, local); local = fmaf(v3, v3, local);
    local = fmaf(v4, v4, local); local = fmaf(v5, v5, local);
    local = fmaf(v6, v6, local); local = fmaf(v7, v7, local);
  } else local = 0.0f;
  if (threadIdx.x < 32U) {
    for (unsigned delta = 8U; delta > 0U; delta >>= 1U)
      local += __shfl_xor_sync(0xffffffffU, local, delta);
    if (threadIdx.x == 0U) reduction[0] = local;
  }
  __syncthreads();

  float mean, mean_eps, inv;
  asm volatile("div.full.f32 %0,%1,%2;" : "=f"(mean) : "f"(reduction[0]), "f"(128.0f));
  mean_eps = mean + 9.9999999747524271e-07f;
  asm volatile("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(mean_eps));
  for (unsigned long long col = threadIdx.x; col < 128ULL; col += blockDim.x) {
    unsigned long long i = row * 128ULL + col;
    dif_store(y, i, dif_load(x, i) * inv * (dif_load(weight, col) + 0.000000000e+00f));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
