
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
// Shared-vector-delta rms_norm_modulate, BF16 6144-wide rows, 512 threads,
// Triton reduction-tile 2048 order: y = (1 + (vector + delta_scale)) *
// (x * inv * (weight + offset)) + (vector + delta_shift), every step a
// separately rounded PTX op.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vector, const dif_scalar* delta, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 2ULL) return;
  unsigned long long base = row * 6144ULL + (unsigned long long)tid * 4ULL;
  float a0, a1, a2, a3;
  float m0 = dif_load(x, base + 2048ULL), m1 = dif_load(x, base + 2049ULL), m2 = dif_load(x, base + 2050ULL),
        m3 = dif_load(x, base + 2051ULL);
  float l0 = dif_load(x, base), l1 = dif_load(x, base + 1ULL), l2 = dif_load(x, base + 2ULL), l3 = dif_load(x, base + 3ULL);
  a0 = fmaf(l0, l0, m0 * m0); a1 = fmaf(l1, l1, m1 * m1); a2 = fmaf(l2, l2, m2 * m2); a3 = fmaf(l3, l3, m3 * m3);
  float h0 = dif_load(x, base + 4096ULL), h1 = dif_load(x, base + 4097ULL), h2 = dif_load(x, base + 4098ULL),
        h3 = dif_load(x, base + 4099ULL);
  a0 = fmaf(h0, h0, a0); a1 = fmaf(h1, h1, a1); a2 = fmaf(h2, h2, a2); a3 = fmaf(h3, h3, a3);
  float local = ((a0 + a1) + a2) + a3;
  for (unsigned delta_step = 16U; delta_step > 0U; delta_step >>= 1U) local += __shfl_xor_sync(0xffffffffU, local, delta_step);
  unsigned lane = tid & 31U, warp = tid >> 5U;
  if (lane == 0U) reduction[warp] = local;
  __syncthreads();
  if (warp == 0U) {
    local = lane < 16U ? reduction[lane] : 0.0f;
    for (unsigned delta_step = 8U; delta_step > 0U; delta_step >>= 1U) local += __shfl_xor_sync(0xffffffffU, local, delta_step);
    if (lane == 0U) reduction[0] = local;
  }
  __syncthreads();
  float mean, mean_eps, inv;
  asm volatile("div.full.f32 %0,%1,%2;" : "=f"(mean) : "f"(reduction[0]), "f"(6144.0f));
  mean_eps = mean + 9.9999999747524271e-07f;
  asm volatile("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(mean_eps));
  unsigned long long shared_base = (row / 2ULL) * 6144ULL;
  for (unsigned long long col = tid; col < 6144ULL; col += blockDim.x) {
    unsigned long long i = row * 6144ULL + col;
    float xv = dif_load(x, i), wv = dif_load(weight, col), base_value = dif_load(vector, shared_base + col),
          scale_delta = dif_load(delta, col), shift_delta = dif_load(delta, 6144ULL + col);
    float normalized, weighted, scale, scale_one, shift, result;
    asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(normalized) : "f"(xv), "f"(inv));
    float weight_value = wv + 1.000000000e+00f;
    asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(weighted) : "f"(normalized), "f"(weight_value));
    asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(scale) : "f"(base_value), "f"(scale_delta));
    scale_one = scale + 1.0f;
    asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(shift) : "f"(base_value), "f"(shift_delta));
    result = fmaf(scale_one, weighted, shift);
    dif_store(y, i, result);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
