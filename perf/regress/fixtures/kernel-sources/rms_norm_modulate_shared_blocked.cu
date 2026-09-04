
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
// Triton reduction-tile 8192 order (8 contiguous values per thread plus a
// 256-thread tail with mul.rn squares); modulation as in the chunked form.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vector, const dif_scalar* delta, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 2ULL) return;
  unsigned long long base = row * 6144ULL + (unsigned long long)tid * 8ULL;
  float v0 = dif_load(x, base), v1 = dif_load(x, base + 1ULL), v2 = dif_load(x, base + 2ULL), v3 = dif_load(x, base + 3ULL),
        v4 = dif_load(x, base + 4ULL), v5 = dif_load(x, base + 5ULL), v6 = dif_load(x, base + 6ULL), v7 = dif_load(x, base + 7ULL);
  float local = v1 * v1;
  local = fmaf(v0, v0, local); local = fmaf(v2, v2, local); local = fmaf(v3, v3, local); local = fmaf(v4, v4, local);
  local = fmaf(v5, v5, local); local = fmaf(v6, v6, local); local = fmaf(v7, v7, local);
  if (tid < 256U) {
    float square, value;
    value = dif_load(x, base + 4096ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4097ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4098ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4099ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4100ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4101ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4102ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4103ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
  }
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
