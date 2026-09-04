
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
  // Rows divisible by 4, blocks of 128..511 threads: the first 128 threads
  // each square 4-column packs, then a 128-wide shared tree with the
  // two-stage fold that keeps the historical summation order.
  if (threadIdx.x < 128U) {
    for (unsigned long long pack = threadIdx.x; pack < 25ULL; pack += 128ULL) {
      unsigned long long col = pack * 4ULL;
      unsigned long long base = row * 100ULL + col;
      float v0 = dif_load(x, base); local += v0 * v0;
      float v1 = dif_load(x, base + 1ULL); local += v1 * v1;
      float v2 = dif_load(x, base + 2ULL); local += v2 * v2;
      float v3 = dif_load(x, base + 3ULL); local += v3 * v3;
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 16U; stride > 0U; stride >>= 1U) {
    unsigned lane = threadIdx.x & 31U;
    if (threadIdx.x < 128U && lane < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U) reduction[0] += reduction[64];
  else if (threadIdx.x == 32U) reduction[32] += reduction[96];
  __syncthreads();
  if (threadIdx.x == 0U) reduction[0] += reduction[32];
  __syncthreads();

  float inv = rsqrtf(reduction[0] / 100.0f + 9.9999999747524271e-07f);
  for (unsigned long long col = threadIdx.x; col < 100ULL; col += blockDim.x) {
    unsigned long long i = row * 100ULL + col;
    dif_store(y, i, dif_load(x, i) * inv * (dif_load(weight, col) + 0.000000000e+00f));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
