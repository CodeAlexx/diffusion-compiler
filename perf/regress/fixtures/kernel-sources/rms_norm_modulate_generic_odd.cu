
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
// Generic rms_norm + (1 + scale) * . + shift with explicit storage-dtype
// rounding after the norm, the modulation product and the shift; the row
// reduction fragment is chosen by the emitter for the width and block size.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned row = blockIdx.x;
  float local = 0.0f;
  if (row >= 4ULL) return;
  for (unsigned long long col = threadIdx.x; col < 6ULL; col += blockDim.x) {
    float v = dif_load(x, (unsigned long long)row * 6ULL + col);
    local += v * v;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }

  float inv = rsqrtf(reduction[0] / 6.0f + 9.9999999747524271e-07f);
  for (unsigned long long col = threadIdx.x; col < 6ULL; col += blockDim.x) {
    unsigned long long i = (unsigned long long)row * 6ULL + col;
    float value = dif_load(x, i) * inv;
    value = dif_round(value);
    float modulation = dif_round(1.0f + dif_load(scale, i));
    value = dif_round(value * modulation);
    dif_store(y, i, value + dif_load(shift, i));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
