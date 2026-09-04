
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
// Layer norm + adaLN modulation, generic path: one block per row, two
// shared-memory tree reductions (mean, then centered variance), then the row
// is normalized, affine transformed and modulated as
// (1 + scale) * normalized + shift with rounding at every step the reference
// applies.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= 4ULL) return;
  float local = 0.0f;
  for (unsigned long long col = threadIdx.x; col < 6ULL; col += blockDim.x) local += dif_load(x, row * 6ULL + col);
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float mean = reduction[0] / 6.0f;
  __syncthreads();
  local = 0.0f;
  for (unsigned long long col = threadIdx.x; col < 6ULL; col += blockDim.x) {
    float centered = dif_load(x, row * 6ULL + col) - mean;
    local += centered * centered;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(reduction[0] / 6.0f + 9.9999999747524271e-07f);
  unsigned long long modulation_row = row / 4ULL;
  for (unsigned long long col = threadIdx.x; col < 6ULL; col += blockDim.x) {
    unsigned long long i = row * 6ULL + col, mi = modulation_row * 6ULL + col;
    float normalized = dif_round((dif_load(x, i) - mean) * inv * dif_load(weight, col) + dif_load(bias, col));
    float one_plus_scale = dif_round(1.0f + dif_load(scale, mi));
    float scaled = dif_round(normalized * one_plus_scale);
    dif_store(y, i, scaled + dif_load(shift, mi));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
