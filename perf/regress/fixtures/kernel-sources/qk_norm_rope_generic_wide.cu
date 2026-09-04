
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
// Generic RMS-normalized, partially rotated (half-split) q/k: one block per
// (sequence, head); the reduction fragment matches the pinned PyTorch
// oracle's grouping, every product is rounded to the storage dtype.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long bh = blockIdx.x;
  if (bh >= 8ULL) return;
  unsigned long long s = bh / 2ULL, h = bh % 2ULL;
  unsigned long long base = (s * 2ULL + h) * 200ULL;
  float local = 0.0f;
  for (unsigned long long d = threadIdx.x; d < 200ULL; d += blockDim.x) {
    float v = dif_load(x, base + d);
    local += v * v;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }

  float inv = rsqrtf(reduction[0] / 200.0f + 9.9999999747524271e-07f);
  for (unsigned long long d = threadIdx.x; d < 200ULL; d += blockDim.x) {
    float value = dif_round(dif_load(x, base + d) * inv * dif_load(weight, d));
    float result = value;
    if (d < 100ULL) {
      float other = dif_round(dif_load(x, base + d + 100ULL) * inv * dif_load(weight, d + 100ULL));
      float left = dif_round(value * dif_load(cosv, s * 100ULL + d));
      float right = dif_round(other * dif_load(sinv, s * 100ULL + d));
      result = dif_round(left - right);
    } else if (d < 200ULL) {
      unsigned long long r = d - 100ULL;
      float other = dif_round(dif_load(x, base + r) * inv * dif_load(weight, r));
      unsigned long long ti = r;
      float left = dif_round(value * dif_load(cosv, s * 100ULL + ti));
      float right = dif_round(other * dif_load(sinv, s * 100ULL + ti));
      result = dif_round(left + right);
    }
    dif_store(y, base + d, result);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
