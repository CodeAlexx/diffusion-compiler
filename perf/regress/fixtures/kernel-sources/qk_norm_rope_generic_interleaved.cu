
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
// Generic RMS-normalized, partially rotated q/k: one block per
// (sequence, head); the reduction fragment matches the pinned PyTorch
// oracle's grouping, every product is rounded to the storage dtype.
// The rotation layout and the row offset into the rotation table are
// generated from the operation's own attributes, so this kernel rotates the
// way the operation says rather than assuming a half split.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_f32* cosv, const dif_f32* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long bh = blockIdx.x;
  if (bh >= 8ULL) return;
  unsigned long long s = bh / 2ULL, h = bh % 2ULL;
  unsigned long long base = (s * 2ULL + h) * 12ULL;
  unsigned long long tb = (s / 4ULL * 6ULL + 2ULL + s % 4ULL) * 6ULL;
  float local = 0.0f;
  // Match the CUDA vectorized RMSNorm reduction used by the pinned PyTorch
  // oracle: one warp consumes four adjacent values per lane, followed by a
  // 16,8,4,2,1 shuffle-style reduction.  The exact grouping matters at BF16
  // boundaries even though it changes only a handful of values.
  if (threadIdx.x < 32U) {
    unsigned long long d = (unsigned long long)threadIdx.x * 4ULL;
    if (d < 12ULL) {
      float v0 = dif_load(x, base + d); local += v0 * v0;
      float v1 = dif_load(x, base + d + 1ULL); local += v1 * v1;
      float v2 = dif_load(x, base + d + 2ULL); local += v2 * v2;
      float v3 = dif_load(x, base + d + 3ULL); local += v3 * v3;
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 16U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }

  float inv = rsqrtf(reduction[0] / 12.0f + 9.9999999747524271e-07f);
  for (unsigned long long d = threadIdx.x; d < 12ULL; d += blockDim.x) {
    float value = dif_round(dif_load(x, base + d) * inv * dif_load(weight, d));
    float result = value;
    if (d < 12ULL) {
      unsigned long long p = d / 2ULL, partner = (d % 2ULL == 0ULL) ? d + 1ULL : d - 1ULL;
      float other = dif_round(dif_load(x, base + partner) * inv * dif_load(weight, partner));
      float c = dif_load_f32(cosv, tb + p);
      float sn = dif_load_f32(sinv, tb + p);
      result = (d % 2ULL == 0ULL) ? (value * c - other * sn) : (other * sn + value * c);
    }
    dif_store(y, base + d, result);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
