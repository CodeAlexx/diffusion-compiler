
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
// Exact attention reference: one block per (query, head) of the whole
// batch; scores reduced through shared memory, a serial softmax on thread 0,
// then the value accumulation. Grouped-query heads read kv head h / group,
// and the key base carries the batch offset when there is a batch.
extern "C" __global__ void dif_op_1(const dif_scalar* q, const dif_scalar* k, const dif_scalar* v, dif_scalar* y) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* probabilities = shared + blockDim.x;
  unsigned long long item = blockIdx.x;
  if (item >= 16ULL) return;
  unsigned long long qs = item / 2ULL, h = item % 2ULL;
  unsigned long long kend = 4ULL;
  for (unsigned long long ks = 0; ks < kend; ++ks) {
    float partial = 0.0f;
    for (unsigned long long d = threadIdx.x; d < 8ULL; d += blockDim.x)
      partial = fmaf(dif_load(q, (qs * 2ULL + h) * 8ULL + d), dif_load(k, ((qs / 4ULL * 4ULL + ks) * 2ULL + h) * 8ULL + d), partial);
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) probabilities[ks] = reduction[0] * 0.35355338454246521f;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float maximum = -3.402823466e+38f;
    for (unsigned long long ks = 0; ks < kend; ++ks) maximum = fmaxf(maximum, probabilities[ks]);
    float denominator = 0.0f;
    for (unsigned long long ks = 0; ks < kend; ++ks) {
      probabilities[ks] = expf(probabilities[ks] - maximum);
      denominator += probabilities[ks];
    }
    for (unsigned long long ks = 0; ks < kend; ++ks) probabilities[ks] /= denominator;
  }
  __syncthreads();
  for (unsigned long long d = threadIdx.x; d < 8ULL; d += blockDim.x) {
    float acc = 0.0f;
    for (unsigned long long ks = 0; ks < kend; ++ks)
      acc = fmaf(probabilities[ks], dif_load(v, ((qs / 4ULL * 4ULL + ks) * 2ULL + h) * 8ULL + d), acc);
    dif_store(y, (qs * 2ULL + h) * 8ULL + d, acc);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
