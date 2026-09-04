
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
// Direct 1-D convolution (or its transpose), one thread per output element,
// with padding handled by the generated sampler expression (zero or
// replicate outside the input) and optional bias.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* w, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 32ULL) {
    unsigned long long o = i % 8ULL;
    unsigned long long oc = (i / 8ULL) % 4ULL;
    unsigned long long b = i / 32ULL;
    unsigned long long group = oc / 4ULL;
    float acc = 0.0f;
    for (unsigned long long ic = 0; ic < 2ULL; ++ic) {
      const dif_scalar* xrow = x + ((b * 2ULL) + (group * 2ULL + ic)) * 8ULL;
      const dif_scalar* wrow = w + ((oc * 2ULL) + ic) * 3ULL;
      long long start = (long long)(o * 1ULL);
      for (unsigned long long k = 0; k < 3ULL; ++k) {
        long long p = start + (long long)(k * 1ULL);
        acc += (((p - 1LL) >= 0LL && (p - 1LL) < 8LL) ? dif_load(xrow, (unsigned long long)(p - 1LL)) : 0.0f) * dif_load(wrow, k);
      }
    }

    
    dif_store(y, i, acc);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
