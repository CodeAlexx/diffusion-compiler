
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
// Backward of RMS-normalized, partially rotated q/k: the upstream gradient
// is rotated back (the generated rot(k) expressions), the norm backward
// follows, and the optional weight gradient is reduced by the first D threads.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64ULL) {
    unsigned long long row = i / 8ULL, d = i % 8ULL, rb = row * 8ULL, tb = (row / 2ULL) * 4ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float value = dif_load(x, rb + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / 8.0f + 9.999999975e-07f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float rotated_gradient = (k < 4ULL ? dif_load(grad_output, rb + k) * dif_load(cosv, tb + k) + dif_load(grad_output, rb + k + 4ULL) * dif_load(sinv, tb + k) : (k < 8ULL ? -dif_load(grad_output, rb + k - 4ULL) * dif_load(sinv, tb + k - 4ULL) + dif_load(grad_output, rb + k) * dif_load(cosv, tb + k - 4ULL) : dif_load(grad_output, rb + k)));
      dot = fmaf(rotated_gradient * dif_load(weight, k), dif_load(x, rb + k), dot);
    }
    float own_rotated = (d < 4ULL ? dif_load(grad_output, rb + d) * dif_load(cosv, tb + d) + dif_load(grad_output, rb + d + 4ULL) * dif_load(sinv, tb + d) : (d < 8ULL ? -dif_load(grad_output, rb + d - 4ULL) * dif_load(sinv, tb + d - 4ULL) + dif_load(grad_output, rb + d) * dif_load(cosv, tb + d - 4ULL) : dif_load(grad_output, rb + d)));
    float value = dif_load(x, i);
    dif_store(grad_input, i, own_rotated * dif_load(weight, d) * inv - value * inv * inv * inv * dot / 8.0f);
    
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
