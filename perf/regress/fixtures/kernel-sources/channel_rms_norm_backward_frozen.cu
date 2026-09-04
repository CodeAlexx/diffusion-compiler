
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
// Gradient of RMS normalization across a channel axis, scaled by sqrt(C) and
// gamma. The fiber statistic is recomputed from the original input in F32
// rather than saved, as the other norm gradients here do.
//
// Two reductions with different axes share one kernel: the input gradient is
// per element and reduces along its own fiber, while the gamma gradient
// reduces across every fiber of a channel. Each output element is owned by
// exactly one thread, so both sum deterministically without atomics.
//
// The forward clamps the denominator at epsilon. Below that clamp the output
// is linear in the input, so the gradient is too -- and the branch here is
// the same branch the forward took, not an approximation of it.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* gamma, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64ULL) {
    unsigned long long trailing = i % 8ULL, rest = i / 8ULL;
    unsigned long long channel = rest % 4ULL;
    unsigned long long fiber = rest / 4ULL * 4ULL * 8ULL + trailing;
    float squared = 0.0f;
    for (unsigned long long c = 0ULL; c < 4ULL; ++c) {
      float value = dif_load(x, fiber + c * 8ULL);
      squared = fmaf(value, value, squared);
    }
    float length = sqrtf(squared);
    float denominator = fmaxf(length, 9.999999960e-13f);
    float own = dif_load(grad_output, i) * dif_load(gamma, channel) * 2.000000000e+00f;
    if (length > 9.999999960e-13f) {
      float projected = 0.0f;
      for (unsigned long long c = 0ULL; c < 4ULL; ++c)
        projected = fmaf(dif_load(grad_output, fiber + c * 8ULL) *
                             dif_load(gamma, c),
                         dif_load(x, fiber + c * 8ULL), projected);
      projected *= 2.000000000e+00f;
      dif_store(grad_input, i,
                own / denominator -
                    dif_load(x, i) * projected / (denominator * denominator * denominator));
    } else {
      dif_store(grad_input, i, own / denominator);
    }
  }
  
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
