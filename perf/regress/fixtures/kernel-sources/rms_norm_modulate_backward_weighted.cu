
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
// Backward of rms_norm followed by (1 + scale) modulation, optionally with a
// learned weight; per element recomputation of the row statistics, plus the
// optional weight gradient reduced by the first `columns` threads.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* scale, dif_scalar* grad_input, dif_scalar* grad_scale, dif_scalar* grad_shift, dif_scalar* grad_weight) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 32ULL) {
    unsigned long long row = i / 8ULL, col = i % 8ULL, base = row * 8ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float value = dif_load(x, base + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / 8.0f + 9.999999747e-06f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k)
      dot = fmaf(dif_load(grad_output, base + k) * (1.0f + dif_load(scale, base + k)) * dif_load(weight, k), dif_load(x, base + k), dot);
    float value = dif_load(x, i);
    float upstream = dif_load(grad_output, i);
    float normed_gradient = upstream * (1.0f + dif_load(scale, i));
    float weight_value = dif_load(weight, col);
    dif_store(grad_input, i, normed_gradient * weight_value * inv - value * inv * inv * inv * dot / 8.0f);
    dif_store(grad_scale, i, upstream * value * inv * weight_value);
    dif_store(grad_shift, i, upstream);
        if (i < 8ULL) {
      float acc = 0.0f;
      for (unsigned long long r = 0ULL; r < 4ULL; ++r) {
        unsigned long long rb = r * 8ULL;
        float rss = 0.0f;
        for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
          float rv = dif_load(x, rb + k);
          rss = fmaf(rv, rv, rss);
        }
        float rinv = rsqrtf(rss / 8.0f + 9.999999747e-06f);
        acc = fmaf(dif_load(grad_output, rb + i) * (1.0f + dif_load(scale, rb + i)) * dif_load(x, rb + i), rinv, acc);
      }
      dif_store(grad_weight, i, acc);
    }

  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
