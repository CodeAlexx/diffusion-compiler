
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
// Gradient of layer normalization followed by an adaptive scale and shift:
//   out = ((x-mean)*inv*W + B) * (1+scale) + shift
// Row statistics are recomputed from the original input in F32 rather than
// saved, the way the plain layer-norm gradient beside this one does.
//
// Three different reductions live here, and each output element is owned by
// exactly one thread so all of them sum deterministically without atomics:
// grad_input is elementwise, the affine gradients reduce down every row, and
// the modulation gradients reduce only across the rows sharing one modulation
// row. Reference form -- the row statistics are recomputed inside each
// reduction, which is quadratic and honest about it.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, const dif_scalar* scale, dif_scalar* grad_input, dif_scalar* grad_weight, dif_scalar* grad_bias, dif_scalar* grad_scale, dif_scalar* grad_shift) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 32ULL) {
    unsigned long long row = i / 8ULL, base = row * 8ULL;
    unsigned long long modulation = row / 1ULL * 8ULL;
    float mean = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) mean += dif_load(x, base + k);
    mean /= 8.0f;
    float variance = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float centered = dif_load(x, base + k) - mean;
      variance = fmaf(centered, centered, variance);
    }
    float inv = rsqrtf(variance / 8.0f + 9.99999975e-06f);
    float gradient_mean = 0.0f, projected_mean = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float upstream = dif_load(grad_output, base + k) * (1.0f + dif_load(scale, modulation + k));
      float weighted = upstream * dif_load(weight, k);
      float normalized = (dif_load(x, base + k) - mean) * inv;
      gradient_mean += weighted;
      projected_mean = fmaf(weighted, normalized, projected_mean);
    }
    gradient_mean /= 8.0f;
    projected_mean /= 8.0f;
    unsigned long long column = i - base;
    float upstream = dif_load(grad_output, i) * (1.0f + dif_load(scale, modulation + column));
    float weighted = upstream * dif_load(weight, column);
    float normalized = (dif_load(x, i) - mean) * inv;
    dif_store(grad_input, i, inv * (weighted - gradient_mean - normalized * projected_mean));
  }
  // The affine gradients belong to the whole tensor: one thread per column,
  // scanning every row.
  if (i < 8ULL) {
    float weight_accumulator = 0.0f, bias_accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < 4ULL; ++r) {
      unsigned long long rb = r * 8ULL;
      unsigned long long rm = r / 1ULL * 8ULL;
      float rmean = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) rmean += dif_load(x, rb + k);
      rmean /= 8.0f;
      float rvariance = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
        float centered = dif_load(x, rb + k) - rmean;
        rvariance = fmaf(centered, centered, rvariance);
      }
      float rinv = rsqrtf(rvariance / 8.0f + 9.99999975e-06f);
      float g = dif_load(grad_output, rb + i) * (1.0f + dif_load(scale, rm + i));
      weight_accumulator = fmaf(g, (dif_load(x, rb + i) - rmean) * rinv, weight_accumulator);
      bias_accumulator += g;
    }
    dif_store(grad_weight, i, weight_accumulator);
    dif_store(grad_bias, i, bias_accumulator);
  }
  // The modulation gradients belong to one modulation row: one thread per
  // (modulation row, column), scanning only the rows that share it.
  if (i < 32ULL) {
    unsigned long long m = i / 8ULL, column = i % 8ULL;
    float scale_accumulator = 0.0f, shift_accumulator = 0.0f;
    for (unsigned long long r = m * 1ULL; r < (m + 1ULL) * 1ULL; ++r) {
      unsigned long long rb = r * 8ULL;
      float rmean = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) rmean += dif_load(x, rb + k);
      rmean /= 8.0f;
      float rvariance = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
        float centered = dif_load(x, rb + k) - rmean;
        rvariance = fmaf(centered, centered, rvariance);
      }
      float rinv = rsqrtf(rvariance / 8.0f + 9.99999975e-06f);
      float normalized = (dif_load(x, rb + column) - rmean) * rinv * dif_load(weight, column) + dif_load(bias, column);
      float g = dif_load(grad_output, rb + column);
      scale_accumulator = fmaf(g, normalized, scale_accumulator);
      shift_accumulator += g;
    }
    dif_store(grad_scale, i, scale_accumulator);
    dif_store(grad_shift, i, shift_accumulator);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
