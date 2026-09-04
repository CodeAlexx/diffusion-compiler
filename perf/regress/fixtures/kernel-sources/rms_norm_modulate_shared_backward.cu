
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
// Backward of rms_norm followed by shared-vector modulation:
//   scale = vector[v(row)] + delta[0]   shift = vector[v(row)] + delta[1]
//   out   = (1 + scale) * (x * inv * (weight + offset)) + shift
// The row statistic is recomputed from the original input in F32, as the
// other norm gradients here do.
//
// Four reductions over four different axes live in one kernel because they
// share that statistic: the input gradient is elementwise, the weight
// gradient reduces down every row, the vector gradient reduces across the
// rows sharing one vector, and the delta gradient reduces across everything.
// Each output element is owned by exactly one thread, so all four sum
// deterministically without atomics.
//
// The vector feeds BOTH the scale and the shift, so its gradient carries the
// normalized value plus one -- dropping either term is the easy mistake here.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vec, const dif_scalar* delta, dif_scalar* grad_input, dif_scalar* grad_weight, dif_scalar* grad_vector, dif_scalar* grad_delta) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 32ULL) {
    unsigned long long row = i / 8ULL, col = i % 8ULL, base = row * 8ULL;
    unsigned long long vbase = row / 2ULL * 8ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float value = dif_load(x, base + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / 8.0f + 9.999999747e-06f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
      float modulated = dif_load(grad_output, base + k) *
                        (1.0f + dif_load(vec, vbase + k) + dif_load(delta, k));
      dot = fmaf(modulated * (dif_load(weight, k) + 1.000000000e+00f), dif_load(x, base + k), dot);
    }
    float own = dif_load(grad_output, i) * (1.0f + dif_load(vec, vbase + col) + dif_load(delta, col));
    float weighted = own * (dif_load(weight, col) + 1.000000000e+00f);
    float value = dif_load(x, i);
    dif_store(grad_input, i, weighted * inv - value * inv * inv * inv * dot / 8.0f);
  }
  if (i < 8ULL) {
    float accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < 4ULL; ++r) {
      unsigned long long rb = r * 8ULL;
      unsigned long long vb = r / 2ULL * 8ULL;
      float rss = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
        float rv = dif_load(x, rb + k);
        rss = fmaf(rv, rv, rss);
      }
      float rinv = rsqrtf(rss / 8.0f + 9.999999747e-06f);
      float modulated = dif_load(grad_output, rb + i) *
                        (1.0f + dif_load(vec, vb + i) + dif_load(delta, i));
      accumulator = fmaf(modulated, dif_load(x, rb + i) * rinv, accumulator);
    }
    dif_store(grad_weight, i, accumulator);
  }
  if (i < 16ULL) {
    unsigned long long v = i / 8ULL, col = i % 8ULL;
    float accumulator = 0.0f;
    for (unsigned long long r = v * 2ULL; r < (v + 1ULL) * 2ULL; ++r) {
      unsigned long long rb = r * 8ULL;
      float rss = 0.0f;
      for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
        float rv = dif_load(x, rb + k);
        rss = fmaf(rv, rv, rss);
      }
      float rinv = rsqrtf(rss / 8.0f + 9.999999747e-06f);
      float normalized = dif_load(x, rb + col) * rinv * (dif_load(weight, col) + 1.000000000e+00f);
      accumulator = fmaf(dif_load(grad_output, rb + col), normalized + 1.0f, accumulator);
    }
    dif_store(grad_vector, i, accumulator);
  }
  if (i < 16ULL) {
    unsigned long long chunk = i / 8ULL, col = i % 8ULL;
    float accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < 4ULL; ++r) {
      unsigned long long rb = r * 8ULL;
      float upstream = dif_load(grad_output, rb + col);
      if (chunk == 1ULL) {
        accumulator += upstream;
      } else {
        float rss = 0.0f;
        for (unsigned long long k = 0ULL; k < 8ULL; ++k) {
          float rv = dif_load(x, rb + k);
          rss = fmaf(rv, rv, rss);
        }
        float rinv = rsqrtf(rss / 8.0f + 9.999999747e-06f);
        accumulator = fmaf(upstream, dif_load(x, rb + col) * rinv * (dif_load(weight, col) + 1.000000000e+00f), accumulator);
      }
    }
    dif_store(grad_delta, i, accumulator);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
