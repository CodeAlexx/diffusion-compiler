
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
// rms_norm backward for a FROZEN gain: one block per row.
//
// The row needs two reductions -- the sum of squares and the dot product of
// the weighted gradient with the input -- and they are the same two numbers
// for every element in the row. The reference form has each of the row's
// threads recompute both, which at 6144 columns is 6144 redundant passes.
// The row's values fit in cache, which is the only reason that was
// survivable rather than catastrophic; it was still the single most
// expensive operation in a Krea training step.
//
// The forward has always done it this way. This is the backward catching up.
//
// A trainable gain needs a per-column reduction across rows as well, which
// is a different shape of problem; that case keeps the reference kernel.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, dif_scalar* grad_input) {
  extern __shared__ float reduction[];
  const unsigned long long row = blockIdx.x;
  const unsigned long long base = row * 8ULL;
  float sum_squares = 0.0f, dot = 0.0f;
  for (unsigned long long k = threadIdx.x; k < 8ULL; k += blockDim.x) {
    const float value = dif_load(x, base + k);
    sum_squares = fmaf(value, value, sum_squares);
    dot = fmaf(dif_load(grad_output, base + k) * dif_load(weight, k), value, dot);
  }
  reduction[threadIdx.x] = sum_squares;
  reduction[blockDim.x + threadIdx.x] = dot;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2u; stride > 0u; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      reduction[blockDim.x + threadIdx.x] += reduction[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse = rsqrtf(reduction[0] / 8.0f + 9.999999975e-07f);
  const float row_dot = reduction[blockDim.x];
  for (unsigned long long k = threadIdx.x; k < 8ULL; k += blockDim.x) {
    const float value = dif_load(x, base + k);
    dif_store(grad_input, base + k,
              dif_load(grad_output, base + k) * dif_load(weight, k) * inverse -
                  value * inverse * inverse * inverse * row_dot / 8.0f);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
