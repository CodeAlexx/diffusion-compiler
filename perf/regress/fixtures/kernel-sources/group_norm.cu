
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
// Group normalization, one block per (batch, group): mean and variance over
// the group's channels and every trailing dimension (two shared arrays, the
// unrolled stride lines are generated for the block size), then the affine.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long vector = (unsigned long long)blockIdx.x;
  if (vector >= 2ULL) return;
  unsigned long long lane = threadIdx.x, group = vector % 2ULL, batch = vector / 2ULL,
                     base = (batch * 4ULL + group * 2ULL) * 4ULL;
  float sum = 0.0f, squares = 0.0f;
  for (unsigned long long k = lane; k < 8ULL; k += blockDim.x) {
    float v = dif_load(x, base + k);
    sum += v;
    squares = fmaf(v, v, squares);
  }
  reduction[lane] = sum;
  reduction[blockDim.x + lane] = squares;
  __syncthreads();
  if (lane < 128ULL) {
    reduction[lane] += reduction[lane + 128ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 128ULL];
  }
  __syncthreads();
  if (lane < 64ULL) {
    reduction[lane] += reduction[lane + 64ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 64ULL];
  }
  __syncthreads();
  if (lane < 32ULL) {
    reduction[lane] += reduction[lane + 32ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 32ULL];
  }
  __syncthreads();
  if (lane < 16ULL) {
    reduction[lane] += reduction[lane + 16ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 16ULL];
  }
  __syncthreads();
  if (lane < 8ULL) {
    reduction[lane] += reduction[lane + 8ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 8ULL];
  }
  __syncthreads();
  if (lane < 4ULL) {
    reduction[lane] += reduction[lane + 4ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 4ULL];
  }
  __syncthreads();
  if (lane < 2ULL) {
    reduction[lane] += reduction[lane + 2ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 2ULL];
  }
  __syncthreads();
  if (lane < 1ULL) {
    reduction[lane] += reduction[lane + 1ULL];
    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + 1ULL];
  }
  __syncthreads();

  float mean = reduction[0] / 8.0f;
  float variance = fmaxf(reduction[blockDim.x] / 8.0f - mean * mean, 0.0f);
  float inv = rsqrtf(variance + 9.999999747e-06f);
  for (unsigned long long k = lane; k < 8ULL; k += blockDim.x) {
    unsigned long long channel = group * 2ULL + k / 4ULL;
    float normalized = (dif_load(x, base + k) - mean) * inv;
    dif_store(y, base + k, fmaf(normalized, dif_load(weight, channel), dif_load(bias, channel)));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
