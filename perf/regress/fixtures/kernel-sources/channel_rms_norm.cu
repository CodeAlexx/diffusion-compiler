
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
// Channel-axis RMS norm with the creator's storage boundaries: one block
// per (leading, trailing) vector, a shared-memory sum of squares over the
// channel axis (the unrolled stride lines are generated for the block size),
// then y = round(round(x / max(sqrt(sum), eps)) * sqrt(C)) * gamma.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* gamma, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long vector = (unsigned long long)blockIdx.x;
  if (vector >= 6ULL) return;
  unsigned long long c = threadIdx.x;
  unsigned long long leading = vector / 3ULL, trailing = vector % 3ULL;
  unsigned long long index = (leading * 8ULL + c) * 3ULL + trailing;
  float value = c < 8ULL ? dif_load(x, index) : 0.0f;
  reduction[c] = value * value;
  __syncthreads();
  if (c < 128ULL) reduction[c] += reduction[c + 128ULL];
  __syncthreads();
  if (c < 64ULL) reduction[c] += reduction[c + 64ULL];
  __syncthreads();
  if (c < 32ULL) reduction[c] += reduction[c + 32ULL];
  __syncthreads();
  if (c < 16ULL) reduction[c] += reduction[c + 16ULL];
  __syncthreads();
  if (c < 8ULL) reduction[c] += reduction[c + 8ULL];
  __syncthreads();
  if (c < 4ULL) reduction[c] += reduction[c + 4ULL];
  __syncthreads();
  if (c < 2ULL) reduction[c] += reduction[c + 2ULL];
  __syncthreads();
  if (c < 1ULL) reduction[c] += reduction[c + 1ULL];
  __syncthreads();

  if (c < 8ULL) {
    float denominator = fmaxf(sqrtf(reduction[0]), 9.999999960e-13f);
    float normalized = dif_round(value / denominator);
    float scaled = dif_round(normalized * 2.828427076e+00f);
    dif_store(y, index, scaled * dif_load(gamma, c));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
