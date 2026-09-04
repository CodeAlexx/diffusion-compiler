
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
// Gradient of the chunked row gather. The forward reads one source row per
// selected row and splits its columns into chunks, so the gradient sums every
// selection that named a source row back into it -- written as a gather over
// the SOURCE, one thread per source element scanning the index vector, rather
// than one thread per selection racing with atomicAdd. A duplicated index
// would otherwise sum in scheduling order.
extern "C" __global__ void dif_op_1(const int* indices, const dif_bf16* g0, const dif_bf16* g1, const dif_bf16* g2, const dif_bf16* g3, dif_bf16* grad_values) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 60ULL) {
    unsigned long long row = i / 12ULL, col = i % 12ULL;
    unsigned long long chunk = col / 3ULL, offset = col % 3ULL;
    const dif_bf16* source = chunk == 0ULL ? g0 : chunk == 1ULL ? g1 : chunk == 2ULL ? g2 : g3;
    float accumulator = 0.0f;
    for (unsigned long long s = 0ULL; s < 6ULL; ++s)
      if ((unsigned long long)indices[s] == row)
        accumulator += dif_load_bf16(source, s * 3ULL + offset);
    dif_store_bf16(grad_values, i, accumulator);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
