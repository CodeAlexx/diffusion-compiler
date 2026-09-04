
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
// INT5 dequantization: 5-bit codes packed bit-contiguously along each row
// (row stride 5*K/8 bytes), one scale per `group` columns, optional per-column
// scales.
extern "C" __global__ void dif_op_1(const unsigned char* packed, const dif_scalar* scales, const dif_scalar* column_scales, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 512ULL) {
    unsigned long long row = i / 128ULL, col = i % 128ULL,
                       bit = col * 5ULL, bi = row * 80ULL + bit / 8ULL;
    unsigned int shift = (unsigned int)(bit & 7ULL);
    unsigned int word = packed[bi];
    if (shift + 5U > 8U)
      word |= ((unsigned int)packed[bi + 1ULL]) << 8U;
    unsigned int encoded = (word >> shift) & 31U;
    int q = encoded < 16U ? (int)encoded : (int)encoded - 32;
    float scale = dif_load(scales, row * 2ULL + col / 64ULL);
    float value = (float)q * scale;
    value *= dif_load(column_scales, col);
    dif_store(y, i, value);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
