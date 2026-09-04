
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
// Blackwell MXFP8 quantization: per 32-column block, |max|/448 rounded UP to
// a power of two (UE8M0), values converted with cvt.rn.satfinite.e4m3x2;
// the scale byte lands in the cuBLASLt 128x4 tiled layout.
extern "C" __global__ void dif_op_1(const dif_bf16* x, unsigned char* q, unsigned char* scales) {
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 4ULL) return;
  unsigned lane = tid & 31U;
  unsigned warp = tid >> 5U;
  for (unsigned long long block = warp; block < 2ULL; block += 8ULL) {
    unsigned long long column = block * 32ULL + lane;
    float value = column < 64ULL ? dif_load_bf16(x, row * 64ULL + column) : 0.0f;
    float maximum = fabsf(value);
    for (unsigned offset = 16U; offset > 0U; offset >>= 1U)
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    maximum = __shfl_sync(0xffffffffU, maximum, 0U);
    float target = maximum / 448.0f;
    unsigned bits = __float_as_uint(target) & 0x7fffffffU;
    unsigned exponent = bits >> 23U;
    unsigned mantissa = bits & 0x7fffffU;
    unsigned encoded = (bits == 0U) ? 0U : ((exponent == 0U) ? (mantissa > 0x400000U ? 1U : 0U) : exponent + (mantissa != 0U));
    unsigned char encoded_scale = (unsigned char)(encoded > 254U ? 254U : encoded);
    float scale = ldexpf(1.0f, (int)encoded_scale - 127);
    if (lane == 0U) {
      unsigned long long tile_outer = row / 128ULL;
      unsigned long long tile_inner = (block / 4ULL) * 4ULL;
      unsigned long long within = (row % 32ULL) * 16ULL + ((row % 128ULL) / 32ULL) * 4ULL + block % 4ULL;
      unsigned long long scale_offset = (tile_inner + tile_outer * 4ULL) * 128ULL + within;
      scales[scale_offset] = encoded_scale;
    }
    if (column < 64ULL) {
      float divided = value / scale;
      unsigned short pair;
      asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(pair) : "f"(divided), "f"(0.0f));
      q[row * 64ULL + column] = (unsigned char)pair;
    }
  }
}
