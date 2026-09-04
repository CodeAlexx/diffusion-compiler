
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
// [B,C,T,H,W] <-> rows [B*T/pt*H/ph*W/pw, C*pt*ph*pw]; the same index map
// serves patchify and unpatchify, only the transfer direction differs.
extern "C" __global__ void dif_op_1(const dif_scalar* input, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 128ULL) {
    unsigned long long row = i / 16ULL, column = i % 16ULL, outer = row,
                       patch_x = outer % 2ULL;
    outer /= 2ULL;
    unsigned long long patch_y = outer % 2ULL;
    outer /= 2ULL;
    unsigned long long patch_frame = outer % 2ULL, batch = outer / 2ULL,
                       inner = column, offset_x = inner % 2ULL;
    inner /= 2ULL;
    unsigned long long offset_y = inner % 2ULL;
    inner /= 2ULL;
    unsigned long long offset_t = inner % 2ULL;
    inner /= 2ULL;
    unsigned long long channel = inner, frame = patch_frame * 2ULL + offset_t,
                       y = patch_y * 2ULL + offset_y, x = patch_x * 2ULL + offset_x,
                       volume_index = ((((batch * 2ULL + channel) * 4ULL + frame) * 4ULL + y) * 4ULL + x);
    dif_store(output, i, dif_load(input, volume_index));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
