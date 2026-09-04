
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
// Gradient of the rotary embedding: the rotation transpose. A pair (first,
// second) rotated by (c,s) comes back through (c,-s; s,c), and the tail past
// the rotated range passes straight through. The pairing and the table dtype
// are generated from the operation's own attributes, so this differentiates
// the rotation that actually ran.
extern "C" __global__ void dif_op_1(const dif_bf16* grad_output, const dif_f32* cosv, const dif_f32* sinv, dif_bf16* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64ULL) {
    unsigned long long d = i % 8ULL, rest = i / 8ULL;
    unsigned long long base = rest * 8ULL, token = rest / 2ULL;
    unsigned long long table = token * 3ULL;
    if (d >= 6ULL) {
      dif_store_bf16(grad_input, i, dif_load_bf16(grad_output, i));
    } else {
      unsigned long long pair = d < 3ULL ? d : d - 3ULL;
      bool leading = d < 3ULL;
      unsigned long long partner = leading ? d + 3ULL : d - 3ULL;
      float c = dif_load_f32(cosv, table + pair);
      float s = dif_load_f32(sinv, table + pair);
      float own = dif_load_bf16(grad_output, i);
      float other = dif_load_bf16(grad_output, base + partner);
      // Leading lane: g_first*c + g_second*s. Trailing lane: g_second*c - g_first*s.
      dif_store_bf16(grad_input, i, leading ? own * c + other * s : own * c - other * s);
    }
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
