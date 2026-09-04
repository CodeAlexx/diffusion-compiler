
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
// Rotary embedding on [B,L,H,D] with f32 cos/sin tables [B,L,P]. The
// layout (interleaved or half-split) is folded into the pair/partner index
// expressions; every multiply-add is a separately rounded PTX op.
extern "C" __global__ void dif_op_1(const dif_bf16* x, const dif_f32* cosine, const dif_f32* sine, dif_bf16* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64ULL) {
    unsigned long long d = i % 8ULL, outer = i / 8ULL, token = (outer / 2ULL) % 4ULL,
                       batch = outer / (2ULL * 4ULL);
    if (d < 8ULL) {
      unsigned long long pair = d / 2ULL, base = i - d, table = (batch * 4ULL + token) * 4ULL + pair;
      float even = dif_load_bf16(x, base + 2ULL * pair), odd = dif_load_bf16(x, base + 2ULL * pair + 1ULL), c = dif_load_f32(cosine, table),
            s = dif_load_f32(sine, table), first, second, result;
      if ((d & 1ULL)) {
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(even), "f"(s));
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(odd), "f"(c));
        asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
      } else {
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(even), "f"(c));
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(odd), "f"(s));
        asm volatile("sub.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
      }
      dif_store_bf16(y, i, result);
    } else dif_store_bf16(y, i, dif_load_bf16(x, i));
  }
}
