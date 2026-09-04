
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
// Gradient of reflect padding. The forward is a gather: every output element
// reads exactly one input element, and the elements just inside each edge are
// read more than once. So the gradient is a scatter-add -- and it is written
// as a gather over the INPUT, one thread per input element summing the output
// positions that read it, rather than one thread per output racing with
// atomicAdd. The sum is then deterministic, which a training run depends on.
//
// Reflection is separable, and per axis an input coordinate is read from at
// most three output coordinates: the identity copy, the low reflection, and
// the high one. The bound is three per axis, so at most twenty-seven loads,
// whatever the padding is. A rank-4 tensor is the rank-5 form with a depth of
// one, which makes its flat layout identical and needs no separate kernel.
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 240ULL) {
    unsigned long long w = i % 6ULL, rest = i / 6ULL;
    unsigned long long h = rest % 5ULL;
    rest /= 5ULL;
    unsigned long long t = rest % 4ULL, plane = rest / 4ULL;
    unsigned long long ot[3], oh[3], ow[3];
    int nt = 0, nh = 0, nw = 0;
    ot[nt++] = t + 2ULL;
    if (t >= 1ULL && t <= 2ULL) ot[nt++] = 2ULL - t;
    if (1ULL >= 1ULL && t + 1ULL + 1ULL >= 4ULL && t + 2ULL <= 4ULL) ot[nt++] = 2ULL + 2ULL * 4ULL - 2ULL - t;
    oh[nh++] = h + 1ULL;
    if (h >= 1ULL && h <= 1ULL) oh[nh++] = 1ULL - h;
    if (3ULL >= 1ULL && h + 1ULL + 3ULL >= 5ULL && h + 2ULL <= 5ULL) oh[nh++] = 1ULL + 2ULL * 5ULL - 2ULL - h;
    ow[nw++] = w + 2ULL;
    if (w >= 1ULL && w <= 2ULL) ow[nw++] = 2ULL - w;
    if (2ULL >= 1ULL && w + 1ULL + 2ULL >= 6ULL && w + 2ULL <= 6ULL) ow[nw++] = 2ULL + 2ULL * 6ULL - 2ULL - w;
    float accumulator = 0.0f;
    for (int a = 0; a < nt; ++a)
      for (int b = 0; b < nh; ++b)
        for (int c = 0; c < nw; ++c)
          accumulator += dif_load(grad_output, ((plane * 7ULL + ot[a]) * 9ULL + oh[b]) * 10ULL + ow[c]);
    dif_store(grad_input, i, accumulator);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
