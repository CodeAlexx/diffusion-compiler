
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
// Port of Serenity's accepted fused BF16 RMSNorm + AdaLN modulation (256
// threads): the BF16 norm boundary is preserved, then modulation in F32 with
// only the final BF16 store. Scale/shift rows broadcast over rows_per_vector.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= 4ULL) return;
  float local = 0.0f;
  for (unsigned long long col = tid; col < 8ULL; col += 256ULL) {
    float value = dif_load(x, row * 8ULL + col);
    local = __fadd_rn(local, __fmul_rn(value, value));
  }
  reduction[tid] = local;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + active]);
    __syncthreads();
  }
  float inv = rsqrtf(__fadd_rn(__fdiv_rn(reduction[0], 8.0f), 9.9999999747524271e-07f));
  unsigned long long vector = (row / 1ULL) * 8ULL;
  for (unsigned long long col = tid; col < 8ULL; col += 256ULL) {
    unsigned long long i = row * 8ULL + col;
    float normed = dif_round(dif_load(x, i) * inv * dif_load(weight, col));
    float result = (1.0f + dif_load(scale, vector + col)) * normed + dif_load(shift, vector + col);
    dif_store(y, i, result);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
