
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
// Vectorized rms_norm for 128-wide bf16 rows with 128-thread blocks
// (Implementation 2): one warp squares 4 values per lane with mul.rn,
// shuffles the sum, and every thread scales its column.
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= 4ULL) return;
  unsigned tid = threadIdx.x;
  float sigma2 = 0.0f;
  if (tid < 32U) {
    unsigned long long base = row * 128ULL + (unsigned long long)tid;
    float v0 = dif_load(x, base), v1 = dif_load(x, base + 32ULL),
          v2 = dif_load(x, base + 64ULL), v3 = dif_load(x, base + 96ULL), s0, s1, s2, s3;
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s0) : "f"(v0));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s1) : "f"(v1));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s2) : "f"(v2));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s3) : "f"(v3));
    sigma2 = ((s0 + s1) + s2) + s3;
    for (unsigned offset = 1U; offset < 32U; offset <<= 1U)
      sigma2 = sigma2 + __shfl_down_sync(0xffffffffU, sigma2, offset);
  }
  if (tid == 0U) reduction[0] = sigma2 * 0.0078125f;
  __syncthreads();
  float inverse = rsqrtf(reduction[0] + 9.9999999747524271e-07f);
  if (tid < 128U) {
    unsigned long long index = row * 128ULL + tid;
    dif_store(y, index, dif_load(weight, tid) * (inverse * dif_load(x, index)));
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
