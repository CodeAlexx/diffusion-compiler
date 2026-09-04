
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
// Direct packed-INT5 Linear (Implementation 3) fused with its dequantizer:
// one block per output row, 256 threads stride K unpacking 5-bit codes, one
// F32 accumulator per flattened input row (M <= 32; the per-row lines are
// generated), warp shuffles, then a cross-warp total per row.
extern "C" __global__ void dif_op_2(const dif_scalar* x, const unsigned char* packed, const dif_scalar* scales, const dif_scalar* column_scales, const dif_scalar* bias, dif_scalar* y) {
  extern __shared__ float partials[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x, lane = tid & 31U, warp = tid >> 5U;
  if (row >= 8ULL) return;
  unsigned long long source_row = row;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (unsigned long long col = tid; col < 128ULL; col += 256ULL) {
    unsigned long long bit = col * 5ULL, bi = source_row * 80ULL + bit / 8ULL;
    unsigned shift = (unsigned)(bit & 7ULL);
    unsigned word = packed[bi];
    if (shift + 5U > 8U) word |= ((unsigned)packed[bi + 1ULL]) << 8U;
    unsigned encoded = (word >> shift) & 31U;
    int q = encoded < 16U ? (int)encoded : (int)encoded - 32;
    float w = (float)q * dif_load(scales, source_row * 2ULL + col / 64ULL);
    w *= dif_load(column_scales, col);
    acc0 = fmaf(dif_load(x, 0ULL + col), w, acc0);
    acc1 = fmaf(dif_load(x, 128ULL + col), w, acc1);
    acc2 = fmaf(dif_load(x, 256ULL + col), w, acc2);
    acc3 = fmaf(dif_load(x, 384ULL + col), w, acc3);

  }
  for (unsigned offset = 16U; offset > 0U; offset >>= 1U) {
    acc0 += __shfl_down_sync(0xffffffffU, acc0, offset);
    acc1 += __shfl_down_sync(0xffffffffU, acc1, offset);
    acc2 += __shfl_down_sync(0xffffffffU, acc2, offset);
    acc3 += __shfl_down_sync(0xffffffffU, acc3, offset);

  }
  if (lane == 0U) {
    partials[warp * 4ULL + 0ULL] = acc0;
    partials[warp * 4ULL + 1ULL] = acc1;
    partials[warp * 4ULL + 2ULL] = acc2;
    partials[warp * 4ULL + 3ULL] = acc3;

  }
  __syncthreads();
  if (warp == 0U && lane < 4ULL) {
    float total = 0.0f;
    for (unsigned source_warp = 0U; source_warp < 8U; ++source_warp) total += partials[source_warp * 4ULL + lane];
    total += dif_load(bias, row);
    dif_store(y, (unsigned long long)lane * 8ULL + row, total);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
