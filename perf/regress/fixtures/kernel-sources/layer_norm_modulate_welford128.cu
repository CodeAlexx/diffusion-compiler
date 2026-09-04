
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
// Layer norm + adaLN modulation, BF16 rows whose width is a multiple of four,
// block 128: a per-op Welford state (mean, M2, count) is accumulated four
// packed values at a time, combined across the warp with shuffles and across
// the four warps through shared memory, then the row is normalized, affine
// transformed and modulated as (1 + scale) * normalized + shift with BF16
// rounding at every step the reference applies. The helper names carry the
// operation id so several instances can share one module.
struct dif_welford_mod_1 {
  float mean;
  float sigma2;
  float count;
};
extern "C" __device__ __forceinline__ dif_welford_mod_1 dif_welford_mod_online_1(float value, dif_welford_mod_1 current) {
  float delta = value - current.mean;
  float count = current.count + 1.0f;
  float mean = current.mean + delta * (1.0f / count);
  return {mean, current.sigma2 + delta * (value - mean), count};
}
extern "C" __device__ __forceinline__ dif_welford_mod_1 dif_welford_mod_combine_1(dif_welford_mod_1 data_b, dif_welford_mod_1 data_a) {
  float delta = data_b.mean - data_a.mean;
  float count = data_a.count + data_b.count;
  if (count > 0.0f) {
    float coefficient = 1.0f / count;
    float n_a = data_a.count * coefficient;
    float n_b = data_b.count * coefficient;
    return {n_a * data_a.mean + n_b * data_b.mean, data_a.sigma2 + data_b.sigma2 + delta * delta * data_a.count * n_b, count};
  }
  return {0.0f, 0.0f, 0.0f};
}
extern "C" __global__ void dif_op_1(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= 4ULL) return;
  unsigned tid = threadIdx.x, lane = tid & 31U, warp = tid >> 5U;
  dif_welford_mod_1 state = {0.0f, 0.0f, 0.0f};
  for (unsigned long long pack = tid; pack < 2ULL; pack += 128ULL) {
    unsigned long long base = row * 8ULL + pack * 4ULL;
    state = dif_welford_mod_online_1(dif_load(x, base), state);
    state = dif_welford_mod_online_1(dif_load(x, base + 1ULL), state);
    state = dif_welford_mod_online_1(dif_load(x, base + 2ULL), state);
    state = dif_welford_mod_online_1(dif_load(x, base + 3ULL), state);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dif_welford_mod_1 other = {__shfl_down_sync(0xffffffffU, state.mean, offset), __shfl_down_sync(0xffffffffU, state.sigma2, offset), __shfl_down_sync(0xffffffffU, state.count, offset)};
    state = dif_welford_mod_combine_1(state, other);
  }
  float* meansigma = reduction;
  float* counts = reduction + 4U;
  for (unsigned offset = 2U; offset > 0U; offset >>= 1U) {
    if (lane == 0U && warp >= offset && warp < 2U * offset) {
      unsigned target = warp - offset;
      meansigma[2U * target] = state.mean;
      meansigma[2U * target + 1U] = state.sigma2;
      counts[target] = state.count;
    }
    __syncthreads();
    if (lane == 0U && warp < offset) {
      dif_welford_mod_1 other = {meansigma[2U * warp], meansigma[2U * warp + 1U], counts[warp]};
      state = dif_welford_mod_combine_1(state, other);
    }
    __syncthreads();
  }
  if (tid == 0U) {
    meansigma[0] = state.mean;
    meansigma[1] = state.sigma2 / 8.0f;
  }
  __syncthreads();
  float mean = meansigma[0];
  float inverse = rsqrtf(meansigma[1] + 9.9999999747524271e-07f);
  unsigned long long modulation_row = row / 4ULL;
  for (unsigned long long pack = tid; pack < 2ULL; pack += 128ULL) {
    unsigned long long base = row * 8ULL + pack * 4ULL;
    for (unsigned inner = 0U; inner < 4U; ++inner) {
      unsigned long long column = pack * 4ULL + inner, index = base + inner, modulation_index = modulation_row * 8ULL + column;
      float normalized = inverse * (dif_load(x, index) - mean);
      normalized = dif_round(dif_load(weight, column) * normalized + dif_load(bias, column));
      float one_plus_scale = dif_round(1.0f + dif_load(scale, modulation_index));
      float scaled = dif_round(normalized * one_plus_scale);
      dif_store(y, index, scaled + dif_load(shift, modulation_index));
    }
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
