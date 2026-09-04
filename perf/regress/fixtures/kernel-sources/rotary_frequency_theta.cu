
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
// Generic multi-axis rotary tables: each output pair maps to (axis,
// component); omega = 1 / (theta * ntk) ^ (2 * component / axis_dim).
extern "C" __global__ void dif_op_1(const dif_f32* positions, const int* pair_axes, const int* pair_indices, const int* axis_dims, dif_f32* cosine, dif_f32* sine) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 12ULL) {
    unsigned long long pair = i % 3ULL, token = (i / 3ULL) % 4ULL,
                       batch = i / (3ULL * 4ULL);
    int axis = pair_axes[pair], component = pair_indices[pair], axis_dim = axis_dims[axis];
    float scale = (2.0f * (float)component) / (float)axis_dim;
    float omega = 1.0f / powf((float)(1000000000 * 1.5), scale);
    float angle = dif_load_f32(positions, (batch * 4ULL + token) * 2ULL + (unsigned long long)axis) * omega;
    dif_store_f32(cosine, i, cosf(angle));
    dif_store_f32(sine, i, sinf(angle));
  }
}
