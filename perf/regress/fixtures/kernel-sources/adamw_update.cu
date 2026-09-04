
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
#define dif_scalar dif_f32
#define dif_load dif_load_f32
#define dif_store dif_store_f32
#define dif_round dif_round_f32
// Decoupled AdamW with typed parameter/gradient storage and F32 moments.
// The decay multiplies the parameter BEFORE the moment update is subtracted
// and never folds into the gradient (flame's LoRA-A runaway lesson).
extern "C" __global__ void dif_op_1(const dif_f32* parameter, const dif_bf16* gradient, const dif_f32* first, const dif_f32* second, const int* completed_steps, dif_f32* updated, dif_f32* updated_first, dif_f32* updated_second) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 8ULL) {
    float step = (float)(completed_steps[0] + 1), beta1 = 8.999999762e-01f, beta2 = 9.990000129e-01f;
    float grad = dif_load_bf16(gradient, i) * 5.000000000e-01f;
    float m = beta1 * dif_load_f32(first, i) + (1.0f - beta1) * grad;
    float v = beta2 * dif_load_f32(second, i) + (1.0f - beta2) * grad * grad;
    float bias1 = 1.0f - powf(beta1, step);
    float bias2_sqrt = sqrtf(1.0f - powf(beta2, step));
    float decayed = dif_load_f32(parameter, i) * (1.0f - 9.999999747e-05f * 9.999999776e-03f);
    float denominator = sqrtf(v) / bias2_sqrt + 9.999999939e-09f;
    float value = decayed - (9.999999747e-05f / bias1) * m / denominator;
    dif_store_f32(updated, i, value);
    dif_store_f32(updated_first, i, m);
    dif_store_f32(updated_second, i, v);
  }
}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
