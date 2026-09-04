
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
extern "C" __global__ void dif_op_1(const dif_scalar* grad_output,const dif_scalar* x,dif_scalar* grad_input){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<64ULL){unsigned long long row=i/16ULL,col=i%16ULL;if(col<0ULL||col>=16ULL){dif_store(grad_input,i,0.0f);return;}unsigned long long lane=col-0ULL,cw=lane<8ULL?lane:lane-8ULL,base=row*16ULL+0ULL;float value=dif_load(x,base+8ULL+cw);float gate=dif_load(x,base+0ULL+cw);float sigmoid=1.0f/(1.0f+expf(-gate));float upstream=dif_load(grad_output,row*8ULL+cw);int is_value_slot=lane>=8ULL;float gradient=is_value_slot?gate*sigmoid*upstream:sigmoid*(1.0f+gate*(1.0f-sigmoid))*value*upstream;dif_store(grad_input,i,gradient);}}
#undef dif_scalar
#undef dif_load
#undef dif_store
#undef dif_round
