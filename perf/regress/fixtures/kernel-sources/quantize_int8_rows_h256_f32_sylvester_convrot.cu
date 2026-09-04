
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
extern "C" __global__ void dif_op_1(const dif_bf16* x0,signed char* q,float* scales){extern __shared__ float values[];__shared__ float maximums[256];unsigned long long row=blockIdx.x;unsigned tid=threadIdx.x;if(row>=2ULL)return;unsigned long long base=row*512ULL;for(unsigned long long column=tid;column<512ULL;column+=256ULL){float value=dif_load_bf16(x0,row*512ULL+column-0ULL);values[column]=value;}__syncthreads();for(unsigned stage=0U;stage<4U;++stage){unsigned stride=1U<<(2U*stage);for(unsigned long long tuple=tid;tuple<512ULL/4ULL;tuple+=256ULL){unsigned long long group=(tuple/64ULL)*256ULL;unsigned lane=(unsigned)(tuple%64ULL);unsigned offset=(lane%stride)+(lane/stride)*(4U*stride);unsigned long long i=group+offset;float x0=values[i],x1=values[i+stride],x2=values[i+2U*stride],x3=values[i+3U*stride];values[i]=0.5f*(x0+x1+x2+x3);values[i+stride]=0.5f*(x0-x1+x2-x3);values[i+2U*stride]=0.5f*(x0+x1-x2-x3);values[i+3U*stride]=0.5f*(x0-x1-x2+x3);}__syncthreads();}float maximum=0.0f;for(unsigned long long column=tid;column<512ULL;column+=256ULL)maximum=fmaxf(maximum,fabsf(values[column]));maximums[tid]=maximum;__syncthreads();for(unsigned active=128U;active>0U;active>>=1U){if(tid<active)maximums[tid]=fmaxf(maximums[tid],maximums[tid+active]);__syncthreads();}float scale=fmaxf(maximums[0]*1.000000000e+00f/127.0f,1.0e-30f);float scale_bf16=dif_round_bf16(scale);if(tid==0U)scales[row]=scale;__syncthreads();for(unsigned long long column=tid;column<512ULL;column+=256ULL){float value=values[column];float divided=value/scale;int encoded=(int)nearbyintf(divided);encoded=encoded>127?127:(encoded<-127?-127:encoded);q[base+column]=(signed char)encoded;}}
