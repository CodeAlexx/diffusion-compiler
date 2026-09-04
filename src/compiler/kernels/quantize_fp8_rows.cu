// Dynamic symmetric per-row E4M3 quantization: |max| / 448 as the scale,
// round-to-nearest saturating conversion through cvt.rn.satfinite.e4m3x2.
extern "C" __global__ void ${function}(const dif_bf16* x, unsigned char* q, float* scales) {
  __shared__ float maximums[256];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * ${columns}ULL;
  float maximum = 0.0f;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    maximum = fmaxf(maximum, fabsf(dif_load_bf16(x, base + column)));
  }
  maximums[tid] = maximum;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) maximums[tid] = fmaxf(maximums[tid], maximums[tid + active]);
    __syncthreads();
  }
  float scale = fmaxf(maximums[0] / 448.0f, 1.0e-30f);
  if (tid == 0U) scales[row] = scale;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float value = dif_load_bf16(x, base + column) / scale;
    unsigned short pair;
    asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(pair) : "f"(value), "f"(0.0f));
    q[base + column] = (unsigned char)pair;
  }
}
