// Port of Serenity's accepted fused BF16 RMSNorm + AdaLN modulation (256
// threads): the BF16 norm boundary is preserved, then modulation in F32 with
// only the final BF16 store. Scale/shift rows broadcast over rows_per_vector.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  float local = 0.0f;
  for (unsigned long long col = tid; col < ${cols}ULL; col += 256ULL) {
    float value = dif_load(x, row * ${cols}ULL + col);
    local = __fadd_rn(local, __fmul_rn(value, value));
  }
  reduction[tid] = local;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + active]);
    __syncthreads();
  }
  float inv = rsqrtf(__fadd_rn(__fdiv_rn(reduction[0], ${cols}.0f), ${epsilon}f));
  unsigned long long vector = (row / ${rows_per_vector}ULL) * ${cols}ULL;
  for (unsigned long long col = tid; col < ${cols}ULL; col += 256ULL) {
    unsigned long long i = row * ${cols}ULL + col;
    float normed = dif_round(dif_load(x, i) * inv * dif_load(weight, col));
    float result = (1.0f + dif_load(scale, vector + col)) * normed + dif_load(shift, vector + col);
    dif_store(y, i, result);
  }
}
