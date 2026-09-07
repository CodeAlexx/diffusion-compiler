// Direct-alpha Snake on NCL storage, with independent eager F32 boundaries.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* alpha, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long c = (i / ${length}ULL) % ${channels}ULL;
    float a = dif_load(alpha, c);
    float inv = 1.0f / (a + ${epsilon}f);
    float xv = dif_load(x, i);
    float s = sinf(a * xv);
    dif_store(y, i, __fadd_rn(xv, __fmul_rn(inv, __fmul_rn(s, s))));
  }
}
