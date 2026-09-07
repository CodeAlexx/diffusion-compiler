// Straight single-axis translation of serenitymojo/ops/reduce.mojo:
// one thread/output, ascending reduced index, F32 accumulate then divide.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long base = (i / ${inner}ULL) * ${reduced}ULL * ${inner}ULL + i % ${inner}ULL;
    float sum = 0.0f;
    for (unsigned long long r = 0; r < ${reduced}ULL; ++r)
      sum = __fadd_rn(sum, dif_load(x, base + r * ${inner}ULL));
    dif_store(y, i, __fdiv_rn(sum, (float)${reduced}ULL));
  }
}
