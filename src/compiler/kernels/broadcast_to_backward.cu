// Gradient of a broadcast: one thread per SOURCE position, summing every
// gradient position that position was expanded into. The source coordinates
// give the base offset; the generated inner loop walks the expanded axes.
extern "C" __global__ void ${function}(const dif_scalar* g, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long source = i;
    unsigned long long base = 0ULL;
${decompose}
    float total = 0.0f;
    for (unsigned long long repeat = 0ULL; repeat < ${repeats}ULL; ++repeat) {
      unsigned long long remainder = repeat;
      unsigned long long offset = 0ULL;
${expand}
      total += dif_load(g, base + offset);
    }
    dif_store(y, i, total);
  }
}
