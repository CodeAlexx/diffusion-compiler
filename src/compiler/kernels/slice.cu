// Contiguous window along one axis; the per-axis lines are generated.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long coordinate = i, source = 0ULL, at = 0ULL;
${axes}
    dif_store(y, i, dif_load(x, source));
  }
}
