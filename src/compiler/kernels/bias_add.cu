extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* bias, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) dif_store(y, i, dif_load(x, i) + dif_load(bias, i % ${width}ULL));
}
