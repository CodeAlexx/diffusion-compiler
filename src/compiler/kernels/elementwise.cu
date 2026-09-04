extern "C" __global__ void ${function}(const dif_scalar* a, const dif_scalar* b, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) dif_store(y, i, ${expression});
}
