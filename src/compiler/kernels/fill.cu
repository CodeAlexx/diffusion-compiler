extern "C" __global__ void ${function}(dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) dif_store(y, i, ${value}f);
}
