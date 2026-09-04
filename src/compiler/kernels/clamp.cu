extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) dif_store(y, i, fminf(${upper}f, fmaxf(${lower}f, dif_load(x, i))));
}
