extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float v = dif_load(x, i);
    dif_store(y, i, 1.0f / (1.0f + expf(-v)));
  }
}
