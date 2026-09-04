// Gradient of the gated linear unit activation. The derivative fragment is
// generated from the same approximation attribute the forward carried, so
// this differentiates the closed form that actually ran.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* g, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float v = dif_load(x, i);
    float d;
    ${derivative}
    dif_store(y, i, dif_load(g, i) * d);
  }
}
