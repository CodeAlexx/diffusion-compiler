// Gradient of the logistic sigmoid: s * (1 - s) at the forward's own input,
// recomputed rather than saved. One elementwise pass, the same shape as the
// SiLU gradient beside it.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* grad_output, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float sigmoid = 1.0f / (1.0f + expf(-dif_load(x, i)));
    dif_store(grad_input, i, dif_load(grad_output, i) * sigmoid * (1.0f - sigmoid));
  }
}
