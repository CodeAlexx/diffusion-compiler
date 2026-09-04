// Addmm bias mode: the output is prefilled with the broadcast bias and the
// GEMM accumulates onto it (beta = 1). x and weight are the GEMM operands.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, dif_scalar* y) {
  (void)x;
  (void)weight;
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) dif_store(y, i, dif_load(bias, i % ${width}ULL));
}
