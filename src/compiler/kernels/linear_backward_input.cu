// grad_input[row, k] = sum_n grad_output[row, n] * weight[n, k] (reference
// form: one thread per output element, serial fmaf accumulation).
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* weight, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${inner}ULL, column = i % ${inner}ULL;
    float value = 0.0f;
    for (unsigned long long output = 0ULL; output < ${outputs}ULL; ++output)
      value = fmaf(dif_load(grad_output, row * ${outputs}ULL + output), dif_load(weight, output * ${inner}ULL + column), value);
    dif_store(grad_input, i, value);
  }
}
