// grad_weight[n, k] = sum_rows grad_output[row, n] * input[row, k]
// (reference form: one thread per element, serial fmaf accumulation).
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* input, dif_scalar* grad_weight) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long output = i / ${inner}ULL, column = i % ${inner}ULL;
    float value = 0.0f;
    for (unsigned long long row = 0ULL; row < ${rows}ULL; ++row)
      value = fmaf(dif_load(grad_output, row * ${outputs}ULL + output), dif_load(input, row * ${inner}ULL + column), value);
    dif_store(grad_weight, i, value);
  }
}
