// One thread per column sums the output gradient down the rows.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, dif_scalar* grad_bias) {
  unsigned long long column = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (column < ${width}ULL) {
    float value = 0.0f;
    for (unsigned long long row = 0ULL; row < ${rows}ULL; ++row)
      value += dif_load(grad_output, row * ${width}ULL + column);
    dif_store(grad_bias, column, value);
  }
}
