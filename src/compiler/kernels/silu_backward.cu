extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* grad_output, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float value = dif_load(x, i), sigmoid = 1.0f / (1.0f + expf(-value));
    float derivative = sigmoid * (1.0f + value * (1.0f - sigmoid));
    dif_store(grad_input, i, dif_load(grad_output, i) * derivative);
  }
}
