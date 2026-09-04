// grad_input[row, k] = sum_n grad_output[row, n] * weight[n, k] * scale[n].
// The weight is read as INT8 and scaled inside the accumulation, so no
// dequantized copy of it ever exists.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const signed char* weight, const dif_f32* scales, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${inner}ULL, column = i % ${inner}ULL;
    float value = 0.0f;
    for (unsigned long long output = 0ULL; output < ${outputs}ULL; ++output) {
      float weight_value = __fmul_rn((float)weight[output * ${inner}ULL + column], dif_load_f32(scales, output));
      value = fmaf(dif_load(grad_output, row * ${outputs}ULL + output), weight_value, value);
    }
    dif_store(grad_input, i, value);
  }
}
