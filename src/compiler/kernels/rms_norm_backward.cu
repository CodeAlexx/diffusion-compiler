// Reference rms_norm backward: one thread per element recomputes the row
// statistics; the optional weight gradient is reduced over rows by the
// first `columns` threads.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, dif_scalar* grad_input${weight_parameter}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, base = row * ${columns}ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float value = dif_load(x, base + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / ${columns}.0f + ${epsilon}f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k)
      dot = fmaf(dif_load(grad_output, base + k) * dif_load(weight, k), dif_load(x, base + k), dot);
    float value = dif_load(x, i);
    float gradient = dif_load(grad_output, i) * dif_load(weight, i - base) * inv - value * inv * inv * inv * dot / ${columns}.0f;
    dif_store(grad_input, i, gradient);
    ${weight_gradient}
  }
}
