// Backward of RMS-normalized, partially rotated q/k: the upstream gradient
// is rotated back (the generated rot(k) expressions), the norm backward
// follows, and the optional weight gradient is reduced by the first D threads.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* grad_input${weight_parameter}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${dim}ULL, d = i % ${dim}ULL, rb = row * ${dim}ULL, tb = (row / ${heads}ULL) * ${table_width}ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < ${dim}ULL; ++k) {
      float value = dif_load(x, rb + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / ${dim}.0f + ${epsilon}f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < ${dim}ULL; ++k) {
      float rotated_gradient = ${rotated_k};
      dot = fmaf(rotated_gradient * dif_load(weight, k), dif_load(x, rb + k), dot);
    }
    float own_rotated = ${rotated_d};
    float value = dif_load(x, i);
    dif_store(grad_input, i, own_rotated * dif_load(weight, d) * inv - value * inv * inv * inv * dot / ${dim}.0f);
    ${weight_gradient}
  }
}
