// Backward of rms_norm followed by (1 + scale) modulation, optionally with a
// learned weight; per element recomputation of the row statistics, plus the
// optional weight gradient reduced by the first `columns` threads.
extern "C" __global__ void ${function}(${parameters}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, col = i % ${columns}ULL, base = row * ${columns}ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float value = dif_load(x, base + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / ${columns}.0f + ${epsilon}f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k)
      dot = fmaf(dif_load(grad_output, base + k) * (1.0f + dif_load(scale, base + k))${dot_weight}, dif_load(x, base + k), dot);
    float value = dif_load(x, i);
    float upstream = dif_load(grad_output, i);
    float normed_gradient = upstream * (1.0f + dif_load(scale, i));
    float weight_value = ${weight_value}
    dif_store(grad_input, i, normed_gradient * weight_value * inv - value * inv * inv * inv * dot / ${columns}.0f);
    dif_store(grad_scale, i, upstream * value * inv * weight_value);
    dif_store(grad_shift, i, upstream);
    ${weight_gradient}
  }
}
