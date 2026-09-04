// Gradient of layer normalization followed by an adaptive scale and shift:
//   out = ((x-mean)*inv*W + B) * (1+scale) + shift
// Row statistics are recomputed from the original input in F32 rather than
// saved, the way the plain layer-norm gradient beside this one does.
//
// Three different reductions live here, and each output element is owned by
// exactly one thread so all of them sum deterministically without atomics:
// grad_input is elementwise, the affine gradients reduce down every row, and
// the modulation gradients reduce only across the rows sharing one modulation
// row. Reference form -- the row statistics are recomputed inside each
// reduction, which is quadratic and honest about it.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, const dif_scalar* scale, dif_scalar* grad_input, dif_scalar* grad_weight, dif_scalar* grad_bias, dif_scalar* grad_scale, dif_scalar* grad_shift) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, base = row * ${columns}ULL;
    unsigned long long modulation = row / ${rows_per_modulation}ULL * ${columns}ULL;
    float mean = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) mean += dif_load(x, base + k);
    mean /= ${columns}.0f;
    float variance = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float centered = dif_load(x, base + k) - mean;
      variance = fmaf(centered, centered, variance);
    }
    float inv = rsqrtf(variance / ${columns}.0f + ${epsilon}f);
    float gradient_mean = 0.0f, projected_mean = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float upstream = dif_load(grad_output, base + k) * (1.0f + dif_load(scale, modulation + k));
      float weighted = upstream * dif_load(weight, k);
      float normalized = (dif_load(x, base + k) - mean) * inv;
      gradient_mean += weighted;
      projected_mean = fmaf(weighted, normalized, projected_mean);
    }
    gradient_mean /= ${columns}.0f;
    projected_mean /= ${columns}.0f;
    unsigned long long column = i - base;
    float upstream = dif_load(grad_output, i) * (1.0f + dif_load(scale, modulation + column));
    float weighted = upstream * dif_load(weight, column);
    float normalized = (dif_load(x, i) - mean) * inv;
    dif_store(grad_input, i, inv * (weighted - gradient_mean - normalized * projected_mean));
  }
  // The affine gradients belong to the whole tensor: one thread per column,
  // scanning every row.
  if (i < ${columns}ULL) {
    float weight_accumulator = 0.0f, bias_accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
      unsigned long long rb = r * ${columns}ULL;
      unsigned long long rm = r / ${rows_per_modulation}ULL * ${columns}ULL;
      float rmean = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) rmean += dif_load(x, rb + k);
      rmean /= ${columns}.0f;
      float rvariance = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
        float centered = dif_load(x, rb + k) - rmean;
        rvariance = fmaf(centered, centered, rvariance);
      }
      float rinv = rsqrtf(rvariance / ${columns}.0f + ${epsilon}f);
      float g = dif_load(grad_output, rb + i) * (1.0f + dif_load(scale, rm + i));
      weight_accumulator = fmaf(g, (dif_load(x, rb + i) - rmean) * rinv, weight_accumulator);
      bias_accumulator += g;
    }
    dif_store(grad_weight, i, weight_accumulator);
    dif_store(grad_bias, i, bias_accumulator);
  }
  // The modulation gradients belong to one modulation row: one thread per
  // (modulation row, column), scanning only the rows that share it.
  if (i < ${modulation_count}ULL) {
    unsigned long long m = i / ${columns}ULL, column = i % ${columns}ULL;
    float scale_accumulator = 0.0f, shift_accumulator = 0.0f;
    for (unsigned long long r = m * ${rows_per_modulation}ULL; r < (m + 1ULL) * ${rows_per_modulation}ULL; ++r) {
      unsigned long long rb = r * ${columns}ULL;
      float rmean = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) rmean += dif_load(x, rb + k);
      rmean /= ${columns}.0f;
      float rvariance = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
        float centered = dif_load(x, rb + k) - rmean;
        rvariance = fmaf(centered, centered, rvariance);
      }
      float rinv = rsqrtf(rvariance / ${columns}.0f + ${epsilon}f);
      float normalized = (dif_load(x, rb + column) - rmean) * rinv * dif_load(weight, column) + dif_load(bias, column);
      float g = dif_load(grad_output, rb + column);
      scale_accumulator = fmaf(g, normalized, scale_accumulator);
      shift_accumulator += g;
    }
    dif_store(grad_scale, i, scale_accumulator);
    dif_store(grad_shift, i, shift_accumulator);
  }
}
