// dx = inv*(gw - mean(gw) - xhat*mean(gw*xhat)); mean/inv recomputed from
// the original input in F32 (flame's non-affine-LN cancellation lesson).
// The first `columns` threads also reduce the affine gradients over rows.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, dif_scalar* grad_input, dif_scalar* grad_weight, dif_scalar* grad_bias) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, base = row * ${columns}ULL;
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
      float weighted = dif_load(grad_output, base + k) * dif_load(weight, k);
      float normalized = (dif_load(x, base + k) - mean) * inv;
      gradient_mean += weighted;
      projected_mean = fmaf(weighted, normalized, projected_mean);
    }
    gradient_mean /= ${columns}.0f;
    projected_mean /= ${columns}.0f;
    float upstream = dif_load(grad_output, i);
    float weighted = upstream * dif_load(weight, i - base);
    float normalized = (dif_load(x, i) - mean) * inv;
    dif_store(grad_input, i, inv * (weighted - gradient_mean - normalized * projected_mean));
    if (i < ${columns}ULL) {
      float weight_accumulator = 0.0f, bias_accumulator = 0.0f;
      for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
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
        float g = dif_load(grad_output, rb + i);
        weight_accumulator = fmaf(g, (dif_load(x, rb + i) - rmean) * rinv, weight_accumulator);
        bias_accumulator += g;
      }
      dif_store(grad_weight, i, weight_accumulator);
      dif_store(grad_bias, i, bias_accumulator);
    }
  }
}
