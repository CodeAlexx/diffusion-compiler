// Generic RMS-normalized, partially rotated (half-split) q/k: one block per
// (sequence, head); the reduction fragment matches the pinned PyTorch
// oracle's grouping, every product is rounded to the storage dtype.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long bh = blockIdx.x;
  if (bh >= ${rows}ULL) return;
  unsigned long long s = bh / ${heads}ULL, h = bh % ${heads}ULL;
  unsigned long long base = (s * ${heads}ULL + h) * ${dim}ULL;
  float local = 0.0f;
${reduction}
  float inv = rsqrtf(reduction[0] / ${dim}.0f + ${epsilon}f);
  for (unsigned long long d = threadIdx.x; d < ${dim}ULL; d += blockDim.x) {
    float value = dif_round(dif_load(x, base + d) * inv * dif_load(weight, d));
    float result = value;
    if (d < ${half}ULL) {
      float other = dif_round(dif_load(x, base + d + ${half}ULL) * inv * dif_load(weight, d + ${half}ULL));
      float left = dif_round(value * dif_load(cosv, s * ${table_width}ULL + d));
      float right = dif_round(other * dif_load(sinv, s * ${table_width}ULL + d));
      result = dif_round(left - right);
    } else if (d < ${rotary}ULL) {
      unsigned long long r = d - ${half}ULL;
      float other = dif_round(dif_load(x, base + r) * inv * dif_load(weight, r));
      unsigned long long ti = ${table_index};
      float left = dif_round(value * dif_load(cosv, s * ${table_width}ULL + ti));
      float right = dif_round(other * dif_load(sinv, s * ${table_width}ULL + ti));
      result = dif_round(left + right);
    }
    dif_store(y, base + d, result);
  }
}
