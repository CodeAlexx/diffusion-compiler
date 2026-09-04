// Generic RMS-normalized, partially rotated q/k: one block per
// (sequence, head); the reduction fragment matches the pinned PyTorch
// oracle's grouping, every product is rounded to the storage dtype.
// The rotation layout and the row offset into the rotation table are
// generated from the operation's own attributes, so this kernel rotates the
// way the operation says rather than assuming a half split.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const ${table_scalar}* cosv, const ${table_scalar}* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long bh = blockIdx.x;
  if (bh >= ${rows}ULL) return;
  unsigned long long s = bh / ${heads}ULL, h = bh % ${heads}ULL;
  unsigned long long base = (s * ${heads}ULL + h) * ${dim}ULL;
  unsigned long long tb = ${table_base};
  float local = 0.0f;
${reduction}
  float inv = rsqrtf(reduction[0] / ${dim}.0f + ${epsilon}f);
  for (unsigned long long d = threadIdx.x; d < ${dim}ULL; d += blockDim.x) {
    float value = dif_round(dif_load(x, base + d) * inv * dif_load(weight, d));
    float result = value;
    ${rotation}
    dif_store(y, base + d, result);
  }
}
