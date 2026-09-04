// Channel-axis RMS norm with the creator's storage boundaries: one block
// per (leading, trailing) vector, a shared-memory sum of squares over the
// channel axis (the unrolled stride lines are generated for the block size),
// then y = round(round(x / max(sqrt(sum), eps)) * sqrt(C)) * gamma.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* gamma, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long vector = (unsigned long long)blockIdx.x;
  if (vector >= ${vectors}ULL) return;
  unsigned long long c = threadIdx.x;
  unsigned long long leading = vector / ${inner}ULL, trailing = vector % ${inner}ULL;
  unsigned long long index = (leading * ${channels}ULL + c) * ${inner}ULL + trailing;
  float value = c < ${channels}ULL ? dif_load(x, index) : 0.0f;
  reduction[c] = value * value;
  __syncthreads();
${reduction}
  if (c < ${channels}ULL) {
    float denominator = fmaxf(sqrtf(reduction[0]), ${epsilon}f);
    float normalized = dif_round(value / denominator);
    float scaled = dif_round(normalized * ${scale}f);
    dif_store(y, index, scaled * dif_load(gamma, c));
  }
}
