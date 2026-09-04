// Group normalization, one block per (batch, group): mean and variance over
// the group's channels and every trailing dimension (two shared arrays, the
// unrolled stride lines are generated for the block size), then the affine.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long vector = (unsigned long long)blockIdx.x;
  if (vector >= ${vectors}ULL) return;
  unsigned long long lane = threadIdx.x, group = vector % ${groups}ULL, batch = vector / ${groups}ULL,
                     base = (batch * ${channels}ULL + group * ${channels_per_group}ULL) * ${inner}ULL;
  float sum = 0.0f, squares = 0.0f;
  for (unsigned long long k = lane; k < ${elements}ULL; k += blockDim.x) {
    float v = dif_load(x, base + k);
    sum += v;
    squares = fmaf(v, v, squares);
  }
  reduction[lane] = sum;
  reduction[blockDim.x + lane] = squares;
  __syncthreads();
${reduction}
  float mean = reduction[0] / ${elements}.0f;
  float variance = fmaxf(reduction[blockDim.x] / ${elements}.0f - mean * mean, 0.0f);
  float inv = rsqrtf(variance + ${epsilon}f);
  for (unsigned long long k = lane; k < ${elements}ULL; k += blockDim.x) {
    unsigned long long channel = group * ${channels_per_group}ULL + k / ${inner}ULL;
    float normalized = (dif_load(x, base + k) - mean) * inv;
    dif_store(y, base + k, fmaf(normalized, dif_load(weight, channel), dif_load(bias, channel)));
  }
}
