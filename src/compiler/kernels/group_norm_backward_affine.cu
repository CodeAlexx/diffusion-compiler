// Weight and bias gradients of group normalization. One block per channel,
// reducing over the batch and the trailing dimensions, so each output is
// written by exactly one block and the sum is deterministic. The group
// statistics are recomputed per batch the way the forward computed them.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* g, dif_scalar* grad_weight, dif_scalar* grad_bias) {
  extern __shared__ float reduction[];
  unsigned long long channel = (unsigned long long)blockIdx.x;
  if (channel >= ${channels}ULL) return;
  unsigned long long lane = threadIdx.x, group = channel / ${channels_per_group}ULL;
  float weight_total = 0.0f, bias_total = 0.0f;
  for (unsigned long long batch = 0ULL; batch < ${batch}ULL; ++batch) {
    unsigned long long group_base = (batch * ${channels}ULL + group * ${channels_per_group}ULL) * ${inner}ULL;
    float sum = 0.0f, squares = 0.0f;
    for (unsigned long long k = lane; k < ${elements}ULL; k += blockDim.x) {
      float v = dif_load(x, group_base + k);
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
    __syncthreads();
    unsigned long long base = (batch * ${channels}ULL + channel) * ${inner}ULL;
    for (unsigned long long i = lane; i < ${inner}ULL; i += blockDim.x) {
      float gradient = dif_load(g, base + i);
      weight_total = fmaf(gradient, (dif_load(x, base + i) - mean) * inv, weight_total);
      bias_total += gradient;
    }
  }
  reduction[lane] = weight_total;
  reduction[blockDim.x + lane] = bias_total;
  __syncthreads();
${reduction}
  if (lane == 0ULL) {
    dif_store(grad_weight, channel, reduction[0]);
    dif_store(grad_bias, channel, reduction[blockDim.x]);
  }
}
