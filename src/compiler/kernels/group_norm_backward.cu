// Gradient of group normalization with respect to its input. One block per
// (batch, group): the first reduction recomputes the group statistics the
// forward used, the second carries the two projections the normalization
// backward needs, and the write is exclusive to the block.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* g, dif_scalar* y) {
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
  __syncthreads();
  float grad_sum = 0.0f, grad_dot = 0.0f;
  for (unsigned long long k = lane; k < ${elements}ULL; k += blockDim.x) {
    unsigned long long channel = group * ${channels_per_group}ULL + k / ${inner}ULL;
    float scaled = dif_load(g, base + k) * dif_load(weight, channel);
    grad_sum += scaled;
    grad_dot = fmaf(scaled, (dif_load(x, base + k) - mean) * inv, grad_dot);
  }
  reduction[lane] = grad_sum;
  reduction[blockDim.x + lane] = grad_dot;
  __syncthreads();
${reduction}
  float mean_grad = reduction[0] / ${elements}.0f;
  float dot_grad = reduction[blockDim.x] / ${elements}.0f;
  for (unsigned long long k = lane; k < ${elements}ULL; k += blockDim.x) {
    unsigned long long channel = group * ${channels_per_group}ULL + k / ${inner}ULL;
    float normalized = (dif_load(x, base + k) - mean) * inv;
    float scaled = dif_load(g, base + k) * dif_load(weight, channel);
    dif_store(y, base + k, inv * (scaled - mean_grad - normalized * dot_grad));
  }
}
