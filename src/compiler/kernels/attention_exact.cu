// Exact attention reference: one block per (query, head) of the whole
// batch; scores reduced through shared memory, a serial softmax on thread 0,
// then the value accumulation. Grouped-query heads read kv head h / group,
// and the key base carries the batch offset when there is a batch.
extern "C" __global__ void ${function}(const dif_scalar* q, const dif_scalar* k, const dif_scalar* v, dif_scalar* y) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* probabilities = shared + blockDim.x;
  unsigned long long item = blockIdx.x;
  if (item >= ${items}ULL) return;
  unsigned long long qs = item / ${heads}ULL, h = item % ${heads}ULL;
  unsigned long long kend = ${kend};
  for (unsigned long long ks = 0; ks < kend; ++ks) {
    float partial = 0.0f;
    for (unsigned long long d = threadIdx.x; d < ${dim}ULL; d += blockDim.x)
      partial = fmaf(dif_load(q, (qs * ${heads}ULL + h) * ${dim}ULL + d), dif_load(k, ${key_base} + d), partial);
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) probabilities[ks] = reduction[0] * ${scale}f;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float maximum = -3.402823466e+38f;
    for (unsigned long long ks = 0; ks < kend; ++ks) maximum = fmaxf(maximum, probabilities[ks]);
    float denominator = 0.0f;
    for (unsigned long long ks = 0; ks < kend; ++ks) {
      probabilities[ks] = expf(probabilities[ks] - maximum);
      denominator += probabilities[ks];
    }
    for (unsigned long long ks = 0; ks < kend; ++ks) probabilities[ks] /= denominator;
  }
  __syncthreads();
  for (unsigned long long d = threadIdx.x; d < ${dim}ULL; d += blockDim.x) {
    float acc = 0.0f;
    for (unsigned long long ks = 0; ks < kend; ++ks)
      acc = fmaf(probabilities[ks], dif_load(v, ${key_base} + d), acc);
    dif_store(y, (qs * ${heads}ULL + h) * ${dim}ULL + d, acc);
  }
}
