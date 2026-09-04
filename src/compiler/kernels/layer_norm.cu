// Generic layer norm: two shared-memory reductions (mean, then centered
// variance) with the block's threads striding the row.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= ${rows}ULL) return;
  float local = 0.0f;
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x)
    local += dif_load(x, row * ${columns}ULL + col);
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float mean = reduction[0] / ${columns}.0f;
  __syncthreads();
  local = 0.0f;
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) {
    float centered = dif_load(x, row * ${columns}ULL + col) - mean;
    local += centered * centered;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(reduction[0] / ${columns}.0f + ${epsilon}f);
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) {
    unsigned long long i = row * ${columns}ULL + col;
    dif_store(y, i, (dif_load(x, i) - mean) * inv * dif_load(weight, col) + dif_load(bias, col));
  }
}
