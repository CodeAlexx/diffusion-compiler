// Layer norm + adaLN modulation, generic path: one block per row, two
// shared-memory tree reductions (mean, then centered variance), then the row
// is normalized, affine transformed and modulated as
// (1 + scale) * normalized + shift with rounding at every step the reference
// applies.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= ${rows}ULL) return;
  float local = 0.0f;
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) local += dif_load(x, row * ${columns}ULL + col);
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
  unsigned long long modulation_row = row / ${rows_per_modulation}ULL;
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) {
    unsigned long long i = row * ${columns}ULL + col, mi = modulation_row * ${columns}ULL + col;
    float normalized = dif_round((dif_load(x, i) - mean) * inv * dif_load(weight, col) + dif_load(bias, col));
    float one_plus_scale = dif_round(1.0f + dif_load(scale, mi));
    float scaled = dif_round(normalized * one_plus_scale);
    dif_store(y, i, scaled + dif_load(shift, mi));
  }
}
