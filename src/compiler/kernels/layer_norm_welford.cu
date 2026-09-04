// Layer norm for bf16 rows divisible by 4 with 128-thread blocks: a
// per-thread Welford state over 4-value packs, warp-shuffle combine, then a
// two-level shared combine; helpers are suffixed per operation.
struct dif_welford_${suffix} { float mean; float sigma2; float count; };
extern "C" __device__ __forceinline__ dif_welford_${suffix} dif_welford_online_${suffix}(float value, dif_welford_${suffix} current) {
  float delta = value - current.mean;
  float count = current.count + 1.0f;
  float mean = current.mean + delta * (1.0f / count);
  return {mean, current.sigma2 + delta * (value - mean), count};
}
extern "C" __device__ __forceinline__ dif_welford_${suffix} dif_welford_combine_${suffix}(dif_welford_${suffix} data_b, dif_welford_${suffix} data_a) {
  float delta = data_b.mean - data_a.mean;
  float count = data_a.count + data_b.count;
  if (count > 0.0f) {
    float coefficient = 1.0f / count;
    float n_a = data_a.count * coefficient;
    float n_b = data_b.count * coefficient;
    return {n_a * data_a.mean + n_b * data_b.mean, data_a.sigma2 + data_b.sigma2 + delta * delta * data_a.count * n_b, count};
  }
  return {0.0f, 0.0f, 0.0f};
}
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* bias, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned tid = threadIdx.x, lane = tid & 31U, warp = tid >> 5U;
  dif_welford_${suffix} state = {0.0f, 0.0f, 0.0f};
  for (unsigned long long pack = tid; pack < ${packs}ULL; pack += 128ULL) {
    unsigned long long base = row * ${columns}ULL + pack * 4ULL;
    state = dif_welford_online_${suffix}(dif_load(x, base), state);
    state = dif_welford_online_${suffix}(dif_load(x, base + 1ULL), state);
    state = dif_welford_online_${suffix}(dif_load(x, base + 2ULL), state);
    state = dif_welford_online_${suffix}(dif_load(x, base + 3ULL), state);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dif_welford_${suffix} other = {__shfl_down_sync(0xffffffffU, state.mean, offset),
                                   __shfl_down_sync(0xffffffffU, state.sigma2, offset),
                                   __shfl_down_sync(0xffffffffU, state.count, offset)};
    state = dif_welford_combine_${suffix}(state, other);
  }
  float* meansigma = reduction;
  float* counts = reduction + 4U;
  for (unsigned offset = 2U; offset > 0U; offset >>= 1U) {
    if (lane == 0U && warp >= offset && warp < 2U * offset) {
      unsigned target = warp - offset;
      meansigma[2U * target] = state.mean;
      meansigma[2U * target + 1U] = state.sigma2;
      counts[target] = state.count;
    }
    __syncthreads();
    if (lane == 0U && warp < offset) {
      dif_welford_${suffix} other = {meansigma[2U * warp], meansigma[2U * warp + 1U], counts[warp]};
      state = dif_welford_combine_${suffix}(state, other);
    }
    __syncthreads();
  }
  if (tid == 0U) {
    meansigma[0] = state.mean;
    meansigma[1] = state.sigma2 / ${columns}.0f;
  }
  __syncthreads();
  float mean = meansigma[0];
  float inverse = rsqrtf(meansigma[1] + ${epsilon}f);
  for (unsigned long long pack = tid; pack < ${packs}ULL; pack += 128ULL) {
    unsigned long long base = row * ${columns}ULL + pack * 4ULL;
    for (unsigned inner = 0U; inner < 4U; ++inner) {
      unsigned long long index = base + inner;
      float normalized = inverse * (dif_load(x, index) - mean);
      dif_store(y, index, dif_load(weight, pack * 4ULL + inner) * normalized + dif_load(bias, pack * 4ULL + inner));
    }
  }
}
