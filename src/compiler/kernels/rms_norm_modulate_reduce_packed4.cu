  // Match PyTorch's vectorized CUDA RMSNorm: 128 logical threads, four
  // adjacent values per vector, per-warp 16..1 reduction, then the four
  // warp totals are combined as (warp0 + warp2) + (warp1 + warp3).
  if (threadIdx.x < 128U) {
    for (unsigned long long pack = threadIdx.x; pack < ${packs}ULL; pack += 128ULL) {
      unsigned long long col = pack * 4ULL;
      unsigned long long base = (unsigned long long)row * ${cols}ULL + col;
      float v0 = dif_load(x, base); local += v0 * v0;
      float v1 = dif_load(x, base + 1ULL); local += v1 * v1;
      float v2 = dif_load(x, base + 2ULL); local += v2 * v2;
      float v3 = dif_load(x, base + 3ULL); local += v3 * v3;
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 16U; stride > 0U; stride >>= 1U) {
    unsigned lane = threadIdx.x & 31U;
    if (threadIdx.x < 128U && lane < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U) reduction[0] += reduction[64];
  else if (threadIdx.x == 32U) reduction[32] += reduction[96];
  __syncthreads();
  if (threadIdx.x == 0U) reduction[0] += reduction[32];
  __syncthreads();
