  // Match the CUDA vectorized RMSNorm reduction used by the pinned PyTorch
  // oracle: one warp consumes four adjacent values per lane, followed by a
  // 16,8,4,2,1 shuffle-style reduction.  The exact grouping matters at BF16
  // boundaries even though it changes only a handful of values.
  if (threadIdx.x < 32U) {
    unsigned long long d = (unsigned long long)threadIdx.x * 4ULL;
    if (d < ${dim}ULL) {
      float v0 = dif_load(x, base + d); local += v0 * v0;
      float v1 = dif_load(x, base + d + 1ULL); local += v1 * v1;
      float v2 = dif_load(x, base + d + 2ULL); local += v2 * v2;
      float v3 = dif_load(x, base + d + 3ULL); local += v3 * v3;
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 16U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
