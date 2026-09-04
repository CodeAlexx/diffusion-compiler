  // 128-wide rows, 128 threads: 16 lanes take 8 values each, one warp folds.
  if (threadIdx.x < 16U) {
    unsigned long long base = row * 128ULL + (unsigned long long)threadIdx.x * 8ULL;
    float v0 = dif_load(x, base), v1 = dif_load(x, base + 1ULL),
          v2 = dif_load(x, base + 2ULL), v3 = dif_load(x, base + 3ULL),
          v4 = dif_load(x, base + 4ULL), v5 = dif_load(x, base + 5ULL),
          v6 = dif_load(x, base + 6ULL), v7 = dif_load(x, base + 7ULL);
    local = v1 * v1; local = fmaf(v0, v0, local);
    local = fmaf(v2, v2, local); local = fmaf(v3, v3, local);
    local = fmaf(v4, v4, local); local = fmaf(v5, v5, local);
    local = fmaf(v6, v6, local); local = fmaf(v7, v7, local);
  } else local = 0.0f;
  if (threadIdx.x < 32U) {
    for (unsigned delta = 8U; delta > 0U; delta >>= 1U)
      local += __shfl_xor_sync(0xffffffffU, local, delta);
    if (threadIdx.x == 0U) reduction[0] = local;
  }
  __syncthreads();
