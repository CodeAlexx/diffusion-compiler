  // 6144-wide rows, 512 threads: three 2048-column chunks accumulated
  // per lane in four accumulators (Triton reduction-tile 2048 order).
  unsigned long long base = row * ${columns}ULL + (unsigned long long)threadIdx.x * 4ULL;
  float a0, a1, a2, a3;
  float m0 = dif_load(x, base + 2048ULL), m1 = dif_load(x, base + 2049ULL),
        m2 = dif_load(x, base + 2050ULL), m3 = dif_load(x, base + 2051ULL);
  float l0 = dif_load(x, base), l1 = dif_load(x, base + 1ULL),
        l2 = dif_load(x, base + 2ULL), l3 = dif_load(x, base + 3ULL);
  a0 = fmaf(l0, l0, m0 * m0); a1 = fmaf(l1, l1, m1 * m1);
  a2 = fmaf(l2, l2, m2 * m2); a3 = fmaf(l3, l3, m3 * m3);
  float h0 = dif_load(x, base + 4096ULL), h1 = dif_load(x, base + 4097ULL),
        h2 = dif_load(x, base + 4098ULL), h3 = dif_load(x, base + 4099ULL);
  a0 = fmaf(h0, h0, a0); a1 = fmaf(h1, h1, a1);
  a2 = fmaf(h2, h2, a2); a3 = fmaf(h3, h3, a3);
  local = ((a0 + a1) + a2) + a3;
  for (unsigned delta = 16U; delta > 0U; delta >>= 1U)
    local += __shfl_xor_sync(0xffffffffU, local, delta);
  unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
  if (lane == 0U) reduction[warp] = local;
  __syncthreads();
  if (warp == 0U) {
    local = lane < 16U ? reduction[lane] : 0.0f;
    for (unsigned delta = 8U; delta > 0U; delta >>= 1U)
      local += __shfl_xor_sync(0xffffffffU, local, delta);
    if (lane == 0U) reduction[0] = local;
  }
  __syncthreads();
