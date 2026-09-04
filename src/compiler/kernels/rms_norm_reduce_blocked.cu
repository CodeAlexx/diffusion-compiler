  // 6144-wide rows, 512 threads: 8 contiguous values per thread plus a
  // 256-thread tail over the last 2048 columns, squared with mul.rn to match
  // the Triton reference rounding; warp shuffles then a 16-warp fold.
  unsigned long long base = row * ${columns}ULL + (unsigned long long)threadIdx.x * 8ULL;
  float v0 = dif_load(x, base), v1 = dif_load(x, base + 1ULL),
        v2 = dif_load(x, base + 2ULL), v3 = dif_load(x, base + 3ULL),
        v4 = dif_load(x, base + 4ULL), v5 = dif_load(x, base + 5ULL),
        v6 = dif_load(x, base + 6ULL), v7 = dif_load(x, base + 7ULL);
  local = v1 * v1; local = fmaf(v0, v0, local);
  local = fmaf(v2, v2, local); local = fmaf(v3, v3, local);
  local = fmaf(v4, v4, local); local = fmaf(v5, v5, local);
  local = fmaf(v6, v6, local); local = fmaf(v7, v7, local);
  if (threadIdx.x < 256U) {
    float square, value;
    value = dif_load(x, base + 4096ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4097ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4098ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4099ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4100ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4101ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4102ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local; value = dif_load(x, base + 4103ULL);
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value));
    local = square + local;
  }
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
