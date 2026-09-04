// Shared-vector-delta rms_norm_modulate, BF16 6144-wide rows, 512 threads,
// Triton reduction-tile 8192 order (8 contiguous values per thread plus a
// 256-thread tail with mul.rn squares); modulation as in the chunked form.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vector, const dif_scalar* delta, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * ${cols}ULL + (unsigned long long)tid * 8ULL;
  float v0 = dif_load(x, base), v1 = dif_load(x, base + 1ULL), v2 = dif_load(x, base + 2ULL), v3 = dif_load(x, base + 3ULL),
        v4 = dif_load(x, base + 4ULL), v5 = dif_load(x, base + 5ULL), v6 = dif_load(x, base + 6ULL), v7 = dif_load(x, base + 7ULL);
  float local = v1 * v1;
  local = fmaf(v0, v0, local); local = fmaf(v2, v2, local); local = fmaf(v3, v3, local); local = fmaf(v4, v4, local);
  local = fmaf(v5, v5, local); local = fmaf(v6, v6, local); local = fmaf(v7, v7, local);
  if (tid < 256U) {
    float square, value;
    value = dif_load(x, base + 4096ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4097ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4098ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4099ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4100ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4101ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4102ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
    value = dif_load(x, base + 4103ULL); asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(square) : "f"(value)); local = square + local;
  }
  for (unsigned delta_step = 16U; delta_step > 0U; delta_step >>= 1U) local += __shfl_xor_sync(0xffffffffU, local, delta_step);
  unsigned lane = tid & 31U, warp = tid >> 5U;
  if (lane == 0U) reduction[warp] = local;
  __syncthreads();
  if (warp == 0U) {
    local = lane < 16U ? reduction[lane] : 0.0f;
    for (unsigned delta_step = 8U; delta_step > 0U; delta_step >>= 1U) local += __shfl_xor_sync(0xffffffffU, local, delta_step);
    if (lane == 0U) reduction[0] = local;
  }
  __syncthreads();
  float mean, mean_eps, inv;
  asm volatile("div.full.f32 %0,%1,%2;" : "=f"(mean) : "f"(reduction[0]), "f"(6144.0f));
  mean_eps = mean + ${epsilon}f;
  asm volatile("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(mean_eps));
  unsigned long long shared_base = (row / ${rows_per_vector}ULL) * 6144ULL;
  for (unsigned long long col = tid; col < 6144ULL; col += blockDim.x) {
    unsigned long long i = row * 6144ULL + col;
    float xv = dif_load(x, i), wv = dif_load(weight, col), base_value = dif_load(vector, shared_base + col),
          scale_delta = dif_load(delta, col), shift_delta = dif_load(delta, 6144ULL + col);
    float normalized, weighted, scale, scale_one, shift, result;
    asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(normalized) : "f"(xv), "f"(inv));
    float weight_value = wv + ${weight_offset}f;
    asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(weighted) : "f"(normalized), "f"(weight_value));
    asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(scale) : "f"(base_value), "f"(scale_delta));
    scale_one = scale + 1.0f;
    asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(shift) : "f"(base_value), "f"(shift_delta));
    result = fmaf(scale_one, weighted, shift);
    dif_store(y, i, result);
  }
}
