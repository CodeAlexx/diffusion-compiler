// Shared-vector-delta rms_norm_modulate, BF16 6144-wide rows, 512 threads,
// Triton reduction-tile 2048 order: y = (1 + (vector + delta_scale)) *
// (x * inv * (weight + offset)) + (vector + delta_shift), every step a
// separately rounded PTX op.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vector, const dif_scalar* delta, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * 6144ULL + (unsigned long long)tid * 4ULL;
  float a0, a1, a2, a3;
  float m0 = dif_load(x, base + 2048ULL), m1 = dif_load(x, base + 2049ULL), m2 = dif_load(x, base + 2050ULL),
        m3 = dif_load(x, base + 2051ULL);
  float l0 = dif_load(x, base), l1 = dif_load(x, base + 1ULL), l2 = dif_load(x, base + 2ULL), l3 = dif_load(x, base + 3ULL);
  a0 = fmaf(l0, l0, m0 * m0); a1 = fmaf(l1, l1, m1 * m1); a2 = fmaf(l2, l2, m2 * m2); a3 = fmaf(l3, l3, m3 * m3);
  float h0 = dif_load(x, base + 4096ULL), h1 = dif_load(x, base + 4097ULL), h2 = dif_load(x, base + 4098ULL),
        h3 = dif_load(x, base + 4099ULL);
  a0 = fmaf(h0, h0, a0); a1 = fmaf(h1, h1, a1); a2 = fmaf(h2, h2, a2); a3 = fmaf(h3, h3, a3);
  float local = ((a0 + a1) + a2) + a3;
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
