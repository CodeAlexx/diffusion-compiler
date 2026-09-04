// Implementation 2: BF16 128-wide heads, interleaved rotary with a
// half-width f32 table: one warp squares 4 values per lane (mul.rn), every
// thread owns one element; even/odd partners are rotated with separately
// rounded PTX ops after the BF16 norm boundary.
extern "C" __global__ void ${function}(const dif_bf16* x, const dif_bf16* weight, const dif_f32* cosv, const dif_f32* sinv, dif_bf16* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * 128ULL;
  float sigma2 = 0.0f;
  if (tid < 32U) {
    unsigned long long lane_base = base + (unsigned long long)tid;
    float v0 = dif_load_bf16(x, lane_base), v1 = dif_load_bf16(x, lane_base + 32ULL),
          v2 = dif_load_bf16(x, lane_base + 64ULL), v3 = dif_load_bf16(x, lane_base + 96ULL), s0, s1, s2, s3;
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s0) : "f"(v0));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s1) : "f"(v1));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s2) : "f"(v2));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s3) : "f"(v3));
    sigma2 = ((s0 + s1) + s2) + s3;
    for (unsigned offset = 1U; offset < 32U; offset <<= 1U) sigma2 = sigma2 + __shfl_down_sync(0xffffffffU, sigma2, offset);
  }
  if (tid == 0U) reduction[0] = sigma2 * 0.0078125f;
  __syncthreads();
  float inv = rsqrtf(reduction[0] + ${epsilon}f);
  if (tid < 128U) {
    unsigned pair = tid / 2U, even = pair * 2U;
    float e = dif_round_bf16(dif_load_bf16(x, base + even) * inv);
    e = dif_round_bf16(e * dif_load_bf16(weight, even));
    float o = dif_round_bf16(dif_load_bf16(x, base + even + 1U) * inv);
    o = dif_round_bf16(o * dif_load_bf16(weight, even + 1U));
    unsigned long long token = row / ${heads}ULL,
                       table_token = (token / ${input_sequence}ULL) * ${table_sequence}ULL + ${table_start}ULL + token % ${input_sequence}ULL,
                       table = table_token * ${table_width}ULL + pair;
    float c = dif_load_f32(cosv, table), s = dif_load_f32(sinv, table), first, second, result;
    if (tid & 1U) {
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(e), "f"(s));
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(o), "f"(c));
      asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
    } else {
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(e), "f"(c));
      asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(o), "f"(s));
      asm volatile("sub.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
    }
    dif_store_bf16(y, base + tid, result);
  }
}
