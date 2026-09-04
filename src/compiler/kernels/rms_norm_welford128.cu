// Vectorized rms_norm for 128-wide bf16 rows with 128-thread blocks
// (Implementation 2): one warp squares 4 values per lane with mul.rn,
// shuffles the sum, and every thread scales its column.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned tid = threadIdx.x;
  float sigma2 = 0.0f;
  if (tid < 32U) {
    unsigned long long base = row * 128ULL + (unsigned long long)tid;
    float v0 = dif_load(x, base), v1 = dif_load(x, base + 32ULL),
          v2 = dif_load(x, base + 64ULL), v3 = dif_load(x, base + 96ULL), s0, s1, s2, s3;
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s0) : "f"(v0));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s1) : "f"(v1));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s2) : "f"(v2));
    asm volatile("mul.rn.f32 %0,%1,%1;" : "=f"(s3) : "f"(v3));
    sigma2 = ((s0 + s1) + s2) + s3;
    for (unsigned offset = 1U; offset < 32U; offset <<= 1U)
      sigma2 = sigma2 + __shfl_down_sync(0xffffffffU, sigma2, offset);
  }
  if (tid == 0U) reduction[0] = sigma2 * 0.0078125f;
  __syncthreads();
  float inverse = rsqrtf(reduction[0] + ${epsilon}f);
  if (tid < 128U) {
    unsigned long long index = row * 128ULL + tid;
    dif_store(y, index, dif_load(weight, tid) * (inverse * dif_load(x, index)));
  }
}
