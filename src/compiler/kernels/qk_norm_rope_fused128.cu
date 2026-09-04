// Port of Serenity's accepted MiniMax-H3 fused Q/K RMSNorm + partial-RoPE
// primitive: one lane owns one head value of a 128-wide BF16 head, a
// 128-lane F32 reduction, the BF16 normalization boundary, then half-split
// rotation of the first `rotary` values.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* cosv, const dif_scalar* sinv, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * 128ULL;
  float local = 0.0f;
  for (unsigned col = tid; col < 128U; col += 128U) {
    float value = dif_load(x, base + col);
    local = __fadd_rn(local, __fmul_rn(value, value));
  }
  reduction[tid] = local;
  __syncthreads();
  for (unsigned active = 64U; active > 0U; active >>= 1U) {
    if (tid < active) reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + active]);
    __syncthreads();
  }
  float inv = rsqrtf(__fadd_rn(__fdiv_rn(reduction[0], 128.0f), ${epsilon}f));
  unsigned long long token = row / ${heads}ULL,
                     table_token = (token / ${input_sequence}ULL) * ${table_sequence}ULL + ${table_start}ULL + token % ${input_sequence}ULL;
  if (tid < ${half}U) {
    unsigned lane = tid;
    float value0 = dif_load(x, base + lane);
    float value1 = dif_load(x, base + lane + ${half}ULL);
    float norm0 = dif_round(value0 * inv * dif_load(weight, lane));
    float norm1 = dif_round(value1 * inv * dif_load(weight, lane + ${half}ULL));
    unsigned long long table = table_token * ${table_width}ULL;
    float result0 = norm0 * dif_load(cosv, table + lane) - norm1 * dif_load(sinv, table + lane);
    float result1 = norm1 * dif_load(cosv, table + lane + ${half}ULL) + norm0 * dif_load(sinv, table + lane + ${half}ULL);
    dif_store(y, base + lane, result0);
    dif_store(y, base + lane + ${half}ULL, result1);
  } else if (tid >= ${rotary}U && tid < 128U) {
    float value = dif_load(x, base + tid);
    dif_store(y, base + tid, value * inv * dif_load(weight, tid));
  }
}
