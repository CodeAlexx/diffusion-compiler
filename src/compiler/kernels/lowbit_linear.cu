// Direct packed-INT5 Linear (Implementation 3) fused with its dequantizer:
// one block per output row, 256 threads stride K unpacking 5-bit codes, one
// F32 accumulator per flattened input row (M <= 32; the per-row lines are
// generated), warp shuffles, then a cross-warp total per row.
extern "C" __global__ void ${function}(${parameters}) {
  extern __shared__ float partials[];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x, lane = tid & 31U, warp = tid >> 5U;
  if (row >= ${n}ULL) return;
  unsigned long long source_row = ${source_row};
${accumulators}
  for (unsigned long long col = tid; col < ${k}ULL; col += 256ULL) {
    unsigned long long bit = col * 5ULL, bi = source_row * ${row_bytes}ULL + bit / 8ULL;
    unsigned shift = (unsigned)(bit & 7ULL);
    unsigned word = packed[bi];
    if (shift + 5U > 8U) word |= ((unsigned)packed[bi + 1ULL]) << 8U;
    unsigned encoded = (word >> shift) & 31U;
    int q = encoded < 16U ? (int)encoded : (int)encoded - 32;
    float w = (float)q * dif_load(scales, source_row * ${groups}ULL + col / ${group}ULL);
    ${column_scale}
${fma}
  }
  for (unsigned offset = 16U; offset > 0U; offset >>= 1U) {
${shuffle}
  }
  if (lane == 0U) {
${partials}
  }
  __syncthreads();
  if (warp == 0U && lane < ${m}ULL) {
    float total = 0.0f;
    for (unsigned source_warp = 0U; source_warp < 8U; ++source_warp) total += partials[source_warp * ${m}ULL + lane];
    ${bias}
    dif_store(y, (unsigned long long)lane * ${n}ULL + row, total);
  }
}
