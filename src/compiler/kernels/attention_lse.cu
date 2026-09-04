// Log-sum-exp of the scaled scores per (query, head), the value the
// decomposed attention backward reuses. Serial F32 reference form.
extern "C" __global__ void ${function}(const ${scalar}* q, const ${scalar}* k, dif_f32* lse) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long qs = i / ${heads}ULL, h = i % ${heads}ULL, qb = (qs * ${heads}ULL + h) * ${dim}ULL,
                       kend = ${kend};
    float maximum = -3.402823466e+38f;
    for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
      float score = 0.0f;
      unsigned long long kb = (ks * ${kv_heads}ULL + ${kv_head}) * ${dim}ULL;
      for (unsigned long long d = 0ULL; d < ${dim}ULL; ++d) score = fmaf(${load}(q, qb + d), ${load}(k, kb + d), score);
      score *= ${scale}f;
      maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0f;
    for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
      float score = 0.0f;
      unsigned long long kb = (ks * ${kv_heads}ULL + ${kv_head}) * ${dim}ULL;
      for (unsigned long long d = 0ULL; d < ${dim}ULL; ++d) score = fmaf(${load}(q, qb + d), ${load}(k, kb + d), score);
      denominator += expf(score * ${scale}f - maximum);
    }
    dif_store_f32(lse, i, maximum + logf(denominator));
  }
}
