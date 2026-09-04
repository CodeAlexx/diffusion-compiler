// Decomposed attention backward (reference form). One thread index covers
// both geometries independently: threads under the query element count
// compute dq at their own flat index, and threads under the key/value
// element count own dk/dv at theirs, accumulating in F32 across every query
// and every query head of their group (grouped-KV gradient accumulation).
// The two guards are separate because cross-attention keys carry their own
// row count, so the key geometry is not the query geometry.  P is recomputed
// from Q,K and the saved F32 logsumexp; delta = rowsum(dO*O) uses the
// forward output. O(S) score recomputations per thread: acceptable at gate
// scale, cuDNN SDPA backward stays future work.
extern "C" __global__ void ${function}(const ${scalar}* grad_output, const ${scalar}* q, const ${scalar}* k, const ${scalar}* v, const ${scalar}* forward_output, const dif_f32* lse, ${scalar}* grad_q, ${scalar}* grad_k, ${scalar}* grad_v) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${dim}ULL, d = i % ${dim}ULL, s = ${s_expr}, h = row % ${heads}ULL, qb = row * ${dim}ULL;
    float dq = 0.0f;
    {
      unsigned long long kh = h / ${group}ULL, kend = ${kend};
      float row_lse = dif_load_f32(lse, row);
      float delta = 0.0f;
      for (unsigned long long e = 0ULL; e < ${dim}ULL; ++e) delta = fmaf(${load}(grad_output, qb + e), ${load}(forward_output, qb + e), delta);
      for (unsigned long long ks = 0ULL; ks < kend; ++ks) {
        unsigned long long kb = ${dq_key_base};
        float score = 0.0f, projected = 0.0f;
        for (unsigned long long e = 0ULL; e < ${dim}ULL; ++e) {
          score = fmaf(${load}(q, qb + e), ${load}(k, kb + e), score);
          projected = fmaf(${load}(grad_output, qb + e), ${load}(v, kb + e), projected);
        }
        float probability = expf(score * ${scale}f - row_lse);
        dq = fmaf(probability * (projected - delta) * ${scale}f, ${load}(k, kb + d), dq);
      }
    }
    ${store}(grad_q, i, dq);
  }
  if (i < ${kv_count}ULL) {
    unsigned long long krow = i / ${dim}ULL, d = i % ${dim}ULL, ks = ${ks_expr}, h = krow % ${kv_heads}ULL, kvb = krow * ${dim}ULL;
    float dk = 0.0f, dv = 0.0f;
    for (unsigned long long g = 0ULL; g < ${group}ULL; ++g) {
      unsigned long long qh = h * ${group}ULL + g;
      for (unsigned long long qs = ${qs_start}; qs < ${sequence}ULL; ++qs) {
        unsigned long long qb = ${peer_query_base};
        float row_lse = dif_load_f32(lse, ${peer_lse_index});
        float score = 0.0f, projected = 0.0f, delta = 0.0f;
        for (unsigned long long e = 0ULL; e < ${dim}ULL; ++e) {
          score = fmaf(${load}(q, qb + e), ${load}(k, kvb + e), score);
          projected = fmaf(${load}(grad_output, qb + e), ${load}(v, kvb + e), projected);
          delta = fmaf(${load}(grad_output, qb + e), ${load}(forward_output, qb + e), delta);
        }
        float probability = expf(score * ${scale}f - row_lse);
        dk = fmaf(probability * (projected - delta) * ${scale}f, ${load}(q, qb + d), dk);
        dv = fmaf(probability, ${load}(grad_output, qb + d), dv);
      }
    }
    ${store}(grad_k, i, dk);
    ${store}(grad_v, i, dv);
  }
}
