// Backward of rms_norm followed by shared-vector modulation:
//   scale = vector[v(row)] + delta[0]   shift = vector[v(row)] + delta[1]
//   out   = (1 + scale) * (x * inv * (weight + offset)) + shift
// The row statistic is recomputed from the original input in F32, as the
// other norm gradients here do.
//
// Four reductions over four different axes live in one kernel because they
// share that statistic: the input gradient is elementwise, the weight
// gradient reduces down every row, the vector gradient reduces across the
// rows sharing one vector, and the delta gradient reduces across everything.
// Each output element is owned by exactly one thread, so all four sum
// deterministically without atomics.
//
// The vector feeds BOTH the scale and the shift, so its gradient carries the
// normalized value plus one -- dropping either term is the easy mistake here.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, const dif_scalar* vec, const dif_scalar* delta, dif_scalar* grad_input, dif_scalar* grad_weight, dif_scalar* grad_vector, dif_scalar* grad_delta) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, col = i % ${columns}ULL, base = row * ${columns}ULL;
    unsigned long long vbase = row / ${rows_per_vector}ULL * ${columns}ULL;
    float ss = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float value = dif_load(x, base + k);
      ss = fmaf(value, value, ss);
    }
    float inv = rsqrtf(ss / ${columns}.0f + ${epsilon}f);
    float dot = 0.0f;
    for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
      float modulated = dif_load(grad_output, base + k) *
                        (1.0f + dif_load(vec, vbase + k) + dif_load(delta, k));
      dot = fmaf(modulated * (dif_load(weight, k) + ${offset}f), dif_load(x, base + k), dot);
    }
    float own = dif_load(grad_output, i) * (1.0f + dif_load(vec, vbase + col) + dif_load(delta, col));
    float weighted = own * (dif_load(weight, col) + ${offset}f);
    float value = dif_load(x, i);
    dif_store(grad_input, i, weighted * inv - value * inv * inv * inv * dot / ${columns}.0f);
  }
  if (i < ${columns}ULL) {
    float accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
      unsigned long long rb = r * ${columns}ULL;
      unsigned long long vb = r / ${rows_per_vector}ULL * ${columns}ULL;
      float rss = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
        float rv = dif_load(x, rb + k);
        rss = fmaf(rv, rv, rss);
      }
      float rinv = rsqrtf(rss / ${columns}.0f + ${epsilon}f);
      float modulated = dif_load(grad_output, rb + i) *
                        (1.0f + dif_load(vec, vb + i) + dif_load(delta, i));
      accumulator = fmaf(modulated, dif_load(x, rb + i) * rinv, accumulator);
    }
    dif_store(grad_weight, i, accumulator);
  }
  if (i < ${vector_count}ULL) {
    unsigned long long v = i / ${columns}ULL, col = i % ${columns}ULL;
    float accumulator = 0.0f;
    for (unsigned long long r = v * ${rows_per_vector}ULL; r < (v + 1ULL) * ${rows_per_vector}ULL; ++r) {
      unsigned long long rb = r * ${columns}ULL;
      float rss = 0.0f;
      for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
        float rv = dif_load(x, rb + k);
        rss = fmaf(rv, rv, rss);
      }
      float rinv = rsqrtf(rss / ${columns}.0f + ${epsilon}f);
      float normalized = dif_load(x, rb + col) * rinv * (dif_load(weight, col) + ${offset}f);
      accumulator = fmaf(dif_load(grad_output, rb + col), normalized + 1.0f, accumulator);
    }
    dif_store(grad_vector, i, accumulator);
  }
  if (i < ${delta_count}ULL) {
    unsigned long long chunk = i / ${columns}ULL, col = i % ${columns}ULL;
    float accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
      unsigned long long rb = r * ${columns}ULL;
      float upstream = dif_load(grad_output, rb + col);
      if (chunk == 1ULL) {
        accumulator += upstream;
      } else {
        float rss = 0.0f;
        for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
          float rv = dif_load(x, rb + k);
          rss = fmaf(rv, rv, rss);
        }
        float rinv = rsqrtf(rss / ${columns}.0f + ${epsilon}f);
        accumulator = fmaf(upstream, dif_load(x, rb + col) * rinv * (dif_load(weight, col) + ${offset}f), accumulator);
      }
    }
    dif_store(grad_delta, i, accumulator);
  }
}
