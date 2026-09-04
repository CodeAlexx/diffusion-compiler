    if (i < ${dim}ULL) {
      float acc = 0.0f;
      for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
        unsigned long long rrb = r * ${dim}ULL, rtb = (r / ${heads}ULL) * ${table_width}ULL;
        float rss = 0.0f;
        for (unsigned long long k = 0ULL; k < ${dim}ULL; ++k) {
          float rv = dif_load(x, rrb + k);
          rss = fmaf(rv, rv, rss);
        }
        float rinv = rsqrtf(rss / ${dim}.0f + ${epsilon}f);
        float rotated_gradient = ${rotated_i};
        acc = fmaf(rotated_gradient * dif_load(x, rrb + i), rinv, acc);
      }
      dif_store(grad_weight, i, acc);
    }
