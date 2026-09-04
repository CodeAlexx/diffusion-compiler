    if (i < ${columns}ULL) {
      float acc = 0.0f;
      for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r) {
        unsigned long long rb = r * ${columns}ULL;
        float rss = 0.0f;
        for (unsigned long long k = 0ULL; k < ${columns}ULL; ++k) {
          float rv = dif_load(x, rb + k);
          rss = fmaf(rv, rv, rss);
        }
        float rinv = rsqrtf(rss / ${columns}.0f + ${epsilon}f);
        acc = fmaf(dif_load(grad_output, rb + i) * dif_load(x, rb + i), rinv, acc);
      }
      dif_store(grad_weight, i, acc);
    }
