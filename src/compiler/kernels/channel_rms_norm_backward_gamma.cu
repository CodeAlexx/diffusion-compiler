  if (i < ${channels}ULL) {
    float accumulator = 0.0f;
    for (unsigned long long f = 0ULL; f < ${fibers}ULL; ++f) {
      unsigned long long fiber = f / ${inner}ULL * ${channels}ULL * ${inner}ULL + f % ${inner}ULL;
      float squared = 0.0f;
      for (unsigned long long c = 0ULL; c < ${channels}ULL; ++c) {
        float value = dif_load(x, fiber + c * ${inner}ULL);
        squared = fmaf(value, value, squared);
      }
      float denominator = fmaxf(sqrtf(squared), ${epsilon}f);
      unsigned long long index = fiber + i * ${inner}ULL;
      accumulator = fmaf(dif_load(grad_output, index),
                         dif_load(x, index) * ${scale}f / denominator,
                         accumulator);
    }
    dif_store(grad_gamma, i, accumulator);
  }
