  if (i < ${update_count}ULL) {
    unsigned long long row = i / ${width}ULL, column = i % ${width}ULL;
    float accumulator = 0.0f;
    for (unsigned long long r = 0ULL; r < ${rows}ULL; ++r)
      if (map[r] >= 0 && (unsigned long long)map[r] == row)
        accumulator += ${load}(grad_output, r * ${width}ULL + column);
    ${store}(grad_updates, i, accumulator);
  }
