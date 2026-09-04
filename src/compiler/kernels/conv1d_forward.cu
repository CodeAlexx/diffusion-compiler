    for (unsigned long long ic = 0; ic < ${in_per_group}ULL; ++ic) {
      const dif_scalar* xrow = x + ((b * ${in_channels}ULL) + (group * ${in_per_group}ULL + ic)) * ${length}ULL;
      const dif_scalar* wrow = w + ((oc * ${in_per_group}ULL) + ic) * ${kernel}ULL;
      long long start = (long long)(o * ${stride}ULL);
      for (unsigned long long k = 0; k < ${kernel}ULL; ++k) {
        long long p = start + (long long)(k * ${dilation}ULL);
        acc += ${sample} * dif_load(wrow, k);
      }
    }
