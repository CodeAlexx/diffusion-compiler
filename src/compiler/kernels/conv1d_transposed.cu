    unsigned long long ocg = oc % ${out_per_group}ULL;
    long long ofull = (long long)o + ${trim_left}LL;
    for (unsigned long long ic = 0; ic < ${in_per_group}ULL; ++ic) {
      const dif_scalar* xrow = x + ((b * ${in_channels}ULL) + (group * ${in_per_group}ULL + ic)) * ${length}ULL;
      const dif_scalar* wrow = w + (((group * ${in_per_group}ULL + ic) * ${out_per_group}ULL) + ocg) * ${kernel}ULL;
      long long imin = (ofull - ${kernel_minus_one}LL + ${stride}LL - 1LL) / ${stride}LL;
      if (imin < 0LL) imin = 0LL;
      long long imax = ofull / ${stride}LL;
      if (imax > ${padded_minus_one}LL) imax = ${padded_minus_one}LL;
      for (long long pi = imin; pi <= imax; ++pi) {
        long long k = ofull - pi * ${stride}LL;
        if (k < 0LL || k >= ${kernel}LL) continue;
        acc += ${sample} * dif_load(wrow, (unsigned long long)k);
      }
    }
