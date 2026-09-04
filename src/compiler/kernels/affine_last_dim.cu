// y = round(x * scale[col]) (+ round(bias[col])) with the storage-dtype
// rounding after each step, matching the creator's eager boundaries.
extern "C" __global__ void ${function}(${parameters}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long col = i % ${width}ULL;
    float value = dif_round(dif_load(x, i) * dif_load(scale, col));
    ${bias}
    dif_store(y, i, value);
  }
}
