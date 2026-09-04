// Scatter-add of the gathered rows' gradients back into the table. One thread
// owns one TABLE element and scans the index vector for the rows that chose
// it, rather than one thread per gathered row racing with atomicAdd: a
// duplicated index would then sum in whatever order the scheduler happened to
// pick, and a training run that cannot reproduce its own gradients is not
// worth much. The index vector is small and stays in cache, so the scan costs
// far less than its arithmetic suggests.
extern "C" __global__ void ${function}(const ${scalar}* g, const int* indices, ${scalar}* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${row_width}ULL, col = i % ${row_width}ULL;
    float accumulator = 0.0f;
    for (unsigned long long m = 0ULL; m < ${gathered_rows}ULL; ++m)
      if ((unsigned long long)indices[m] == row)
        accumulator += ${load}(g, m * ${row_width}ULL + col);
    ${store}(y, i, accumulator);
  }
}
