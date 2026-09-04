// Gradient of the chunked row gather. The forward reads one source row per
// selected row and splits its columns into chunks, so the gradient sums every
// selection that named a source row back into it -- written as a gather over
// the SOURCE, one thread per source element scanning the index vector, rather
// than one thread per selection racing with atomicAdd. A duplicated index
// would otherwise sum in scheduling order.
extern "C" __global__ void ${function}(const int* indices${chunk_parameters}, ${scalar}* grad_values) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${source_width}ULL, col = i % ${source_width}ULL;
    unsigned long long chunk = col / ${width}ULL, offset = col % ${width}ULL;
    const ${scalar}* source = ${chunk_pointer};
    float accumulator = 0.0f;
    for (unsigned long long s = 0ULL; s < ${rows}ULL; ++s)
      if ((unsigned long long)indices[s] == row)
        accumulator += ${load}(source, s * ${width}ULL + offset);
    ${store}(grad_values, i, accumulator);
  }
}
