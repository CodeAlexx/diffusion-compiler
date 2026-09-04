// Rows selected by an i32 index vector; out-of-range indices produce NaN
// so a bad map is visible, never silent.
extern "C" __global__ void ${function}(const dif_scalar* x, const int* indices, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${row_width}ULL, col = i % ${row_width}ULL;
    int source = indices[row];
    if (source >= 0 && source < ${input_rows})
      dif_store(y, i, dif_load(x, (unsigned long long)source * ${row_width}ULL + col));
    else
      dif_store(y, i, __int_as_float(0x7fffffff));
  }
}
