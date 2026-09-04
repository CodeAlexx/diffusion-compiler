// y = base with rows replaced from `updates` where map[row] >= 0; -1 keeps
// the base row; any other out-of-range map value produces NaN.
extern "C" __global__ void ${function}(const dif_scalar* base, const dif_scalar* updates, const int* map, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${row_width}ULL, col = i % ${row_width}ULL;
    int source = map[row];
    if (source == -1)
      dif_store(y, i, dif_load(base, i));
    else if (source >= 0 && source < ${update_rows})
      dif_store(y, i, dif_load(updates, (unsigned long long)source * ${row_width}ULL + col));
    else
      dif_store(y, i, __int_as_float(0x7fffffff));
  }
}
