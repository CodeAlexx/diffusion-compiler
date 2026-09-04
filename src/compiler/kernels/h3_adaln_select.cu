// Select the six adaLN modulation rows [shift, scale, gate x2] for each
// token from the projected table [T, 18H] by the token's table index.
extern "C" __global__ void ${function}(const dif_scalar* projected, const int* indices, dif_scalar* o0, dif_scalar* o1, dif_scalar* o2, dif_scalar* o3, dif_scalar* o4, dif_scalar* o5) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${hidden}ULL, col = i % ${hidden}ULL, table = (unsigned long long)indices[row];
    dif_store(o0, i, dif_load(projected, (table * 6ULL + 0ULL) * ${hidden}ULL + col));
    dif_store(o1, i, dif_load(projected, (table * 6ULL + 1ULL) * ${hidden}ULL + col));
    dif_store(o2, i, dif_load(projected, (table * 6ULL + 2ULL) * ${hidden}ULL + col));
    dif_store(o3, i, dif_load(projected, (table * 6ULL + 3ULL) * ${hidden}ULL + col));
    dif_store(o4, i, dif_load(projected, (table * 6ULL + 4ULL) * ${hidden}ULL + col));
    dif_store(o5, i, dif_load(projected, (table * 6ULL + 5ULL) * ${hidden}ULL + col));
  }
}
