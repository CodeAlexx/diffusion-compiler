// Packed projection weight [3N, K] with per-head (q,k,v) row interleaving
// -> three [N, K] weights.
extern "C" __global__ void ${function}(const dif_scalar* packed, dif_scalar* q, dif_scalar* k, dif_scalar* v) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${hidden}ULL, col = i % ${hidden}ULL, head = row / ${dim}ULL,
                       d = row % ${dim}ULL, base = ((head * 3ULL) * ${dim}ULL + d) * ${hidden}ULL + col;
    dif_store(q, i, dif_load(packed, base));
    dif_store(k, i, dif_load(packed, base + ${dim_hidden}ULL));
    dif_store(v, i, dif_load(packed, base + ${double_dim_hidden}ULL));
  }
}
