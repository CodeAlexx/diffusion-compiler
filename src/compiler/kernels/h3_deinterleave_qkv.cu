// Packed [T, 3*H*D] with per-head (q,k,v) interleaving -> q, k, v [T, H, D].
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* q, dif_scalar* k, dif_scalar* v) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${head_width}ULL, within = i % ${head_width}ULL, head = within / ${dim}ULL,
                       d = within % ${dim}ULL, base = row * ${packed}ULL + head * ${triple_dim}ULL + d;
    dif_store(q, i, dif_load(x, base));
    dif_store(k, i, dif_load(x, base + ${dim}ULL));
    dif_store(v, i, dif_load(x, base + ${double_dim}ULL));
  }
}
