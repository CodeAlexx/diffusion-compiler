// RMS normalization, one block per row: sum of squares reduced into
// reduction[0] by the reduction strategy the emitter selected for the row
// width and block size, then y = x * rsqrt(mean + eps) * (weight + offset).
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* weight, dif_scalar* y) {
  extern __shared__ float reduction[];
  unsigned long long row = blockIdx.x;
  float local = 0.0f;
  if (row >= ${rows}ULL) return;
${reduction}
${inverse}
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) {
    unsigned long long i = row * ${columns}ULL + col;
    dif_store(y, i, dif_load(x, i) * inv * (dif_load(weight, col) + ${weight_offset}f));
  }
}
