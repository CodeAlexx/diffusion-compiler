// y = value * round(silu(gate)); value and gate are the two halves of a
// packed window [start, start + 2*width) of the input's last dimension.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${width}ULL, col = i % ${width}ULL;
    float value = dif_load(x, row * ${input_width}ULL + ${value_offset}ULL + col);
    float gate = dif_load(x, row * ${input_width}ULL + ${gate_offset}ULL + col);
    float activated = dif_round(dif_silu(gate));
    dif_store(y, i, value * activated);
  }
}
