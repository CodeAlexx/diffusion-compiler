// y = residual + round(gate * branch); gate rows broadcast over
// rows_per_gate consecutive input rows.
extern "C" __global__ void ${function}(const dif_scalar* residual, const dif_scalar* branch, const dif_scalar* gate, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${width}ULL,
                       gate_index = (row / ${rows_per_gate}ULL) * ${width}ULL + i % ${width}ULL;
    dif_store(y, i, dif_load(residual, i) + dif_round(dif_load(gate, gate_index) * dif_load(branch, i)));
  }
}
