// Gradient of the gated residual add. The gate may govern several rows at
// once (a DiT block gates every token with one per-sample value), so the
// branch gradient is elementwise against the broadcast gate while the gate
// gradient reduces across the rows that share it. When the gate is not
// broadcast the two guards cover the same elements and each expression
// collapses to the form this kernel always had.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* branch, const dif_scalar* gate, dif_scalar* grad_branch, dif_scalar* grad_gate) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float upstream = dif_load(grad_output, i);
    dif_store(grad_branch, i, upstream * dif_load(gate, ${gate_index}));
  }
  if (i < ${gate_count}ULL) {
    float accumulator = 0.0f;
    ${gate_reduction}
    dif_store(grad_gate, i, accumulator);
  }
}
