extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* branch, const dif_scalar* gate, dif_scalar* grad_branch, dif_scalar* grad_gate) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float upstream = dif_load(grad_output, i);
    dif_store(grad_branch, i, upstream * dif_load(gate, i));
    dif_store(grad_gate, i, upstream * dif_load(branch, i));
  }
}
