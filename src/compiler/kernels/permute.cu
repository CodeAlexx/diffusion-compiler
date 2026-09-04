// General axis permutation: the per-axis lines are generated in reverse
// output-axis order from the Permutation attributes.
extern "C" __global__ void ${function}(const dif_scalar* input, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long remaining = i, source = 0ULL, coordinate;
${axes}
    dif_store(output, i, dif_load(input, source));
  }
}
