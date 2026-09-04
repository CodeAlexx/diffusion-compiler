// Storage-dtype conversion through the typed load/store helpers.
extern "C" __global__ void ${function}(const ${input_type}* x, ${output_type}* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) ${store}(y, i, ${load}(x, i));
}
