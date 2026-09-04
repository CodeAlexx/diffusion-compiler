// Gradient of the mapped row update. A destination row took its value either
// from the base (map < 0) or from an update row, so the base gradient passes
// through exactly where the map declined to substitute, and the update
// gradient sums every destination that named an update row. Two reductions
// over different axes, each output element owned by one thread.
extern "C" __global__ void ${function}(const ${scalar}* grad_output, const int* map, ${scalar}* grad_base${update_parameter}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${width}ULL;
    ${store}(grad_base, i, map[row] < 0 ? ${load}(grad_output, i) : 0.0f);
  }
  ${update_gradient}
}
