// One kernel for a single-consumer pointwise region (ops stamped
// Implementation=2): the staged expressions are generated per region, the
// anchor's expression is stored once through the typed store.
extern "C" __global__ void ${function}(${parameters}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    ${body}
    ${store}(y, i, ${terminal});
  }
}
