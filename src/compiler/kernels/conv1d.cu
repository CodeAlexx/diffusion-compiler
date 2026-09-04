// Direct 1-D convolution (or its transpose), one thread per output element,
// with padding handled by the generated sampler expression (zero or
// replicate outside the input) and optional bias.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* w, ${bias_parameter}dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long o = i % ${out_length}ULL;
    unsigned long long oc = (i / ${out_length}ULL) % ${out_channels}ULL;
    unsigned long long b = i / ${out_stride}ULL;
    unsigned long long group = oc / ${out_per_group}ULL;
    float acc = 0.0f;
${accumulate}
    ${bias}
    dif_store(y, i, acc);
  }
}
