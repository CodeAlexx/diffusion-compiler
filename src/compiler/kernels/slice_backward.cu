// Gradient of an axis slice: the upstream gradient lands in the window the
// slice read, every other position is zero. One thread per input position.
extern "C" __global__ void ${function}(const dif_scalar* g, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long lane = i % ${inner}ULL;
    unsigned long long position = (i / ${inner}ULL) % ${extent}ULL;
    unsigned long long outer = i / (${inner}ULL * ${extent}ULL);
    if (position >= ${start}ULL && position < ${start}ULL + ${window}ULL)
      dif_store(y, i, dif_load(g, (outer * ${window}ULL + (position - ${start}ULL)) * ${inner}ULL + lane));
    else
      dif_store(y, i, 0.0f);
  }
}
