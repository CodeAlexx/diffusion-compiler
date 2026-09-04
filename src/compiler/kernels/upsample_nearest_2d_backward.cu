// Gradient of nearest-neighbour 2-D upsampling. One thread per INPUT
// position, summing the ScaleH x ScaleW output block that read it, so the
// write is exclusive and no atomics are needed.
extern "C" __global__ void ${function}(const dif_scalar* g, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long x = i % ${width}ULL;
    unsigned long long h = (i / ${width}ULL) % ${height}ULL;
    unsigned long long plane = i / (${width}ULL * ${height}ULL);
    float total = 0.0f;
    for (unsigned long long dy = 0ULL; dy < ${scale_h}ULL; ++dy)
      for (unsigned long long dx = 0ULL; dx < ${scale_w}ULL; ++dx)
        total += dif_load(g, (plane * ${out_height}ULL + h * ${scale_h}ULL + dy) * ${out_width}ULL + x * ${scale_w}ULL + dx);
    dif_store(y, i, total);
  }
}
