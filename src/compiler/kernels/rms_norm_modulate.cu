// Generic rms_norm + (1 + scale) * . + shift with explicit storage-dtype
// rounding after the norm, the modulation product and the shift; the row
// reduction fragment is chosen by the emitter for the width and block size.
extern "C" __global__ void ${function}(${parameters}) {
  extern __shared__ float reduction[];
  unsigned row = blockIdx.x;
  float local = 0.0f;
  if (row >= ${rows}ULL) return;
${reduction}
  float inv = rsqrtf(reduction[0] / ${cols}.0f + ${epsilon}f);
  for (unsigned long long col = threadIdx.x; col < ${cols}ULL; col += blockDim.x) {
    unsigned long long i = (unsigned long long)row * ${cols}ULL + col;
    float value = dif_load(x, i) * inv${weight_factor};
    value = dif_round(value);
    float modulation = dif_round(1.0f + dif_load(scale, i));
    value = dif_round(value * modulation);
    dif_store(y, i, value + dif_load(shift, i));
  }
}
