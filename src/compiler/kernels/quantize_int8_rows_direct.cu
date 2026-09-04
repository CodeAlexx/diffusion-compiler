// Dynamic symmetric per-row INT8 quantization, one block of 256 threads per
// row: |max| reduction, scale = max * clip / 127 (guarded), round-to-nearest,
// clamp to [-127, 127]. Optional second code for the residual of the first.
extern "C" __global__ void ${function}(${parameters}) {
  __shared__ float maximums[256];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * ${columns}ULL;
  float maximum = 0.0f;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float value = fabsf(${load_input});
    maximum = fmaxf(maximum, value);
  }
  maximums[tid] = maximum;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) maximums[tid] = fmaxf(maximums[tid], maximums[tid + active]);
    __syncthreads();
  }
  float scale = fmaxf(maximums[0] * ${clip_ratio} / 127.0f, 1.0e-30f);
  if (tid == 0U) scales[row] = scale;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    int value = (int)rintf(${load_input} / scale);
    value = value > 127 ? 127 : (value < -127 ? -127 : value);
    q[base + column] = (signed char)value;
  }
  ${residual2}
}
