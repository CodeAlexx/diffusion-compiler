// ConvRot INT8 row quantization: the row is rotated in shared memory by a
// block-diagonal Hadamard (H256 or H4096, radix-4 butterfly stages; optional
// per-column sign flip keyed by a hash of the column) before the symmetric
// per-row quantization. The placeholder fragments select the butterfly, the scale
// storage rounding, the rounding contract of the codes, and the optional
// residual second code.
extern "C" __global__ void ${function}(${parameters}) {
  extern __shared__ float values[];
  __shared__ float maximums[256];
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned long long base = row * ${columns}ULL;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float value = ${load_input};
    ${signed_rotation}
    values[column] = value;
  }
  __syncthreads();
  for (unsigned stage = 0U; stage < ${rotation_stages}U; ++stage) {
    unsigned stride = 1U << (2U * stage);
    for (unsigned long long tuple = tid; tuple < ${columns}ULL / 4ULL; tuple += 256ULL) {
      unsigned long long group = (tuple / ${rotation_lanes}ULL) * ${rotation_group}ULL;
      unsigned lane = (unsigned)(tuple % ${rotation_lanes}ULL);
      unsigned offset = (lane % stride) + (lane / stride) * (4U * stride);
      unsigned long long i = group + offset;
      float x0 = values[i], x1 = values[i + stride], x2 = values[i + 2U * stride],
            x3 = values[i + 3U * stride];
      ${butterfly}
    }
    __syncthreads();
  }
  float maximum = 0.0f;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL)
    maximum = fmaxf(maximum, fabsf(values[column]));
  maximums[tid] = maximum;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) maximums[tid] = fmaxf(maximums[tid], maximums[tid + active]);
    __syncthreads();
  }
  float scale = fmaxf(maximums[0] * ${clip_ratio} / 127.0f, 1.0e-30f);
  float scale_bf16 = dif_round_bf16(scale);
  if (tid == 0U) scales[row] = ${scale_store};
  __syncthreads();
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    ${encode}
    q[base + column] = (signed char)encoded;
  }
  ${residual2}
}
