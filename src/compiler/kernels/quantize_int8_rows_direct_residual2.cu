  __syncthreads();
  float residual_maximum = 0.0f;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float residual = fmaf(-(float)q[base + column], scale, ${load_input});
    residual_maximum = fmaxf(residual_maximum, fabsf(residual));
  }
  maximums[tid] = residual_maximum;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) maximums[tid] = fmaxf(maximums[tid], maximums[tid + active]);
    __syncthreads();
  }
  float scale2 = fmaxf(maximums[0] / 127.0f, 1.0e-30f);
  if (tid == 0U) scales2[row] = scale2;
  __syncthreads();
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float residual = fmaf(-(float)q[base + column], scale, ${load_input});
    int value = (int)rintf(residual / scale2);
    value = value > 127 ? 127 : (value < -127 ? -127 : value);
    q2[base + column] = (signed char)value;
  }
