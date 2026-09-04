  __syncthreads();
  float stored_scale = scales[row];
  float residual_maximum = 0.0f;
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float value = ${value};
    float residual = fmaf(-(float)q[base + column], stored_scale, value);
    residual_maximum = fmaxf(residual_maximum, fabsf(residual));
  }
  maximums[tid] = residual_maximum;
  __syncthreads();
  for (unsigned active = 128U; active > 0U; active >>= 1U) {
    if (tid < active) maximums[tid] = fmaxf(maximums[tid], maximums[tid + active]);
    __syncthreads();
  }
  float scale2 = fmaxf(maximums[0] / 127.0f, 1.0e-30f);
  float scale2_bf16 = dif_round_bf16(scale2);
  if (tid == 0U) scales2[row] = ${scale2_store};
  __syncthreads();
  float stored_scale2 = scales2[row];
  for (unsigned long long column = tid; column < ${columns}ULL; column += 256ULL) {
    float value = ${value};
    float residual = ${residual};
    float divided = ${divided};
    int encoded = (int)nearbyintf(divided);
    encoded = encoded > 127 ? 127 : (encoded < ${clamp_low} ? ${clamp_low} : encoded);
    q2[base + column] = (signed char)encoded;
  }
