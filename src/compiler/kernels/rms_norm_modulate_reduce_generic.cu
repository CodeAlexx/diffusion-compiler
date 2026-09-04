  for (unsigned long long col = threadIdx.x; col < ${cols}ULL; col += blockDim.x) {
    float v = dif_load(x, (unsigned long long)row * ${cols}ULL + col);
    local += v * v;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
