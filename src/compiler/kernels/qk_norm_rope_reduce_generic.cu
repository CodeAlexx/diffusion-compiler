  for (unsigned long long d = threadIdx.x; d < ${dim}ULL; d += blockDim.x) {
    float v = dif_load(x, base + d);
    local += v * v;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
