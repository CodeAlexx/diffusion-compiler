  // Any width and block size: strided squares, then a shared-memory tree.
  for (unsigned long long col = threadIdx.x; col < ${columns}ULL; col += blockDim.x) {
    float v = dif_load(x, row * ${columns}ULL + col);
    local += v * v;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
