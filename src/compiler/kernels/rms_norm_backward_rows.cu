// rms_norm backward for a FROZEN gain: one block per row.
//
// The row needs two reductions -- the sum of squares and the dot product of
// the weighted gradient with the input -- and they are the same two numbers
// for every element in the row. The reference form has each of the row's
// threads recompute both, which at 6144 columns is 6144 redundant passes.
// The row's values fit in cache, which is the only reason that was
// survivable rather than catastrophic; it was still the single most
// expensive operation in a Krea training step.
//
// The forward has always done it this way. This is the backward catching up.
//
// A trainable gain needs a per-column reduction across rows as well, which
// is a different shape of problem; that case keeps the reference kernel.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* weight, dif_scalar* grad_input) {
  extern __shared__ float reduction[];
  const unsigned long long row = blockIdx.x;
  const unsigned long long base = row * ${columns}ULL;
  float sum_squares = 0.0f, dot = 0.0f;
  for (unsigned long long k = threadIdx.x; k < ${columns}ULL; k += blockDim.x) {
    const float value = dif_load(x, base + k);
    sum_squares = fmaf(value, value, sum_squares);
    dot = fmaf(dif_load(grad_output, base + k) * dif_load(weight, k), value, dot);
  }
  reduction[threadIdx.x] = sum_squares;
  reduction[blockDim.x + threadIdx.x] = dot;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2u; stride > 0u; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      reduction[blockDim.x + threadIdx.x] += reduction[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse = rsqrtf(reduction[0] / ${columns}.0f + ${epsilon}f);
  const float row_dot = reduction[blockDim.x];
  for (unsigned long long k = threadIdx.x; k < ${columns}ULL; k += blockDim.x) {
    const float value = dif_load(x, base + k);
    dif_store(grad_input, base + k,
              dif_load(grad_output, base + k) * dif_load(weight, k) * inverse -
                  value * inverse * inverse * inverse * row_dot / ${columns}.0f);
  }
}
