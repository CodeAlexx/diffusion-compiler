// grad_input[m,k] = sum_n grad_output[m,n] * weight[n,k] * scale[n].
//
// The weight is read as INT8 and scaled inside the accumulation, so no
// dequantized copy of it ever exists. Each block computes one 32x32 tile of
// the result and stages both operands through shared memory, so every value
// loaded is used thirty-two times instead of once -- the naive form ran at a
// tenth of a percent of this card's peak, which at a 12-billion-parameter
// base is the difference between a step and a coffee break.
//
// The summation order over n is unchanged from that naive form: base ascends
// and step ascends within it, so n is still visited in increasing order and
// the result is bit-identical to the kernel this replaced.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const signed char* weight, const dif_f32* scales, dif_scalar* grad_input) {
  // The trailing column of padding keeps the strided reads below off a single
  // shared-memory bank.
  __shared__ float tile_gradient[32][33];
  __shared__ float tile_weight[32][33];
  const unsigned tiles_per_row = (${inner}u + 31u) / 32u;
  const unsigned tile_row = blockIdx.x / tiles_per_row;
  const unsigned tile_column = blockIdx.x % tiles_per_row;
  const unsigned local_row = threadIdx.x / 32u;
  const unsigned local_column = threadIdx.x % 32u;
  const unsigned row = tile_row * 32u + local_row;
  const unsigned column = tile_column * 32u + local_column;
  float value = 0.0f;
  for (unsigned base = 0u; base < ${outputs}u; base += 32u) {
    const unsigned gradient_output = base + local_column;
    tile_gradient[local_row][local_column] =
        (row < ${rows}u && gradient_output < ${outputs}u)
            ? dif_load(grad_output, (unsigned long long)row * ${outputs}ULL + gradient_output)
            : 0.0f;
    const unsigned weight_output = base + local_row;
    tile_weight[local_row][local_column] =
        (weight_output < ${outputs}u && column < ${inner}u)
            ? __fmul_rn((float)weight[(unsigned long long)weight_output * ${inner}ULL + column], dif_load_f32(scales, weight_output))
            : 0.0f;
    __syncthreads();
    for (unsigned step = 0u; step < 32u; ++step)
      value = fmaf(tile_gradient[local_row][step], tile_weight[step][local_column], value);
    __syncthreads();
  }
  if (row < ${rows}u && column < ${inner}u)
    dif_store(grad_input, (unsigned long long)row * ${inner}ULL + column, value);
}
