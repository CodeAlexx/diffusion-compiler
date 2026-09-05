// grad_input[m,k] = sum_n grad_output[m,n] * weight[n,k] * scale[n].
//
// The weight is read as INT8 and scaled inside the accumulation, so no
// dequantized copy of it ever exists -- that copy is the whole cost the
// resident format removes, and paying it once per linear per step would
// hand it straight back.
//
// Register tiling: one block owns a 64x64 tile of the result and each of its
// 256 threads owns a 4x4 patch of that tile. Sixteen fused multiply-adds per
// eight shared-memory reads, where the one-output-per-thread form managed
// one per two. That ratio is the whole difference; a thread that computes a
// single element spends all its time reading operands other threads are
// reading too.
//
// The summation order over n is unchanged: chunks ascend and steps ascend
// within them, so n is still visited in increasing order and the result is
// bit-identical to the simpler form this replaced. The CPU reference fuses
// its multiply and add for the same reason.
#define DIF_TILE 64u
#define DIF_CHUNK 32u
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const signed char* weight, const dif_f32* scales, dif_scalar* grad_input) {
  // Padded by one column so the strided reads below miss the bank conflict.
  __shared__ float tile_gradient[DIF_CHUNK][DIF_TILE + 1u];
  __shared__ float tile_weight[DIF_CHUNK][DIF_TILE + 1u];
  const unsigned tiles_per_row = (${inner}u + DIF_TILE - 1u) / DIF_TILE;
  const unsigned tile_row = (blockIdx.x / tiles_per_row) * DIF_TILE;
  const unsigned tile_column = (blockIdx.x % tiles_per_row) * DIF_TILE;
  // 16x16 threads, each owning a 4x4 patch.
  const unsigned lane_row = (threadIdx.x / 16u) * 4u;
  const unsigned lane_column = (threadIdx.x % 16u) * 4u;
  float value[4][4] = {{0.0f}};
  for (unsigned base = 0u; base < ${outputs}u; base += DIF_CHUNK) {
    // Stage the gradient chunk transposed, so the inner loop reads one row
    // of it per step with unit stride.
    for (unsigned index = threadIdx.x; index < DIF_CHUNK * DIF_TILE; index += 256u) {
      const unsigned local_row = index / DIF_TILE, local_column = index % DIF_TILE;
      const unsigned row = tile_row + local_column, output = base + local_row;
      tile_gradient[local_row][local_column] =
          (row < ${rows}u && output < ${outputs}u)
              ? dif_load(grad_output, (unsigned long long)row * ${outputs}ULL + output)
              : 0.0f;
      const unsigned column = tile_column + local_column;
      tile_weight[local_row][local_column] =
          (output < ${outputs}u && column < ${inner}u)
              ? __fmul_rn((float)weight[(unsigned long long)output * ${inner}ULL + column], dif_load_f32(scales, output))
              : 0.0f;
    }
    __syncthreads();
    for (unsigned step = 0u; step < DIF_CHUNK; ++step) {
      float gradient[4], scaled[4];
      for (unsigned i = 0u; i < 4u; ++i) {
        gradient[i] = tile_gradient[step][lane_row + i];
        scaled[i] = tile_weight[step][lane_column + i];
      }
      for (unsigned i = 0u; i < 4u; ++i)
        for (unsigned j = 0u; j < 4u; ++j)
          value[i][j] = fmaf(gradient[i], scaled[j], value[i][j]);
    }
    __syncthreads();
  }
  for (unsigned i = 0u; i < 4u; ++i) {
    const unsigned row = tile_row + lane_row + i;
    if (row >= ${rows}u) continue;
    for (unsigned j = 0u; j < 4u; ++j) {
      const unsigned column = tile_column + lane_column + j;
      if (column < ${inner}u)
        dif_store(grad_input, (unsigned long long)row * ${inner}ULL + column, value[i][j]);
    }
  }
}
