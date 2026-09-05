// Exact dense non-causal streaming attention (Opcode::Attention
// Implementation 5).  Design: flash-style online softmax with K/V tiles
// streamed through shared memory via cp.async double buffering; Q fragments,
// softmax state, and the output accumulator are register resident, so the
// kernel allocates no global workspace. Both Q K^T and P V use BF16 operands
// and FP32 tensor-core accumulation. Keeping V in its original BF16 dtype
// preserves its exponent range; FP32 accumulation avoids overflowing an
// unnormalized tile sum before the final softmax division.
#include "dif/runtime/exact_stream_attention.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace dif::runtime {
namespace {

thread_local std::string last_error;

const char *set_error(const char *message) noexcept {
  last_error = message;
  return last_error.c_str();
}


constexpr int kHeadDim = 128;
constexpr int kRowChunks = kHeadDim / 8;

__device__ __forceinline__ std::uint32_t smem_address(const void *pointer) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ int swizzle_chunk(int row, int chunk) {
  return chunk ^ (row & 7);
}

__device__ __forceinline__ void cp_async_16(std::uint32_t destination,
                                            const void *source, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                   destination),
               "l"(source), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int Remaining> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(Remaining));
}

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t &r0,
                                            std::uint32_t &r1,
                                            std::uint32_t &r2,
                                            std::uint32_t &r3,
                                            std::uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x4_trans(std::uint32_t &r0,
                                                  std::uint32_t &r1,
                                                  std::uint32_t &r2,
                                                  std::uint32_t &r3,
                                                  std::uint32_t address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(address));
}

__device__ __forceinline__ void mma_bf16(float *d, const std::uint32_t *a,
                                         std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float exp2_approx(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(value));
  return result;
}

template <int Rows, int Threads>
__device__ __forceinline__ void
load_tile(std::uint32_t tile_base, const char *global_base,
          std::int64_t row_stride_bytes, int first_row, int row_limit,
          int thread) {
  constexpr int kChunks = Rows * kRowChunks;
  static_assert(kChunks % Threads == 0);
#pragma unroll
  for (int index = thread; index < kChunks; index += Threads) {
    const int row = index / kRowChunks;
    const int chunk = index % kRowChunks;
    const bool valid = first_row + row < row_limit;
    const std::uint32_t destination =
        tile_base + static_cast<std::uint32_t>(
                        row * (kRowChunks * 16) + swizzle_chunk(row, chunk) * 16);
    const char *source =
        global_base +
        (valid ? static_cast<std::int64_t>(first_row + row) * row_stride_bytes +
                     chunk * 16
               : 0);
    cp_async_16(destination, source, valid);
  }
}

template <int BlockRows, int BlockColumns, int Warps>
__global__ void __launch_bounds__(Warps * 32, 1)
    exact_stream_attention_kernel(const __nv_bfloat16 *query,
                                  const __nv_bfloat16 *key,
                                  const __nv_bfloat16 *value,
                                  __nv_bfloat16 *output, int sequence,
                                  int heads, int kv_heads, float scale) {
  constexpr int kThreads = Warps * 32;
  constexpr int kMTiles = BlockRows / (16 * Warps);
  static_assert(BlockRows == 16 * Warps * kMTiles);
  constexpr int kScoreFragments = BlockColumns / 8;
  constexpr int kScorePairs = kScoreFragments / 2;
  constexpr int kValueSteps = BlockColumns / 16;
  constexpr int kOutputFragments = kHeadDim / 8;
  constexpr int kTileBytes = BlockColumns * kHeadDim * 2;

  extern __shared__ char shared_storage[];
  char *const query_tile = shared_storage;
  char *const key_tiles = shared_storage + BlockRows * kHeadDim * 2;
  char *const value_tiles = key_tiles + 2 * kTileBytes;

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32;
  const int lane = thread % 32;
  const int lane_pair = lane % 4;

  const int query_block = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int batch = static_cast<int>(blockIdx.z);
  const int kv_head = head / (heads / kv_heads);

  const std::int64_t row_stride =
      static_cast<std::int64_t>(heads) * kHeadDim * 2;
  const std::int64_t batch_offset =
      static_cast<std::int64_t>(batch) * sequence * row_stride;
  const char *const query_base = reinterpret_cast<const char *>(query) +
                                 batch_offset +
                                 static_cast<std::int64_t>(head) * kHeadDim * 2;
  const char *const key_base = reinterpret_cast<const char *>(key) +
                               batch_offset +
                               static_cast<std::int64_t>(kv_head) * kHeadDim * 2;
  const char *const value_base =
      reinterpret_cast<const char *>(value) + batch_offset +
      static_cast<std::int64_t>(kv_head) * kHeadDim * 2;

  const int first_query_row = query_block * BlockRows;
  const int tiles = (sequence + BlockColumns - 1) / BlockColumns;

  load_tile<BlockRows, kThreads>(smem_address(query_tile), query_base,
                                 row_stride, first_query_row, sequence, thread);
  cp_async_commit();
  load_tile<BlockColumns, kThreads>(smem_address(key_tiles), key_base,
                                    row_stride, 0, sequence, thread);
  load_tile<BlockColumns, kThreads>(smem_address(value_tiles), value_base,
                                    row_stride, 0, sequence, thread);
  cp_async_commit();
  if (tiles > 1) {
    load_tile<BlockColumns, kThreads>(smem_address(key_tiles + kTileBytes),
                                      key_base, row_stride, BlockColumns,
                                      sequence, thread);
    load_tile<BlockColumns, kThreads>(smem_address(value_tiles + kTileBytes),
                                      value_base, row_stride, BlockColumns,
                                      sequence, thread);
  }
  cp_async_commit();

  // Q fragments for this warp's kMTiles m16 sub-tiles, register resident.
  std::uint32_t query_fragment[kMTiles][8][4];
  cp_async_wait<2>();
  __syncthreads();
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    const int tile_row = (warp * kMTiles + m_tile) * 16;
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int chunk = step * 2;
      const int matrix = lane >> 3;
      const int row =
          tile_row + ((matrix == 0 || matrix == 2) ? 0 : 8) + (lane & 7);
      const int piece = (matrix < 2) ? chunk : chunk + 1;
      ldmatrix_x4(query_fragment[m_tile][step][0],
                  query_fragment[m_tile][step][1],
                  query_fragment[m_tile][step][2],
                  query_fragment[m_tile][step][3],
                  smem_address(query_tile) +
                      static_cast<std::uint32_t>(
                          row * (kRowChunks * 16) +
                          swizzle_chunk(row, piece) * 16));
    }
  }

  float output_accumulator[kMTiles][kOutputFragments][4];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile)
#pragma unroll
    for (int fragment = 0; fragment < kOutputFragments; ++fragment)
#pragma unroll
      for (int piece = 0; piece < 4; ++piece)
        output_accumulator[m_tile][fragment][piece] = 0.0F;
  float running_max[kMTiles][2], running_sum[kMTiles][2];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    running_max[m_tile][0] = running_max[m_tile][1] = -INFINITY;
    running_sum[m_tile][0] = running_sum[m_tile][1] = 0.0F;
  }
  const float log2_scale = scale * 1.4426950408889634F;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile & 1;
    char *const key_tile = key_tiles + stage * kTileBytes;
    char *const value_tile = value_tiles + stage * kTileBytes;
    cp_async_wait<1>();
    __syncthreads();

    float score[kMTiles][kScoreFragments][4];
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile)
#pragma unroll
      for (int fragment = 0; fragment < kScoreFragments; ++fragment)
#pragma unroll
        for (int piece = 0; piece < 4; ++piece)
          score[m_tile][fragment][piece] = 0.0F;

    // Q Kt with a one-step register pipeline on the K fragments.
    {
      const int matrix = lane >> 3;
      const int key_row_base = (matrix < 2 ? 0 : 8) + (lane & 7);
      const bool high_piece = (matrix == 1 || matrix == 3);
      std::uint32_t key_fragment[2][kScorePairs][4];
      const auto fetch_step = [&](int step, std::uint32_t destination[][4]) {
        const int piece = step * 2 + (high_piece ? 1 : 0);
#pragma unroll
        for (int pair = 0; pair < kScorePairs; ++pair) {
          const int row = pair * 16 + key_row_base;
          ldmatrix_x4(destination[pair][0], destination[pair][1],
                      destination[pair][2], destination[pair][3],
                      smem_address(key_tile) +
                          static_cast<std::uint32_t>(
                              row * (kRowChunks * 16) +
                              swizzle_chunk(row, piece) * 16));
        }
      };
      fetch_step(0, key_fragment[0]);
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        if (step + 1 < 8)
          fetch_step(step + 1, key_fragment[(step + 1) & 1]);
#pragma unroll
        for (int pair = 0; pair < kScorePairs; ++pair)
#pragma unroll
          for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
            mma_bf16(score[m_tile][pair * 2], query_fragment[m_tile][step],
                     key_fragment[step & 1][pair][0],
                     key_fragment[step & 1][pair][1]);
            mma_bf16(score[m_tile][pair * 2 + 1], query_fragment[m_tile][step],
                     key_fragment[step & 1][pair][2],
                     key_fragment[step & 1][pair][3]);
          }
      }
    }

    if (tile == tiles - 1 && sequence % BlockColumns != 0) {
      const int base_column = tile * BlockColumns;
#pragma unroll
      for (int fragment = 0; fragment < kScoreFragments; ++fragment) {
        const int column = base_column + fragment * 8 + lane_pair * 2;
#pragma unroll
        for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
          if (column >= sequence)
            score[m_tile][fragment][0] = score[m_tile][fragment][2] =
                -INFINITY;
          if (column + 1 >= sequence)
            score[m_tile][fragment][1] = score[m_tile][fragment][3] =
                -INFINITY;
        }
      }
    }

    std::uint32_t probability[kMTiles][kValueSteps][4];
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
      float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int fragment = 0; fragment < kScoreFragments; ++fragment) {
        tile_max[0] = fmaxf(tile_max[0], fmaxf(score[m_tile][fragment][0],
                                               score[m_tile][fragment][1]));
        tile_max[1] = fmaxf(tile_max[1], fmaxf(score[m_tile][fragment][2],
                                               score[m_tile][fragment][3]));
      }
#pragma unroll
      for (int distance = 1; distance <= 2; distance += distance) {
        tile_max[0] = fmaxf(
            tile_max[0], __shfl_xor_sync(0xFFFFFFFFU, tile_max[0], distance));
        tile_max[1] = fmaxf(
            tile_max[1], __shfl_xor_sync(0xFFFFFFFFU, tile_max[1], distance));
      }
      float alpha[2];
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        const float updated = fmaxf(running_max[m_tile][row], tile_max[row]);
        alpha[row] =
            exp2_approx((running_max[m_tile][row] - updated) * log2_scale);
        running_max[m_tile][row] = updated;
      }
      float partial_sum[2] = {0.0F, 0.0F};
#pragma unroll
      for (int fragment = 0; fragment < kScoreFragments; ++fragment) {
        score[m_tile][fragment][0] = exp2_approx(
            (score[m_tile][fragment][0] - running_max[m_tile][0]) * log2_scale);
        score[m_tile][fragment][1] = exp2_approx(
            (score[m_tile][fragment][1] - running_max[m_tile][0]) * log2_scale);
        score[m_tile][fragment][2] = exp2_approx(
            (score[m_tile][fragment][2] - running_max[m_tile][1]) * log2_scale);
        score[m_tile][fragment][3] = exp2_approx(
            (score[m_tile][fragment][3] - running_max[m_tile][1]) * log2_scale);
        partial_sum[0] +=
            score[m_tile][fragment][0] + score[m_tile][fragment][1];
        partial_sum[1] +=
            score[m_tile][fragment][2] + score[m_tile][fragment][3];
      }
#pragma unroll
      for (int distance = 1; distance <= 2; distance += distance) {
        partial_sum[0] +=
            __shfl_xor_sync(0xFFFFFFFFU, partial_sum[0], distance);
        partial_sum[1] +=
            __shfl_xor_sync(0xFFFFFFFFU, partial_sum[1], distance);
      }
#pragma unroll
      for (int row = 0; row < 2; ++row)
        running_sum[m_tile][row] =
            running_sum[m_tile][row] * alpha[row] + partial_sum[row];
#pragma unroll
      for (int fragment = 0; fragment < kOutputFragments; ++fragment) {
        output_accumulator[m_tile][fragment][0] *= alpha[0];
        output_accumulator[m_tile][fragment][1] *= alpha[0];
        output_accumulator[m_tile][fragment][2] *= alpha[1];
        output_accumulator[m_tile][fragment][3] *= alpha[1];
      }
#pragma unroll
      for (int step = 0; step < kValueSteps; ++step) {
        const __nv_bfloat162 low_rows = __floats2bfloat162_rn(
            score[m_tile][step * 2][0], score[m_tile][step * 2][1]);
        const __nv_bfloat162 high_rows = __floats2bfloat162_rn(
            score[m_tile][step * 2][2], score[m_tile][step * 2][3]);
        const __nv_bfloat162 low_rows_high_k = __floats2bfloat162_rn(
            score[m_tile][step * 2 + 1][0], score[m_tile][step * 2 + 1][1]);
        const __nv_bfloat162 high_rows_high_k = __floats2bfloat162_rn(
            score[m_tile][step * 2 + 1][2], score[m_tile][step * 2 + 1][3]);
        probability[m_tile][step][0] =
            *reinterpret_cast<const std::uint32_t *>(&low_rows);
        probability[m_tile][step][1] =
            *reinterpret_cast<const std::uint32_t *>(&high_rows);
        probability[m_tile][step][2] =
            *reinterpret_cast<const std::uint32_t *>(&low_rows_high_k);
        probability[m_tile][step][3] =
            *reinterpret_cast<const std::uint32_t *>(&high_rows_high_k);
      }
    }

    // P V consumes the original BF16 values and accumulates directly in
    // FP32. Even a 32-term tile can overflow FP16 with finite BF16 inputs.
    {
      const int matrix = lane >> 3;
      const int value_row_base = ((matrix == 0 || matrix == 2) ? 0 : 8) +
                                 (lane & 7);
      const bool high_piece = (matrix >= 2);
      std::uint32_t value_fragment[2][kValueSteps][4];
      const auto fetch_pair = [&](int pair, std::uint32_t destination[][4]) {
        const int piece = pair * 2 + (high_piece ? 1 : 0);
#pragma unroll
        for (int step = 0; step < kValueSteps; ++step) {
          const int row = step * 16 + value_row_base;
          ldmatrix_x4_trans(destination[step][0], destination[step][1],
                            destination[step][2], destination[step][3],
                            smem_address(value_tile) +
                                static_cast<std::uint32_t>(
                                    row * (kRowChunks * 16) +
                                    swizzle_chunk(row, piece) * 16));
        }
      };
      fetch_pair(0, value_fragment[0]);
#pragma unroll
      for (int pair = 0; pair < kOutputFragments / 2; ++pair) {
        if (pair + 1 < kOutputFragments / 2)
          fetch_pair(pair + 1, value_fragment[(pair + 1) & 1]);
#pragma unroll
        for (int step = 0; step < kValueSteps; ++step)
#pragma unroll
          for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
            mma_bf16(output_accumulator[m_tile][pair * 2],
                    probability[m_tile][step],
                    value_fragment[pair & 1][step][0],
                    value_fragment[pair & 1][step][1]);
            mma_bf16(output_accumulator[m_tile][pair * 2 + 1],
                    probability[m_tile][step],
                    value_fragment[pair & 1][step][2],
                    value_fragment[pair & 1][step][3]);
          }
      }
    }

    __syncthreads();
    if (tile + 2 < tiles) {
      load_tile<BlockColumns, kThreads>(
          smem_address(key_tiles + stage * kTileBytes), key_base, row_stride,
          (tile + 2) * BlockColumns, sequence, thread);
      load_tile<BlockColumns, kThreads>(
          smem_address(value_tiles + stage * kTileBytes), value_base,
          row_stride, (tile + 2) * BlockColumns, sequence, thread);
    }
    cp_async_commit();
  }

  char *const output_base = reinterpret_cast<char *>(output) + batch_offset +
                            static_cast<std::int64_t>(head) * kHeadDim * 2;
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    const float inverse_sum[2] = {
        running_sum[m_tile][0] > 0.0F ? 1.0F / running_sum[m_tile][0] : 0.0F,
        running_sum[m_tile][1] > 0.0F ? 1.0F / running_sum[m_tile][1] : 0.0F};
    const int row0 =
        first_query_row + (warp * kMTiles + m_tile) * 16 + lane / 4;
    const int row1 = row0 + 8;
#pragma unroll
    for (int fragment = 0; fragment < kOutputFragments; ++fragment) {
      const int column = fragment * 8 + lane_pair * 2;
      if (row0 < sequence) {
        const __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(
            output_accumulator[m_tile][fragment][0] * inverse_sum[0],
            output_accumulator[m_tile][fragment][1] * inverse_sum[0]));
        *reinterpret_cast<__nv_bfloat162 *>(
            output_base + static_cast<std::int64_t>(row0) * row_stride +
            column * 2) = packed;
      }
      if (row1 < sequence) {
        const __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(
            output_accumulator[m_tile][fragment][2] * inverse_sum[1],
            output_accumulator[m_tile][fragment][3] * inverse_sum[1]));
        *reinterpret_cast<__nv_bfloat162 *>(
            output_base + static_cast<std::int64_t>(row1) * row_stride +
            column * 2) = packed;
      }
    }
  }
}


constexpr int kBlockRows = 64;
constexpr int kBlockColumns = 32;
constexpr int kWarps = 4;
constexpr int kSharedBytes = (kBlockRows + 4 * kBlockColumns) * kHeadDim * 2;

} // namespace

const char *exact_stream_attention_bf16_forward(
    std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
    std::uintptr_t output, std::uint32_t batch, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t key_value_heads,
    std::uint32_t head_dimension, float scale, std::uintptr_t stream) noexcept {
  if (!query || !key || !value || !output)
    return set_error("exact stream attention requires nonnull tensors");
  if (head_dimension != 128U)
    return set_error("exact stream attention requires head dimension 128");
  if (key_value_heads != heads)
    return set_error("exact stream attention requires kv_heads == heads");
  if (batch == 0U || sequence == 0U || heads == 0U)
    return set_error("exact stream attention requires nonzero geometry");
  if (sequence > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      heads > 65535U || batch > 65535U)
    return set_error("exact stream attention geometry exceeds launch limits");
  if (!std::isfinite(scale) || scale <= 0.0F)
    return set_error("exact stream attention requires a positive finite scale");

  auto *kernel = exact_stream_attention_kernel<kBlockRows, kBlockColumns,
                                               kWarps>;
  static const cudaError_t attribute_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (attribute_status != cudaSuccess)
    return set_error("exact stream attention shared-memory opt-in failed");

  const dim3 grid((sequence + kBlockRows - 1U) / kBlockRows, heads, batch);
  kernel<<<grid, kWarps * 32, kSharedBytes,
           reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query),
      reinterpret_cast<const __nv_bfloat16 *>(key),
      reinterpret_cast<const __nv_bfloat16 *>(value),
      reinterpret_cast<__nv_bfloat16 *>(output), static_cast<int>(sequence),
      static_cast<int>(heads), static_cast<int>(heads), scale);
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    last_error = std::string("exact stream attention launch failed: ") +
                 cudaGetErrorString(launch_status);
    return last_error.c_str();
  }
  return nullptr;
}

} // namespace dif::runtime
