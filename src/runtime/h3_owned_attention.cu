// Diffusion Compiler: in-tree owned MiniMax-H3 dense INT8 attention (v4).
//
// Vendored 2026-09-03 from the project owner's Ck-INT8 native kernel
// (h3_dense_int8_v4.cu, ABI v4, sm_86). Independently written, H3-specific
// CUDA; it includes, links, and dispatches no third-party attention code.
// Behind RunOptions::h3_owned_attention the executor binds these entry points
// in place of the dlopen'd DSO route; the classification stays approximate
// and the H3 video/audio quality gates remain the admission bar.
//
// Why v4 exists: on the power-capped RTX 3090 Ti the v3 kernel is bound by
// energy per key tile, not by any pipe (ablations, 2026-08-21): tensor MMAs
// ~44%, ldmatrix smem reads ~16%, K/V L2->smem loads ~16%, softmax ALU ~15%,
// the rest ~9%. v4 attacks the non-MMA terms:
//   * 8 warps x 32 query rows (256 queries per CTA), 64-key tiles: every K and
//     V^T ldmatrix fragment now feeds four MMAs instead of two, halving smem
//     read energy per query, and each CTA streams K/V for twice as many
//     queries, halving L2/DRAM traffic per query.
//   * K is quantized (after the same 128-point Walsh-Hadamard rotation as Q)
//     with ONE scale per 128-key tile, so the row max is an integer
//     reduction and the per-score scale multiply folds into the exp2
//     argument (no per-key scale loads, no per-score FMUL).
//   * V is quantized per channel per head; the scale is applied once in the
//     epilogue, so the P.V accumulate is a single FFMA per element.
//   * u8 probabilities come out of the ex2 already scaled by 255 (log2(255)
//     folded into the exponent bias) and are packed with cvt.pack (two
//     F2IP per four values); row sums use dp4a instead of tensor MMAs.
//   * Probabilities are still scaled per query row per KEY TILE (tile max ->
//     255); running-max-relative scaling (running-max-relative, as in third-party INT8 attention) measured
//     0.99963 vs 0.99987 cosine on the worst H3 capture, so it is not used.
// Numerics otherwise follow v3 (FP32 online softmax in the exp2 domain,
// lazy O rescale, INT8 mma.m16n8k32 for QK^T and P.V, cp.async double
// buffering, one CTA barrier per key tile).

#include "dif/runtime/h3_owned_attention.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

// The kernel is written for sm_86 (opt-in shared memory, INT8 mma.m16n8k32,
// cp.async). The executor admits it only when the running device reports
// exactly this SM; every other device fails closed at prepare.
#define DIF_H3_OWNED_ATTENTION_TARGET_SM 86

namespace {

constexpr int kHeadDim = 128;
constexpr int kScaleTile = 128;  // K scale + padding granularity in keys.
constexpr int kAbiVersion = 4;
constexpr float kInt8Max = 127.0f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kInvSqrtHeadDim = 0.08838834764831845f;
constexpr float kProbabilityLevels = 255.0f;
constexpr float kLog2ProbabilityLevels = 7.994353436858858f;  // log2(255)
constexpr uint32_t kOnesU8x4 = 0x01010101u;

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_16(uint32_t shared, const void* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(shared), "l"(global)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int kPending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPending) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t address, uint32_t (&r)[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address));
}

__device__ __forceinline__ void mma_s8s8(int (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_u8s8(int (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// round-to-nearest-even of four values in [0, 255] into one u8x4 register,
// byte 0 <- a ... byte 3 <- d. ptxas fuses each float->int + pack pair into
// one F2IP.U8 instruction (two per four values).
__device__ __forceinline__ int float_to_int_rn(float x) {
  // Explicit non-FTZ cvt: with --use_fast_math __float2int_rn becomes
  // F2I.FTZ, which ptxas does not fuse with the pack into F2IP.
  int r;
  asm("cvt.rni.s32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint32_t pack_u8x4_rn(float a, float b, float c,
                                                 float d) {
  const int ia = float_to_int_rn(a);
  const int ib = float_to_int_rn(b);
  const int ic = float_to_int_rn(c);
  const int id = float_to_int_rn(d);
  uint32_t r;
  asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, 0;\n" : "=r"(r) : "r"(id), "r"(ic));
  asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;\n"
      : "=r"(r)
      : "r"(ib), "r"(ia), "r"(r));
  return r;
}

__device__ __forceinline__ int8_t quantize_s8(float value, float inverse_scale) {
  float rounded = nearbyintf(value * inverse_scale);
  rounded = fminf(kInt8Max, fmaxf(-kInt8Max, rounded));
  return static_cast<int8_t>(static_cast<int>(rounded));
}

__device__ __forceinline__ uint32_t pack_s8x4(int8_t a, int8_t b, int8_t c,
                                              int8_t d) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(a))) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high) {
  const __nv_bfloat162 packed = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

// ---------------------------------------------------------------------------
// 128-point orthonormal Walsh-Hadamard rotation of one row held by a warp:
// lane l owns channels 4l..4l+3. Q and K use the same transform, so
// (H q).(H k) = q.k.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hadamard128_warp(float (&v)[4], int lane) {
  const float a0 = v[0] + v[1];
  const float a1 = v[0] - v[1];
  const float a2 = v[2] + v[3];
  const float a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
    const bool upper = (lane & bit) != 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, v[c], bit);
      v[c] = upper ? (other - v[c]) : (v[c] + other);
    }
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) v[c] *= kInvSqrtHeadDim;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* source,
                                            float (&v)[4]) {
  const uint32_t low = *reinterpret_cast<const uint32_t*>(source);
  const uint32_t high = *reinterpret_cast<const uint32_t*>(source + 2);
  const __nv_bfloat162 low2 = *reinterpret_cast<const __nv_bfloat162*>(&low);
  const __nv_bfloat162 high2 = *reinterpret_cast<const __nv_bfloat162*>(&high);
  v[0] = __low2float(low2);
  v[1] = __high2float(low2);
  v[2] = __low2float(high2);
  v[3] = __high2float(high2);
}

__device__ __forceinline__ float abs_max4(const float (&v)[4]) {
  return fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
}

// Rotate + quantize one 128-wide row held by a warp with its own row scale
// (Q path). Returns the scale (1.0 for an all-zero row).
__device__ __forceinline__ uint32_t rotate_quantize_row(float (&v)[4], int lane,
                                                        float& scale) {
  hadamard128_warp(v, lane);
  const float row_max = warp_max(abs_max4(v));
  scale = row_max > 0.0f ? row_max / kInt8Max : 1.0f;
  const float inverse_scale = 1.0f / scale;
  return pack_s8x4(quantize_s8(v[0], inverse_scale),
                   quantize_s8(v[1], inverse_scale),
                   quantize_s8(v[2], inverse_scale),
                   quantize_s8(v[3], inverse_scale));
}

// ---------------------------------------------------------------------------
// K quantization: block per 128-key tile (8 warps x 16 rows held rotated in
// registers), one symmetric INT8 scale per tile, zero padding.
// k8: [B,H,S_pad,128]; k_scale: [B,H,S_pad/128].
// ---------------------------------------------------------------------------
constexpr int kKQuantThreads = 256;
constexpr int kKQuantRowsPerWarp = kScaleTile / (kKQuantThreads / 32);  // 16

// Optional K mean-centering (k_mean != nullptr): the per-head mean over the
// sequence is subtracted before rotation and quantization. q.(k - m) differs
// from q.k by a per-query constant, so softmax is unchanged exactly while the
// INT8 range spent on the shared offset is recovered (Mojo Sage lesson).
__global__ void __launch_bounds__(kKQuantThreads) quantize_k_tiles_h128_bf16(
    const __nv_bfloat16* __restrict__ k, int8_t* __restrict__ k8,
    float* __restrict__ k_scale, const float* __restrict__ k_mean, int heads,
    int sequence, int padded_sequence, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_s) {
  __shared__ float warp_maxima[kKQuantThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int key_tile = blockIdx.x;
  const int head = blockIdx.y;
  const int batch_index = blockIdx.z;
  const int key_start = key_tile * kScaleTile;
  const int64_t bh = static_cast<int64_t>(batch_index) * heads + head;
  const __nv_bfloat16* k_head = k + static_cast<int64_t>(batch_index) * k_stride_b +
                                static_cast<int64_t>(head) * k_stride_h;
  float mean4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (k_mean != nullptr) {
    const float4 m = *reinterpret_cast<const float4*>(k_mean + bh * kHeadDim + lane * 4);
    mean4[0] = m.x; mean4[1] = m.y; mean4[2] = m.z; mean4[3] = m.w;
  }

  float rows[kKQuantRowsPerWarp][4];
  float local_max = 0.0f;
#pragma unroll
  for (int r = 0; r < kKQuantRowsPerWarp; ++r) {
    const int key = key_start + warp * kKQuantRowsPerWarp + r;
    rows[r][0] = rows[r][1] = rows[r][2] = rows[r][3] = 0.0f;
    if (key < sequence) {
      load_bf16x4(k_head + static_cast<int64_t>(key) * k_stride_s + lane * 4, rows[r]);
      rows[r][0] -= mean4[0]; rows[r][1] -= mean4[1];
      rows[r][2] -= mean4[2]; rows[r][3] -= mean4[3];
      hadamard128_warp(rows[r], lane);
      local_max = fmaxf(local_max, abs_max4(rows[r]));
    }
  }
  local_max = warp_max(local_max);
  if (lane == 0) warp_maxima[warp] = local_max;
  __syncthreads();
  float tile_max = warp_maxima[0];
#pragma unroll
  for (int w = 1; w < kKQuantThreads / 32; ++w) tile_max = fmaxf(tile_max, warp_maxima[w]);
  const float scale = tile_max > 0.0f ? tile_max / kInt8Max : 1.0f;
  const float inverse_scale = 1.0f / scale;

  uint32_t* k8_tile = reinterpret_cast<uint32_t*>(
      k8 + (bh * padded_sequence + key_start) * kHeadDim);
#pragma unroll
  for (int r = 0; r < kKQuantRowsPerWarp; ++r) {
    const int row = warp * kKQuantRowsPerWarp + r;
    k8_tile[row * (kHeadDim / 4) + lane] =
        pack_s8x4(quantize_s8(rows[r][0], inverse_scale),
                  quantize_s8(rows[r][1], inverse_scale),
                  quantize_s8(rows[r][2], inverse_scale),
                  quantize_s8(rows[r][3], inverse_scale));
  }
  if (threadIdx.x == 0) {
    k_scale[bh * (padded_sequence / kScaleTile) + key_tile] =
        key_start < sequence ? scale : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// K per-head mean, deterministic two-stage reduction (no float atomics, so a
// repeated run is bit-identical): per-128-key-tile partial sums
// [B,H,S_pad/128,128], then a fixed-order sum over tiles divided by the
// sequence length into k_mean [B,H,128].
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kHeadDim) k_tile_channel_sums_bf16(
    const __nv_bfloat16* __restrict__ k, float* __restrict__ partials, int heads,
    int sequence, int padded_sequence, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_s) {
  const int channel = threadIdx.x;
  const int key_tile = blockIdx.x;
  const int head = blockIdx.y;
  const int batch_index = blockIdx.z;
  const int64_t bh = static_cast<int64_t>(batch_index) * heads + head;
  const __nv_bfloat16* k_head = k + static_cast<int64_t>(batch_index) * k_stride_b +
                                static_cast<int64_t>(head) * k_stride_h;
  const int key_start = key_tile * kScaleTile;
  const int key_end = min(key_start + kScaleTile, sequence);
  float sum = 0.0f;
  for (int key = key_start; key < key_end; ++key)
    sum += __bfloat162float(k_head[static_cast<int64_t>(key) * k_stride_s + channel]);
  partials[(bh * (padded_sequence / kScaleTile) + key_tile) * kHeadDim + channel] = sum;
}

__global__ void __launch_bounds__(kHeadDim) k_mean_finalize(
    const float* __restrict__ partials, float* __restrict__ k_mean, int key_tiles,
    int sequence) {
  const int channel = threadIdx.x;
  const int64_t bh = blockIdx.x;
  float sum = 0.0f;
  for (int tile = 0; tile < key_tiles; ++tile)
    sum += partials[(bh * key_tiles + tile) * kHeadDim + channel];
  k_mean[bh * kHeadDim + channel] = sum / static_cast<float>(sequence);
}

// ---------------------------------------------------------------------------
// V quantization, per channel per head, three launches:
//   1. v_channel_absmax: block per (128-key tile, head, batch), thread per
//      channel, atomicMax of the channel |max| (as non-negative float bits)
//      into v_scale[B,H,128] (zeroed first).
//   2. finalize_v_scale: bits -> symmetric INT8 scale.
//   3. quantize_v_tiles: transposed INT8 output [B,H,128,S_pad] in the
//      32-key permuted order expected by the attention kernel's P fragments.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int permute_key32(int virtual_key) {
  const int half = (virtual_key >> 4) & 1;
  const int quad = (virtual_key >> 2) & 3;
  const int item = virtual_key & 3;
  return half * 16 + (item >> 1) * 8 + quad * 2 + (item & 1);
}

constexpr int kVQuantThreads = 128;
constexpr int kVQuantRowStride = kScaleTile + 4;  // int8 bytes, bank-staggered
constexpr int kVQuantSharedBytes =
    kScaleTile * kHeadDim * sizeof(__nv_bfloat16) + kHeadDim * kVQuantRowStride;

// Stage one [128 keys][128 features] BF16 tile into shared memory (zero
// padding beyond the sequence).
__device__ __forceinline__ void stage_v_tile(
    __nv_bfloat16* tile, const __nv_bfloat16* v_head, int key_start, int sequence,
    int64_t v_stride_s, int tid) {
  for (int index = tid; index < kScaleTile * (kHeadDim / 2); index += kVQuantThreads) {
    const int key_row = index / (kHeadDim / 2);
    const int pair = index % (kHeadDim / 2);
    const int key = key_start + key_row;
    uint32_t raw = 0u;
    if (key < sequence) {
      raw = *reinterpret_cast<const uint32_t*>(
          v_head + static_cast<int64_t>(key) * v_stride_s + pair * 2);
    }
    *reinterpret_cast<uint32_t*>(&tile[key_row * kHeadDim + pair * 2]) = raw;
  }
}

__global__ void __launch_bounds__(kVQuantThreads) v_channel_absmax_bf16(
    const __nv_bfloat16* __restrict__ v, unsigned int* __restrict__ v_scale_bits,
    int heads, int sequence, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_s) {
  __shared__ __align__(16) __nv_bfloat16 tile[kScaleTile * kHeadDim];
  const int key_start = blockIdx.x * kScaleTile;
  const int head = blockIdx.y;
  const int batch_index = blockIdx.z;
  const int tid = threadIdx.x;
  if (key_start >= sequence) return;
  const __nv_bfloat16* v_head = v + static_cast<int64_t>(batch_index) * v_stride_b +
                                static_cast<int64_t>(head) * v_stride_h;
  stage_v_tile(tile, v_head, key_start, sequence, v_stride_s, tid);
  __syncthreads();
  const int feature = tid;
  float channel_max = 0.0f;
#pragma unroll 8
  for (int key_row = 0; key_row < kScaleTile; ++key_row) {
    channel_max = fmaxf(
        channel_max, fabsf(__bfloat162float(tile[key_row * kHeadDim + feature])));
  }
  const int64_t bh = static_cast<int64_t>(batch_index) * heads + head;
  // Non-negative floats order like their bit patterns.
  atomicMax(v_scale_bits + bh * kHeadDim + feature, __float_as_uint(channel_max));
}

__global__ void __launch_bounds__(kHeadDim) finalize_v_scale(float* __restrict__ v_scale) {
  float* slot = v_scale + static_cast<int64_t>(blockIdx.x) * kHeadDim + threadIdx.x;
  const float channel_max = __uint_as_float(*reinterpret_cast<unsigned int*>(slot));
  *slot = channel_max > 0.0f ? channel_max / kInt8Max : 1.0f;
}

__global__ void __launch_bounds__(kVQuantThreads) quantize_v_tiles_bf16(
    const __nv_bfloat16* __restrict__ v, int8_t* __restrict__ v8,
    const float* __restrict__ v_scale, int heads, int sequence,
    int padded_sequence, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_s) {
  // The two buffers total 49,664 bytes, above CUDA's 48 KiB static limit.
  // Preserve their layout in opt-in dynamic shared memory.
  extern __shared__ __align__(16) unsigned char v_shared[];
  auto* tile = reinterpret_cast<__nv_bfloat16*>(v_shared);
  auto* transposed = reinterpret_cast<int8_t*>(
      v_shared + kScaleTile * kHeadDim * sizeof(__nv_bfloat16));

  const int key_start = blockIdx.x * kScaleTile;
  const int head = blockIdx.y;
  const int batch_index = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t bh = static_cast<int64_t>(batch_index) * heads + head;
  const __nv_bfloat16* v_head = v + static_cast<int64_t>(batch_index) * v_stride_b +
                                static_cast<int64_t>(head) * v_stride_h;
  stage_v_tile(tile, v_head, key_start, sequence, v_stride_s, tid);
  __syncthreads();

  // Thread = feature channel; quantize in permuted key order.
  const int feature = tid;
  const float inverse_scale = 1.0f / v_scale[bh * kHeadDim + feature];
  uint32_t* transposed_row =
      reinterpret_cast<uint32_t*>(transposed + feature * kVQuantRowStride);
#pragma unroll 4
  for (int group = 0; group < kScaleTile / 4; ++group) {
    int8_t packed[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int virtual_key = group * 4 + item;
      const int key_row = (virtual_key & ~31) + permute_key32(virtual_key & 31);
      packed[item] = quantize_s8(
          __bfloat162float(tile[key_row * kHeadDim + feature]), inverse_scale);
    }
    transposed_row[group] = pack_s8x4(packed[0], packed[1], packed[2], packed[3]);
  }
  __syncthreads();

  // Coalesced write-out: one 128-byte feature row per warp instruction.
  int8_t* v8_head = v8 + bh * kHeadDim * padded_sequence + key_start;
  for (int row = warp; row < kHeadDim; row += kVQuantThreads / 32) {
    const uint32_t value = *reinterpret_cast<const uint32_t*>(
        transposed + row * kVQuantRowStride + lane * 4);
    *reinterpret_cast<uint32_t*>(
        v8_head + static_cast<int64_t>(row) * padded_sequence + lane * 4) = value;
  }
}

// ---------------------------------------------------------------------------
// Attention kernel (SM86)
// ---------------------------------------------------------------------------
#if DIF_H3_OWNED_ATTENTION_TARGET_SM == 86

template <int kRowBytes>
__device__ __forceinline__ int swizzle_chunk(int row, int chunk) {
  static_assert(kRowBytes == 128 || kRowBytes == 64, "unsupported row length");
  if constexpr (kRowBytes == 128) {
    return chunk ^ (row & 7);
  } else {
    return chunk ^ ((row >> 1) & 3);
  }
}

// ldmatrix.x4 lane offset for the MMA A operand of a 16-row slab stored as
// 128-byte swizzled rows: matrices [rows 0-7, chunk 2c], [rows 8-15, 2c],
// [rows 0-7, 2c+1], [rows 8-15, 2c+1]. Per k-step c: address = base ^ (32 c).
__device__ __forceinline__ int ldmatrix_a_lane_offset(int lane) {
  const int matrix = lane >> 3;
  const int row = (lane & 7) + 8 * (matrix & 1);
  const int chunk = matrix >> 1;
  return row * 128 + swizzle_chunk<128>(row, chunk) * 16;
}

// ldmatrix.x4 lane offset for a B operand pair (two n8 blocks = 16 rows) of
// kRowBytes-byte swizzled rows: matrices [rows 0-7, chunk 2c], [rows 0-7,
// 2c+1], [rows 8-15, 2c], [rows 8-15, 2c+1]. Per k-step c: base ^ (32 c).
template <int kRowBytes>
__device__ __forceinline__ int ldmatrix_b_lane_offset(int lane) {
  const int matrix = lane >> 3;
  const int row = (lane & 7) + 8 * (matrix >> 1);
  const int chunk = matrix & 1;
  return row * kRowBytes + swizzle_chunk<kRowBytes>(row, chunk) * 16;
}

template <bool kValue>
struct BoolTag {
  static constexpr bool value = kValue;
};

struct AttentionConfig {
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kRowsPerWarp = 32;
  static constexpr int kMTiles = kRowsPerWarp / 16;  // 2
  static constexpr int kQueries = kWarps * kRowsPerWarp;  // 256
  static constexpr int kKeyTile = 64;
  static constexpr int kKeyBlocks = kKeyTile / 8;   // 8 n8 blocks
  static constexpr int kKeyGroups = kKeyTile / 32;  // 2 k32 groups for P.V
  static constexpr int kVRowBytes = kKeyTile;       // V^T row = 64 keys
  static constexpr int kQBytes = kQueries * kHeadDim;  // 32 KB, resident
  static constexpr int kKBytes = kKeyTile * kHeadDim;  // 8 KB per stage
  static constexpr int kVBytes = kHeadDim * kKeyTile;  // 8 KB per stage
  static constexpr int kKOffset = kQBytes;
  static constexpr int kVOffset = kKOffset + 2 * kKBytes;
  static constexpr int kSharedBytes = kVOffset + 2 * kVBytes;  // 65536
  static_assert(kSharedBytes <= 101376, "exceeds sm_86 opt-in shared memory");
};

// cp.async one 64-key tile: 64 K rows x 8 chunks and 128 V^T rows x 4 chunks
// of 16 bytes -> 2 + 2 chunks per thread, constant trip counts.
__device__ __forceinline__ void stage_key_tile_at(
    unsigned char* k_tile, unsigned char* v_tile, const int8_t* k_head,
    const int8_t* v_head, int key_start, int padded_sequence, int tid) {
  using Config = AttentionConfig;
  constexpr int kKChunks = Config::kKeyTile * 8;
  static_assert(kKChunks % Config::kThreads == 0, "K chunk split");
#pragma unroll
  for (int i = 0; i < kKChunks / Config::kThreads; ++i) {
    const int chunk = tid + i * Config::kThreads;
    const int row = chunk >> 3;
    const int column = chunk & 7;
    cp_async_16(shared_address(k_tile + row * 128 +
                               swizzle_chunk<128>(row, column) * 16),
                k_head + static_cast<int64_t>(key_start + row) * kHeadDim +
                    column * 16);
  }
  constexpr int kVChunksPerRow = Config::kKeyTile / 16;  // 4
  constexpr int kVChunks = kHeadDim * kVChunksPerRow;
  static_assert(kVChunks % Config::kThreads == 0, "V chunk split");
#pragma unroll
  for (int i = 0; i < kVChunks / Config::kThreads; ++i) {
    const int chunk = tid + i * Config::kThreads;
    const int row = chunk / kVChunksPerRow;
    const int column = chunk % kVChunksPerRow;
    cp_async_16(shared_address(v_tile + row * Config::kVRowBytes +
                               swizzle_chunk<Config::kVRowBytes>(row, column) * 16),
                v_head + static_cast<int64_t>(row) * padded_sequence + key_start +
                    column * 16);
  }
}

__device__ __forceinline__ void stage_key_tile(
    unsigned char* shared, int buffer, const int8_t* k_head,
    const int8_t* v_head, int key_start, int padded_sequence, int tid) {
  using Config = AttentionConfig;
  stage_key_tile_at(shared + Config::kKOffset + buffer * Config::kKBytes,
                    shared + Config::kVOffset + buffer * Config::kVBytes, k_head,
                    v_head, key_start, padded_sequence, tid);
}

__device__ __forceinline__ int warp_row_max_int(int value) {
  value = max(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return max(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

__device__ __forceinline__ float warp_row_max_float(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

// One 16-row m-tile: s32 scores (8 n8 blocks x 4) -> u8 probabilities packed
// as two P.V A fragments (k32 groups), plus the row statistics. Row r0 =
// lane/4, r1 = r0 + 8. c0/c1 = per-row log2-domain score scale (> 0).
template <bool kMaskTail>
__device__ __forceinline__ void softmax_mtile(
    const int (&s)[AttentionConfig::kKeyBlocks][4], float c0, float c1,
    float& m0, float& m1, float& alpha0, float& alpha1, float& ps0, float& ps1,
    uint32_t (&probabilities)[AttentionConfig::kKeyGroups][4], int valid_keys,
    int t4) {
  constexpr int kBlocks = AttentionConfig::kKeyBlocks;
  float tile_max0, tile_max1;
  float x[kBlocks][4];
  if constexpr (!kMaskTail) {
    int mi0 = max(s[0][0], s[0][1]);
    int mi1 = max(s[0][2], s[0][3]);
#pragma unroll
    for (int j = 1; j < kBlocks; ++j) {
      mi0 = max(mi0, max(s[j][0], s[j][1]));
      mi1 = max(mi1, max(s[j][2], s[j][3]));
    }
    tile_max0 = __int2float_rn(warp_row_max_int(mi0)) * c0;
    tile_max1 = __int2float_rn(warp_row_max_int(mi1)) * c1;
  } else {
    float max0 = -CUDART_INF_F;
    float max1 = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < kBlocks; ++j) {
      const int key0 = j * 8 + t4 * 2;
      x[j][0] = key0 < valid_keys ? __int2float_rn(s[j][0]) * c0 : -CUDART_INF_F;
      x[j][1] = key0 + 1 < valid_keys ? __int2float_rn(s[j][1]) * c0 : -CUDART_INF_F;
      x[j][2] = key0 < valid_keys ? __int2float_rn(s[j][2]) * c1 : -CUDART_INF_F;
      x[j][3] = key0 + 1 < valid_keys ? __int2float_rn(s[j][3]) * c1 : -CUDART_INF_F;
      max0 = fmaxf(max0, fmaxf(x[j][0], x[j][1]));
      max1 = fmaxf(max1, fmaxf(x[j][2], x[j][3]));
    }
    tile_max0 = warp_row_max_float(max0);
    tile_max1 = warp_row_max_float(max1);
  }
  const float next0 = fmaxf(m0, tile_max0);
  const float next1 = fmaxf(m1, tile_max1);
  alpha0 = fast_exp2(m0 - next0);
  alpha1 = fast_exp2(m1 - next1);
  ps0 = fast_exp2(tile_max0 - next0) * (1.0f / kProbabilityLevels);
  ps1 = fast_exp2(tile_max1 - next1) * (1.0f / kProbabilityLevels);
  m0 = next0;
  m1 = next1;
  // exp2(x - tile_max + log2 255) = 255 * p, rounded to u8 by pack_u8x4_rn.
  const float bias0 = kLog2ProbabilityLevels - tile_max0;
  const float bias1 = kLog2ProbabilityLevels - tile_max1;
#pragma unroll
  for (int group = 0; group < AttentionConfig::kKeyGroups; ++group) {
    float p[4][4];
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const int j = group * 4 + sub;
      if constexpr (!kMaskTail) {
        p[sub][0] = fast_exp2(fmaf(__int2float_rn(s[j][0]), c0, bias0));
        p[sub][1] = fast_exp2(fmaf(__int2float_rn(s[j][1]), c0, bias0));
        p[sub][2] = fast_exp2(fmaf(__int2float_rn(s[j][2]), c1, bias1));
        p[sub][3] = fast_exp2(fmaf(__int2float_rn(s[j][3]), c1, bias1));
      } else {
        p[sub][0] = fast_exp2(x[j][0] + bias0);
        p[sub][1] = fast_exp2(x[j][1] + bias0);
        p[sub][2] = fast_exp2(x[j][2] + bias1);
        p[sub][3] = fast_exp2(x[j][3] + bias1);
      }
    }
    // A fragment: reg0 row r0 virtual k 4t..4t+3 <- keys {2t,2t+1} of block
    // 4g and of block 4g+1; reg1 row r1; reg2/reg3 virtual k 16+4t.. from
    // blocks 4g+2, 4g+3. V^T is stored with the matching key permutation.
    probabilities[group][0] = pack_u8x4_rn(p[0][0], p[0][1], p[1][0], p[1][1]);
    probabilities[group][1] = pack_u8x4_rn(p[0][2], p[0][3], p[1][2], p[1][3]);
    probabilities[group][2] = pack_u8x4_rn(p[2][0], p[2][1], p[3][0], p[3][1]);
    probabilities[group][3] = pack_u8x4_rn(p[2][2], p[2][3], p[3][2], p[3][3]);
  }
}

__global__ void __launch_bounds__(AttentionConfig::kThreads, 1) attention_int8_sm86(
    const __nv_bfloat16* __restrict__ q, const int8_t* __restrict__ k8,
    const float* __restrict__ k_scale, const int8_t* __restrict__ v8,
    const float* __restrict__ v_scale, __nv_bfloat16* __restrict__ output,
    int heads, int sequence, int padded_sequence, float attention_scale,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_s,
    int64_t out_stride_b, int64_t out_stride_h, int64_t out_stride_s) {
  using Config = AttentionConfig;
  constexpr int kMTiles = Config::kMTiles;
  constexpr int kBlocks = Config::kKeyBlocks;
  constexpr int kGroups = Config::kKeyGroups;
  constexpr int kKeyTile = Config::kKeyTile;

  extern __shared__ __align__(128) unsigned char shared[];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t4 = lane & 3;
  const int head = blockIdx.y;
  const int batch_index = blockIdx.z;
  const int cta_query0 = blockIdx.x * Config::kQueries;
  const int warp_row0 = warp * Config::kRowsPerWarp;
  const int64_t bh = static_cast<int64_t>(batch_index) * heads + head;
  const __nv_bfloat16* q_head = q + static_cast<int64_t>(batch_index) * q_stride_b +
                                static_cast<int64_t>(head) * q_stride_h;
  const int8_t* k_head = k8 + bh * padded_sequence * kHeadDim;
  const int8_t* v_head = v8 + bh * kHeadDim * padded_sequence;
  const float* k_scale_head = k_scale + bh * (padded_sequence / kScaleTile);
  const float* v_scale_head = v_scale + bh * kHeadDim;
  __nv_bfloat16* out_head = output + static_cast<int64_t>(batch_index) * out_stride_b +
                            static_cast<int64_t>(head) * out_stride_h;
  // ---- prologue: stage tile 0, rotate+quantize this warp's 32 Q rows ----
  stage_key_tile(shared, 0, k_head, v_head, 0, padded_sequence, tid);
  cp_async_commit();

  unsigned char* q_shared = shared;
  float qs[kMTiles][2] = {{1.0f, 1.0f}, {1.0f, 1.0f}};
#pragma unroll 4
  for (int r = 0; r < Config::kRowsPerWarp; ++r) {
    const int local_row = warp_row0 + r;
    const int query = cta_query0 + local_row;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (query < sequence) {
      load_bf16x4(q_head + static_cast<int64_t>(query) * q_stride_s + lane * 4, v);
    }
    float scale;
    const uint32_t packed = rotate_quantize_row(v, lane, scale);
    *reinterpret_cast<uint32_t*>(
        q_shared + local_row * 128 +
        swizzle_chunk<128>(local_row, lane >> 2) * 16 + t4 * 4) = packed;
    if (r == g) qs[0][0] = scale;
    if (r == g + 8) qs[0][1] = scale;
    if (r == g + 16) qs[1][0] = scale;
    if (r == g + 24) qs[1][1] = scale;
  }

  // Per-row log2-domain factor without the per-tile K scale.
  float qc[kMTiles][2];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    qc[mt][0] = qs[mt][0] * attention_scale * kLog2e;
    qc[mt][1] = qs[mt][1] * attention_scale * kLog2e;
  }
  const int a_lane_offset = ldmatrix_a_lane_offset(lane);
  const int k_lane_offset = ldmatrix_b_lane_offset<128>(lane);
  const int v_lane_offset = ldmatrix_b_lane_offset<Config::kVRowBytes>(lane);
  const uint32_t q_warp_address = shared_address(q_shared + warp_row0 * 128);

  float o[kMTiles][16][4];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      o[mt][j][0] = 0.0f;
      o[mt][j][1] = 0.0f;
      o[mt][j][2] = 0.0f;
      o[mt][j][3] = 0.0f;
    }
  }
  float m[kMTiles][2] = {{-CUDART_INF_F, -CUDART_INF_F}, {-CUDART_INF_F, -CUDART_INF_F}};
  float d[kMTiles][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  cp_async_wait<0>();
  __syncthreads();

  // ---- main loop over 64-key tiles ----
  // Only the last tile with keys can be partial (S_pad - S < 128), so it is
  // peeled out of the hot loop; the loop body carries no tail masking.
  const int tiles_with_keys = (sequence + kKeyTile - 1) / kKeyTile;
  auto process_tile = [&](const int tile, auto mask_tag) {
    constexpr bool kMaskTail = decltype(mask_tag)::value;
    const int buffer = tile & 1;
    const int key_start = tile * kKeyTile;
    if (tile + 1 < tiles_with_keys) {
      stage_key_tile(shared, buffer ^ 1, k_head, v_head, key_start + kKeyTile,
                     padded_sequence, tid);
      cp_async_commit();
    }
    const float k_tile_scale = __ldg(k_scale_head + (key_start / kScaleTile));
    const uint32_t k_tile_address =
        shared_address(shared + Config::kKOffset + buffer * Config::kKBytes);
    const uint32_t v_tile_address =
        shared_address(shared + Config::kVOffset + buffer * Config::kVBytes);

    // QK^T: 2 m-tiles x 64 keys, s32. Each K fragment feeds four MMAs.
    int scores[kMTiles][kBlocks][4];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
      for (int j = 0; j < kBlocks; ++j) {
        scores[mt][j][0] = 0;
        scores[mt][j][1] = 0;
        scores[mt][j][2] = 0;
        scores[mt][j][3] = 0;
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint32_t q_fragment[kMTiles][4];
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        ldmatrix_x4(q_warp_address + mt * 16 * 128 + (a_lane_offset ^ (32 * c)),
                    q_fragment[mt]);
      }
#pragma unroll
      for (int jj = 0; jj < kBlocks / 2; ++jj) {
        uint32_t k_fragment[4];
        ldmatrix_x4(k_tile_address + jj * 16 * 128 + (k_lane_offset ^ (32 * c)),
                    k_fragment);
#pragma unroll
        for (int mt = 0; mt < kMTiles; ++mt) {
          mma_s8s8(scores[mt][2 * jj], q_fragment[mt], k_fragment[0], k_fragment[1]);
          mma_s8s8(scores[mt][2 * jj + 1], q_fragment[mt], k_fragment[2],
                   k_fragment[3]);
        }
      }
    }

    // Online softmax per m-tile -> u8 probabilities in P.V A-fragment layout.
    float alpha[kMTiles][2];
    float ps[kMTiles][2];
    uint32_t probabilities[kMTiles][kGroups][4];
    const int valid_keys = sequence - key_start;
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      const float c0 = qc[mt][0] * k_tile_scale;
      const float c1 = qc[mt][1] * k_tile_scale;
      softmax_mtile<kMaskTail>(scores[mt], c0, c1, m[mt][0], m[mt][1],
                               alpha[mt][0], alpha[mt][1], ps[mt][0], ps[mt][1],
                               probabilities[mt], valid_keys, t4);
    }

    // Row sums of the u8 probabilities (dp4a). Each thread keeps its own
    // partial over its 16 keys per row; the quad is reduced in the epilogue
    // (alpha is the same for all four lanes of a row, so this commutes).
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      unsigned int sum0 = 0u;
      unsigned int sum1 = 0u;
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        sum0 = __dp4a(probabilities[mt][group][0], kOnesU8x4, sum0);
        sum0 = __dp4a(probabilities[mt][group][2], kOnesU8x4, sum0);
        sum1 = __dp4a(probabilities[mt][group][1], kOnesU8x4, sum1);
        sum1 = __dp4a(probabilities[mt][group][3], kOnesU8x4, sum1);
      }
      d[mt][0] = fmaf(d[mt][0], alpha[mt][0],
                      __int2float_rn(static_cast<int>(sum0)) * ps[mt][0]);
      d[mt][1] = fmaf(d[mt][1], alpha[mt][1],
                      __int2float_rn(static_cast<int>(sum1)) * ps[mt][1]);
    }

    // Lazy rescale of the output accumulators.
    const bool rescale = (alpha[0][0] != 1.0f) || (alpha[0][1] != 1.0f) ||
                         (alpha[1][0] != 1.0f) || (alpha[1][1] != 1.0f);
    if (__any_sync(0xffffffffu, rescale)) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          o[mt][j][0] *= alpha[mt][0];
          o[mt][j][1] *= alpha[mt][0];
          o[mt][j][2] *= alpha[mt][1];
          o[mt][j][3] *= alpha[mt][1];
        }
      }
    }

    // P.V in four 32-feature quarters; each V fragment feeds four MMAs.
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      int pv[kMTiles][4][4];
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          pv[mt][j][0] = 0;
          pv[mt][j][1] = 0;
          pv[mt][j][2] = 0;
          pv[mt][j][3] = 0;
        }
      }
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          uint32_t v_fragment[4];
          ldmatrix_x4(v_tile_address +
                          (quarter * 2 + jj) * 16 * Config::kVRowBytes +
                          (v_lane_offset ^ (32 * group)),
                      v_fragment);
#pragma unroll
          for (int mt = 0; mt < kMTiles; ++mt) {
            mma_u8s8(pv[mt][2 * jj], probabilities[mt][group], v_fragment[0],
                     v_fragment[1]);
            mma_u8s8(pv[mt][2 * jj + 1], probabilities[mt][group], v_fragment[2],
                     v_fragment[3]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int block = quarter * 4 + j;
          o[mt][block][0] = fmaf(__int2float_rn(pv[mt][j][0]), ps[mt][0], o[mt][block][0]);
          o[mt][block][1] = fmaf(__int2float_rn(pv[mt][j][1]), ps[mt][0], o[mt][block][1]);
          o[mt][block][2] = fmaf(__int2float_rn(pv[mt][j][2]), ps[mt][1], o[mt][block][2]);
          o[mt][block][3] = fmaf(__int2float_rn(pv[mt][j][3]), ps[mt][1], o[mt][block][3]);
        }
      }
    }

    cp_async_wait<0>();
    __syncthreads();
  };

  for (int tile = 0; tile + 1 < tiles_with_keys; ++tile) {
    process_tile(tile, BoolTag<false>{});
  }
  if (sequence - (tiles_with_keys - 1) * kKeyTile < kKeyTile) {
    process_tile(tiles_with_keys - 1, BoolTag<true>{});
  } else {
    process_tile(tiles_with_keys - 1, BoolTag<false>{});
  }

  // Finish the row sums: reduce the per-lane partials across the quad.
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float value = d[mt][r];
      value += __shfl_xor_sync(0xffffffffu, value, 1);
      value += __shfl_xor_sync(0xffffffffu, value, 2);
      d[mt][r] = value;
    }
  }

  // ---- epilogue: O * v_scale[channel] / d ----
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    const float inv0 = 1.0f / d[mt][0];
    const float inv1 = 1.0f / d[mt][1];
    const int query0 = cta_query0 + warp_row0 + mt * 16 + g;
    const int query1 = query0 + 8;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int feature = j * 8 + t4 * 2;
      const float2 vs = __ldg(reinterpret_cast<const float2*>(v_scale_head + feature));
      if (query0 < sequence) {
        *reinterpret_cast<uint32_t*>(
            out_head + static_cast<int64_t>(query0) * out_stride_s + feature) =
            pack_bf16x2(o[mt][j][0] * vs.x * inv0, o[mt][j][1] * vs.y * inv0);
      }
      if (query1 < sequence) {
        *reinterpret_cast<uint32_t*>(
            out_head + static_cast<int64_t>(query1) * out_stride_s + feature) =
            pack_bf16x2(o[mt][j][2] * vs.x * inv1, o[mt][j][3] * vs.y * inv1);
      }
    }
  }
}

#endif  // SM86

}  // namespace

namespace dif::runtime::h3_owned_attention {

// ---------------------------------------------------------------------------
// Host entry points (ABI v4 shapes, in-tree)
// ---------------------------------------------------------------------------

int abi_version() { return kAbiVersion; }

int target_sm() {
  return DIF_H3_OWNED_ATTENTION_TARGET_SM;
}

const char *cuda_error(int status) {
  return cudaGetErrorString(static_cast<cudaError_t>(status));
}

// k8: [B,H,S_pad,128] int8 (rotated rows, one scale per 128-key tile);
// k_scale: [B,H,S_pad/128] f32; v8: [B,H,128,S_pad] int8 (permuted keys);
// v_scale: [B,H,128] f32 (per channel per head).
namespace {
int quantize_kv_impl(
    const void* k, const void* v, void* k8, void* k_scale, void* v8,
    void* v_scale, void* k_mean_partials, void* k_mean, int batch, int heads,
    int sequence, int padded_sequence, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_s, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_s, void* stream) {
  if (!k || !v || !k8 || !k_scale || !v8 || !v_scale || batch <= 0 ||
      heads <= 0 || sequence <= 0 || padded_sequence < sequence ||
      padded_sequence % kScaleTile != 0 || k_stride_b <= 0 || k_stride_h <= 0 ||
      k_stride_s <= 0 || v_stride_b <= 0 || v_stride_h <= 0 || v_stride_s <= 0 ||
      (k_stride_b | k_stride_h | k_stride_s | v_stride_b | v_stride_h |
       v_stride_s) % 2 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const int key_tiles = padded_sequence / kScaleTile;
  const dim3 tile_grid(key_tiles, heads, batch);
  cudaError_t status = cudaSuccess;
  if (k_mean != nullptr) {
    if (k_mean_partials == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    k_tile_channel_sums_bf16<<<tile_grid, kHeadDim, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(k), static_cast<float*>(k_mean_partials),
        heads, sequence, padded_sequence, k_stride_b, k_stride_h, k_stride_s);
    status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    k_mean_finalize<<<batch * heads, kHeadDim, 0, cuda_stream>>>(
        static_cast<const float*>(k_mean_partials), static_cast<float*>(k_mean),
        key_tiles, sequence);
    status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
  }
  quantize_k_tiles_h128_bf16<<<tile_grid, kKQuantThreads, 0, cuda_stream>>>(
      static_cast<const __nv_bfloat16*>(k), static_cast<int8_t*>(k8),
      static_cast<float*>(k_scale), static_cast<const float*>(k_mean), heads,
      sequence, padded_sequence, k_stride_b, k_stride_h, k_stride_s);
  status = cudaGetLastError();
  if (status != cudaSuccess) return static_cast<int>(status);

  const size_t scale_bytes =
      static_cast<size_t>(batch) * heads * kHeadDim * sizeof(float);
  status = cudaMemsetAsync(v_scale, 0, scale_bytes, cuda_stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  v_channel_absmax_bf16<<<tile_grid, kVQuantThreads, 0, cuda_stream>>>(
      static_cast<const __nv_bfloat16*>(v), static_cast<unsigned int*>(v_scale),
      heads, sequence, v_stride_b, v_stride_h, v_stride_s);
  status = cudaGetLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  finalize_v_scale<<<batch * heads, kHeadDim, 0, cuda_stream>>>(
      static_cast<float*>(v_scale));
  status = cudaGetLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaFuncSetAttribute(quantize_v_tiles_bf16,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kVQuantSharedBytes);
  if (status != cudaSuccess) return static_cast<int>(status);
  quantize_v_tiles_bf16<<<tile_grid, kVQuantThreads, kVQuantSharedBytes, cuda_stream>>>(
      static_cast<const __nv_bfloat16*>(v), static_cast<int8_t*>(v8),
      static_cast<const float*>(v_scale), heads, sequence, padded_sequence,
      v_stride_b, v_stride_h, v_stride_s);
  return static_cast<int>(cudaGetLastError());
}
}  // namespace

int quantize_kv_bf16(
    const void* k, const void* v, void* k8, void* k_scale, void* v8,
    void* v_scale, int batch, int heads, int sequence, int padded_sequence,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_s,
    void* stream) {
  return quantize_kv_impl(k, v, k8, k_scale, v8, v_scale, nullptr, nullptr,
                          batch, heads, sequence, padded_sequence, k_stride_b,
                          k_stride_h, k_stride_s, v_stride_b, v_stride_h,
                          v_stride_s, stream);
}

int quantize_kv_centered_bf16(
    const void* k, const void* v, void* k8, void* k_scale, void* v8,
    void* v_scale, void* k_mean_partials, void* k_mean, int batch, int heads,
    int sequence, int padded_sequence, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_s, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_s, void* stream) {
  if (!k_mean_partials || !k_mean) return static_cast<int>(cudaErrorInvalidValue);
  return quantize_kv_impl(k, v, k8, k_scale, v8, v_scale, k_mean_partials,
                          k_mean, batch, heads, sequence, padded_sequence,
                          k_stride_b, k_stride_h, k_stride_s, v_stride_b,
                          v_stride_h, v_stride_s, stream);
}

int attention_bf16(
    const void* q, const void* k8, const void* k_scale, const void* v8,
    const void* v_scale, void* output, int batch, int heads, int sequence,
    int padded_sequence, float attention_scale, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_s, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_s, void* stream) {
  if (!q || !k8 || !k_scale || !v8 || !v_scale || !output || batch <= 0 ||
      heads <= 0 || sequence <= 0 || padded_sequence < sequence ||
      padded_sequence % kScaleTile != 0 || !std::isfinite(attention_scale) ||
      attention_scale <= 0.0f || q_stride_b <= 0 || q_stride_h <= 0 ||
      q_stride_s <= 0 || out_stride_b <= 0 || out_stride_h <= 0 ||
      out_stride_s <= 0 ||
      (q_stride_b | q_stride_h | q_stride_s | out_stride_b | out_stride_h |
       out_stride_s) % 2 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
#if DIF_H3_OWNED_ATTENTION_TARGET_SM != 86
  return static_cast<int>(cudaErrorNotSupported);
#else
  using Config = AttentionConfig;
  auto kernel = attention_int8_sm86;
  constexpr int kSharedBytes = Config::kSharedBytes;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (status != cudaSuccess) return static_cast<int>(status);
  const dim3 grid((sequence + Config::kQueries - 1) / Config::kQueries, heads,
                  batch);
  kernel<<<grid, Config::kThreads, kSharedBytes,
           static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const int8_t*>(k8),
      static_cast<const float*>(k_scale), static_cast<const int8_t*>(v8),
      static_cast<const float*>(v_scale), static_cast<__nv_bfloat16*>(output),
      heads, sequence, padded_sequence, attention_scale, q_stride_b, q_stride_h,
      q_stride_s, out_stride_b, out_stride_h, out_stride_s);
  return static_cast<int>(cudaGetLastError());
#endif
}

// Kept for ABI continuity with the v3 status reporting: warps, key tile,
// whether Q lives in registers (0: re-read from shared memory each tile).
int attention_config(
    int* warps, int* key_tile, int* q_in_registers) {
  if (!warps || !key_tile || !q_in_registers) return static_cast<int>(cudaErrorInvalidValue);
  *warps = 8;
  *key_tile = 64;
  *q_in_registers = 0;
  return 0;
}

// Queries per CTA, query rows per warp, key tile, K scale tile (keys),
// V scale granularity (0 = per channel per head).
int attention_geometry(
    int* queries_per_cta, int* rows_per_warp, int* key_tile, int* k_scale_keys,
    int* v_scale_keys) {
  if (!queries_per_cta || !rows_per_warp || !key_tile || !k_scale_keys ||
      !v_scale_keys) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *queries_per_cta = 256;
  *rows_per_warp = 32;
  *key_tile = 64;
  *k_scale_keys = kScaleTile;
  *v_scale_keys = 0;
  return 0;
}

}  // namespace dif::runtime::h3_owned_attention
