// fastload.h -- ABI of the checkpoint loader written in fastload_x86.S.
//
// The loader moves a safetensors data section from disk into ONE device
// arena with the CPU out of the byte path: io_uring reads land in pinned
// slots, each completed slot is DMA'd tensor by tensor to its arena offset,
// and the slot is recycled once the GPU has drained it. Two read modes:
//   direct    O_DIRECT, 16 MiB requests, 8 in flight. The page cache is
//             neither consulted nor filled (nothing charged to the cgroup).
//   buffered  plain reads, for a file whose pages are already resident.
// Mode 0 probes residency of the requested pages with mincore and picks one.
// Reads cover only the requested spans rounded to pages, merging adjacent
// ranges and splitting at 16 MiB. Unselected gaps in a shard are not read.
//
// Checkpoints stored as F16 or F32 are converted to BF16 in the pinned slot
// by the AVX2 kernels in the same file before the DMA; BF16 checkpoints
// never touch a CPU byte loop.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// One tensor (or a piece of one). Spans MUST be sorted by file_off and must
// not overlap. `nbytes` is the byte count IN THE FILE; the device size is
// nbytes for convert 0 and 1, nbytes/2 for convert 2.
//   convert 0: raw bytes           1: F16 -> BF16           2: F32 -> BF16
struct serenity_fastload_span {
  uint64_t file_off;   // absolute offset in the file
  uint64_t nbytes;     // bytes in the file
  uint64_t dst_off;    // byte offset in the device arena, or the absolute
                       // device address when arena_device/arena_bytes are 0
  uint32_t convert;
  uint32_t host;       // 0: DMA to the device; 1: CPU copy to host memory at
                       //    arena + dst_off (conversions apply either way)
};

struct serenity_fastload_stats {
  uint64_t bytes_read;      // bytes delivered by the reads (aligned superset)
  uint64_t chunks;          // read requests issued
  uint64_t mode_used;       // 1 direct, 2 buffered
  uint64_t error_code;      // 0 on success, else the negative return value
  uint64_t total_ns;        // wall time of the call
  uint64_t resident_pages;  // requested-page residency probe (mode 0 only)
  uint64_t total_pages;
  uint64_t h2d_copies;      // actual CUDA copy submissions (chunk splits count)
};

// The assembly uses AVX2 for its byte loops and F16C for half conversion.
// Keep capability selection outside the assembly entry point so unsupported
// x86 hosts take the normal compiler loader instead of trapping with SIGILL.
static inline int serenity_fastload_cpu_supported(void) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
#else
  return 0;
#endif
}

// Production policy: retain the prior loader unless DIF_FASTLOAD=1 explicitly
// opts in to experimental fastload on a supported host. Keep this policy in
// one shared helper so resident, streamed, and tool-side host materialization
// cannot silently acquire different defaults. Subordinate flags cannot opt in.
static inline int serenity_fastload_runtime_enabled(void) {
  const char *toggle;
  if (!serenity_fastload_cpu_supported())
    return 0;
  toggle = getenv("DIF_FASTLOAD");
  return toggle != NULL && strcmp(toggle, "1") == 0;
}

static inline int serenity_fastload_streamed_runtime_enabled(void) {
  const char *toggle;
  if (!serenity_fastload_runtime_enabled())
    return 0;
  toggle = getenv("DIF_FASTLOAD_STREAMED");
  return toggle == NULL || strcmp(toggle, "0") != 0;
}

// Everything the loader needs, in one struct. The caller owns both
// descriptors (either may be -1): `fd` is a plain read descriptor, `fd_direct`
// the same file opened with O_DIRECT. Mode 0 probes residency through `fd`
// and uses `fd_direct` when the file is cold; mode 1 forces direct, mode 2
// forces plain. The loader never opens or closes a file.
// With arena_device == 0 and arena_bytes == 0 every span's dst_off is an
// absolute device address.
struct serenity_fastload_request {
  int64_t fd;
  int64_t fd_direct;
  const struct serenity_fastload_span *spans;
  uint64_t nspans;
  void *arena_device;
  uint64_t arena_bytes;
  void *cuda_stream;   // CUstream
  int64_t mode;
  struct serenity_fastload_stats *stats;  // may be NULL
  char *err;                              // may be NULL
  uint64_t err_capacity;
};

// Returns 0 on success, negative on failure (see error_code / err text):
//  -1 no usable descriptor  -2 io_uring_setup failed  -3 mmap failed
//  -4 read failed           -5 short read inside a span -6 CUDA call failed
//  -7 bad span table        -8 pinned allocation failed -9 span outside arena
// Uses the CUDA driver API (cuMemcpyHtoDAsync, cuMemHostAlloc, events) on the
// caller's current context. Thread-safe: concurrent callers are serialized
// on a process-wide lock (the pinned ring is shared), so their I/O queues.
int serenity_fastload_run(const struct serenity_fastload_request *request);

// The CPU byte loops, exported for their own gate (fastload_test.cpp).
void serenity_f32_to_bf16_rne(const float *src, uint16_t *dst, uint64_t n);
void serenity_f16_to_bf16_rne(const uint16_t *src, uint16_t *dst, uint64_t n);
void serenity_nt_memcpy(void *dst, const void *src, uint64_t nbytes);

#ifdef __cplusplus
}
#endif
