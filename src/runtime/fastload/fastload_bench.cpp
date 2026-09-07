// fastload_bench.cpp -- gate and benchmark for the assembler loader.
//
// Loads a checkpoint through serenity_fastload_run into one device arena,
// prints the timing, then (unless told not to) compares EVERY device byte
// against a plain pread of the file (converted by the scalar reference for
// F16/F32 spans). This is the correctness gate for fastload_ring.S.
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/local/cuda-12.4/include fastload_bench.cpp \
//       fastload_x86.S fastload_ring.S -L/usr/local/cuda-12.4/lib64 -lcudart -lcuda \
//       -o /tmp/fastload_bench
// Run:
//   python3 fastload_spans.py <ckpt.safetensors> /tmp/spans
//   /tmp/fastload_bench <ckpt.safetensors> /tmp/spans.bin
//       <mode 0|1|2|current-raw|current-host|streamed-host> [noverify|host] [raw]
// `current-raw` reproduces the compiler's mapped prefetch + pageable H2D path
// and ignores conversion spans, providing a like-for-like transport A/B.
// `current-host` reproduces the streamed prefetcher's per-tensor O_DIRECT
// staging boundary and writes into the same ordinary host arena as `1 host`.
// `streamed-host` calls fastload once per tensor, matching how the generic
// streamed prefetcher stages large tensors without reading gaps between them.
// DIF_FASTLOAD_EVICT=1 asks the kernel to discard the file cache first.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "dif/runtime/fastload.h"
#include "dif/runtime/tensor.hpp"

static uint16_t ref_f32_to_bf16(uint32_t bits) {
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}
static uint32_t ref_f16_to_f32_bits(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  if (exp == 0) {
    if (man == 0) return sign << 31;
    int e = -1; uint32_t m = man;
    do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
    return (sign << 31) | ((127u - 15u - e) << 23) | ((m & 0x3FFu) << 13);
  }
  if (exp == 31) return (sign << 31) | 0x7F800000u | (man << 13);
  return (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)
#define CUK(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char *s_ = nullptr; cuGetErrorString(e_, &s_); std::printf("CUDA driver error %s at %s:%d\n", s_ ? s_ : "unknown", __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
  if (argc < 4) { std::printf("usage: fastload_bench <ckpt> <spans.bin> <mode 0|1|2|current-raw|current-host|streamed-host> [noverify|host] [raw]\n"); return 2; }
  const char *path = argv[1];
  const std::string_view mode_name = argv[3];
  const bool current_device = mode_name == "current-raw";
  const bool current_host = mode_name == "current-host";
  const bool streamed_host = mode_name == "streamed-host";
  const bool current = current_device || current_host;
  const int mode = current || streamed_host ? -1 : std::atoi(argv[3]);
  if (!current && !streamed_host &&
      (mode_name.size() != 1U || mode < 0 || mode > 2)) {
    std::printf("mode must be 0, 1, 2, current-raw, current-host, or streamed-host\n"); return 2;
  }
  if (!serenity_fastload_cpu_supported()) {
    std::printf("fastload requires AVX2 and F16C\n");
    return 77;
  }
  bool verify = true;
  bool to_host = current_host || streamed_host;
  bool raw_only = current || streamed_host;
  for (int index = 4; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "noverify")
      verify = false;
    else if (option == "host")
      to_host = true;
    else if (option == "raw")
      raw_only = true;
    else {
      std::printf("unknown option %s\n", argv[index]); return 2;
    }
  }
  if (current_device && to_host) {
    std::printf("current-raw is a device transport comparator; host is unsupported\n");
    return 2;
  }

  std::vector<serenity_fastload_span> spans;
  {
    FILE *f = std::fopen(argv[2], "rb");
    if (!f) { std::printf("cannot open %s\n", argv[2]); return 2; }
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1) return 2;
    spans.resize(n);
    if (std::fread(spans.data(), sizeof(serenity_fastload_span), n, f) != n) return 2;
    std::fclose(f);
  }
  std::uint64_t ignored_spans = 0U, ignored_bytes = 0U;
  if (raw_only) {
    std::vector<serenity_fastload_span> raw;
    raw.reserve(spans.size());
    for (const auto &span : spans) {
      if (span.convert == 0U)
        raw.push_back(span);
      else {
        ++ignored_spans;
        ignored_bytes += span.nbytes;
      }
    }
    spans = std::move(raw);
  }
  uint64_t arena_bytes = 0, file_bytes = 0, dev_bytes = 0;
  for (const auto &s : spans) {
    const uint64_t dev = s.convert == 2 ? s.nbytes / 2 : s.nbytes;
    arena_bytes = std::max(arena_bytes, s.dst_off + dev);
    file_bytes += s.nbytes; dev_bytes += dev;
  }
  arena_bytes = (arena_bytes + 4095) & ~4095ull;

  void *arena = nullptr;
  std::vector<uint8_t> host_arena;
  if (to_host) {
    host_arena.assign(arena_bytes, 0xA5);
    arena = host_arena.data();
    for (auto &s : spans) s.host = 1;
  } else {
    CK(cudaMalloc(&arena, arena_bytes));
    CK(cudaMemset(arena, 0xA5, arena_bytes));  // poison: a missed byte will show
  }
  cudaStream_t stream;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  serenity_fastload_stats st{};
  char err[256] = {0};
  const auto t0 = std::chrono::steady_clock::now();
  int rc = 0;
  double resident = -1.0;
  if (current) {
    const auto storage = dif::runtime::map_readonly_file(path);
    if (std::getenv("DIF_FASTLOAD_EVICT"))
      storage->evict(0U, storage->size());
    resident = storage->resident_fraction(0U, storage->size());
    if (current_host) {
      for (const auto &span : spans) {
        auto *destination = static_cast<std::uint8_t *>(arena) + span.dst_off;
        if (!storage->read_direct(span.file_off, span.nbytes, destination))
          std::memcpy(destination, storage->data() + span.file_off,
                      span.nbytes);
      }
    } else {
      for (const auto &span : spans)
        storage->prefetch(span.file_off, span.nbytes);
      for (const auto &span : spans) {
        const auto destination =
            reinterpret_cast<CUdeviceptr>(arena) + span.dst_off;
        CUK(cuMemcpyHtoDAsync(destination, storage->data() + span.file_off,
                              span.nbytes, stream));
      }
      CUK(cuStreamSynchronize(stream));
    }
  } else {
    const int fd_plain = open(path, O_RDONLY | O_CLOEXEC);
    const int fd_direct = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd_plain < 0) {
      std::printf("cannot open %s\n", path); return 2;
    }
    if (std::getenv("DIF_FASTLOAD_EVICT"))
      (void)posix_fadvise(fd_plain, 0, 0, POSIX_FADV_DONTNEED);
    if (streamed_host) {
      for (const auto &source_span : spans) {
        auto span = source_span;
        span.dst_off = reinterpret_cast<std::uint64_t>(
            static_cast<std::uint8_t *>(arena) + source_span.dst_off);
        span.host = 1U;
        serenity_fastload_request req{};
        req.fd = fd_plain; req.fd_direct = fd_direct;
        req.spans = &span; req.nspans = 1U;
        req.mode = 1;
        serenity_fastload_stats one{};
        req.stats = &one; req.err = err; req.err_capacity = sizeof err;
        rc = serenity_fastload_run(&req);
        st.bytes_read += one.bytes_read;
        st.chunks += one.chunks;
        st.total_ns += one.total_ns;
        st.mode_used = one.mode_used;
        if (rc != 0) {
          st.error_code = one.error_code;
          break;
        }
      }
    } else {
      serenity_fastload_request req{};
      req.fd = fd_plain; req.fd_direct = fd_direct;
      req.spans = spans.data(); req.nspans = spans.size();
      req.arena_device = arena; req.arena_bytes = arena_bytes;
      req.cuda_stream = stream; req.mode = mode;
      req.stats = &st; req.err = err; req.err_capacity = sizeof err;
      rc = serenity_fastload_run(&req);
    }
    if (fd_direct >= 0) close(fd_direct);
    close(fd_plain);
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("loader=%s rc=%d mode_used=%llu chunks=%llu h2d_copies=%llu bytes_read=%llu resident=%llu/%llu resident_fraction=%.6f ignored_conversion_spans=%llu ignored_conversion_bytes=%llu\n",
              current_host ? "current-host" :
                  current_device ? "current-raw" :
                      streamed_host ? "streamed-host" : "fastload", rc,
              (unsigned long long)st.mode_used,
              (unsigned long long)st.chunks,
              (unsigned long long)st.h2d_copies,
              (unsigned long long)st.bytes_read,
              (unsigned long long)st.resident_pages,
              (unsigned long long)st.total_pages, resident,
              (unsigned long long)ignored_spans,
              (unsigned long long)ignored_bytes);
  std::printf("loader wall %.3f s  (%.2f GB/s of file bytes, %llu spans, %.2f GB to device)\n", wall,
              file_bytes / wall / 1e9, (unsigned long long)spans.size(), dev_bytes / 1e9);
  if (rc != 0) { std::printf("error: %s (code %llu)\n", err, (unsigned long long)st.error_code); return 1; }
  if (!verify) return 0;

  // ---- gate: every device byte against a pread of the file
  const int fd = open(path, O_RDONLY);
  if (fd < 0) { std::printf("cannot open %s for verification\n", path); return 1; }
  std::vector<uint8_t> file_buf, ref_buf, dev_buf;
  uint64_t bad_spans = 0, bad_bytes = 0, checked = 0;
  const auto v0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < spans.size(); ++i) {
    const auto &s = spans[i];
    file_buf.resize(s.nbytes);
    uint64_t got = 0;
    while (got < s.nbytes) {
      const ssize_t r = pread(fd, file_buf.data() + got, s.nbytes - got, s.file_off + got);
      if (r <= 0) { std::printf("pread failed at span %zu\n", i); return 1; }
      got += r;
    }
    const uint8_t *ref = file_buf.data();
    uint64_t dev = s.nbytes;
    if (s.convert == 1) {
      ref_buf.resize(s.nbytes);
      auto *o = reinterpret_cast<uint16_t *>(ref_buf.data());
      const auto *in = reinterpret_cast<const uint16_t *>(file_buf.data());
      for (uint64_t k = 0; k < s.nbytes / 2; ++k) o[k] = ref_f32_to_bf16(ref_f16_to_f32_bits(in[k]));
      ref = ref_buf.data();
    } else if (s.convert == 2) {
      dev = s.nbytes / 2;
      ref_buf.resize(dev);
      auto *o = reinterpret_cast<uint16_t *>(ref_buf.data());
      const auto *in = reinterpret_cast<const uint32_t *>(file_buf.data());
      for (uint64_t k = 0; k < dev / 2; ++k) o[k] = ref_f32_to_bf16(in[k]);
      ref = ref_buf.data();
    }
    dev_buf.resize(dev);
    if (to_host)
      std::memcpy(dev_buf.data(), static_cast<uint8_t *>(arena) + s.dst_off, dev);
    else
      CK(cudaMemcpy(dev_buf.data(), static_cast<uint8_t *>(arena) + s.dst_off, dev, cudaMemcpyDeviceToHost));
    if (std::memcmp(dev_buf.data(), ref, dev) != 0) {
      ++bad_spans;
      for (uint64_t k = 0; k < dev; ++k) bad_bytes += dev_buf[k] != ref[k];
      if (bad_spans <= 5) std::printf("MISMATCH span %zu file_off=%llu nbytes=%llu convert=%u\n", i, (unsigned long long)s.file_off, (unsigned long long)s.nbytes, s.convert);
    }
    checked += dev;
  }
  const double vwall = std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count();
  std::printf("gate: %llu device bytes checked in %.1f s, %llu bad spans, %llu bad bytes -> %s\n",
              (unsigned long long)checked, vwall, (unsigned long long)bad_spans, (unsigned long long)bad_bytes,
              bad_spans ? "FAIL" : "PASS");
  return bad_spans ? 1 : 0;
}
