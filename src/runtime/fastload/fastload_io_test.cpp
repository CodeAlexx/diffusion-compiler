#include "dif/runtime/fastload.h"

#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void check(CUresult result) {
  require(result == CUDA_SUCCESS, "CUDA call failed");
}
constexpr std::uint64_t mib = 1U << 20U;
std::uint64_t page_end(std::uint64_t value) {
  return (value + 4095U) & ~4095ULL;
}
} // namespace

int main() {
  if (!serenity_fastload_cpu_supported() || cuInit(0) != CUDA_SUCCESS)
    return 77;
  CUdevice device;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS)
    return 77;
  CUcontext context;
  check(cuDevicePrimaryCtxRetain(&context, device));
  check(cuCtxSetCurrent(context));
  CUstream stream;
  check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  char directory[] = "/tmp/dif-fastload-io-XXXXXX";
  require(mkdtemp(directory) != nullptr, "mkdtemp failed");
  const auto path = std::filesystem::path(directory) / "spans.bin";
  int fd = -1, direct = -1;
  CUdeviceptr arena = 0;
  int result = 1;
  try {
    std::vector<std::uint8_t> source(96U * mib + 37U);
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = static_cast<std::uint8_t>((i * 37U) ^ (i >> 9U));
    std::vector<serenity_fastload_span> spans;
    std::uint64_t destination = 31U;
    auto add = [&](std::uint64_t offset, std::uint64_t bytes, unsigned convert) {
      spans.push_back({offset, bytes, destination, convert, 0U});
      destination += (convert == 2 ? bytes / 2 : bytes) + 43U;
    };
    add(13U, 101U, 0U);
    add(114U, 73U, 0U); // same page and adjacent payloads
    add(4093U, 17U * mib + 57U, 0U); // unaligned, crosses the 16 MiB boundary
    for (unsigned i = 0; i < 32U; ++i)
      add(20U * mib + i * 2U * mib + 12U, mib + i * 2U, 0U);
    // In-slot F32/F16 conversions, disjoint source ranges on one page.
    add(90U * mib + 12U, 28U, 2U);
    add(90U * mib + 42U, 18U, 1U);
    const float floats[] = {1.F, -2.F, 0.F, 0.1F, 65536.F, -0.25F, 3.F};
    std::memcpy(source.data() + 90U * mib + 12U, floats, sizeof(floats));
    const std::uint16_t halves[] = {0x3c00, 0xc000, 0, 0x3800, 0x4400,
                                  0x8000, 0x3400, 0xbc00, 0x4000};
    std::memcpy(source.data() + 90U * mib + 42U, halves, sizeof(halves));
    add(source.size() - 19U, 19U, 0U); // short aligned read at EOF is legal
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(source.data()), source.size());
    file.close();
    require(!file.fail(), "fixture write failed");
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    direct = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    require(fd >= 0, "fixture open failed");
    check(cuMemAlloc(&arena, destination));
    std::vector<std::uint8_t> host(destination), actual(destination);
    // Independent byte references plus sentinels in every destination gap.
    std::vector<std::uint8_t> expected_host(destination, 0xA5),
        expected_device(destination, 0xA5);
    std::uint64_t read_bound = 0U;
    auto end = std::uint64_t{0U};
    for (std::size_t i = 0; i < spans.size(); ++i) {
      const auto &span = spans[i];
      const auto lo = span.file_off & ~4095ULL;
      const auto hi = std::min<std::uint64_t>(source.size(),
                                            page_end(span.file_off + span.nbytes));
      read_bound += hi - std::max<std::uint64_t>(lo, end);
      end = hi;
      auto &expected = i % 2 ? expected_host : expected_device;
      if (span.convert == 0) {
        std::memcpy(expected.data() + span.dst_off,
                    source.data() + span.file_off, span.nbytes);
      } else if (span.convert == 2) {
        for (std::size_t j = 0; j < span.nbytes / 4U; ++j) {
          std::uint32_t bits;
          std::memcpy(&bits, source.data() + span.file_off + j * 4U, 4U);
          const auto value = static_cast<std::uint16_t>(
              (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
          std::memcpy(expected.data() + span.dst_off + j * 2U, &value, 2U);
        }
      } else {
        const std::uint16_t reference[] = {0x3f80, 0xc000, 0, 0x3f00, 0x4080,
                                         0x8000, 0x3e80, 0xbf80, 0x4000};
        std::memcpy(expected.data() + span.dst_off, reference, sizeof(reference));
      }
    }
    for (const auto mode : {0, 2, 1}) {
      if (mode == 1 && direct < 0)
        continue;
      std::fill(host.begin(), host.end(), 0xA5);
      check(cuMemsetD8(arena, 0xA5, destination));
      auto request_spans = spans;
      for (std::size_t i = 0; i < spans.size(); ++i) {
        request_spans[i].host = i % 2;
        request_spans[i].dst_off += i % 2
            ? reinterpret_cast<std::uintptr_t>(host.data()) : arena;
      }
      serenity_fastload_stats stats{};
      serenity_fastload_request request{};
      request.fd = fd;
      request.fd_direct = direct;
      request.spans = request_spans.data();
      request.nspans = request_spans.size();
      request.cuda_stream = stream;
      request.mode = mode;
      request.stats = &stats;
      require(serenity_fastload_run(&request) == 0, "sparse load failed");
      check(cuMemcpyDtoH(actual.data(), arena, destination));
      require(actual == expected_device, "device bytes or gap sentinels differ");
      require(host == expected_host, "host bytes or gap sentinels differ");
      require(stats.bytes_read == read_bound, "loader read unrequested gaps");
      if (mode == 0)
        require(stats.total_pages == page_end(read_bound) / 4096U &&
                stats.resident_pages == stats.total_pages && stats.mode_used == 2,
                "residency probe scanned gaps or double-counted shared pages");
      require(stats.chunks > 8, "fixture must recycle every ring slot");
      std::printf("sparse mode=%d spans=%zu chunks=%llu bytes_read=%llu PASS\n",
                  mode, spans.size(), (unsigned long long)stats.chunks,
                  (unsigned long long)stats.bytes_read);
      auto invalid = request_spans.back();
      invalid.file_off = source.size() + 4096U;
      request.spans = &invalid;
      request.nspans = 1;
      require(serenity_fastload_run(&request) == -5, "out-of-file span accepted");
      invalid.file_off = UINT64_MAX - 4U;
      require(serenity_fastload_run(&request) == -7, "overflowing span accepted");
    }
    // A tiny warm selection in an otherwise cold shard must use the cache.
    // The unselected pages must affect neither the decision nor probe cost.
    require(fsync(fd) == 0, "fixture fsync failed");
    require(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0,
            "fixture eviction failed");
    std::uint8_t byte{};
    require(pread(fd, &byte, 1, 13) == 1, "warm selection failed");
    auto selected = spans.front();
    selected.dst_off += arena;
    serenity_fastload_stats stats{};
    serenity_fastload_request request{};
    request.fd = fd;
    request.fd_direct = direct;
    request.spans = &selected;
    request.nspans = 1;
    request.cuda_stream = stream;
    request.stats = &stats;
    require(serenity_fastload_run(&request) == 0 && stats.total_pages == 1 &&
            stats.resident_pages == 1 && stats.mode_used == 2,
            "sparse warm selection was judged by unrelated cold pages");
    if (direct >= 0) {
      require(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0,
              "cold selection eviction failed");
      require(serenity_fastload_run(&request) == 0 && stats.total_pages == 1 &&
              stats.resident_pages == 0 && stats.mode_used == 1,
              "sparse cold selection did not use direct IO");
    }
    result = 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
  }
  if (arena) (void)cuMemFree(arena);
  if (direct >= 0) close(direct);
  if (fd >= 0) close(fd);
  std::filesystem::remove(path);
  std::filesystem::remove(directory);
  (void)cuStreamDestroy(stream);
  (void)cuDevicePrimaryCtxRelease(device);
  return result;
}
