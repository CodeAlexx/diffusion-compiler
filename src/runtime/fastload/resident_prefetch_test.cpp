#include "resident_prefetch.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

using dif::runtime::fastload::ResidentBatch;
using dif::runtime::fastload::ResidentPrefetch;
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void check(CUresult rc) { require(rc == CUDA_SUCCESS, "CUDA failed"); }
}

int main() {
  if (!serenity_fastload_cpu_supported() || cuInit(0) != CUDA_SUCCESS) return 77;
  CUdevice device;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 77;
  CUcontext context{};
  check(cuDevicePrimaryCtxRetain(&context, device));
  check(cuCtxSetCurrent(context));
  CUstream consumer{};
  check(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING));
  char directory[] = "/tmp/dif-resident-prefetch-XXXXXX";
  require(mkdtemp(directory), "mkdtemp failed");
  const auto path = std::filesystem::path(directory) / "weights.bin";
  int fd = -1, direct = -1, rc = 1;
  CUdeviceptr arena{};
  try {
    constexpr std::size_t mib = 1U << 20U, jobs = 12, stride = 2 * mib;
    std::vector<unsigned char> source(jobs * stride);
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = static_cast<unsigned char>(i * 19U + (i >> 11U));
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(source.data()), source.size());
    file.close();
    require(!file.fail(), "fixture write failed");
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    direct = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    require(fd >= 0, "open failed");
    require(fsync(fd) == 0, "fsync failed");
    check(cuMemAlloc(&arena, source.size()));
    std::vector<ResidentBatch> batches(jobs);
    std::vector<unsigned char> expected(source.size(), 0xA5), actual(source.size());
    std::uint64_t expected_bytes = 0;
    for (std::size_t i = 0; i < jobs; ++i) {
      auto &batch = batches[i];
      batch.fd = fd;
      batch.direct_fd = direct;
      // Two disjoint, unaligned spans and untouched sentinel gaps per batch.
      for (auto offset : {13U, static_cast<unsigned>(mib + 37U)}) {
        const auto begin = i * stride + offset;
        const auto size = mib / 2U + i;
        batch.spans.push_back({begin, size, arena + begin, 0U, 0U});
        std::copy_n(source.begin() + begin, size, expected.begin() + begin);
        expected_bytes += size;
      }
    }
    for (const bool cold : {false, true}) {
      if (cold && direct < 0) continue;
      if (cold) require(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0,
                        "fixture eviction failed");
      check(cuMemsetD8(arena, 0xA5, source.size()));
      ResidentPrefetch prefetch(context, batches, 2);
      prefetch.start();
      for (std::size_t i = 0; i < jobs; ++i) {
        const auto ready = prefetch.wait(i);
        require(ready.event, "batch fell back");
        check(cuStreamWaitEvent(consumer, ready.event, 0));
        if (i == 0) {
          // Calling start again must not restart or overwrite earlier jobs.
          prefetch.start();
          require(prefetch.wait(i).event == ready.event, "readiness not reused");
        }
      }
      check(cuStreamSynchronize(consumer));
      check(cuMemcpyDtoH(actual.data(), arena, source.size()));
      require(actual == expected, "device bytes or sentinel gaps differ");
      const auto receipt = prefetch.take_receipt();
      require(receipt.bytes == expected_bytes && receipt.batches == jobs &&
              receipt.failures == 0 && receipt.copies >= jobs * 2,
              "receipt does not match one-time uploads");
      require(prefetch.take_receipt().bytes == 0, "receipt counted twice");
      std::printf("resident prefetch cold=%d bytes=%llu batches=%llu PASS\n",
                  cold, (unsigned long long)receipt.bytes,
                  (unsigned long long)receipt.batches);
    }
    {
      ResidentPrefetch cancelled(context, batches, 2);
      cancelled.start();
      require(cancelled.wait(0).event, "cancel test first batch failed");
      cancelled.stop();
      require(cancelled.take_receipt().batches <= 2,
              "prefetch exceeded the demand window");
    }
    {
      auto refused = batches;
      refused[0].spans[0].file_off = source.size() + 4096;
      ResidentPrefetch fallback(context, std::move(refused), 2);
      fallback.start();
      require(!fallback.wait(0).event && !fallback.wait(jobs - 1).event,
              "invalid request published ready or stranded a waiter");
      require(fallback.take_receipt().failures == 1, "missing fallback receipt");
      // A drained refusal leaves destinations safe for the normal uploader.
      check(cuMemcpyHtoDAsync(arena, source.data(), source.size(), consumer));
      check(cuStreamSynchronize(consumer));
      check(cuMemcpyDtoH(actual.data(), arena, source.size()));
      require(actual == source, "fallback upload raced the worker");
    }
    // Destruction without any start is also a normal prepared-only lifetime.
    { ResidentPrefetch unused(context, batches); }
    rc = 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
  }
  if (arena) (void)cuMemFree(arena);
  if (direct >= 0) close(direct);
  if (fd >= 0) close(fd);
  std::filesystem::remove(path);
  std::filesystem::remove(directory);
  (void)cuStreamDestroy(consumer);
  (void)cuDevicePrimaryCtxRelease(device);
  return rc;
}
