#pragma once

#include "dif/runtime/fastload.h"
#include <cuda.h>

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace dif::runtime::fastload {

// Model-neutral, immutable file-to-final-device upload jobs. The owner keeps
// descriptors, context and destination allocations alive until destruction.
struct ResidentBatch {
  int fd{-1};
  int direct_fd{-1};
  std::vector<serenity_fastload_span> spans;
};

struct PrefetchReceipt {
  std::uint64_t bytes{}, direct_bytes{}, batches{}, copies{}, failures{};
  double load_ms{}, wait_ms{}; // overlapping worker/consumer clocks, not additive
};

class ResidentPrefetch {
public:
  struct Ready { CUevent event{}; bool direct{}; };
  ResidentPrefetch(CUcontext context, std::vector<ResidentBatch> batches,
                   std::size_t lookahead = 2);
  ~ResidentPrefetch();
  ResidentPrefetch(const ResidentPrefetch &) = delete;
  ResidentPrefetch &operator=(const ResidentPrefetch &) = delete;

  void start();
  // Waits for this batch only; the caller queues a CUDA wait on the returned
  // event before consuming its weights. Null means use the old upload path.
  Ready wait(std::size_t batch);
  PrefetchReceipt take_receipt();
  void stop();

private:
  void work() noexcept;
  CUcontext context_{};
  CUstream stream_{};
  std::vector<ResidentBatch> batches_;
  std::vector<CUevent> ready_;
  std::vector<unsigned char> status_; // 0 pending, 1 ready, 2 fallback
  std::vector<bool> direct_;
  std::size_t lookahead_{}, allowed_{};
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread worker_;
  bool started_{}, stopping_{};
  std::exception_ptr fatal_;
  PrefetchReceipt receipt_;
};
} // namespace dif::runtime::fastload
