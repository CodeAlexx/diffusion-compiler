#include "resident_prefetch.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

namespace dif::runtime::fastload {
namespace {
void check(CUresult rc, const char *action) {
  if (rc != CUDA_SUCCESS)
    fail(std::string(action) + ": CUDA error " + std::to_string(rc));
}
double since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}
} // namespace

ResidentPrefetch::ResidentPrefetch(CUcontext context,
                                  std::vector<ResidentBatch> batches,
                                  std::size_t lookahead)
    : context_(context), batches_(std::move(batches)),
      ready_(batches_.size()), status_(batches_.size()),
      direct_(batches_.size()), lookahead_(lookahead) {
  if (!context || lookahead == 0)
    fail("resident prefetch requires a context and positive lookahead");
}

ResidentPrefetch::~ResidentPrefetch() {
  stop(); // no destinations, descriptors or context may die before this join
  if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
    for (auto event : ready_)
      if (event) (void)cuEventDestroy(event);
    if (stream_) (void)cuStreamDestroy(stream_);
    CUcontext previous{};
    (void)cuCtxPopCurrent(&previous);
  }
}

void ResidentPrefetch::start() {
  std::lock_guard lock(mutex_);
  if (started_ || stopping_) return;
  allowed_ = std::min(lookahead_, batches_.size());
  worker_ = std::thread([this] { work(); });
  started_ = true;
}

ResidentPrefetch::Ready ResidentPrefetch::wait(std::size_t batch) {
  const auto begin = std::chrono::steady_clock::now();
  std::unique_lock lock(mutex_);
  if (!started_ || batch >= batches_.size())
    fail("invalid resident prefetch wait");
  allowed_ = std::max(allowed_, batch +
      std::min(lookahead_, batches_.size() - batch));
  changed_.notify_all();
  changed_.wait(lock, [&] { return status_[batch] || fatal_ || stopping_; });
  receipt_.wait_ms += since(begin);
  if (fatal_) std::rethrow_exception(fatal_);
  return status_[batch] == 1 ? Ready{ready_[batch], direct_[batch]} : Ready{};
}

PrefetchReceipt ResidentPrefetch::take_receipt() {
  std::lock_guard lock(mutex_);
  if (fatal_) std::rethrow_exception(fatal_);
  return std::exchange(receipt_, {});
}

void ResidentPrefetch::stop() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void ResidentPrefetch::work() noexcept {
  bool pushed = false;
  try {
    check(cuCtxPushCurrent(context_), "resident prefetch context");
    pushed = true;
    check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
          "resident prefetch stream");
    for (std::size_t i = 0; i < batches_.size(); ++i) {
      {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return stopping_ || i < allowed_; });
        if (stopping_) break;
      }
      check(cuEventCreate(&ready_[i], CU_EVENT_DISABLE_TIMING),
            "resident prefetch event");
      const auto &batch = batches_[i];
      serenity_fastload_request request{};
      request.fd = batch.fd;
      request.fd_direct = batch.direct_fd;
      request.spans = batch.spans.data();
      request.nspans = batch.spans.size();
      request.cuda_stream = stream_;
      request.mode = 0;
      serenity_fastload_stats stats{};
      request.stats = &stats;
      const auto begin = std::chrono::steady_clock::now();
      const int rc = serenity_fastload_run(&request);
      // A refused/partial read must finish any issued DMA before the owner
      // overwrites its destinations through the fallback. Never use global
      // executor telemetry here: those counters belong to the consumer thread.
      if (rc != 0)
        check(cuStreamSynchronize(stream_), "resident prefetch failure drain");
      else
        check(cuEventRecord(ready_[i], stream_), "resident prefetch ready");
      const auto elapsed = since(begin);
      {
        std::lock_guard lock(mutex_);
        receipt_.load_ms += elapsed;
        if (rc != 0) {
          ++receipt_.failures;
          std::fill(status_.begin() + static_cast<std::ptrdiff_t>(i),
                    status_.end(), 2);
          changed_.notify_all();
          break;
        }
        std::uint64_t bytes = 0;
        for (const auto &span : batch.spans) bytes += span.nbytes;
        receipt_.bytes += bytes;
        if (stats.mode_used == 1) receipt_.direct_bytes += bytes;
        ++receipt_.batches;
        receipt_.copies += stats.h2d_copies;
        direct_[i] = stats.mode_used == 1;
        status_[i] = 1;
      }
      changed_.notify_all();
    }
    check(cuStreamSynchronize(stream_), "resident prefetch final drain");
  } catch (...) {
    std::lock_guard lock(mutex_);
    fatal_ = std::current_exception();
    changed_.notify_all();
  }
  if (pushed) {
    CUcontext previous{};
    (void)cuCtxPopCurrent(&previous);
  }
}
} // namespace dif::runtime::fastload
