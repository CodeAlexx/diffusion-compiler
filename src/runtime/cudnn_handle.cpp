#include "dif/runtime/cudnn_handle.hpp"

#include "dif/support/error.hpp"

namespace dif::runtime {
namespace {

struct HandleOwner {
  cudnnHandle_t handle{};

  HandleOwner() {
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS)
      fail("cudnnCreate");
  }
  HandleOwner(const HandleOwner &) = delete;
  HandleOwner &operator=(const HandleOwner &) = delete;
  ~HandleOwner() {
    if (handle)
      (void)cudnnDestroy(handle);
  }
};

} // namespace

cudnnHandle_t shared_cudnn_handle() {
  thread_local HandleOwner owner;
  return owner.handle;
}

} // namespace dif::runtime
