#include "dif/runtime/executor.hpp"

#include "dif/support/error.hpp"

namespace dif::runtime {

std::unique_ptr<Executor> make_cuda_executor(int) {
  fail("CUDA backend was not built");
}

bool cuda_available() { return false; }

} // namespace dif::runtime
