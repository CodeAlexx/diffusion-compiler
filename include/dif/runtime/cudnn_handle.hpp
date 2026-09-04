#pragma once

#include <cudnn.h>

namespace dif::runtime {

// One cuDNN handle per thread, created on first use and kept for the life of
// the thread.
//
// cudnnCreate costs tens of milliseconds and allocates its own internal
// workspace, so a handle per plan made preparation scale with the number of
// distinct convolution and attention shapes: a model with forty convolution
// shapes paid for forty handles before it ran a single kernel. The plans set
// their stream at execute, so sharing one handle changes no result.
cudnnHandle_t shared_cudnn_handle();

} // namespace dif::runtime
