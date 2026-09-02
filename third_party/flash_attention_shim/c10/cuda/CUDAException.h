#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#define C10_CUDA_CHECK(expression)                                             \
  do {                                                                         \
    const auto dif_flash_status = (expression);                                \
    if (dif_flash_status != cudaSuccess)                                       \
      throw std::runtime_error(std::string("FlashAttention CUDA error: ") +    \
                               cudaGetErrorString(dif_flash_status));           \
  } while (false)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
